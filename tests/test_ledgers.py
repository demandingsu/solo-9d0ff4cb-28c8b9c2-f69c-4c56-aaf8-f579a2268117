from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import csv_ingest, db, ledgers, service
from app.config import Settings
from app.main import create_app

PORT = {
    "code": "CNSHA-PD",
    "name": "上海港浦东",
    "min_lat": 31.0,
    "max_lat": 31.4,
    "min_lon": 121.2,
    "max_lon": 121.8,
}
INSIDE = (31.2, 121.5)
DAY1 = "2026-01-01"
DAY2 = "2026-01-02"


def utc(day: int, hour: int, minute: int = 0) -> str:
    return f"2026-01-{day:02d}T{hour:02d}:{minute:02d}:00Z"


def stay(vessel: str, day: int, start_min: int, end_min: int) -> list:
    """In-port points every 10 min from start_min to end_min (inclusive)."""
    return [
        (vessel, utc(day, *divmod(t, 60)), *INSIDE)
        for t in range(start_min, end_min + 1, 10)
    ]


def csv_bytes(rows: list[tuple[str, str, float, float]]) -> bytes:
    lines = ["vessel,time,lat,lon"]
    lines += [f"{v},{t},{lat},{lon}" for v, t, lat, lon in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    app = create_app(Settings(db_path=str(tmp_path / "test.sqlite3")))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def registered(client: TestClient) -> TestClient:
    response = client.post("/ports", json=PORT)
    assert response.status_code == 201, response.text
    return client


def upload(
    client: TestClient,
    rows: list[tuple[str, str, float, float]],
    *,
    key: str,
    port: str = PORT["code"],
):
    return client.post(
        f"/ports/{port}/uploads",
        files={"file": ("ais.csv", csv_bytes(rows), "text/csv")},
        headers={"Idempotency-Key": key},
    )


def publish(
    client: TestClient,
    delivery: str,
    date_str: str,
    factors: list[dict],
    *,
    port: str = PORT["code"],
):
    return client.post(
        f"/ports/{port}/emission-ledgers",
        json={"delivery_no": delivery, "date": date_str, "factors": factors},
    )


def factor(vessel: str, grams_per_hour) -> dict:
    return {"vessel_id": vessel, "grams_per_hour": grams_per_hour}


def get_ledger(client: TestClient, delivery: str, *, port: str = PORT["code"]):
    return client.get(f"/ports/{port}/emission-ledgers/{delivery}")


def table_count(client: TestClient, table: str) -> int:
    db_path = client.app.state.settings.db_path  # type: ignore[attr-defined]
    connection = sqlite3.connect(db_path)
    try:
        return connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        connection.close()


def test_publish_basic_accounting_and_evidence(registered: TestClient) -> None:
    # SHIP-A 00:00-01:00 (3600 s) and SHIP-B 02:00-02:30 (1800 s).
    upload(registered, stay("SHIP-A", 1, 0, 60), key="a")
    upload(registered, stay("SHIP-B", 1, 120, 150), key="b")

    response = publish(
        registered,
        "DEL-1",
        DAY1,
        [factor("SHIP-A", 1200), factor("SHIP-B", 100)],
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["replayed"] is False
    ledger = body["ledger"]
    assert ledger["delivery_no"] == "DEL-1"
    assert ledger["date"] == DAY1
    assert ledger["day_end"] - ledger["day_start"] == 86_400
    assert ledger["vessel_count"] == 2
    assert ledger["total_grams"] == 1250  # 1200*1h + 100*0.5h

    vessels = {v["vessel_id"]: v for v in ledger["vessels"]}
    assert vessels["SHIP-A"]["allocated_seconds"] == 3600
    assert vessels["SHIP-A"]["emission_grams"] == 1200
    assert vessels["SHIP-B"]["allocated_seconds"] == 1800
    assert vessels["SHIP-B"]["emission_grams"] == 50

    # Evidence: event id, allocated seconds and factor are stored per slice.
    current = registered.get(f"/ports/{PORT['code']}/events").json()["events"]
    by_vessel = {e["vessel_id"]: e for e in current}
    line_a = vessels["SHIP-A"]["events"][0]
    assert line_a["event_id"] == by_vessel["SHIP-A"]["event_id"]
    assert line_a["allocated_seconds"] == 3600
    assert line_a["factor_grams_per_hour"] == "1200"
    assert line_a["start_time"] == by_vessel["SHIP-A"]["start_time"]

    # The same ledger is retrievable by delivery number.
    fetched = get_ledger(registered, "DEL-1")
    assert fetched.status_code == 200
    assert fetched.json()["ledger"] == ledger


def test_cross_day_split_and_day_boundaries(registered: TestClient) -> None:
    rows = []
    # SHIP-A stays 23:00 day1 -> 01:00 day2 (spans midnight).
    rows += stay("SHIP-A", 1, 23 * 60, 23 * 60 + 50)
    rows += stay("SHIP-A", 2, 0, 60)
    # SHIP-B stays exactly [00:00, 00:30] on day2: day2 only.
    rows += stay("SHIP-B", 2, 0, 30)
    # SHIP-C stays 23:00 -> 24:00 exactly: day1 only, zero overlap with day2.
    rows += stay("SHIP-C", 1, 23 * 60, 23 * 60 + 50)
    rows += [("SHIP-C", utc(2, 0, 0), *INSIDE)]
    upload(registered, rows, key="cross")

    day1 = publish(
        registered, "DEL-D1", DAY1, [factor("SHIP-A", 3600), factor("SHIP-C", 60)]
    )
    assert day1.status_code == 201, day1.text
    day2 = publish(
        registered, "DEL-D2", DAY2, [factor("SHIP-A", 3600), factor("SHIP-B", 60)]
    )
    assert day2.status_code == 201, day2.text

    vessels1 = {v["vessel_id"]: v for v in day1.json()["ledger"]["vessels"]}
    assert set(vessels1) == {"SHIP-A", "SHIP-C"}  # SHIP-B has no day1 overlap
    assert vessels1["SHIP-A"]["allocated_seconds"] == 3600  # 23:00 -> 24:00
    assert vessels1["SHIP-A"]["emission_grams"] == 3600
    assert vessels1["SHIP-C"]["allocated_seconds"] == 3600
    assert vessels1["SHIP-C"]["emission_grams"] == 60

    vessels2 = {v["vessel_id"]: v for v in day2.json()["ledger"]["vessels"]}
    assert set(vessels2) == {"SHIP-A", "SHIP-B"}  # SHIP-C ends at 00:00 sharp
    assert vessels2["SHIP-A"]["allocated_seconds"] == 3600  # 00:00 -> 01:00
    assert vessels2["SHIP-B"]["allocated_seconds"] == 1800
    assert vessels2["SHIP-B"]["emission_grams"] == 30


def test_rounding_is_per_vessel_then_summed(registered: TestClient) -> None:
    # Each vessel: 1800 s * 1 g/h = 0.5 g -> rounds up to 1 g per vessel.
    # The port total must be the sum of rounded vessel values (2), not the
    # rounded exact grand total (1).
    upload(registered, stay("SHIP-A", 1, 0, 30), key="ra")
    upload(registered, stay("SHIP-B", 1, 120, 150), key="rb")

    ledger = publish(
        registered, "DEL-R", DAY1, [factor("SHIP-A", 1), factor("SHIP-B", 1)]
    ).json()["ledger"]
    vessels = {v["vessel_id"]: v for v in ledger["vessels"]}
    assert vessels["SHIP-A"]["emission_grams"] == 1
    assert vessels["SHIP-B"]["emission_grams"] == 1
    assert ledger["total_grams"] == 2


def test_fractional_factor_exactness(registered: TestClient) -> None:
    upload(registered, stay("SHIP-A", 1, 0, 30), key="frac")
    # 1800 s * 0.1 g/h = 0.05 g -> 0 g (below the half-up boundary).
    low = publish(registered, "DEL-F1", DAY1, [factor("SHIP-A", 0.1)])
    assert low.json()["ledger"]["vessels"][0]["emission_grams"] == 0
    # 1800 s * 3 g/h = 1.5 g -> 2 g (exactly on the boundary, rounds up).
    exact = publish(registered, "DEL-F3", DAY1, [factor("SHIP-A", 3)])
    assert exact.json()["ledger"]["vessels"][0]["emission_grams"] == 2


def test_missing_factor_rejects_whole_publish(registered: TestClient) -> None:
    upload(registered, stay("SHIP-A", 1, 0, 60), key="ma")
    upload(registered, stay("SHIP-B", 1, 120, 150), key="mb")

    response = publish(registered, "DEL-M", DAY1, [factor("SHIP-A", 100)])
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "missing_emission_factor"
    assert error["details"]["vessels"] == ["SHIP-B"]

    # Nothing was persisted: no ledger, no vessel lines, no event slices.
    assert table_count(registered, "emission_ledgers") == 0
    assert table_count(registered, "emission_ledger_vessels") == 0
    assert table_count(registered, "emission_ledger_events") == 0
    assert get_ledger(registered, "DEL-M").status_code == 404

    # The same delivery number stays usable after the rejection.
    fixed = publish(
        registered, "DEL-M", DAY1, [factor("SHIP-A", 100), factor("SHIP-B", 10)]
    )
    assert fixed.status_code == 201


def test_replay_same_delivery_returns_original(registered: TestClient) -> None:
    upload(registered, stay("SHIP-A", 1, 0, 60), key="ia")
    factors = [factor("SHIP-A", 1200)]

    first = publish(registered, "DEL-I", DAY1, factors)
    assert first.status_code == 201
    replay = publish(registered, "DEL-I", DAY1, factors)
    assert replay.status_code == 200
    assert replay.json()["replayed"] is True
    assert replay.json()["ledger"] == first.json()["ledger"]

    # Numerically equal factors spelled differently are the same input.
    respelled = publish(registered, "DEL-I", DAY1, [factor("SHIP-A", 1200.0)])
    assert respelled.status_code == 200
    assert (
        respelled.json()["ledger"]["ledger_id"]
        == first.json()["ledger"]["ledger_id"]
    )

    # Only one ledger exists after the replays.
    assert table_count(registered, "emission_ledgers") == 1


def test_same_delivery_different_content_conflicts(registered: TestClient) -> None:
    upload(registered, stay("SHIP-A", 1, 0, 60), key="ca")
    assert publish(registered, "DEL-C", DAY1, [factor("SHIP-A", 100)]).status_code == 201

    different_factor = publish(registered, "DEL-C", DAY1, [factor("SHIP-A", 200)])
    assert different_factor.status_code == 409
    assert different_factor.json()["error"]["code"] == "delivery_no_reused"

    different_date = publish(registered, "DEL-C", DAY2, [factor("SHIP-A", 100)])
    assert different_date.status_code == 409

    # The original ledger is untouched.
    assert get_ledger(registered, "DEL-C").json()["ledger"]["total_grams"] == 100


def test_late_points_leave_old_ledger_untouched(registered: TestClient) -> None:
    # Baseline: SHIP-A 00:00 -> 01:00 (3600 s in day1).
    upload(registered, stay("SHIP-A", 1, 0, 60), key="la")
    first = publish(registered, "DEL-L1", DAY1, [factor("SHIP-A", 60)])
    old_ledger = first.json()["ledger"]
    old_event_id = old_ledger["vessels"][0]["events"][0]["event_id"]
    assert old_ledger["vessels"][0]["allocated_seconds"] == 3600
    assert old_ledger["total_grams"] == 60

    # Late points extend the stay to 01:20; the old event is retracted.
    late = [("SHIP-A", utc(1, 1, 10), *INSIDE), ("SHIP-A", utc(1, 1, 20), *INSIDE)]
    assert upload(registered, late, key="late").status_code == 201
    current = registered.get(f"/ports/{PORT['code']}/events").json()["events"]
    assert len(current) == 1
    assert current[0]["event_id"] != old_event_id
    assert current[0]["duration_seconds"] == 4800

    # The old ledger keeps its original evidence (retracted event id included).
    fetched = get_ledger(registered, "DEL-L1")
    assert fetched.json()["ledger"] == old_ledger
    evidence = fetched.json()["ledger"]["vessels"][0]["events"][0]
    assert evidence["event_id"] == old_event_id

    # A new delivery number measures the latest events.
    second = publish(registered, "DEL-L2", DAY1, [factor("SHIP-A", 60)])
    new_ledger = second.json()["ledger"]
    assert new_ledger["vessels"][0]["allocated_seconds"] == 4800
    assert new_ledger["total_grams"] == 80
    assert (
        new_ledger["vessels"][0]["events"][0]["event_id"]
        == current[0]["event_id"]
    )


def test_zero_activity_day_publishes_empty_ledger(registered: TestClient) -> None:
    upload(registered, stay("SHIP-A", 1, 0, 60), key="za")
    response = publish(registered, "DEL-Z", DAY2, [factor("SHIP-A", 100)])
    assert response.status_code == 201
    ledger = response.json()["ledger"]
    assert ledger["vessel_count"] == 0
    assert ledger["total_grams"] == 0
    assert ledger["vessels"] == []


def test_request_validation(registered: TestClient) -> None:
    upload(registered, stay("SHIP-A", 1, 0, 60), key="va")

    duplicate = publish(
        registered, "DEL-V1", DAY1, [factor("SHIP-A", 1), factor("SHIP-A", 2)]
    )
    assert duplicate.status_code == 422

    negative = publish(registered, "DEL-V2", DAY1, [factor("SHIP-A", -1)])
    assert negative.status_code == 422

    bad_date = publish(registered, "DEL-V3", "2026-02-30", [factor("SHIP-A", 1)])
    assert bad_date.status_code == 422

    bad_delivery = publish(registered, "DEL/1", DAY1, [factor("SHIP-A", 1)])
    assert bad_delivery.status_code == 422

    empty_factors = registered.post(
        f"/ports/{PORT['code']}/emission-ledgers",
        json={"delivery_no": "DEL-V4", "date": DAY1, "factors": []},
    )
    assert empty_factors.status_code == 422

    # None of the invalid requests left a ledger behind.
    assert table_count(registered, "emission_ledgers") == 0


def test_unknown_port_and_ledger_are_404(registered: TestClient) -> None:
    assert (
        publish(registered, "DEL-X", DAY1, [factor("S", 1)], port="NOPE").status_code
        == 404
    )
    assert get_ledger(registered, "DEL-X").status_code == 404
    assert get_ledger(registered, "DEL-X", port="NOPE").status_code == 404


def test_delivery_numbers_are_scoped_per_port(tmp_path: Path) -> None:
    app = create_app(Settings(db_path=str(tmp_path / "ports.sqlite3")))
    with TestClient(app) as client:
        for code, shift in (("PORT-A", 0.0), ("PORT-B", 5.0)):
            assert client.post(
                "/ports",
                json={
                    **PORT,
                    "code": code,
                    "min_lat": PORT["min_lat"] + shift,
                    "max_lat": PORT["max_lat"] + shift,
                },
            ).status_code == 201
        rows_a = [
            ("SHIP-A", utc(1, 0, m), INSIDE[0], INSIDE[1]) for m in (0, 10, 20, 30)
        ]
        rows_b = [
            ("SHIP-A", utc(1, 0, m), INSIDE[0] + 5.0, INSIDE[1])
            for m in (0, 10, 20, 30)
        ]
        upload(client, rows_a, key="pa", port="PORT-A")
        upload(client, rows_b, key="pb", port="PORT-B")

        # Same delivery number in two ports: independent ledgers, no conflict.
        one = publish(client, "DEL-SHARED", DAY1, [factor("SHIP-A", 60)], port="PORT-A")
        two = publish(client, "DEL-SHARED", DAY1, [factor("SHIP-A", 120)], port="PORT-B")
        assert one.status_code == 201 and two.status_code == 201
        assert one.json()["ledger"]["total_grams"] == 30
        assert two.json()["ledger"]["total_grams"] == 60
        assert (
            get_ledger(client, "DEL-SHARED", port="PORT-A").json()["ledger"]["total_grams"]
            == 30
        )
        assert (
            get_ledger(client, "DEL-SHARED", port="PORT-B").json()["ledger"]["total_grams"]
            == 60
        )


def test_interleaved_upload_and_publish_never_mix_event_states(
    tmp_path: Path,
) -> None:
    """An upload and a publish on shared storage commit in some order.

    The ledger must reflect exactly one consistent event state: either the
    pre-upload event (id E1, 3600 s) or the post-upload event (id E2,
    4800 s) -- never a mix such as E1 with 4800 s.
    """
    path = str(tmp_path / "race.sqlite3")
    db.initialise(path)
    setup = db.connect(path)
    try:
        service.register_port(
            setup,
            code="P",
            name="race",
            min_lat_e6=31_000_000,
            max_lat_e6=31_400_000,
            min_lon_e6=121_200_000,
            max_lon_e6=121_800_000,
        )
        base_rows = [
            csv_ingest.RawPointRow(
                i, "SHIP-A", 1_800_000_000 + i * 600, 31_200_000, 121_500_000
            )
            for i in range(7)  # 3600 s stay
        ]
        service.ingest(
            setup, port_code="P", upload_key="base", content=b"base", rows=base_rows
        )
        old_event_id = service.list_events(setup, port_code="P")[0][0]["event_id"]
    finally:
        setup.close()

    ledger_date = datetime.fromtimestamp(1_800_000_000, timezone.utc).date()
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []
    published: list[dict] = []

    def uploader() -> None:
        conn = db.connect(path)
        try:
            late = [
                csv_ingest.RawPointRow(
                    10 + i,
                    "SHIP-A",
                    1_800_000_000 + (7 + i) * 600,
                    31_200_000,
                    121_500_000,
                )
                for i in range(2)  # extends the stay to 4800 s
            ]
            barrier.wait(timeout=10)
            service.ingest(
                conn, port_code="P", upload_key="late", content=b"late", rows=late
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            conn.close()

    def publisher() -> None:
        conn = db.connect(path)
        try:
            barrier.wait(timeout=10)
            ledger, _ = ledgers.publish_ledger(
                conn,
                port_code="P",
                delivery_no="DEL-RACE",
                ledger_date=ledger_date,
                factors=[("SHIP-A", "60")],
            )
            published.append(ledger)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            conn.close()

    threads = [threading.Thread(target=uploader), threading.Thread(target=publisher)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    assert len(published) == 1
    vessel = published[0]["vessels"][0]
    line = vessel["events"][0]
    if line["event_id"] == old_event_id:
        # Published before the upload committed: old event, 3600 s, 60 g.
        assert vessel["allocated_seconds"] == 3600
        assert published[0]["total_grams"] == 60
    else:
        # Published after the upload committed: new event, 4800 s, 80 g.
        assert vessel["allocated_seconds"] == 4800
        assert published[0]["total_grams"] == 80
        assert line["event_id"] != old_event_id


def test_crash_mid_publish_leaves_no_partial_ledger(
    registered: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    upload(registered, stay("SHIP-A", 1, 0, 60), key="ca")

    def boom(*args, **kwargs):
        raise RuntimeError("simulated crash after ledger inserts")

    monkeypatch.setattr(ledgers, "_ledger_payload", boom)
    with pytest.raises(RuntimeError):
        publish(registered, "DEL-CRASH", DAY1, [factor("SHIP-A", 60)])

    assert table_count(registered, "emission_ledgers") == 0
    assert table_count(registered, "emission_ledger_vessels") == 0
    assert table_count(registered, "emission_ledger_events") == 0
    assert get_ledger(registered, "DEL-CRASH").status_code == 404


def test_two_instances_share_ledgers_and_idempotency(tmp_path: Path) -> None:
    """Publish via one instance, re-query and replay through the other."""
    db_path = str(tmp_path / "instances.sqlite3")
    settings = Settings(db_path=db_path)
    app_one, app_two = create_app(settings), create_app(settings)
    with TestClient(app_one) as one, TestClient(app_two) as two:
        for client in (one, two):
            assert client.post("/ports", json=PORT).status_code in (200, 201)
        upload(one, stay("SHIP-A", 1, 0, 60), key="shared-upload")

        first = publish(one, "DEL-X1", DAY1, [factor("SHIP-A", 60)])
        assert first.status_code == 201

        # The other instance reads the same committed ledger.
        fetched = get_ledger(two, "DEL-X1")
        assert fetched.status_code == 200
        assert fetched.json()["ledger"] == first.json()["ledger"]

        # Replay through the other instance returns the original ledger.
        replay = publish(two, "DEL-X1", DAY1, [factor("SHIP-A", 60)])
        assert replay.status_code == 200
        assert (
            replay.json()["ledger"]["ledger_id"]
            == first.json()["ledger"]["ledger_id"]
        )

        # And a conflicting reuse is rejected there as well.
        conflict = publish(two, "DEL-X1", DAY1, [factor("SHIP-A", 61)])
        assert conflict.status_code == 409
