"""Per-port daily emission ledgers (靠泊排放日账).

An accountant publishes a ledger for one port and one UTC date by submitting
per-vessel emission factors (grams per hour).  Each vessel's accounted
emission is::

    factor[grams/hour] * allocated_seconds / 3600

where ``allocated_seconds`` is the total overlap between the vessel's *current
active* stay events and the half-open UTC day ``[day 00:00, next day 00:00)``.
The per-vessel value is rounded half-up to whole grams; the port total is the
sum of those rounded vessel values (never a re-rounded grand total).

Consistency contract
--------------------
* A publish runs inside one ``BEGIN IMMEDIATE`` transaction: the active-event
  snapshot it measures and the ledger rows it writes commit together.  An
  upload interleaving from another instance either commits entirely before
  the publish (and is measured) or after (and is not); the two never mix.
* Any failure (missing factor, conflict, crash) rolls the whole publish back:
  no half-written ledger is ever visible.
* ``(port_id, delivery_no)`` is unique and checked inside the write
  transaction.  Re-sending the same delivery number with the same input
  returns the original ledger; a different input under a used delivery number
  is rejected with 409.
* Ledgers are immutable evidence: they store the event ids, allocated seconds
  and factors they were computed from.  Late AIS points that retract an event
  never rewrite a published ledger; a new delivery number measures the latest
  events instead.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from decimal import Decimal
from fractions import Fraction
from typing import Any, Iterator

from . import service
from .db import write_transaction
from .errors import AppError

SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86_400


def day_window(ledger_date: date) -> tuple[int, int]:
    """Half-open UTC day ``[start, end)`` as epoch seconds."""
    start = datetime(
        ledger_date.year, ledger_date.month, ledger_date.day,
        tzinfo=timezone.utc,
    )
    day_start = int(start.timestamp())
    return day_start, day_start + SECONDS_PER_DAY


def canonical_factor(value: Decimal) -> str:
    """Canonical plain-decimal text for a submitted factor.

    Numerically equal factors (``1200`` / ``1200.0`` / ``1.2e3``) map to the
    same text, so idempotency compares values, not spellings.
    """
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def round_half_up_grams(factor_text: str, seconds: int) -> int:
    """Exact grams for ``factor * seconds / 3600``, rounded half-up.

    Uses :class:`~fractions.Fraction` so the rounding boundary (x.5) is
    decided exactly, independent of binary/decimal float noise.
    """
    exact = Fraction(factor_text) * seconds / SECONDS_PER_HOUR
    return (2 * exact.numerator + exact.denominator) // (2 * exact.denominator)


def allocate(
    event_rows: list[sqlite3.Row], day_start: int, day_end: int
) -> dict[str, list[dict[str, Any]]]:
    """Group positive-overlap event slices per vessel.

    ``event_rows`` must be the port's active events.  A slice keeps the event
    id and its in-day seconds; events that only touch the day boundary
    contribute nothing.
    """
    per_vessel: dict[str, list[dict[str, Any]]] = {}
    for row in event_rows:
        seconds = min(row["end_time"], day_end) - max(row["start_time"], day_start)
        if seconds <= 0:
            continue
        per_vessel.setdefault(row["vessel_id"], []).append(
            {
                "event_id": row["id"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "allocated_seconds": seconds,
            }
        )
    return per_vessel


def _request_hash(ledger_date: date, factors: list[tuple[str, str]]) -> str:
    canonical = json.dumps(
        {
            "date": ledger_date.isoformat(),
            "factors": sorted([vessel, factor] for vessel, factor in factors),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@contextmanager
def _transaction_if_needed(
    conn: sqlite3.Connection,
) -> Iterator[sqlite3.Connection]:
    """Join an existing write transaction, or start one (see app.service)."""
    if conn.in_transaction:
        yield conn
        return
    with write_transaction(conn):
        yield conn


def publish_ledger(
    conn: sqlite3.Connection,
    *,
    port_code: str,
    delivery_no: str,
    ledger_date: date,
    factors: list[tuple[str, str]],
) -> tuple[dict[str, Any], bool]:
    """Publish a ledger atomically.  Returns ``(ledger, replayed)``.

    ``factors`` holds ``(vessel_id, canonical_factor_text)`` pairs, already
    validated and de-duplicated by the API layer.  The whole measurement and
    all inserts commit in one write transaction; any AppError raised here
    leaves no ledger rows behind.
    """
    request_sha = _request_hash(ledger_date, factors)
    day_start, day_end = day_window(ledger_date)

    with _transaction_if_needed(conn):
        port = service.get_port(conn, port_code)

        existing = conn.execute(
            "SELECT * FROM emission_ledgers WHERE port_id = ? AND delivery_no = ?",
            (port.id, delivery_no),
        ).fetchone()
        if existing is not None:
            if existing["request_sha256"] != request_sha:
                raise AppError(
                    409,
                    "delivery_no_reused",
                    (
                        f"delivery_no {delivery_no!r} was already used for a "
                        "different ledger request"
                    ),
                )
            return _ledger_payload(conn, existing, port_code=port.code), True

        # Only events with a positive overlap can match: end > start for every
        # stored event, so start < day_end and end > day_start <=> overlap > 0.
        event_rows = conn.execute(
            """
            SELECT * FROM events
            WHERE port_id = ? AND start_time < ? AND end_time > ?
            ORDER BY vessel_id, start_time, id
            """,
            (port.id, day_end, day_start),
        ).fetchall()
        allocations = allocate(event_rows, day_start, day_end)

        factor_map = dict(factors)
        missing = sorted(v for v in allocations if v not in factor_map)
        if missing:
            raise AppError(
                422,
                "missing_emission_factor",
                (
                    "vessels with stays on this date are missing an emission "
                    "factor; the whole publish was rejected"
                ),
                {"date": ledger_date.isoformat(), "vessels": missing},
            )

        vessel_lines: list[tuple[str, str, int, int, list[dict[str, Any]]]] = []
        total_grams = 0
        for vessel_id in sorted(allocations):
            slices = allocations[vessel_id]
            factor_text = factor_map[vessel_id]
            seconds = sum(s["allocated_seconds"] for s in slices)
            grams = round_half_up_grams(factor_text, seconds)
            total_grams += grams
            vessel_lines.append((vessel_id, factor_text, seconds, grams, slices))

        cursor = conn.execute(
            """
            INSERT INTO emission_ledgers
                (port_id, delivery_no, ledger_date, day_start, day_end,
                 request_sha256, vessel_count, total_grams, published_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                port.id,
                delivery_no,
                ledger_date.isoformat(),
                day_start,
                day_end,
                request_sha,
                len(vessel_lines),
                total_grams,
                service.utc_now_iso(),
            ),
        )
        ledger_id = int(cursor.lastrowid)
        for vessel_id, factor_text, seconds, grams, slices in vessel_lines:
            conn.execute(
                """
                INSERT INTO emission_ledger_vessels
                    (ledger_id, vessel_id, factor_grams_per_hour,
                     allocated_seconds, emission_grams)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ledger_id, vessel_id, factor_text, seconds, grams),
            )
            for s in slices:
                conn.execute(
                    """
                    INSERT INTO emission_ledger_events
                        (ledger_id, vessel_id, event_id, event_start, event_end,
                         allocated_seconds, factor_grams_per_hour)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ledger_id,
                        vessel_id,
                        s["event_id"],
                        s["start_time"],
                        s["end_time"],
                        s["allocated_seconds"],
                        factor_text,
                    ),
                )
        row = conn.execute(
            "SELECT * FROM emission_ledgers WHERE id = ?", (ledger_id,)
        ).fetchone()
        return _ledger_payload(conn, row, port_code=port.code), False


def get_ledger(
    conn: sqlite3.Connection, *, port_code: str, delivery_no: str
) -> dict[str, Any]:
    port = service.get_port(conn, port_code)
    row = conn.execute(
        "SELECT * FROM emission_ledgers WHERE port_id = ? AND delivery_no = ?",
        (port.id, delivery_no),
    ).fetchone()
    if row is None:
        raise AppError(
            404,
            "ledger_not_found",
            f"no ledger with delivery_no {delivery_no!r} for port {port_code!r}",
        )
    return _ledger_payload(conn, row, port_code=port.code)


def _ledger_payload(
    conn: sqlite3.Connection, ledger_row: sqlite3.Row, *, port_code: str
) -> dict[str, Any]:
    vessel_rows = conn.execute(
        "SELECT * FROM emission_ledger_vessels WHERE ledger_id = ? "
        "ORDER BY vessel_id",
        (ledger_row["id"],),
    ).fetchall()
    event_rows = conn.execute(
        "SELECT * FROM emission_ledger_events WHERE ledger_id = ? "
        "ORDER BY vessel_id, event_start, event_id",
        (ledger_row["id"],),
    ).fetchall()
    events_by_vessel: dict[str, list[dict[str, Any]]] = {}
    for row in event_rows:
        events_by_vessel.setdefault(row["vessel_id"], []).append(
            {
                "event_id": row["event_id"],
                "start_time": row["event_start"],
                "end_time": row["event_end"],
                "allocated_seconds": row["allocated_seconds"],
                "factor_grams_per_hour": row["factor_grams_per_hour"],
            }
        )
    return {
        "ledger_id": ledger_row["id"],
        "port_id": ledger_row["port_id"],
        "port_code": port_code,
        "delivery_no": ledger_row["delivery_no"],
        "date": ledger_row["ledger_date"],
        "day_start": ledger_row["day_start"],
        "day_end": ledger_row["day_end"],
        "vessel_count": ledger_row["vessel_count"],
        "total_grams": ledger_row["total_grams"],
        "published_at": ledger_row["published_at"],
        "vessels": [
            {
                "vessel_id": v["vessel_id"],
                "factor_grams_per_hour": v["factor_grams_per_hour"],
                "allocated_seconds": v["allocated_seconds"],
                "emission_grams": v["emission_grams"],
                "events": events_by_vessel.get(v["vessel_id"], []),
            }
            for v in vessel_rows
        ],
    }
