"""Reusable sanity checks for normalized SEC ownership data.

The module follows the project's "flag, never silently drop" convention. It
provides two checks:

* ownership-position continuity across rows and filings, with optional split
  factors; and
* reported transaction price against exact-day, adjacent-trading-day, and
  weekly market ranges.

Both functions return one result row for every input row and accept the shared
canonical names plus compatible historical Form 4 column names.
"""

from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


CONTINUITY_OUTPUT_COLUMNS = [
    "input_row_number", "issuer_cik", "issuer_ticker", "owner_cik", "owner_name",
    "security_key", "security_title", "table", "direct_or_indirect",
    "nature_of_ownership", "event_date", "filing_date", "accession", "form",
    "sequence_in_filing", "previous_accession", "previous_event_date",
    "previous_reported_position", "shares_acquired", "shares_disposed",
    "net_transaction_shares", "split_adjustment_factor",
    "expected_position_after", "reported_position_after", "implied_position_before",
    "position_difference", "continuity_status", "continuity_reason",
    "position_state_updated",
    "economic_row_id", "owner_row_id", "owner_attribution_status", "amendment_status",
]

PRICE_OUTPUT_COLUMNS = [
    "input_row_number", "issuer_cik", "issuer_ticker", "owner_cik", "owner_name",
    "accession", "form", "transaction_date", "transaction_code", "security_title",
    "reported_price", "exact_date", "exact_low", "exact_high", "exact_close",
    "adjacent_start", "adjacent_end", "adjacent_low", "adjacent_high",
    "week_start", "week_end", "weekly_low", "weekly_high", "matched_window",
    "distance_from_exact_close", "distance_from_nearest_boundary",
    "price_check_status", "price_check_reason",
    "economic_row_id", "owner_row_id", "reported_transaction_value",
    "derived_transaction_value", "transaction_value_status",
]


def _present(value) -> bool:
    """Return True when a value is neither None, NaN nor blank text."""
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip() != ""


def _value(row: dict, *names: str, default=""):
    """Return the first non-blank value among ``names`` in a row, else ``default``."""
    for name in names:
        value = row.get(name)
        if _present(value):
            return value
    return default


def _decimal(value) -> Optional[Decimal]:
    """Parse a value such as ``"$1,250.50"`` to an exact Decimal; None if missing or invalid."""
    if not _present(value):
        return None
    cleaned = str(value).strip().replace(",", "").replace("$", "")
    try:
        number = Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


def _number(value: Optional[Decimal]):
    """Convert a Decimal to float, or pd.NA when missing."""
    return float(value) if value is not None else pd.NA


def _date(value) -> pd.Timestamp:
    """Parse a value to a timezone-naive date at midnight; NaT if invalid."""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    timestamp = pd.Timestamp(parsed)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.normalize()


def _date_text(value) -> str:
    """Return a value's date as ``YYYY-MM-DD``, or ``""`` when it is not a date."""
    parsed = _date(value)
    return "" if pd.isna(parsed) else parsed.date().isoformat()


def _truthy(value) -> bool:
    """Interpret flags such as ``1``, ``true``, ``yes`` or ``y`` as True."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _normalized_text(value) -> str:
    """Lower-case text and collapse whitespace; missing values become ``""``."""
    return re.sub(r"\s+", " ", str(value if _present(value) else "").strip().lower())


def _normalized_security(value) -> str:
    """Turn a security title into a stable key (``common_stock`` for common shares)."""
    compact = _normalized_text(value)
    if compact in {"common stock", "common shares"}:
        return "common_stock"
    return re.sub(r"[^a-z0-9]+", "_", compact).strip("_") or "unknown_security"


def _owner_key(row: dict) -> str:
    """Identify the reporting owner by CIK, or by normalized name when the CIK is missing."""
    cik = str(_value(row, "owner_cik")).strip().lstrip("0")
    return f"cik:{cik}" if cik else f"name:{_normalized_text(_value(row, 'owner_name'))}"


def _issuer_key(row: dict) -> str:
    """Identify the issuer by CIK, or by ticker when the CIK is missing."""
    cik = str(_value(row, "issuer_cik")).strip().lstrip("0")
    ticker = _normalized_text(_value(row, "issuer_ticker", "ticker"))
    return f"cik:{cik}" if cik else f"ticker:{ticker}"


def _security_key(row: dict) -> tuple[str, ...]:
    """Return the ownership bucket a row's position belongs to.

    Issuer, owner, table, security, direct/indirect and nature of ownership;
    derivative rows also include exercise price, dates and underlying security
    so economically different instruments are never combined.
    """
    table = _normalized_text(_value(row, "table", "record_kind"))
    base = [
        _issuer_key(row),
        _owner_key(row),
        table,
        _normalized_security(_value(row, "security_title")),
        _normalized_text(_value(row, "direct_or_indirect")),
        _normalized_text(_value(row, "nature_of_ownership")),
    ]
    if table == "derivative":
        base.extend(
            [
                _normalized_text(_value(row, "conversion_or_exercise_price")),
                _date_text(_value(row, "exercise_date")),
                _date_text(_value(row, "expiration_date")),
                _normalized_security(_value(row, "underlying_security_title")),
            ]
        )
    return tuple(base)


def _security_key_text(row: dict) -> str:
    """Return the security part of the bucket key (without issuer and owner) as text."""
    return "|".join(_security_key(row)[2:])


def _event_date(row: dict) -> pd.Timestamp:
    """Return the row's date: transaction date, else report date, else filing date."""
    return _date(
        _value(
            row,
            "transaction_date",
            "period_of_report",
            "report_date",
            "edgar_filing_date",
            "filing_date",
        )
    )


def _row_kind(row: dict) -> str:
    """Return shared row kind, including the historical prototype schema."""
    kind = _normalized_text(_value(row, "row_kind"))
    table = _normalized_text(_value(row, "table", "record_kind"))
    if not kind and table in {"non_derivative", "derivative"}:
        # The earlier prototype stores the table in record_kind and writes
        # transactions and holdings to separate CSVs, so transaction rows have
        # no explicit row_kind column.
        return "transaction"
    return kind


def _filing_date(row: dict) -> pd.Timestamp:
    """Return the EDGAR filing date of a row."""
    return _date(_value(row, "edgar_filing_date", "filing_date"))


def _accession(row: dict) -> str:
    """Return the accession number under any of its column spellings."""
    return str(_value(row, "accession", "accession_number", "accessionNumber"))


def _sequence(row: dict, fallback: int) -> int:
    """Return the row's order within its filing, or ``fallback`` when missing."""
    value = _decimal(_value(row, "sequence_in_filing"))
    return int(value) if value is not None else fallback


def _split_records(split_adjustments: Optional[pd.DataFrame]) -> list[dict]:
    """Validate split adjustments and parse their dates and factors.

    Every row needs an ``effective_date`` and a positive ``split_factor``.
    """
    if split_adjustments is None or split_adjustments.empty:
        return []
    required = {"effective_date", "split_factor"}
    missing = required.difference(split_adjustments.columns)
    if missing:
        raise ValueError(f"Split adjustments are missing columns: {', '.join(sorted(missing))}")
    output = []
    for raw in split_adjustments.to_dict("records"):
        effective = _date(raw.get("effective_date"))
        factor = _decimal(raw.get("split_factor"))
        if pd.isna(effective) or factor is None or factor <= 0:
            raise ValueError("Every split adjustment needs a valid effective_date and positive split_factor")
        output.append({**raw, "_effective_date": effective, "_factor": factor})
    return output


def _matches_adjustment(row: dict, adjustment: dict) -> bool:
    """Return True when a split adjustment applies to the row's security.

    Matches on PERMNO when the adjustment has one, else on issuer CIK, else on
    ticker; an adjustment with no identifier applies to every row.
    """
    # A security-level factor must never spill into another share class.
    if _present(adjustment.get("permno")):
        return _decimal(row.get("permno")) == _decimal(adjustment.get("permno"))
    adjustment_cik = str(adjustment.get("issuer_cik", "")).strip().lstrip("0")
    row_cik = str(_value(row, "issuer_cik")).strip().lstrip("0")
    if adjustment_cik:
        return bool(row_cik) and adjustment_cik == row_cik
    adjustment_ticker = _normalized_text(adjustment.get("issuer_ticker", adjustment.get("ticker", "")))
    row_ticker = _normalized_text(_value(row, "issuer_ticker", "ticker"))
    return not adjustment_ticker or adjustment_ticker == row_ticker


def _split_factor_between(
    adjustments: list[dict], previous_date: pd.Timestamp, current_date: pd.Timestamp, row: dict
) -> Decimal:
    """Multiply the split factors effective after ``previous_date`` up to ``current_date``."""
    factor = Decimal("1")
    if pd.isna(previous_date) or pd.isna(current_date):
        return factor
    for adjustment in adjustments:
        if (
            previous_date < adjustment["_effective_date"] <= current_date
            and _matches_adjustment(row, adjustment)
        ):
            factor *= adjustment["_factor"]
    return factor


def _available_at(row: dict) -> pd.Timestamp:
    """Return when the filing became public, as a UTC timestamp.

    Uses the EDGAR acceptance time (New York time when no zone is given). With
    only a filing date, uses the end of that day in New York.
    """
    value = _value(row, "acceptance_datetime")
    if value:
        stamp = pd.to_datetime(value, errors="coerce")
        if pd.isna(stamp):
            return pd.NaT
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("America/New_York")
        return stamp.tz_convert("UTC")
    day = _filing_date(row)
    if pd.isna(day):
        return pd.NaT
    # A date alone gives no intraday availability. Use local end-of-day.
    return (day + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)).tz_localize(
        "America/New_York").tz_convert("UTC")


def resolve_amendments(rows: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """Annotate a versioned snapshot without guessing amendment replacements.

    Only ``amends_owner_row_id`` is an automatic row-level linkage. The linked
    original must be earlier, uniquely identified, and in the same owner/security
    bucket. A date-of-original-submission or shared accession alone is NOT a row
    link. Every row remains in output. ``record_active`` is suitable only for the
    requested snapshot; historical signals must request their own ``as_of``.
    """
    result = rows.copy().reset_index(drop=True)
    result["amendment_status"] = "original"
    result["record_active"] = False
    result["superseded_by_owner_row_id"] = ""
    if result.empty:
        return result
    cutoff = pd.Timestamp(as_of) if as_of is not None else None
    if cutoff is not None:
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    records = result.to_dict("records")
    ids = {}
    for index, row in enumerate(records):
        row_id = _value(row, "owner_row_id")
        if row_id:
            ids.setdefault(str(row_id), []).append(index)
    available = [_available_at(row) for row in records]
    ordering = sorted(range(len(records)), key=lambda i: (
        available[i] if not pd.isna(available[i]) else pd.Timestamp.max.tz_localize("UTC"), i))
    for index in ordering:
        row = records[index]
        is_amendment = _truthy(row.get("is_amendment")) or "/" in str(_value(row, "form", "document_type"))
        if pd.isna(available[index]) or (cutoff is not None and available[index] > cutoff):
            result.at[index, "amendment_status"] = "not_available"
            continue
        if not is_amendment:
            result.at[index, "record_active"] = True
            continue
        result.at[index, "amendment_status"] = "unresolved_quarantined"
        target_id = str(_value(row, "amends_owner_row_id"))
        candidates = ids.get(target_id, [])
        if len(candidates) != 1 or not _present(row.get("owner_row_id")):
            continue
        target = candidates[0]
        if (pd.isna(available[target]) or available[target] >= available[index]
                or _security_key(records[target]) != _security_key(row)
                or not result.at[target, "record_active"]):
            continue
        if _value(row, "owner_attribution_status") in {
            "joint_owner_unallocated",
            "legacy_joint_owner_unresolved",
            "unknown_owner",
        }:
            continue
        result.at[index, "amendment_status"] = "explicit_row_link"
        result.at[index, "record_active"] = True
        result.at[target, "record_active"] = False
        result.at[target, "superseded_by_owner_row_id"] = str(row["owner_row_id"])
    return result


def check_ownership_continuity(
    rows: pd.DataFrame,
    split_adjustments: Optional[pd.DataFrame] = None,
    tolerance: float = 1e-6,
    *,
    as_of=None,
) -> pd.DataFrame:
    """Check reported ownership balances across normalized filing rows.

    Unlinked amendments are reported as ``SKIP_AMENDMENT`` and do not update the
    running position. Explicit row-linked corrections only reset their directly
    preceding state from the correction's availability time. Earlier outputs are
    never rewritten. Missing-value rows are retained and, when a reported ending
    position exists, establish a new baseline for subsequent checks.

    ``split_adjustments`` is optional and uses one row per effective event. A
    2-for-1 split is represented by ``split_factor=2``. Required columns are
    ``effective_date`` and ``split_factor``; ``issuer_cik`` or ``ticker`` may be
    supplied to scope the event.
    """
    if rows.empty:
        return pd.DataFrame(columns=CONTINUITY_OUTPUT_COLUMNS)

    adjustments = _split_records(split_adjustments)
    linked_rows = resolve_amendments(rows, as_of=as_of)
    tolerance_decimal = Decimal(str(tolerance))
    records = []
    for input_order, raw in enumerate(linked_rows.to_dict("records"), start=1):
        record = dict(raw)
        record["_input_order"] = input_order
        record["_event_date"] = _event_date(record)
        record["_filing_date"] = _filing_date(record)
        record["_available_at"] = _available_at(record)
        record["_sequence"] = _sequence(record, input_order)
        record["_group_key"] = _security_key(record)
        records.append(record)

    records.sort(
        key=lambda row: (
            row["_group_key"],
            pd.Timestamp.max.tz_localize("UTC") if pd.isna(row["_available_at"]) else row["_available_at"],
            _accession(row),
            pd.Timestamp.max if pd.isna(row["_event_date"]) else row["_event_date"],
            row["_sequence"],
            row["_input_order"],
        )
    )

    state: dict[tuple[str, ...], dict] = {}
    output = []
    for row in records:
        group_key = row["_group_key"]
        previous = state.get(group_key)
        event_date = row["_event_date"]
        filing_date = row["_filing_date"]
        accession = _accession(row)
        form = str(_value(row, "form", "document_type"))
        row_kind = _row_kind(row)
        direction = str(_value(row, "acquired_disposed")).strip().upper()
        shares = _decimal(_value(row, "shares", "transaction_shares"))
        reported = _decimal(
            _value(row, "shares_owned_following", "shares_owned_after", "shares_owned")
        )
        is_amendment = _truthy(row.get("is_amendment")) or "/" in form

        valid_transaction = (
            row_kind == "transaction" and shares is not None and shares >= 0 and direction in {"A", "D"}
        )
        acquired = shares if valid_transaction and direction == "A" else Decimal("0")
        disposed = shares if valid_transaction and direction == "D" else Decimal("0")
        net = acquired - disposed if valid_transaction else (Decimal("0") if row_kind == "holding" else None)
        factor = Decimal("1")
        expected = implied = difference = None
        status = ""
        reason = ""
        update_state = False

        owner_status = str(_value(row, "owner_attribution_status"))
        if row["amendment_status"] == "not_available" and as_of is not None:
            status = "NOT_AVAILABLE_AS_OF"
            reason = "Filing was not available at the requested observation time."
        elif owner_status in {
            "joint_owner_unallocated",
            "legacy_joint_owner_unresolved",
            "unknown_owner",
        } or "|" in str(_value(row, "owner_cik")):
            status = "NOT_TESTABLE_OWNER_ATTRIBUTION"
            reason = "Joint or missing owner allocation cannot establish an individual position."
        elif is_amendment and row["amendment_status"] == "explicit_row_link":
            if previous and previous["owner_row_id"] == _value(row, "amends_owner_row_id") and reported is not None:
                status = "AMENDMENT_BASELINE_RESET"
                reason = "Explicit correction updates its immediately preceding row from amendment availability onward."
                update_state = True
            else:
                status = "REVIEW_AMENDMENT_INTERVENING_STATE"
                reason = "Explicit row link exists, but later positions or a missing balance prevent safe state replacement."
        elif is_amendment:
            status = "SKIP_AMENDMENT"
            reason = "Amendments are retained but excluded from the running position by default."
        elif pd.isna(event_date):
            status = "NOT_TESTABLE_MISSING_DATE"
            reason = "No transaction, report, or filing date is available."
        elif row_kind not in {"transaction", "holding"}:
            status = "NOT_TESTABLE_ROW_KIND"
            reason = f"Row kind {row_kind or '<blank>'} is not a transaction or holding."
        elif previous is not None and event_date < previous["event_date"]:
            status = "NOT_TESTABLE_LATE_DISCLOSURE"
            reason = "A newly available historical event cannot overwrite a more recent position."
        elif reported is None:
            status = "NOT_TESTABLE_MISSING_POSITION"
            reason = "The reported position after the event is missing or non-numeric."
        elif row_kind == "transaction" and (shares is None or shares < 0 or direction not in {"A", "D"}):
            status = "RESET_BASELINE_MISSING_TRANSACTION"
            reason = "Transaction quantity or acquisition/disposition direction is missing; reported position becomes a new baseline."
            update_state = True
        elif previous is None:
            status = "BASELINE_NO_PRIOR" if row_kind == "transaction" else "BASELINE_HOLDING"
            reason = "No earlier comparable reported position is available."
            implied = reported - net
            update_state = True
        else:
            factor = _split_factor_between(adjustments, previous["event_date"], event_date, row)
            expected = previous["position"] * factor + net
            implied = (reported - net) / factor
            difference = reported - expected
            if abs(difference) <= tolerance_decimal:
                difference = Decimal("0")
                status = "PASS"
                reason = "Reported position reconciles with the previous position and intervening transaction."
            else:
                status = "FLAG_REVIEW" if factor != 1 else "FLAG_REVIEW_UNADJUSTED"
                reason = (
                    "Reported position does not reconcile after supplied split adjustments."
                    if factor != 1
                    else "Reported position does not reconcile; no split adjustment was applied."
                )
            update_state = True

        if update_state and reported is not None:
            state[group_key] = {
                "position": reported,
                "event_date": event_date,
                "accession": accession,
                "owner_row_id": _value(row, "owner_row_id"),
            }

        output.append(
            {
                "input_row_number": row["_input_order"],
                "issuer_cik": _value(row, "issuer_cik"),
                "issuer_ticker": _value(row, "issuer_ticker", "ticker"),
                "owner_cik": _value(row, "owner_cik"),
                "owner_name": _value(row, "owner_name"),
                "security_key": _security_key_text(row),
                "security_title": _value(row, "security_title"),
                "table": _value(row, "table", "record_kind"),
                "direct_or_indirect": _value(row, "direct_or_indirect"),
                "nature_of_ownership": _value(row, "nature_of_ownership"),
                "event_date": _date_text(event_date),
                "filing_date": _date_text(filing_date),
                "accession": accession,
                "form": form,
                "sequence_in_filing": row["_sequence"],
                "previous_accession": previous["accession"] if previous else "",
                "previous_event_date": _date_text(previous["event_date"]) if previous else "",
                "previous_reported_position": _number(previous["position"]) if previous else pd.NA,
                "shares_acquired": _number(acquired if shares is not None else None),
                "shares_disposed": _number(disposed if shares is not None else None),
                "net_transaction_shares": _number(net),
                "split_adjustment_factor": _number(factor),
                "expected_position_after": _number(expected),
                "reported_position_after": _number(reported),
                "implied_position_before": _number(implied),
                "position_difference": _number(difference),
                "continuity_status": status,
                "continuity_reason": reason,
                "position_state_updated": update_state,
                "economic_row_id": _value(row, "economic_row_id"),
                "owner_row_id": _value(row, "owner_row_id"),
                "owner_attribution_status": owner_status,
                "amendment_status": row["amendment_status"],
            }
        )

    result = pd.DataFrame(output, columns=CONTINUITY_OUTPUT_COLUMNS)
    return result.sort_values("input_row_number").reset_index(drop=True)


def _resolve_column(frame: pd.DataFrame, explicit: Optional[str], candidates: Iterable[str], label: str) -> str:
    """Return ``explicit`` if given, else the first of ``candidates`` present in ``frame``."""
    if explicit:
        if explicit not in frame.columns:
            raise ValueError(f"{label} column {explicit!r} is not present")
        return explicit
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"Could not identify {label} column; tried: {', '.join(candidates)}")


def _range_values(frame: pd.DataFrame) -> tuple:
    """Return the lowest low and highest high over the rows of a price window."""
    if frame.empty:
        return (pd.NA, pd.NA)
    low = frame["_low"].min(skipna=True)
    high = frame["_high"].max(skipna=True)
    return (pd.NA if pd.isna(low) else float(low), pd.NA if pd.isna(high) else float(high))


def _inside(price: Decimal, low, high) -> bool:
    """Return True when ``price`` lies within [low, high] (both must be known)."""
    return _present(low) and _present(high) and Decimal(str(low)) <= price <= Decimal(str(high))


def _boundary_distance(price: Decimal, low, high):
    """Return how far ``price`` lies outside [low, high] (0 inside, NA if unknown)."""
    if not _present(low) or not _present(high):
        return pd.NA
    low_d, high_d = Decimal(str(low)), Decimal(str(high))
    if price < low_d:
        return float(price - low_d)
    if price > high_d:
        return float(price - high_d)
    return 0.0


def check_transaction_prices(
    transactions: pd.DataFrame,
    market_data: Optional[pd.DataFrame] = None,
    *,
    transaction_security_col: Optional[str] = None,
    market_security_col: Optional[str] = None,
    market_date_col: Optional[str] = None,
    market_low_col: Optional[str] = None,
    market_high_col: Optional[str] = None,
    market_close_col: Optional[str] = None,
    comparable_codes: Iterable[str] = ("P", "S"),
    max_adjacent_calendar_days: int = 7,
) -> pd.DataFrame:
    """Compare reported prices with exact, adjacent, and weekly market ranges.

    Adjacent range means the nearest previous trading day through the nearest
    next trading day, including the exact date when present. Weekly range uses
    the Monday-Sunday calendar week containing the transaction date.
    """
    if transactions.empty:
        return pd.DataFrame(columns=PRICE_OUTPUT_COLUMNS)
    if max_adjacent_calendar_days < 0:
        raise ValueError("max_adjacent_calendar_days must be nonnegative")
    if market_data is None or market_data.empty:
        market = pd.DataFrame(columns=["_security", "_date", "_low", "_high", "_close"])
    else:
        market_security = _resolve_column(
            market_data, market_security_col, ("permno", "PERMNO", "ticker", "Ticker", "crsp_ticker"), "market security"
        )
        market_date = _resolve_column(
            market_data, market_date_col, ("date", "crsp_date", "DlyCalDt", "dlycaldt"), "market date"
        )
        market_low = _resolve_column(
            market_data, market_low_col, ("low", "crsp_low", "DlyLow", "dlylow", "bidlo"), "market low"
        )
        market_high = _resolve_column(
            market_data, market_high_col, ("high", "crsp_high", "DlyHigh", "dlyhigh", "askhi"), "market high"
        )
        close_candidates = ("close", "crsp_close", "DlyClose", "dlyclose", "prc", "DlyPrc")
        market_close = market_close_col
        if not market_close:
            market_close = next((name for name in close_candidates if name in market_data.columns), None)
        market = pd.DataFrame(
            {
                "_security": market_data[market_security].map(_normalized_text),
                "_date": pd.to_datetime(market_data[market_date], errors="coerce").dt.normalize(),
                "_low": pd.to_numeric(market_data[market_low], errors="coerce").abs(),
                "_high": pd.to_numeric(market_data[market_high], errors="coerce").abs(),
                "_close": (
                    pd.to_numeric(market_data[market_close], errors="coerce").abs()
                    if market_close
                    else pd.Series(pd.NA, index=market_data.index, dtype="Float64")
                ),
            }
        ).dropna(subset=["_security", "_date"])
        if market.duplicated(["_security", "_date"]).any():
            raise ValueError("Market data contain duplicate security/date observations")
        market = market.sort_values(["_security", "_date"]).reset_index(drop=True)

    tx_security = _resolve_column(
        transactions,
        transaction_security_col,
        ("permno", "PERMNO", "issuer_ticker", "ticker", "issuer_cik"),
        "transaction security",
    )
    comparable = {str(code).upper() for code in comparable_codes}
    market_groups = {key: group.reset_index(drop=True) for key, group in market.groupby("_security", sort=False)}
    output = []

    for input_order, row in enumerate(transactions.to_dict("records"), start=1):
        security = _normalized_text(row.get(tx_security, ""))
        tx_date = _date(_value(row, "transaction_date"))
        code = str(_value(row, "transaction_code")).strip().upper()
        price = _decimal(_value(row, "price_per_share", "transaction_price_per_share"))
        row_kind = _row_kind(row)
        table = _normalized_text(_value(row, "table", "record_kind"))

        exact = pd.DataFrame(columns=market.columns)
        adjacent = pd.DataFrame(columns=market.columns)
        weekly = pd.DataFrame(columns=market.columns)
        week_start = week_end = pd.NaT
        group = market_groups.get(security)
        if group is not None and not pd.isna(tx_date):
            exact = group.loc[group["_date"].eq(tx_date)]
            span = pd.Timedelta(days=max_adjacent_calendar_days)
            before = group.loc[group["_date"].between(tx_date - span, tx_date, inclusive="left")].tail(1)
            after = group.loc[group["_date"].between(tx_date, tx_date + span, inclusive="right")].head(1)
            adjacent = pd.concat([before, exact, after], ignore_index=True).drop_duplicates("_date")
            week_start = tx_date - pd.Timedelta(days=tx_date.weekday())
            week_end = week_start + pd.Timedelta(days=6)
            weekly = group.loc[group["_date"].between(week_start, week_end)]

        exact_low, exact_high = _range_values(exact)
        adjacent_low, adjacent_high = _range_values(adjacent)
        weekly_low, weekly_high = _range_values(weekly)
        exact_close = exact["_close"].iloc[0] if len(exact) and _present(exact["_close"].iloc[0]) else pd.NA
        status = ""
        reason = ""
        matched = ""
        boundary_low = boundary_high = pd.NA

        is_non_derivative = table in {"", "non_derivative"}
        if row_kind not in {"", "transaction"} or not is_non_derivative or code not in comparable:
            status = "NOT_MARKET_COMPARABLE"
            reason = "Only non-derivative purchase/sale transaction codes are compared with public-market ranges."
        elif price is None:
            status = "NO_REPORTED_PRICE"
            reason = "The filing has no numeric transaction price."
        elif price <= 0:
            status = "NONPOSITIVE_REPORTED_PRICE"
            reason = "A purchase or sale needs positive reported price for a market comparison."
        elif pd.isna(tx_date):
            status = "NO_TRANSACTION_DATE"
            reason = "The filing has no valid transaction date."
        elif not security:
            status = "NO_SECURITY_IDENTIFIER"
            reason = "No ticker or configured security identifier is available."
        elif group is None or (exact.empty and adjacent.empty and weekly.empty):
            status = "NO_MARKET_DATA"
            reason = "No market observations are available around the transaction date."
        elif _inside(price, exact_low, exact_high):
            status, matched = "PASS_EXACT_DAY_RANGE", "exact_day"
            reason = "Reported price is inside the exact transaction-day low/high range."
            boundary_low, boundary_high = exact_low, exact_high
        elif _inside(price, adjacent_low, adjacent_high):
            status, matched = "PASS_ADJACENT_DAY_RANGE", "adjacent_trading_days"
            reason = "Reported price is outside the exact range but inside the previous-to-next trading-day range."
            boundary_low, boundary_high = adjacent_low, adjacent_high
        elif _inside(price, weekly_low, weekly_high):
            status, matched = "PASS_WEEKLY_RANGE", "calendar_week"
            reason = "Reported price is outside narrower ranges but inside the transaction-week range."
            boundary_low, boundary_high = weekly_low, weekly_high
        else:
            status, matched = "FLAG_OUTSIDE_WEEKLY_RANGE", "calendar_week"
            reason = "Reported price is outside all available exact, adjacent, and weekly ranges; review transaction context."
            boundary_low, boundary_high = weekly_low, weekly_high

        distance_close = (
            float(price - Decimal(str(exact_close)))
            if price is not None and _present(exact_close)
            else pd.NA
        )
        shares = _decimal(_value(row, "shares", "transaction_shares"))
        derived_value = shares * price if shares is not None and price is not None else None
        reported_value = _decimal(_value(row, "reported_transaction_value"))
        value_status = "NO_INDEPENDENT_REPORTED_VALUE"
        if reported_value is not None:
            value_status = ("NOT_TESTABLE_VALUE" if derived_value is None else
                            "PASS_REPORTED_VALUE" if abs(reported_value - derived_value) <= Decimal("0.01")
                            else "FLAG_REPORTED_VALUE_DIFFERENCE")
        output.append(
            {
                "input_row_number": input_order,
                "issuer_cik": _value(row, "issuer_cik"),
                "issuer_ticker": _value(row, "issuer_ticker", "ticker"),
                "owner_cik": _value(row, "owner_cik"),
                "owner_name": _value(row, "owner_name"),
                "accession": _accession(row),
                "form": _value(row, "form", "document_type"),
                "transaction_date": _date_text(tx_date),
                "transaction_code": code,
                "security_title": _value(row, "security_title"),
                "reported_price": _number(price),
                "exact_date": _date_text(tx_date) if not exact.empty else "",
                "exact_low": exact_low,
                "exact_high": exact_high,
                "exact_close": exact_close,
                "adjacent_start": _date_text(adjacent["_date"].min()) if not adjacent.empty else "",
                "adjacent_end": _date_text(adjacent["_date"].max()) if not adjacent.empty else "",
                "adjacent_low": adjacent_low,
                "adjacent_high": adjacent_high,
                "week_start": _date_text(week_start),
                "week_end": _date_text(week_end),
                "weekly_low": weekly_low,
                "weekly_high": weekly_high,
                "matched_window": matched,
                "distance_from_exact_close": distance_close,
                "distance_from_nearest_boundary": (
                    _boundary_distance(price, boundary_low, boundary_high) if price is not None else pd.NA
                ),
                "price_check_status": status,
                "price_check_reason": reason,
                "economic_row_id": _value(row, "economic_row_id"),
                "owner_row_id": _value(row, "owner_row_id"),
                "reported_transaction_value": _number(reported_value),
                "derived_transaction_value": _number(derived_value),
                "transaction_value_status": value_status,
            }
        )

    return pd.DataFrame(output, columns=PRICE_OUTPUT_COLUMNS)


def _status_counts(frame: pd.DataFrame, column: str) -> dict:
    """Count rows by value of ``column``, NaN included, as a sorted dict."""
    if frame.empty:
        return {}
    return {str(key): int(value) for key, value in frame[column].value_counts(dropna=False).sort_index().items()}


def main() -> int:
    """Run continuity and price checks for an input table and write results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions", type=Path, required=True, help="Normalized ownership CSV")
    parser.add_argument("--market-data", type=Path, help="Optional daily market-price CSV")
    parser.add_argument("--split-adjustments", type=Path, help="Optional split-factor CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("data/quality_checks"))
    args = parser.parse_args()

    transactions = pd.read_csv(args.transactions, dtype=str)
    splits = pd.read_csv(args.split_adjustments, dtype=str) if args.split_adjustments else None
    continuity = check_ownership_continuity(transactions, splits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    continuity_path = args.output_dir / "ownership_continuity.csv"
    continuity.to_csv(continuity_path, index=False)

    summary = {
        "input_rows": int(len(transactions)),
        "ownership_continuity_statuses": _status_counts(continuity, "continuity_status"),
    }
    if args.market_data:
        market = pd.read_csv(args.market_data)
        prices = check_transaction_prices(transactions, market)
        prices.to_csv(args.output_dir / "price_interval_checks.csv", index=False)
        summary["price_check_statuses"] = _status_counts(prices, "price_check_status")

    (args.output_dir / "quality_check_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
