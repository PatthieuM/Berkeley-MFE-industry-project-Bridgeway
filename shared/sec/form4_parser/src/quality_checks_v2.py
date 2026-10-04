"""Version 2 quality checks for a separately reviewed rerun.

The module follows the project's "flag, never silently drop" convention. It
provides two checks:

* ownership-position continuity across rows and filings, with optional split
  factors; and
* reported transaction price against exact-day, adjacent-trading-day, and
  weekly market ranges.

Both functions retain every input row. Explicit unsuccessful parse statuses are
not used to establish positions or judge prices; a missing parse status remains
compatible with historical input tables. Calendar dates preserve the date in
their supplied timezone. This module does not replace the frozen continuity
implementation or the published audit; adopt its continuity results only after
a separate audit run has been reviewed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

QUALITY_CHECKS_VERSION = 2
SUCCESSFUL_PARSE_STATUSES = {"ok", "ok_legacy_xml", "ok_legacy_table"}


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
    """Keep the supplied calendar date, without converting it to another timezone."""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    timestamp = pd.Timestamp(parsed)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
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


def _identifier(value, *, numeric=False) -> str:
    """Normalize positive integral CIK/PERMNOs; invalid numeric identifiers are missing."""
    if not _present(value):
        return ""
    if numeric:
        number = _decimal(value)
        if number is not None and number > 0 and number == number.to_integral_value():
            return str(int(number))
        return ""
    return _normalized_text(value)


def _numeric_key(value) -> str:
    number = _decimal(value)
    return format(number.normalize(), "f") if number is not None else _normalized_text(value)


def _market_identifier(value, column: str) -> str:
    return _identifier(value, numeric=column.lower() in {"permno", "issuer_cik", "cik"})


def _security_columns(transactions, market, transaction_column, market_column):
    """Choose matching identifier namespaces, preferring PERMNO when both have it."""
    families = (
        (("permno", "PERMNO"), ("permno", "PERMNO")),
        (("issuer_ticker", "ticker"), ("ticker", "Ticker", "crsp_ticker")),
        (("issuer_cik", "cik"), ("issuer_cik", "cik")),
    )
    if transaction_column and market_column:
        return (
            _resolve_column(transactions, transaction_column, (), "transaction security"),
            _resolve_column(market, market_column, (), "market security"),
        )
    for transaction_names, market_names in families:
        if transaction_column and transaction_column not in transaction_names:
            continue
        if market_column and market_column not in market_names:
            continue
        tx_name = next((name for name in transaction_names if name in transactions.columns
                        and (not transaction_column or name == transaction_column)), None)
        market_name = next((name for name in market_names if name in market.columns
                            and (not market_column or name == market_column)), None)
        if tx_name and market_name:
            return tx_name, market_name
    raise ValueError("No shared security identifier; specify matching transaction and market security columns")


def _unreviewed_parse(row: dict) -> bool:
    status = _normalized_text(_value(row, "parse_status"))
    return bool(status) and status not in SUCCESSFUL_PARSE_STATUSES


def _normalized_security(value) -> str:
    """Turn a security title into a stable key (``common_stock`` for common shares)."""
    compact = _normalized_text(value)
    if compact in {"common stock", "common shares"}:
        return "common_stock"
    return re.sub(r"[^a-z0-9]+", "_", compact).strip("_") or "unknown_security"


def _owner_key(row: dict) -> str:
    """Identify the reporting owner by CIK, or by normalized name when the CIK is missing."""
    cik = _identifier(_value(row, "owner_cik"), numeric=True)
    return f"cik:{cik}" if cik else f"name:{_normalized_text(_value(row, 'owner_name'))}"


def _issuer_key(row: dict) -> str:
    """Identify the issuer by CIK, or by ticker when the CIK is missing."""
    cik = _identifier(_value(row, "issuer_cik"), numeric=True)
    ticker = _normalized_text(_value(row, "issuer_ticker", "ticker"))
    return f"cik:{cik}" if cik else f"ticker:{ticker}"


def _security_key(row: dict) -> tuple[str, ...]:
    """Return the ownership bucket a row's position belongs to.

    Issuer, owner, table, security, direct/indirect and nature of ownership;
    derivative rows also include exercise price, dates and underlying security
    to distinguish instruments when those fields are available.
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
                _numeric_key(_value(row, "conversion_or_exercise_price")),
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
        for identifier in ("permno", "issuer_cik"):
            if _present(raw.get(identifier)):
                value = _decimal(raw[identifier])
                if value is None or value <= 0 or value != value.to_integral_value():
                    raise ValueError(f"Split {identifier} must be a positive integer when supplied")
        output.append({**raw, "_effective_date": effective, "_factor": factor})
    return output


def _matches_adjustment(row: dict, adjustment: dict) -> bool:
    """Return True when a split adjustment applies to the row's security.

    Matches on PERMNO when the adjustment has one, else on issuer CIK, else on
    ticker; an adjustment with no identifier applies to every row.
    """
    # Prefer PERMNO when supplied so the adjustment is scoped to that security.
    if _present(adjustment.get("permno")):
        return _identifier(row.get("permno"), numeric=True) == _identifier(adjustment["permno"], numeric=True)
    adjustment_cik = _identifier(_value(adjustment, "issuer_cik"), numeric=True)
    row_cik = _identifier(_value(row, "issuer_cik"), numeric=True)
    if adjustment_cik:
        return bool(row_cik) and adjustment_cik == row_cik
    adjustment_ticker = _normalized_text(_value(adjustment, "issuer_ticker", "ticker"))
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
        if _unreviewed_parse(row):
            result.at[index, "amendment_status"] = "unreviewed_parse"
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
    supplied to scope the event; ``permno`` takes precedence when supplied.
    ``tolerance`` is a finite, nonnegative absolute number of shares (default
    0.000001). Explicit unsuccessful parse statuses cannot update the baseline.
    """
    tolerance_decimal = _decimal(tolerance)
    if tolerance_decimal is None or tolerance_decimal < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if rows.empty:
        return pd.DataFrame(columns=CONTINUITY_OUTPUT_COLUMNS)

    adjustments = _split_records(split_adjustments)
    linked_rows = resolve_amendments(rows, as_of=as_of)
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
        elif _unreviewed_parse(row):
            status = "NOT_TESTABLE_PARSE_STATUS"
            reason = "Unsuccessful or unresolved parsing cannot establish a position."
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
    """Return bounds from complete, finite, positive low/high observations only."""
    if frame.empty:
        return (pd.NA, pd.NA)
    valid = frame["_low"].map(lambda value: _present(value) and math.isfinite(float(value)))
    valid &= frame["_high"].map(lambda value: _present(value) and math.isfinite(float(value)))
    valid &= frame["_low"].gt(0) & frame["_high"].ge(frame["_low"])
    frame = frame.loc[valid]
    if frame.empty:
        return (pd.NA, pd.NA)
    low = frame["_low"].min(skipna=True)
    high = frame["_high"].max(skipna=True)
    return (pd.NA if pd.isna(low) else float(low), pd.NA if pd.isna(high) else float(high))


def _inside(price: Decimal, low, high) -> bool:
    """Return True when ``price`` lies within [low, high] (both must be known)."""
    return _present(low) and _present(high) and Decimal(str(low)) <= price <= Decimal(str(high))


def _boundary_distance(price: Decimal, low, high):
    """Return signed boundary distance: negative below, positive above, zero inside."""
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

    Adjacent range uses the nearest supplied observations before and after the
    transaction, each within ``max_adjacent_calendar_days`` (default 7), plus
    the exact date when present. Weekly range uses its Monday-Sunday calendar
    week. Bounds are inclusive; default comparable codes are P and S.
    Shares times price is derived independently of market data. A separately
    reported transaction value matches within an absolute tolerance of 0.01.
    Automatic security matching uses a namespace present on both tables,
    preferring PERMNO, then ticker, then CIK. Supply both column arguments for
    a custom identifier namespace.
    """
    if transactions.empty:
        return pd.DataFrame(columns=PRICE_OUTPUT_COLUMNS)
    if max_adjacent_calendar_days < 0:
        raise ValueError("max_adjacent_calendar_days must be nonnegative")
    tx_security = None
    if market_data is None or market_data.empty:
        market = pd.DataFrame(columns=["_security", "_date", "_low", "_high", "_close"])
    else:
        tx_security, market_security = _security_columns(
            transactions, market_data, transaction_security_col, market_security_col
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
                "_security": market_data[market_security].map(lambda value: _market_identifier(value, market_security)),
                "_date": market_data[market_date].map(_date),
                "_low": pd.to_numeric(market_data[market_low], errors="coerce").abs(),
                "_high": pd.to_numeric(market_data[market_high], errors="coerce").abs(),
                "_close": (
                    pd.to_numeric(market_data[market_close], errors="coerce").abs()
                    if market_close
                    else pd.Series(pd.NA, index=market_data.index, dtype="Float64")
                ),
            }
        ).dropna(subset=["_security", "_date"])
        market = market.loc[market["_security"].ne("")]
        if market.duplicated(["_security", "_date"]).any():
            raise ValueError("Market data contain duplicate security/date observations")
        market = market.sort_values(["_security", "_date"]).reset_index(drop=True)

    tx_security = tx_security or _resolve_column(
        transactions,
        transaction_security_col,
        ("permno", "PERMNO", "issuer_ticker", "ticker", "issuer_cik"),
        "transaction security",
    )
    comparable = {str(code).upper() for code in comparable_codes}
    market_groups = {key: group.reset_index(drop=True) for key, group in market.groupby("_security", sort=False)}
    output = []

    for input_order, row in enumerate(transactions.to_dict("records"), start=1):
        security = _market_identifier(row.get(tx_security, ""), tx_security)
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
        exact_close = _number(_decimal(exact["_close"].iloc[0])) if len(exact) else pd.NA
        status = ""
        reason = ""
        matched = ""
        boundary_low = boundary_high = pd.NA
        available_ranges = [
            (window, low, high)
            for window, low, high in (
                ("exact_day", exact_low, exact_high),
                ("adjacent_trading_days", adjacent_low, adjacent_high),
                ("calendar_week", weekly_low, weekly_high),
            )
            if _present(low) and _present(high)
        ]

        is_non_derivative = table in {"", "non_derivative"}
        if _unreviewed_parse(row):
            status = "NOT_TESTABLE_PARSE_STATUS"
            reason = "Unsuccessful or unresolved parsing cannot support a price comparison."
        elif row_kind not in {"", "transaction"} or not is_non_derivative or code not in comparable:
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
        elif not available_ranges:
            status = "NO_MARKET_DATA"
            reason = "No complete, finite, positive market low/high ranges are available around the transaction date."
        elif _inside(price, exact_low, exact_high):
            status, matched = "PASS_EXACT_DAY_RANGE", "exact_day"
            reason = "Reported price is inside the exact transaction-day low/high range."
            boundary_low, boundary_high = exact_low, exact_high
        elif _inside(price, adjacent_low, adjacent_high):
            status, matched = "PASS_ADJACENT_DAY_RANGE", "adjacent_trading_days"
            reason = "Reported price did not match the exact-day range and is inside the adjacent-observation range."
            boundary_low, boundary_high = adjacent_low, adjacent_high
        elif _inside(price, weekly_low, weekly_high):
            status, matched = "PASS_WEEKLY_RANGE", "calendar_week"
            reason = "Reported price did not match narrower ranges and is inside the transaction-week range."
            boundary_low, boundary_high = weekly_low, weekly_high
        else:
            matched, boundary_low, boundary_high = available_ranges[-1]
            status = {"calendar_week": "FLAG_OUTSIDE_WEEKLY_RANGE",
                      "adjacent_trading_days": "FLAG_OUTSIDE_ADJACENT_RANGE",
                      "exact_day": "FLAG_OUTSIDE_EXACT_RANGE"}[matched]
            reason = "Reported price is outside all usable supplied ranges; review transaction context."

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


def main(argv=None) -> int:
    """Write version 2 checks to a separate output directory for review."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions", type=Path, required=True, help="Normalized ownership CSV")
    parser.add_argument("--market-data", type=Path, help="Optional daily market-price CSV")
    parser.add_argument("--split-adjustments", type=Path, help="Optional split-factor CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for version 2 results")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("--output-dir must not already exist; keep earlier audit outputs unchanged")

    transactions = pd.read_csv(args.transactions, dtype=str)
    splits = pd.read_csv(args.split_adjustments, dtype=str) if args.split_adjustments else None
    continuity = check_ownership_continuity(transactions, splits)
    args.output_dir.mkdir(parents=True)
    continuity_path = args.output_dir / "ownership_continuity.csv"
    continuity.to_csv(continuity_path, index=False)

    summary = {
        "quality_checks_version": QUALITY_CHECKS_VERSION,
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
