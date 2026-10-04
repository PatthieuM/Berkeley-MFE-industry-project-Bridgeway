"""Compute event-window returns and issuer-by-month clustered inference.

Inputs are event tables, daily security levels, and a benchmark level series; outputs
feed summary tables and cumulative-path figures in the signal research pipeline.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from shared.config import Config, DEFAULT_CONFIG


def _dates(values) -> pd.DatetimeIndex:
    """Parse values to timezone-naive dates normalized to midnight; bad values become NaT."""
    result = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce"))
    if result.tz is not None:
        result = result.tz_convert(None)
    return result.normalize()


def event_returns(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: pd.Series,
    horizons: Sequence[int],
) -> pd.DataFrame:
    """Compute close-to-close stock and XBI returns over identical sessions."""
    required_events = {"event_id", "security_id", "filing_date"}
    required_prices = {"date", "security_id", "close"}
    if missing := required_events.difference(events.columns):
        raise ValueError(f"events missing columns: {sorted(missing)}")
    if missing := required_prices.difference(prices.columns):
        raise ValueError(f"prices missing columns: {sorted(missing)}")
    if events["event_id"].duplicated().any():
        raise ValueError("event_id must be unique")
    hs = tuple(int(value) for value in horizons)
    if not hs or any(value <= 0 for value in hs):
        raise ValueError("horizons must be positive")

    event_frame = events.copy()
    event_frame["filing_date"] = _dates(event_frame["filing_date"])
    bench = pd.to_numeric(benchmark.copy(), errors="coerce")
    bench.index = _dates(bench.index)
    bench = bench.sort_index()
    if bench.index.has_duplicates:
        raise ValueError("benchmark dates must be unique")
    calendar = bench.index

    price_frame = prices.copy()
    price_frame["date"] = _dates(price_frame["date"])
    if price_frame.duplicated(["date", "security_id"]).any():
        raise ValueError("prices must be unique by date/security_id")
    close_frame = price_frame.pivot(index="date", columns="security_id", values="close").reindex(calendar)
    close_frame = close_frame.apply(pd.to_numeric, errors="coerce")
    securities = close_frame.columns
    close = close_frame.to_numpy(dtype=float, copy=False)
    n_days, n_securities = close.shape

    if "delisted" in price_frame:
        terminal_frame = (
            price_frame.assign(delisted=price_frame["delisted"].eq(True))
            .pivot(index="date", columns="security_id", values="delisted")
            .reindex(index=calendar, columns=securities, fill_value=False).fillna(False)
        )
        terminal = terminal_frame.to_numpy(dtype=bool, copy=False)
    else:
        terminal = np.zeros((n_days, n_securities), dtype=bool)

    invalid = ~np.isfinite(close) | (close <= 0)
    invalid_prefix = np.vstack(
        [np.zeros((1, n_securities), dtype=np.int32), np.cumsum(invalid, axis=0, dtype=np.int32)]
    )
    terminal_prefix = np.vstack(
        [np.zeros((1, n_securities), dtype=np.int32), np.cumsum(terminal, axis=0, dtype=np.int32)]
    )
    bench_values = bench.to_numpy(float)
    bench_invalid_prefix = np.r_[0, np.cumsum(~np.isfinite(bench_values) | (bench_values <= 0), dtype=np.int32)]

    repeated = np.repeat(np.arange(len(event_frame)), len(hs))
    horizon_values = np.tile(np.asarray(hs, dtype=np.int64), len(event_frame))
    # Entry is the first benchmark session strictly after filing, preventing look-ahead.
    entry_by_event = calendar.searchsorted(event_frame["filing_date"].to_numpy(), side="right")
    entry = entry_by_event[repeated].astype(np.int64, copy=False)
    end = entry + horizon_values
    security_by_event = securities.get_indexer(event_frame["security_id"])
    security = security_by_event[repeated]
    in_bounds = (entry < n_days) & (end < n_days) & (security >= 0)

    calendar_values = calendar.to_numpy(dtype="datetime64[ns]")
    entry_dates = np.full(len(repeated), np.datetime64("NaT"), dtype="datetime64[ns]")
    end_dates = np.full(len(repeated), np.datetime64("NaT"), dtype="datetime64[ns]")
    entry_ok, end_ok = entry < n_days, end < n_days
    entry_dates[entry_ok] = calendar_values[entry[entry_ok]]
    end_dates[end_ok] = calendar_values[end[end_ok]]

    delisted = np.zeros(len(repeated), dtype=bool)
    stock_bad = np.ones(len(repeated), dtype=bool)
    valid_rows = np.flatnonzero(in_bounds)
    if valid_rows.size:
        start_i, end_i, sec_i = entry[valid_rows], end[valid_rows], security[valid_rows]
        stock_bad[valid_rows] = (invalid_prefix[end_i + 1, sec_i] - invalid_prefix[start_i, sec_i]) != 0
        delisted[valid_rows] = (terminal_prefix[end_i + 1, sec_i] - terminal_prefix[start_i, sec_i]) != 0
    bench_bad = np.ones(len(repeated), dtype=bool)
    bench_rows = np.flatnonzero(end < n_days)
    if bench_rows.size:
        bench_bad[bench_rows] = (
            bench_invalid_prefix[end[bench_rows] + 1] - bench_invalid_prefix[entry[bench_rows]]
        ) != 0
    complete = in_bounds & ~stock_bad & ~bench_bad & ~delisted

    stock_return = np.full(len(repeated), np.nan)
    benchmark_return = np.full(len(repeated), np.nan)
    good = np.flatnonzero(complete)
    if good.size:
        stock_return[good] = close[end[good], security[good]] / close[entry[good], security[good]] - 1
        benchmark_return[good] = bench_values[end[good]] / bench_values[entry[good]] - 1
    result = event_frame.iloc[repeated].reset_index(drop=True)
    result["entry_date"] = entry_dates
    result["horizon"] = horizon_values
    result["end_date"] = end_dates
    result["stock_return"] = stock_return
    result["benchmark_return"] = benchmark_return
    result["excess_return"] = stock_return - benchmark_return
    result["delisted"] = delisted
    result["partial_path"] = ~complete
    return result


def _cluster_component(scores: np.ndarray, labels: np.ndarray) -> float:
    """Return the variance term for one clustering dimension (e.g. issuer or month).

    Sums the per-observation scores within each cluster and returns
    G / (G - 1) times the sum of squared cluster sums (G = number of clusters).
    NaN when there are fewer than two clusters.
    """
    codes, unique = pd.factorize(labels, sort=False)
    if len(unique) < 2:
        return np.nan
    sums = np.bincount(codes, weights=scores)
    return float(len(unique) / (len(unique) - 1) * np.square(sums).sum())


def summary_stats(frame: pd.DataFrame, value: str = "excess_return") -> dict:
    """Return descriptive statistics and two-way clustered two-sided inference."""
    valid = pd.to_numeric(frame[value], errors="coerce").notna() & frame["filing_date"].notna()
    current = frame.loc[valid].copy()
    n = len(current)
    empty = {
        "N": 0, "mean": np.nan, "median": np.nan, "hit_rate": np.nan,
        "clustered_se": np.nan, "ci_low": np.nan, "ci_high": np.nan,
        "t": np.nan, "p_two_sided": np.nan, "df": np.nan,
        "issuer_clusters": 0, "month_clusters": 0,
    }
    if n == 0:
        return empty
    values = pd.to_numeric(current[value], errors="coerce").to_numpy(float)
    effect = float(values.mean())
    issuers = current["issuer_cik"].to_numpy()
    months = pd.to_datetime(current["filing_date"]).dt.to_period("M").astype(str).to_numpy()
    scores = (values - effect) / n
    intersections = pd.MultiIndex.from_arrays([issuers, months])
    # Cameron-Gelbach-Miller inclusion-exclusion combines the two cluster dimensions.
    variance = (
        _cluster_component(scores, issuers)
        + _cluster_component(scores, months)
        - _cluster_component(scores, intersections)
    )
    standard_error = float(np.sqrt(max(0.0, variance))) if np.isfinite(variance) else np.nan
    issuer_clusters, month_clusters = pd.unique(issuers).size, pd.unique(months).size
    degrees = min(issuer_clusters, month_clusters) - 1
    t_value = effect / standard_error if standard_error > 0 else np.nan
    critical = stats.t.ppf(0.975, degrees) if degrees > 0 else np.nan
    p_value = 2 * stats.t.sf(abs(t_value), degrees) if degrees > 0 and np.isfinite(t_value) else np.nan
    return {
        "N": n,
        "mean": effect,
        "median": float(np.median(values)),
        "hit_rate": float(np.mean(values > 0)),
        "clustered_se": standard_error,
        "ci_low": effect - critical * standard_error,
        "ci_high": effect + critical * standard_error,
        "t": t_value,
        "p_two_sided": float(p_value),
        "df": degrees,
        "issuer_clusters": issuer_clusters,
        "month_clusters": month_clusters,
    }


def summarize_periods(
    outcomes: pd.DataFrame,
    signal: str,
    config: Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Summarize each configured horizon within the configured study periods."""
    rows = []
    for period, low, high in config.periods:
        period_frame = outcomes.loc[pd.to_datetime(outcomes["filing_date"]).dt.year.between(low, high)]
        for horizon in config.horizons:
            stats_row = summary_stats(period_frame.loc[period_frame["horizon"].eq(horizon)])
            rows.append({"signal": signal, "horizon": horizon, "period": period, **stats_row})
    return pd.DataFrame(rows)


def evaluate_signals(
    signals: Dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    benchmark: pd.Series,
    config: Config = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Evaluate named event tables and return pooled summaries plus event outcomes."""
    price_input = prices[["date", "security_id", "close", "delisted"]]
    summaries, outcomes = [], {}
    for name, events in signals.items():
        event_input = events.rename(columns={"permno": "security_id"})
        result = event_returns(event_input, price_input, benchmark, config.horizons)
        outcomes[name] = result
        summaries.append(summarize_periods(result, name, config))
    return pd.concat(summaries, ignore_index=True), outcomes


def cumulative_average_path(
    signals: Dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    benchmark: pd.Series,
    config: Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Return each signal's mean cumulative excess-return path by session."""
    rows = [{"signal": name, "session": 0, "N": len(events), "mean_excess": 0.0}
            for name, events in signals.items()]
    for name, events in signals.items():
        event_input = events.rename(columns={"permno": "security_id"})
        result = event_returns(event_input, prices, benchmark, range(1, config.path_horizon + 1))
        grouped = result.groupby("horizon")["excess_return"].agg(["count", "mean"]).reset_index()
        for row in grouped.itertuples(index=False):
            rows.append({"signal": name, "session": int(row.horizon), "N": int(row.count), "mean_excess": row.mean})
    return pd.DataFrame(rows)
