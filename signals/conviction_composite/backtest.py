"""
backtest.py -- robustness testing + calendar-time backtest (Task 3).

The event-level t-stats in ``analysis.ipynb`` are overstated: forward returns
overlap in time and cluster within firms, so the trades are far from the
independent observations a naive t-stat assumes. This module fixes that with a
**calendar-time portfolio**: each signalled trade opens an equal-weight position
held ``horizon`` trading days, and the portfolio's *daily* return is the mean of
its open positions. Overlap is handled by construction, so the resulting monthly
series has ~independent observations and honest significance.

It provides, for any subset of events:
  * calendar_daily_returns  -- the daily portfolio return series
  * perf_stats              -- CAGR, monthly-based Sharpe, Newey-West t-stat,
                               max drawdown, and alpha/beta vs the XBI benchmark
  * cluster_bootstrap       -- firm-block bootstrap CI on event-level mean return

Data cleaning: daily returns are winsorized to [-90%, +200%] to strip unadjusted
reverse-split artifacts (e.g. a raw yfinance +1.3e6 one-day "return"); genuine
biotech single-day moves are well inside that band.

Survivorship caveat (direction matters): the free feed drops delisted names.
In a LONG book those are mostly failures (-> -100%), so long returns are biased
*up*; in a SHORT book the same drops bias short profits *down*. Magnitudes here
are therefore optimistic for longs / conservative for shorts -- the CRSP feed
(delisting returns) is the fix. Statistical *significance* is far more robust
than the point estimates.
"""
from __future__ import annotations

import glob
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

RET_FLOOR, RET_CAP = -0.90, 2.0     # daily-return winsorization
NW_LAG = 6                          # Newey-West lag (months)


def load_panels(cache_dir: str) -> dict[str, pd.DataFrame]:
    """Load cached per-ticker daily price files into ``{ticker: frame}``.

    Each frame is indexed by date and its ``ret`` column is winsorized to
    [RET_FLOOR, RET_CAP]. Empty files and files without a usable date index are
    skipped. Only used with the legacy yfinance cache; the CRSP scripts build
    panels with ``crsp_reprice.load_crsp_panels`` instead.
    """
    panels: dict[str, pd.DataFrame] = {}
    for f in glob.glob(os.path.join(cache_dir, "*.parquet")):
        tk = os.path.basename(f).split("_")[0]
        p = pd.read_parquet(f)
        if len(p) == 0:
            continue
        if not isinstance(p.index, pd.DatetimeIndex):
            if "Date" in p.columns:
                p = p.set_index("Date")
            elif "close" in p.columns:
                continue
            else:
                p = p.set_index(p.columns[0])
        p.index = pd.to_datetime(p.index)
        p = p.sort_index()
        p["ret"] = p["ret"].clip(lower=RET_FLOOR, upper=RET_CAP)
        panels[tk] = p
    return panels


def _views(panels):
    """Convert panels to ``(dates, returns)`` NumPy arrays for fast lookups in the backtest loop."""
    return {tk: (p.index.values.astype("datetime64[ns]"),
                 p["ret"].to_numpy("float64")) for tk, p in panels.items()}


def calendar_daily_returns(events: pd.DataFrame, horizon_days: int,
                           panels: dict[str, pd.DataFrame],
                           side_col: str = "side", long_short: bool = False,
                           weight_col: str | None = None) -> pd.Series:
    """Daily equal-weight calendar-time portfolio return for ``events``.

    Each position earns its stock's daily return from the bar after entry_date
    through ``horizon_days`` later. If ``long_short``, positions are signed by
    ``side_col``; otherwise all are treated long (analyze one side at a time).
    """
    views = _views(panels)
    # num[d] = sum of weighted returns of positions open on day d; den[d] = sum of
    # absolute weights. The day's portfolio return is num / den (equal weight).
    num, den = defaultdict(float), defaultdict(float)
    ev = events.dropna(subset=["entry_date"]).copy()
    ev["entry_date"] = pd.to_datetime(ev["entry_date"]).values.astype("datetime64[ns]")
    for _, r in ev.iterrows():
        v = views.get(r["ticker"])
        if v is None:
            continue
        dates, ret = v
        # Position is bought at the entry-date close, so it earns returns from the
        # next session through `horizon_days` sessions later (capped at the last day).
        pos = int(np.searchsorted(dates, np.datetime64(r["entry_date"]), side="left"))
        sgn = r[side_col] if long_short else 1.0
        w = (r[weight_col] if weight_col else 1.0) * sgn
        end = min(pos + horizon_days, len(ret) - 1)
        for j in range(pos + 1, end + 1):
            rj = ret[j]
            if np.isfinite(rj):
                num[dates[j]] += w * rj
                den[dates[j]] += abs(w)
    if not num:
        return pd.Series(dtype="float64")
    idx = sorted(num)
    return pd.Series([num[d] / den[d] if den[d] else 0.0 for d in idx],
                     index=pd.DatetimeIndex(idx)).sort_index()


def _nw_t(monthly: np.ndarray, lag: int = NW_LAG) -> float:
    """Return the Newey-West t-statistic of the mean of a monthly return series.

    Uses Bartlett weights up to ``lag`` months so that autocorrelation from
    overlapping positions does not inflate significance. Returns NaN for fewer
    than three months or a zero standard error.
    """
    n = len(monthly)
    if n < 3:
        return np.nan
    dev = monthly - monthly.mean()
    var = (dev @ dev) / n
    for k in range(1, lag + 1):
        if k < n:
            var += 2 * (1 - k / (lag + 1)) * (dev[k:] @ dev[:-k]) / n
    se = np.sqrt(var / n)
    return monthly.mean() / se if se > 0 else np.nan


def perf_stats(daily: pd.Series, bench: pd.Series | None = None,
               name: str = "strategy", freq: int = 252) -> dict:
    """Annualized performance + Newey-West t + alpha/beta vs benchmark."""
    d = daily.dropna()
    if len(d) < 20:
        return {"name": name, "n_days": len(d)}
    # Statistics use monthly compounded returns, which are close to independent.
    monthly = (1 + d).resample("ME").prod() - 1
    cum = (1 + d).cumprod()
    cagr = cum.iloc[-1] ** (freq / len(d)) - 1
    ann_vol = monthly.std() * np.sqrt(12)
    sharpe = (monthly.mean() * 12) / ann_vol if ann_vol > 0 else np.nan
    dd = (cum / cum.cummax() - 1).min()
    out = {"name": name, "n_days": len(d), "n_months": len(monthly),
           "cagr": cagr, "ann_vol": ann_vol, "sharpe": sharpe,
           "t_nw": _nw_t(monthly.values), "max_dd": dd,
           "hit_month": (monthly > 0).mean(),
           "start": str(d.index.min().date()), "end": str(d.index.max().date())}
    if bench is not None:
        bm = (1 + bench.reindex(d.index).fillna(0)).resample("ME").prod() - 1
        pair = pd.concat([monthly.rename("s"), bm.rename("b")], axis=1).dropna()
        if len(pair) > 12:
            # Monthly OLS of strategy on XBI: slope = beta, intercept = monthly alpha.
            beta, alpha_m = np.polyfit(pair["b"], pair["s"], 1)
            resid = pair["s"] - (beta * pair["b"] + alpha_m)
            se_a = resid.std() / np.sqrt(len(pair))
            out.update(beta_xbi=beta, alpha_ann=(1 + alpha_m) ** 12 - 1,
                       alpha_t=alpha_m / se_a if se_a > 0 else np.nan)
    return out


def cluster_bootstrap(events: pd.DataFrame, col: str = "resid_ret_6m",
                      nboot: int = 2000, seed: int = 42) -> dict:
    """Firm-block bootstrap: resample whole tickers with replacement so the CI
    respects within-firm return correlation (naive t ignores it)."""
    sub = events.dropna(subset=[col])
    firms = sub["ticker"].unique()
    by_firm = {f: sub.loc[sub.ticker == f, col].values for f in firms}
    rng = np.random.default_rng(seed)
    means = np.empty(nboot)
    for b in range(nboot):
        pick = rng.choice(firms, size=len(firms), replace=True)
        means[b] = np.concatenate([by_firm[f] for f in pick]).mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    # Two-sided p-value: twice the share of bootstrap means on the far side of zero.
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {"mean": sub[col].mean(), "ci_lo": lo, "ci_hi": hi, "p": p, "n": len(sub)}
