"""Run the official insider-signal analysis and publish aggregate results.

Universe, SEC event, price, and XBI inputs become CSV tables, Markdown prose,
figures, and presentation-ready summaries without exporting licensed rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shared.config import Config, DEFAULT_CONFIG
from shared.events.events import load_insider_events, load_prices, load_universe, load_xbi, price_levels
from shared.engine.evaluate import cumulative_average_path, evaluate_signals
from signals.smallcap_purchases.signals import build_signals


V6_REFERENCE = {
    "N": 2684,
    "mean": 0.015091292205902546,
    "t": 3.910953407089037,
    "early_mean": 0.016137531859513846,
    "late_mean": 0.014046610615811153,
}


def _analysis_signals(signals: Dict[str, pd.DataFrame], config: Config) -> Dict[str, pd.DataFrame]:
    """Keep only events filed inside the study window (``study_start`` to ``study_end``)."""
    start, end = pd.Timestamp(config.study_start), pd.Timestamp(config.study_end)
    return {
        name: frame.loc[pd.to_datetime(frame["filing_date"]).between(start, end)].copy()
        for name, frame in signals.items()
    }


def run_analysis(config: Config = DEFAULT_CONFIG) -> dict:
    """Load pipeline inputs and return signals, returns, and coverage diagnostics."""
    universe = load_universe(config)
    events, event_audit = load_insider_events(config, universe, return_audit=True)
    relevant_permnos = (
        set(universe["permno"].dropna().astype(int))
        if config.price_source == "universe_full"
        else set(events["permno"].dropna().astype(int))
    )
    daily = load_prices(config.price_source, config, relevant_permnos)
    benchmark = load_xbi(config)
    signals, breakpoints, _ = build_signals(events, daily, pd.DatetimeIndex(benchmark.index), config)
    signals = _analysis_signals(signals, config)
    levels = price_levels(daily)
    results_table, outcomes = evaluate_signals(signals, levels, benchmark, config)

    all_study_events = pd.concat(
        [frame[["event_id", "permno", "issuer_cik", "filing_date"]].assign(signal=name)
         for name, frame in signals.items()], ignore_index=True
    )
    unique_signal_events = all_study_events.drop_duplicates("event_id")
    five = pd.concat(
        [frame.loc[frame["horizon"].eq(5), ["event_id", "excess_return"]]
         for frame in outcomes.values()], ignore_index=True
    ).drop_duplicates("event_id")
    complete_five = int(five["excess_return"].notna().sum())
    universe_permnos = set(universe["permno"].dropna().astype(int))
    priced_permnos = set(daily["permno"].dropna().astype(int))
    coverage = {
        "price_source": config.price_source,
        "universe_permnos": len(universe_permnos),
        "universe_permnos_with_prices": len(universe_permnos & priced_permnos),
        "universe_permno_price_share": len(universe_permnos & priced_permnos) / len(universe_permnos),
        "study_events": len(unique_signal_events),
        "study_events_complete_h5": complete_five,
        "study_event_complete_h5_share": complete_five / len(unique_signal_events),
    }
    return {
        "universe": universe,
        "events": events,
        "event_audit": event_audit,
        "daily": daily,
        "benchmark": benchmark,
        "signals": signals,
        "breakpoints": breakpoints,
        "levels": levels,
        "results_table": results_table,
        "outcomes": outcomes,
        "coverage": coverage,
    }


def per_100_distribution(s1_outcomes: pd.DataFrame) -> pd.DataFrame:
    """Summarize five-session S1 stock, benchmark, and excess values per $100."""
    usable = s1_outcomes.loc[
        s1_outcomes["horizon"].eq(5) & s1_outcomes["excess_return"].notna()
    ]
    metrics = {
        "stock_terminal_value": 100 * (1 + usable["stock_return"]),
        "xbi_terminal_value": 100 * (1 + usable["benchmark_return"]),
        "excess_dollars": 100 * usable["excess_return"],
    }
    rows = []
    for metric, values in metrics.items():
        rows.append({
            "metric": metric, "N": len(values), "mean": values.mean(),
            "p10": values.quantile(0.10), "p25": values.quantile(0.25),
            "median": values.median(), "p75": values.quantile(0.75),
            "p90": values.quantile(0.90),
        })
    return pd.DataFrame(rows)


def _plot_horizon_means(results: pd.DataFrame, output: Path) -> None:
    """Plot S1-S3 mean excess return by horizon, with 95% clustered confidence bands (2009-2025)."""
    full = results.loc[results["period"].eq("2009-2025")]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = {"S1": "#1b7f79", "S2": "#355c9a", "S3": "#bd4f6c"}
    for signal, frame in full.groupby("signal"):
        current = frame.sort_values("horizon")
        ax.plot(current["horizon"], 100 * current["mean"], marker="o", label=signal, color=colors[signal])
        ax.fill_between(
            current["horizon"],
            100 * current["ci_low"],
            100 * current["ci_high"],
            alpha=0.15,
            color=colors[signal],
        )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set(xlabel="Trading sessions after entry", ylabel="Mean excess return (percentage points)")
    ax.set_xticks(sorted(full["horizon"].unique()))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_cumulative_path(path: pd.DataFrame, output: Path) -> None:
    """Plot each signal's average cumulative excess return, session by session after entry."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = {"S1": "#1b7f79", "S2": "#355c9a", "S3": "#bd4f6c"}
    for signal, frame in path.groupby("signal"):
        current = frame.sort_values("session")
        ax.plot(current["session"], 100 * current["mean_excess"], label=signal, color=colors[signal])
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set(xlabel="Trading sessions after entry", ylabel="Average cumulative excess return (pp)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_per_100(s1_outcomes: pd.DataFrame, output: Path) -> None:
    """Histogram of S1's 5-session excess dollars per $100 invested, with mean and median marked."""
    usable = s1_outcomes.loc[s1_outcomes["horizon"].eq(5) & s1_outcomes["excess_return"].notna()]
    values = 100 * usable["excess_return"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(values, bins=50, color="#1b7f79", alpha=0.85)
    ax.axvline(values.mean(), color="black", linestyle="--", label=f"Mean ${values.mean():.2f}")
    ax.axvline(values.median(), color="#bd4f6c", linestyle=":", label=f"Median ${values.median():.2f}")
    ax.set(xlabel="Excess dollars after 5 sessions per $100 invested", ylabel="Issuer-days")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _format_results_markdown(
    results: pd.DataFrame,
    per100: pd.DataFrame,
    coverage: dict,
    breakpoints: pd.DataFrame,
    ssn_results: pd.DataFrame | None = None,
    conditioning: pd.DataFrame | None = None,
) -> str:
    """Render ``RESULTS.md``: event-study table, per-$100 view, coverage, takeaways, S4 and
    conditioning tables.
    """
    lines = [
        "# Insider-signal results",
        "",
        "Official run: `price_source=\"universe_full\"`. Returns are stock total return minus XBI total return over identical exchange sessions; entry is the first session close strictly after filing.",
        "",
        "## Event-study results",
        "",
        "| Signal | Horizon | Period | N | Mean pp | Median pp | Hit rate | 95% clustered CI pp | t | two-sided p |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results.sort_values(["signal", "horizon", "period"]).itertuples(index=False):
        lines.append(
            f"| {row.signal} | {row.horizon} | {row.period} | {row.N:,} | {100*row.mean:.3f} | "
            f"{100*row.median:.3f} | {100*row.hit_rate:.1f}% | [{100*row.ci_low:.3f}, {100*row.ci_high:.3f}] | "
            f"{row.t:.2f} | {row.p_two_sided:.4g} |"
        )
    excess = per100.set_index("metric").loc["excess_dollars"]
    lines += [
        "",
        "## S1 five-session per-$100 view",
        "",
        f"Across {int(excess.N):,} complete S1 paths, $100 invested in the stock rather than XBI produced mean excess value of ${excess['mean']:.2f}; median ${excess['median']:.2f}; 10th/90th percentiles ${excess.p10:.2f}/${excess.p90:.2f}.",
        "",
        "## Coverage",
        "",
        f"- Universe PERMNO price coverage: {coverage['universe_permnos_with_prices']:,}/{coverage['universe_permnos']:,} ({coverage['universe_permno_price_share']:.1%}).",
        f"- Complete five-session paths: {coverage['study_events_complete_h5']:,}/{coverage['study_events']:,} unique study issuer-days ({coverage['study_event_complete_h5_share']:.1%}).",
        f"- Every breakpoint passes `training_max_year < year`; {len(breakpoints)} annual cutoffs cover 2009–2025.",
        "",
        "## Plain takeaways",
        "",
    ]
    five = results.loc[(results["horizon"].eq(5)) & (results["period"].eq("2009-2025"))].set_index("signal")
    for signal, label in [("S1", "small-cap purchases"), ("S2", "all purchases"), ("S3", "sales")]:
        row = five.loc[signal]
        direction = "positive" if row["mean"] > 0 else "negative"
        significance = (
            "statistically distinguishable from zero"
            if row["p_two_sided"] < 0.05
            else "not statistically distinguishable from zero"
        )
        lines.append(
            f"- {label.title()}: {100*row['mean']:.3f} pp over five sessions (N={int(row.N):,}), {direction} and {significance} at 5%. "
        )
    s1 = five.loc["S1"]
    lines += [
        f"- Relative to the v6 cache result (N={V6_REFERENCE['N']:,}, {100*V6_REFERENCE['mean']:.3f} pp), full-universe pricing changes N by {int(s1.N)-V6_REFERENCE['N']:+,} and the mean by {100*(s1['mean']-V6_REFERENCE['mean']):+.3f} pp.",
        "- The difference is coverage, not a changed signal rule: the frozen v6 report records cached prices for only 503/784 event PERMNOs and 3,334 mapped purchase issuer-days with no cached rows. `universe_full` covers all 953 universe PERMNOs, restoring lagged feature eligibility and forward paths.",
        "",
        "## Definitions and limitations",
        "",
        "S1 is S2 restricted to the lagged-cap lower tercile using expanding prior-years-only purchase breakpoints. S2 is every mapped purchase issuer-day. S3 is every mapped issuer-day with at least one non-derivative transaction code S. Original Form 4 only; amendments are excluded; one observation is issuer × filing date × direction.",
        "",
        "Historical GICS is effective-dated but not publication-vintage. Current-CIK fallback intervals remain a historical-identity limitation. Incomplete or delisted paths are missing, never zero-filled. Confidence intervals and p-values use issuer × filing-month two-way clustered inference. No row-level licensed data are written here.",
    ]
    if ssn_results is not None:
        lines += [
            "",
            "## S4 seller-silence results",
            "",
            "The pre-registered S4 headlines remain 63 and 126 sessions; all other horizons are descriptive. The estimand is SSN minus SSS-only.",
            "",
            "| Horizon | Period | N SSN | N SSS-only | SSN mean pp | SSS-only mean pp | Difference pp | t | two-sided p |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in ssn_results.sort_values(["horizon", "period"]).itertuples(index=False):
            lines.append(
                f"| {row.horizon} | {row.period} | {row.N_SSN:,} | {row.N_SSS_only:,} | "
                f"{100*row.SSN_mean:.3f} | {100*row.SSS_only_mean:.3f} | {100*row.difference:.3f} | "
                f"{row.t:.2f} | {row.p_two_sided:.4g} |"
            )
    if conditioning is not None:
        lines += [
            "",
            "## Descriptive conditioning",
            "",
            "These 2009–2025 splits use the same XBI excess returns and issuer × filing-month clustered inference. They are descriptive, not additional confirmatory tests. Buyer-role groups are exclusive with senior executives taking precedence, followed by other officers; `directors_only` and `ten_percent_owners_only` contain no officer category. Dollar terciles use only purchase events from years before the filing year. Seller classes use the Cohen–Malloy–Pomorski three-prior-year same-calendar-month rule, frozen at each year start.",
            "",
            "| Signal | Dimension | Group | Horizon | N | Mean pp | 95% clustered CI pp | t | two-sided p |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in conditioning.sort_values(["signal", "dimension", "group", "horizon"]).itertuples(index=False):
            lines.append(
                f"| {row.signal} | {row.dimension} | {row.group} | {row.horizon} | {row.N:,} | "
                f"{100*row.mean:.3f} | [{100*row.ci_low:.3f}, {100*row.ci_high:.3f}] | "
                f"{row.t:.2f} | {row.p_two_sided:.4g} |"
            )
    return "\n".join(lines) + "\n"


def write_report(
    analysis: dict,
    config: Config = DEFAULT_CONFIG,
    ssn_results: pd.DataFrame | None = None,
    conditioning: pd.DataFrame | None = None,
) -> dict:
    """Write aggregate result tables and Markdown, then return headline metrics."""
    results_dir = config.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    results = analysis["results_table"].copy()
    results.to_csv(results_dir / "results_table.csv", index=False)
    per100 = per_100_distribution(analysis["outcomes"]["S1"])
    markdown = _format_results_markdown(
        results, per100, analysis["coverage"], analysis["breakpoints"], ssn_results, conditioning
    )
    (results_dir / "RESULTS.md").write_text(markdown, encoding="utf-8")
    five = results.loc[(results["horizon"].eq(5)) & (results["period"].eq("2009-2025"))]
    return {
        "price_source": config.price_source,
        "five_session": five[["signal", "N", "mean", "t", "p_two_sided"]].to_dict("records"),
        "coverage": analysis["coverage"],
    }


def run_official() -> dict:
    """Run the full-universe analysis, extensions, sanity checks, and reports."""
    config = DEFAULT_CONFIG.with_price_source("universe_full")
    analysis = run_analysis(config)
    from signals.seller_silence.ssn import run_ssn
    from signals.smallcap_purchases.conditioning import conditioning_table
    from shared.engine.sanity import run_sanity

    ssn = run_ssn(analysis["universe"], analysis["levels"], analysis["benchmark"], config)
    ssn["results"].to_csv(config.results_dir / "ssn_results.csv", index=False)
    conditioning = conditioning_table(analysis["outcomes"], config)
    conditioning.to_csv(config.results_dir / "conditioning.csv", index=False)
    summary = write_report(analysis, config, ssn["results"], conditioning)

    sanity = run_sanity(analysis, ssn, config)
    summary["S4_primary"] = ssn["results"].loc[
        ssn["results"]["period"].eq("2009-2025") & ssn["results"]["horizon"].isin([63, 126]),
        ["horizon", "N_SSN", "N_SSS_only", "difference", "t", "p_two_sided"],
    ].to_dict("records")
    summary["sanity_flag_rows"] = int(sanity["summary"]["flagged"].fillna(0).sum())
    return summary


if __name__ == "__main__":
    print(json.dumps(run_official(), indent=2))


# The functions below are an additive presentation layer.  They deliberately
# reuse the tested signal, return, and inference functions above without
# changing the official analysis or its results_table.csv output.


def _extended_outcomes(analysis: dict, horizons=(1, 2, 3, 5, 10, 21, 42, 63, 126)) -> Dict[str, pd.DataFrame]:
    """Compute descriptive extra horizons with the tested event-return engine."""
    from shared.engine.evaluate import event_returns

    prices = analysis["levels"][["date", "security_id", "close", "delisted"]]
    return {
        signal: event_returns(events.rename(columns={"permno": "security_id"}), prices,
                              analysis["benchmark"], horizons)
        for signal, events in analysis["signals"].items()
    }


def _summaries_by_horizon(outcomes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return summary statistics (N, mean, t, ...) for each signal and horizon."""
    from shared.engine.evaluate import summary_stats

    rows = []
    for signal, frame in outcomes.items():
        for horizon, current in frame.groupby("horizon", sort=True):
            rows.append({"signal": signal, "horizon": int(horizon), **summary_stats(current)})
    return pd.DataFrame(rows)


def _month_cluster_bootstrap_path(
    s1_outcomes: pd.DataFrame, seed: int = 240925, replications: int = 1000
) -> pd.DataFrame:
    """S1 mean path with a reproducible filing-month cluster bootstrap."""
    current = s1_outcomes.copy()
    current["filing_month"] = pd.to_datetime(current["filing_date"]).dt.to_period("M").astype(str)
    horizons = sorted(current["horizon"].unique())
    grouped = current.groupby(["filing_month", "horizon"])["excess_return"].agg(["sum", "count"])
    months = sorted(current["filing_month"].unique())
    sums = grouped["sum"].unstack("horizon").reindex(index=months, columns=horizons, fill_value=0).to_numpy(float)
    counts = grouped["count"].unstack("horizon").reindex(index=months, columns=horizons, fill_value=0).to_numpy(float)
    rng = np.random.default_rng(seed)
    boot = np.empty((replications, len(horizons)), dtype=float)
    for index in range(replications):
        draw = rng.integers(0, len(months), size=len(months))
        denominator = counts[draw].sum(axis=0)
        boot[index] = np.divide(sums[draw].sum(axis=0), denominator,
                                out=np.full(len(horizons), np.nan), where=denominator > 0)
    grouped_mean = current.groupby("horizon")["excess_return"].agg(["count", "mean"])
    rows = [{"session": 0, "N": current["event_id"].nunique(), "mean_excess": 0.0,
             "ci_low": 0.0, "ci_high": 0.0}]
    for column, horizon in enumerate(horizons):
        rows.append({
            "session": int(horizon), "N": int(grouped_mean.loc[horizon, "count"]),
            "mean_excess": float(grouped_mean.loc[horizon, "mean"]),
            "ci_low": float(np.nanquantile(boot[:, column], 0.025)),
            "ci_high": float(np.nanquantile(boot[:, column], 0.975)),
        })
    return pd.DataFrame(rows)


def _full_s1_path_outcomes(analysis: dict, config: Config) -> pd.DataFrame:
    """Compute every S1 session from 1 through 63 with the tested return function."""
    from shared.engine.evaluate import event_returns

    events = analysis["signals"]["S1"].rename(columns={"permno": "security_id"})
    prices = analysis["levels"][["date", "security_id", "close", "delisted"]]
    return event_returns(events, prices, analysis["benchmark"], range(1, config.path_horizon + 1))


def _yearly_s1(s1_outcomes: pd.DataFrame) -> pd.DataFrame:
    """Return S1's 5-session summary statistics for each filing year."""
    from shared.engine.evaluate import summary_stats

    five = s1_outcomes.loc[s1_outcomes["horizon"].eq(5)].copy()
    five["year"] = pd.to_datetime(five["filing_date"]).dt.year
    return pd.DataFrame([{"year": int(year), **summary_stats(frame)}
                         for year, frame in five.groupby("year", sort=True)])


def _size_tables(s2_outcomes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize S2 by lagged-size tercile: returns the 5-session table and the all-horizon table."""
    from shared.engine.evaluate import summary_stats

    usable = s2_outcomes.loc[s2_outcomes["size_bucket"].isin(["small", "mid", "large"])].copy()
    rows = []
    for (bucket, horizon), frame in usable.groupby(["size_bucket", "horizon"], sort=False):
        rows.append({"size_tercile": bucket, "horizon": int(horizon), **summary_stats(frame)})
    all_horizons = pd.DataFrame(rows)
    five = all_horizons.loc[all_horizons["horizon"].eq(5)].copy()
    return five, all_horizons


def _distribution_tables(s1_outcomes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, int]:
    """Describe how concentrated S1's 5-session payoff is.

    Returns percentiles of excess dollars per $100, the mean after trimming the
    best trades or the top-contributing issuers, and related concentration
    figures.
    """
    five = s1_outcomes.loc[s1_outcomes["horizon"].eq(5) & s1_outcomes["excess_return"].notna()].copy()
    dollars = 100 * five["excess_return"]
    quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    percentiles = pd.DataFrame({
        "percentile": [f"p{int(q*100):02d}" for q in quantiles],
        "excess_dollars_per_100": [float(dollars.quantile(q)) for q in quantiles],
    })
    ranked = five.assign(excess_dollars=100 * five["excess_return"])
    issuer_contribution = ranked.groupby("issuer_cik")["excess_dollars"].sum().sort_values(ascending=False)
    top_five_issuers = set(issuer_contribution.head(5).index)
    top_ten_sum = float(issuer_contribution.head(10).sum())
    total_sum = float(issuer_contribution.sum())
    rows = [
        {"measure": "All S1 trades", "value": float(dollars.mean()), "unit": "mean excess $ per $100"},
        {
            "measure": "After removing top 1% of trades",
            "value": float(dollars[dollars.le(dollars.quantile(0.99))].mean()),
            "unit": "mean excess $ per $100",
        },
        {
            "measure": "After removing top 5% of trades",
            "value": float(dollars[dollars.le(dollars.quantile(0.95))].mean()),
            "unit": "mean excess $ per $100",
        },
        {
            "measure": "After dropping 5 largest-contribution issuers",
            "value": float(ranked.loc[~ranked["issuer_cik"].isin(top_five_issuers), "excess_dollars"].mean()),
            "unit": "mean excess $ per $100",
        },
        {
            "measure": "Share of total excess from top 10 issuers",
            "value": top_ten_sum / total_sum if total_sum else np.nan,
            "unit": "share",
        },
    ]
    outside = int(((dollars < -50) | (dollars > 100)).sum())
    return percentiles, pd.DataFrame(rows), dollars, outside


def _example_month(s1_signal: pd.DataFrame) -> pd.Period:
    """Pick the 2025 month with the most S1 events (earliest if tied) as the worked example."""
    in_2025 = s1_signal.loc[pd.to_datetime(s1_signal["filing_date"]).dt.year.eq(2025)].copy()
    months = pd.to_datetime(in_2025["filing_date"]).dt.to_period("M")
    counts = months.value_counts().sort_index()
    maximum = counts.max()
    return counts.loc[counts.eq(maximum)].index.min()


def _join_nonempty(values) -> str:
    """Join the distinct non-blank values, sorted, with ``"; "``."""
    return "; ".join(sorted({str(value).strip() for value in values if pd.notna(value) and str(value).strip()}))


def _load_example_metadata(example_events: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Load SEC-only labels and purchase amounts for the fixed-rule example month."""
    year = int(pd.to_datetime(example_events["filing_date"]).dt.year.iloc[0])
    submissions, transactions, owners = [], [], []
    for sub_path in sorted(config.normalized_sec_dir.glob(f"year={year}/quarter=*/submissions.parquet")):
        sub = pd.read_parquet(sub_path, columns=[
            "accession", "filing_date", "form", "issuer_cik", "issuer_name", "ticker", "is_amendment"
        ])
        sub["filing_date"] = pd.to_datetime(sub["filing_date"]).dt.normalize()
        sub = sub.loc[sub["form"].eq("4") & ~sub["is_amendment"].fillna(False)]
        keys = example_events[["issuer_cik", "filing_date"]].drop_duplicates()
        sub = sub.merge(keys, on=["issuer_cik", "filing_date"], how="inner")
        if sub.empty:
            continue
        accessions = set(sub["accession"])
        tx = pd.read_parquet(sub_path.parent / "transactions.parquet", columns=[
            "accession", "transaction_key", "transaction_code", "trade_value", "shares", "price"
        ])
        tx = tx.loc[tx["accession"].isin(accessions) & tx["transaction_code"].eq("P")]
        tx = tx.drop_duplicates("transaction_key", keep="last")
        owner = pd.read_parquet(sub_path.parent / "owners.parquet", columns=[
            "accession", "owner_cik", "role", "officer_title"
        ])
        owner = owner.loc[owner["accession"].isin(accessions) & owner["owner_cik"].notna()]
        submissions.append(sub)
        transactions.append(tx)
        owners.append(owner)
    sub = pd.concat(submissions, ignore_index=True)
    tx = pd.concat(transactions, ignore_index=True)
    owner = pd.concat(owners, ignore_index=True)
    tx["purchase_amount"] = pd.to_numeric(tx["trade_value"], errors="coerce")
    fallback = pd.to_numeric(tx["shares"], errors="coerce") * pd.to_numeric(tx["price"], errors="coerce")
    tx["purchase_amount"] = tx["purchase_amount"].fillna(fallback)
    amount = tx.groupby("accession", as_index=False)["purchase_amount"].sum(min_count=1)
    owner["role_label"] = owner["role"].fillna("").astype(str)
    title = owner["officer_title"].fillna("").astype(str).str.strip()
    owner.loc[title.ne(""), "role_label"] = owner.loc[title.ne(""), "role_label"] + " (" + title[title.ne("")] + ")"
    role = owner.groupby("accession")["role_label"].agg(_join_nonempty).rename("roles").reset_index()
    filings = sub.merge(amount, on="accession", how="inner").merge(role, on="accession", how="inner")
    return filings.groupby(["issuer_cik", "filing_date"], as_index=False).agg(
        ticker=("ticker", _join_nonempty), company=("issuer_name", _join_nonempty),
        insider_roles=("roles", _join_nonempty), purchase_amount=("purchase_amount", "sum"),
    )


def _example_trade_table(
    analysis: dict, extended: Dict[str, pd.DataFrame], month: pd.Period, config: Config
) -> pd.DataFrame:
    """Build the per-trade table for the example month: dates, roles, prices and returns."""
    five = extended["S1"].loc[extended["S1"]["horizon"].eq(5)].copy()
    five = five.loc[pd.to_datetime(five["filing_date"]).dt.to_period("M").eq(month)]
    metadata = _load_example_metadata(five[["issuer_cik", "filing_date"]], config)
    table = five.merge(metadata, on=["issuer_cik", "filing_date"], how="left")
    closes = analysis["daily"][["permno", "date", "dlyclose"]].drop_duplicates(["permno", "date"])
    entry = closes.rename(columns={"permno": "security_id", "date": "entry_date", "dlyclose": "entry_close"})
    exit_ = closes.rename(columns={"permno": "security_id", "date": "end_date", "dlyclose": "exit_close"})
    table = table.merge(entry, on=["security_id", "entry_date"], how="left")
    table = table.merge(exit_, on=["security_id", "end_date"], how="left")
    table["dollars_per_100"] = 100 * table["excess_return"]
    return table[[
        "event_id", "security_id", "ticker", "company", "filing_date", "entry_date", "end_date",
        "insider_roles", "purchase_amount", "entry_close", "exit_close", "stock_return",
        "benchmark_return", "excess_return", "dollars_per_100", "issuer_cik",
    ]].sort_values(["filing_date", "ticker"]).reset_index(drop=True)


def _example_path_data(example: pd.DataFrame, analysis: dict) -> pd.DataFrame:
    """Price paths from 10 sessions before to 10 after entry for three example trades.

    The trades are the one closest to the median excess return, the best and
    the worst of the example month.
    """
    valid = example.loc[example["excess_return"].notna()].copy()
    median = valid["excess_return"].median()
    choices = {
        "median-like": valid.loc[(valid["excess_return"] - median).abs().idxmin()],
        "best": valid.loc[valid["excess_return"].idxmax()],
        "worst": valid.loc[valid["excess_return"].idxmin()],
    }
    calendar = pd.DatetimeIndex(analysis["benchmark"].index)
    levels = analysis["levels"]
    rows = []
    for selection, trade in choices.items():
        position = int(calendar.get_loc(pd.Timestamp(trade["entry_date"])))
        positions = np.arange(max(0, position - 10), min(len(calendar), position + 11))
        dates = calendar[positions]
        sessions = positions - position
        stock = levels.loc[
            levels["security_id"].eq(trade["security_id"]) & levels["date"].isin(dates),
            ["date", "close"],
        ].set_index("date")["close"].reindex(dates)
        benchmark = analysis["benchmark"].reindex(dates)
        label = f"{trade['ticker']} ({pd.Timestamp(trade['filing_date']):%b %d})"
        for series_name, values in (("Stock", stock), ("XBI", benchmark)):
            base = float(values.loc[pd.Timestamp(trade["entry_date"])])
            for session, date, value in zip(sessions, dates, values):
                rows.append({
                    "selection": selection, "trade_label": label, "event_id": trade["event_id"],
                    "series": series_name, "session": int(session), "date": date,
                    "rebased_value": 100 * float(value) / base if pd.notna(value) else np.nan,
                })
    order = pd.CategoricalDtype(["median-like", "best", "worst"], ordered=True)
    result = pd.DataFrame(rows)
    result["selection"] = result["selection"].astype(order)
    return result.sort_values(["selection", "series", "session"])


def format_p_value(value: float) -> str:
    """Notebook-safe p-value formatting: small values are never rounded to zero."""
    return "—" if pd.isna(value) else f"{value:.2e}"


def format_inference_table(frame: pd.DataFrame, leading: tuple[str, ...]) -> pd.DataFrame:
    """Return a compact, presentation-ready table with explicit units."""
    out = frame.copy()
    out["mean (pp)"] = out["mean"].map(lambda x: f"{100*x:+.2f}")
    out["median (pp)"] = out["median"].map(lambda x: f"{100*x:+.2f}")
    out["hit rate"] = out["hit_rate"].map(lambda x: f"{100*x:.1f}%")
    out["95% CI (pp)"] = [f"[{100*a:+.2f}, {100*b:+.2f}]" for a, b in zip(out["ci_low"], out["ci_high"])]
    out["t"] = out["t"].map(lambda x: f"{x:.2f}")
    out["p"] = out["p_two_sided"].map(format_p_value)
    columns = list(leading) + ["N", "mean (pp)", "median (pp)", "hit rate", "95% CI (pp)",
                                "t", "p", "issuer_clusters", "month_clusters"]
    return out[columns]


def format_example_table(example: pd.DataFrame) -> pd.DataFrame:
    """Format all example trades and append the requested summary row."""
    out = pd.DataFrame({
        "Ticker": example["ticker"], "Company": example["company"],
        "Filing date": pd.to_datetime(example["filing_date"]).dt.strftime("%Y-%m-%d"),
        "Entry date": pd.to_datetime(example["entry_date"]).dt.strftime("%Y-%m-%d"),
        "Insider role(s)": example["insider_roles"],
        "Purchase amount": example["purchase_amount"].map(lambda x: f"${x:,.0f}" if pd.notna(x) else "—"),
        "Entry close": example["entry_close"].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "—"),
        "Exit close (5 sessions)": example["exit_close"].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "—"),
        "Stock return": example["stock_return"].map(lambda x: f"{100*x:+.2f}%"),
        "XBI return": example["benchmark_return"].map(lambda x: f"{100*x:+.2f}%"),
        "Excess return": example["excess_return"].map(lambda x: f"{100*x:+.2f} pp"),
        "$ result per $100": example["dollars_per_100"].map(lambda x: f"${x:+.2f}"),
    })
    valid = example["excess_return"].dropna()
    summary = {column: "" for column in out.columns}
    summary.update({
        "Ticker": "SUMMARY", "Company": f"N={len(valid)}; hits={100*(valid > 0).mean():.1f}%",
        "Excess return": f"mean {100*valid.mean():+.2f} pp",
        "$ result per $100": f"median ${100*valid.median():+.2f}",
    })
    return pd.concat([out, pd.DataFrame([summary])], ignore_index=True)


def build_presentation_outputs(analysis: dict, config: Config = DEFAULT_CONFIG) -> dict:
    """Compute notebook tables, save aggregate CSVs, and render every requested PNG."""
    from shared.engine import figures

    results_dir = config.results_dir
    figure_dir = results_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    extended = _extended_outcomes(analysis)
    horizons = _summaries_by_horizon(extended)
    s1_path = _month_cluster_bootstrap_path(_full_s1_path_outcomes(analysis, config))
    yearly = _yearly_s1(extended["S1"])
    size_five, size_horizons = _size_tables(extended["S2"])
    percentiles, concentration, dollars, outside = _distribution_tables(extended["S1"])

    annual_counts = pd.concat([
        frame.assign(signal=signal, year=pd.to_datetime(frame["filing_date"]).dt.year)
        for signal, frame in analysis["signals"].items()
    ]).groupby(["year", "signal"]).size().rename("events").reset_index()
    headline = analysis["results_table"].loc[
        analysis["results_table"]["horizon"].eq(5)
        & analysis["results_table"]["period"].eq("2009-2025")
    ].copy()
    subperiods = analysis["results_table"].loc[
        analysis["results_table"]["signal"].eq("S1")
        & analysis["results_table"]["horizon"].eq(5)
    ].copy()
    coverage = pd.DataFrame([
        {"measure": "Universe securities (PERMNOs)", "value": analysis["coverage"]["universe_permnos"]},
        {"measure": "Universe securities with prices", "value": analysis["coverage"]["universe_permnos_with_prices"]},
        *({"measure": f"{signal} events", "value": len(frame)} for signal, frame in analysis["signals"].items()),
    ])

    month = _example_month(analysis["signals"]["S1"])
    example = _example_trade_table(analysis, extended, month, config)
    example_paths = _example_path_data(example, analysis)
    s1_five = extended["S1"].loc[extended["S1"]["horizon"].eq(5),
                                  ["security_id", "entry_date", "excess_return"]]
    entry_closes = analysis["daily"][["permno", "date", "dlyclose"]].rename(
        columns={"permno": "security_id", "date": "entry_date", "dlyclose": "entry_close"}
    )
    s1_with_price = s1_five.merge(entry_closes, on=["security_id", "entry_date"], how="left")
    low_price_share = float(s1_with_price.loc[s1_with_price["excess_return"].notna(), "entry_close"].lt(5).mean())

    aggregate_tables = {
        "events_by_year.csv": annual_counts, "headline_table.csv": headline,
        "horizons_descriptive.csv": horizons, "s1_cumulative_path_ci.csv": s1_path,
        "s1_yearly.csv": yearly, "s1_subperiods.csv": subperiods,
        "size_gradient_h5.csv": size_five, "size_horizon.csv": size_horizons,
        "distribution_percentiles.csv": percentiles, "concentration_summary.csv": concentration,
    }
    for filename, frame in aggregate_tables.items():
        frame.to_csv(results_dir / filename, index=False)

    figures.events_by_year(annual_counts, figure_dir / "events_by_year.png")
    figures.horizon_profile(horizons, figure_dir / "horizon_profile.png")
    figures.cumulative_s1(s1_path, figure_dir / "s1_cumulative_path_ci.png")
    figures.yearly_stability(yearly, figure_dir / "s1_yearly_stability.png")
    figures.size_gradient(size_five, figure_dir / "size_gradient_h5.png")
    figures.size_horizon_heatmap(size_horizons, figure_dir / "size_horizon_heatmap.png")
    figures.per_100_histogram(dollars, outside, figure_dir / "s1_per_100_histogram.png")
    figures.example_paths(example_paths, figure_dir / "example_month_paths.png")

    peak = s1_path.loc[s1_path["mean_excess"].idxmax()]
    return {
        "coverage": coverage, "events_by_year": annual_counts, "headline": headline,
        "horizons": horizons, "s1_path": s1_path, "yearly": yearly,
        "positive_year_share": float(yearly["mean"].gt(0).mean()), "subperiods": subperiods,
        "size_five": size_five, "size_horizons": size_horizons,
        "percentiles": percentiles, "concentration": concentration,
        "histogram_outside": outside, "example_month": str(month), "example": example,
        "example_paths": example_paths, "peak_session": int(peak["session"]),
        "peak_mean": float(peak["mean_excess"]), "low_price_share": low_price_share,
        "figures_dir": figure_dir,
    }
