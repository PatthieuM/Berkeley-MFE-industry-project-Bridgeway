"""Render static figures from insider-signal summary tables.

Each public plotting function accepts a prepared table and writes an image file for
the research deliverable; no analytical tables are modified.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SIGNAL_COLORS = {"S1": "#16817A", "S2": "#3569A8", "S3": "#C05874"}
SIZE_COLORS = {"small": "#16817A", "mid": "#6E91B8", "large": "#C05874"}
SOURCE = "Source: SEC Form 4, CRSP, universe_v4; 2009–2025"


def set_presentation_style() -> None:
    """Apply the shared Matplotlib style used by all deliverable figures."""
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.titlesize": 13,
        "axes.labelsize": 10, "font.size": 10,
        "legend.frameon": False, "grid.alpha": 0.22,
        "savefig.bbox": "tight", "savefig.dpi": 180,
    })


def _finish(fig: plt.Figure, output: Path) -> None:
    """Stamp the source note on a figure, save it to ``output`` and close it."""
    fig.text(0.01, 0.008, SOURCE, fontsize=8, color="#666666")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def events_by_year(table: pd.DataFrame, output: Path) -> None:
    """Write a stacked chart of S1-S3 event counts by filing year."""
    set_presentation_style()
    pivot = table.pivot(index="year", columns="signal", values="events").fillna(0)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bottom = np.zeros(len(pivot))
    for signal in ("S1", "S2", "S3"):
        values = pivot.get(signal, pd.Series(0, index=pivot.index)).to_numpy()
        ax.bar(pivot.index, values, bottom=bottom, color=SIGNAL_COLORS[signal], label=signal, width=0.78)
        bottom += values
    ax.set_title("Signal events by filing year")
    ax.set_xlabel("Filing year")
    ax.set_ylabel("Issuer–filing-date events")
    ax.set_xticks(pivot.index)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(ncol=3)
    ax.grid(axis="y")
    _finish(fig, output)


def horizon_profile(table: pd.DataFrame, output: Path) -> None:
    """Write signal mean excess returns and confidence bands by horizon."""
    set_presentation_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.9))
    for signal in ("S1", "S2", "S3"):
        current = table.loc[table["signal"].eq(signal)].sort_values("horizon")
        x = current["horizon"].to_numpy(float)
        mean = 100 * current["mean"].to_numpy(float)
        low = 100 * current["ci_low"].to_numpy(float)
        high = 100 * current["ci_high"].to_numpy(float)
        ax.plot(x, mean, marker="o", linewidth=2, color=SIGNAL_COLORS[signal], label=signal)
        ax.fill_between(x, low, high, color=SIGNAL_COLORS[signal], alpha=0.13)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.axvline(5, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_title("Excess returns by horizon (descriptive beyond pre-registered 5 sessions)")
    ax.set_xlabel("Trading sessions after entry")
    ax.set_ylabel("Mean excess return (percentage points)")
    ax.set_xticks(sorted(table["horizon"].unique()))
    ax.legend(ncol=3)
    ax.grid(axis="y")
    _finish(fig, output)


def cumulative_s1(path: pd.DataFrame, output: Path) -> None:
    """Write the S1 cumulative excess-return path with its confidence band."""
    set_presentation_style()
    current = path.sort_values("session")
    fig, ax = plt.subplots(figsize=(8.5, 4.9))
    x = current["session"].to_numpy(float)
    mean = 100 * current["mean_excess"].to_numpy(float)
    low = 100 * current["ci_low"].to_numpy(float)
    high = 100 * current["ci_high"].to_numpy(float)
    ax.plot(x, mean, color=SIGNAL_COLORS["S1"], linewidth=2.2, label="S1 mean")
    ax.fill_between(x, low, high, color=SIGNAL_COLORS["S1"], alpha=0.18, label="95% month-cluster bootstrap CI")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.axvline(5, color="#777777", linewidth=0.8, linestyle="--", label="Pre-registered horizon")
    peak = current.loc[current["mean_excess"].idxmax()]
    ax.scatter([peak["session"]], [100 * peak["mean_excess"]], color=SIGNAL_COLORS["S1"], zorder=3)
    ax.annotate(f"Peak: {int(peak['session'])} sessions, {100*peak['mean_excess']:+.2f} pp",
                (peak["session"], 100 * peak["mean_excess"]), xytext=(8, 10), textcoords="offset points")
    ax.set_title("S1 cumulative average excess-return path")
    ax.set_xlabel("Trading sessions after entry")
    ax.set_ylabel("Mean cumulative excess return (percentage points)")
    ax.legend(loc="best")
    ax.grid(axis="y")
    _finish(fig, output)


def yearly_stability(table: pd.DataFrame, output: Path) -> None:
    """Write annual S1 five-session means with confidence intervals."""
    set_presentation_style()
    current = table.sort_values("year")
    means = 100 * current["mean"].to_numpy(float)
    low = 100 * current["ci_low"].to_numpy(float)
    high = 100 * current["ci_high"].to_numpy(float)
    errors = np.vstack([means - low, high - means])
    colors = [SIGNAL_COLORS["S1"] if value > 0 else "#A9A9A9" for value in means]
    fig, ax = plt.subplots(figsize=(10, 4.9))
    ax.bar(current["year"], means, color=colors, width=0.76)
    ax.errorbar(current["year"], means, yerr=errors, fmt="none", ecolor="#333333", capsize=2, linewidth=0.8)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("S1 five-session mean by filing year")
    ax.set_xlabel("Filing year")
    ax.set_ylabel("Mean excess return (percentage points)")
    ax.set_xticks(current["year"])
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y")
    _finish(fig, output)


def size_gradient(table: pd.DataFrame, output: Path) -> None:
    """Write five-session returns across lagged market-cap terciles."""
    set_presentation_style()
    order = ["small", "mid", "large"]
    current = table.set_index("size_tercile").reindex(order)
    means = 100 * current["mean"].to_numpy(float)
    errors = np.vstack([means - 100 * current["ci_low"].to_numpy(float),
                        100 * current["ci_high"].to_numpy(float) - means])
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.bar([x.title() for x in order], means, color=[SIZE_COLORS[x] for x in order], width=0.62)
    ax.errorbar(range(3), means, yerr=errors, fmt="none", ecolor="#333333", capsize=4, linewidth=1)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("All purchases: five-session excess return by lagged-cap tercile")
    ax.set_xlabel("Lagged market-cap tercile (prior-years-only breakpoints)")
    ax.set_ylabel("Mean excess return (percentage points)")
    ax.grid(axis="y")
    _finish(fig, output)


def size_horizon_heatmap(table: pd.DataFrame, output: Path) -> None:
    """Write a heatmap of mean returns by size tercile and horizon."""
    set_presentation_style()
    order = ["small", "mid", "large"]
    pivot = table.pivot(index="size_tercile", columns="horizon", values="mean").reindex(order) * 100
    values = pivot.to_numpy(float)
    bound = max(0.5, float(np.nanmax(np.abs(values))))
    fig, ax = plt.subplots(figsize=(9.2, 3.7))
    image = ax.imshow(values, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:+.2f}", ha="center", va="center",
                    color="white" if abs(values[i, j]) > 0.55 * bound else "#222222", fontsize=9)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(order)), [x.title() for x in order])
    ax.set_title("All purchases: size tercile × return horizon")
    ax.set_xlabel("Trading sessions after entry")
    ax.set_ylabel("Lagged market-cap tercile")
    cbar = fig.colorbar(image, ax=ax, shrink=0.85)
    cbar.set_label("Mean excess return (percentage points)")
    _finish(fig, output)


def per_100_histogram(values: pd.Series, outside: int, output: Path) -> None:
    """Write the clipped distribution of excess dollars per $100 invested."""
    set_presentation_style()
    clean = pd.to_numeric(values, errors="coerce").dropna()
    shown = clean.clip(-50, 100)
    fig, ax = plt.subplots(figsize=(8.5, 4.9))
    ax.hist(shown, bins=np.linspace(-50, 100, 51), color=SIGNAL_COLORS["S1"], alpha=0.9)
    ax.axvline(clean.mean(), color="#222222", linestyle="--", label=f"Mean {clean.mean():+.2f}")
    ax.axvline(clean.median(), color=SIGNAL_COLORS["S3"], linestyle=":", label=f"Median {clean.median():+.2f}")
    ax.set_xlim(-50, 100)
    ax.set_title("S1 five-session result per $100 invested")
    ax.set_xlabel("Excess result versus XBI (dollars per $100)")
    ax.set_ylabel("Issuer–filing-date events")
    ax.text(0.99, 0.96, f"{outside:,} observations outside [−50, +100] are clipped to the boundaries",
            transform=ax.transAxes, ha="right", va="top", fontsize=9)
    ax.legend()
    ax.grid(axis="y")
    _finish(fig, output)


def example_paths(paths: pd.DataFrame, output: Path) -> None:
    """Write selected stock and benchmark paths around representative S1 events."""
    set_presentation_style()
    selected = list(dict.fromkeys(paths["selection"].tolist()))
    fig, axes = plt.subplots(1, len(selected), figsize=(12, 4.3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, selection in zip(axes, selected):
        current = paths.loc[paths["selection"].eq(selection)]
        title = current["trade_label"].iloc[0]
        for series, frame in current.groupby("series", sort=False):
            color = SIGNAL_COLORS["S1"] if series == "Stock" else "#555555"
            style = "-" if series == "Stock" else "--"
            ax.plot(frame["session"], frame["rebased_value"], color=color, linestyle=style,
                    linewidth=2, label=series)
        ax.axvline(0, color="#777777", linewidth=0.8)
        ax.axvline(5, color="#777777", linewidth=0.8, linestyle=":")
        ax.set_title(f"{selection.title()}: {title}", fontsize=10)
        ax.set_xlabel("Sessions from entry")
        ax.grid(axis="y")
    axes[0].set_ylabel("Total-return index (entry = 100)")
    axes[-1].legend(loc="best")
    fig.suptitle("Illustration: median-like, best, and worst S1 trades in the example month", fontweight="bold")
    _finish(fig, output)


def seller_silence(table: pd.DataFrame, output: Path) -> None:
    """Plot full-period SSN and SSS-only mean excess returns by horizon."""
    set_presentation_style()
    current = table.loc[table["period"].eq("2009-2025")].sort_values("horizon")
    fig, ax = plt.subplots(figsize=(8.5, 4.9))
    x = current["horizon"].to_numpy(float)
    ax.plot(x, 100 * current["SSN_mean"], marker="o", linewidth=2.1,
            color=SIGNAL_COLORS["S1"], label="SSN")
    ax.plot(x, 100 * current["SSS_only_mean"], marker="o", linewidth=2.1,
            color=SIGNAL_COLORS["S3"], label="SSS-only")
    ax.axhline(0, color="#333333", linewidth=0.8)
    for horizon in (63, 126):
        ax.axvline(horizon, color="#777777", linewidth=0.7, linestyle="--")
    ax.set_title("S4 seller silence: excess returns by horizon")
    ax.set_xlabel("Trading sessions after entry")
    ax.set_ylabel("Mean excess return (percentage points)")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(axis="y")
    _finish(fig, output)
