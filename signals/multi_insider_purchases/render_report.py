"""Render saved aggregate JSON as Markdown without running an analysis."""
import argparse
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTECTED = {HERE / name for name in ("run_split.py", "results.json", "RESULTS.md", "train_ledger.csv")}


def number(value, digits=2, signed=False):
    if not isinstance(value, (float, int)) or not math.isfinite(value):
        return "—"
    return format(value, f"{'+' if signed else ''}.{digits}f")


def config(row):
    return f"w={row['w']}, k={row['k']}, h={row['h']}, {row['est']}"


def estimate(row, field="mean_pp"):
    if row.get("status") not in ("ok", "descriptive", "descriptive_secondary"):
        return f"Unavailable ({row.get('status', 'status not recorded')}; n={row.get('n', '—')})"
    return (f"{number(row.get(field), signed=True)} pp; SE {number(row.get('se_pp'))} "
            f"({row.get('se_source', '—')}); t={number(row.get('t'))}; "
            f"two-sided p={number(row.get('p_two'), 3)}; n={row.get('n', row.get('n_clustered', '—'))}")


def interval(row):
    return f"[{number(row.get('ci_low_pp'), signed=True)}, {number(row.get('ci_high_pp'), signed=True)}] pp"


def render(res, source="saved results.json", source_sha=None):
    """Render saved results, including statuses with no available estimate."""
    md = ["# Multi-insider purchases: results, readable version of RESULTS.md", "",
          f"Source: {source}. All estimates below are read from that saved result.", "",
          f"Recorded decision: **{res.get('decision', 'Not recorded')}**", ""]
    if source_sha:
        md += [f"Source SHA-256: `{source_sha}`.", ""]
    selected = res.get("selected")
    if selected:
        md += [f"Selected configuration: **{config(selected)}**.", ""]
    test = res.get("test")
    if test:
        period = " to ".join(res.get("stages", {}).get("test", []))
        md += [f"## Test result ({period})", "", estimate(test), ""]
        if test.get("status") == "ok":
            md += [f"95% interval: {interval(test)}. {test['issuers']} issuers; {test['quarters']} quarters.", ""]
    md += ["## Training and validation", "",
           "Validation retains the largest positive t-statistic with one-sided p < .10 among the three selected training candidates.", "",
           "| Validation candidate | Mean (pp) | SE (pp) | t | One-sided p | n | Status |",
           "|---|---:|---:|---:|---:|---:|---|"]
    for row in res.get("validation", []):
        md.append(f"| {config(row)} | {number(row.get('mean_pp'), signed=True)} | {number(row.get('se_pp'))} | "
                  f"{number(row.get('t'))} | {number(row.get('p_one'), 3)} | {row.get('n', '—')} | {row.get('status', '—')} |")
    md += ["", "Top 10 of 56 training configurations. "
           "Yes marks a wins100 row whose mean and SE equal its mean-estimator twin to 10 decimals; such rows were skipped when picking the top three.", "",
           "| Configuration | Mean (pp) | SE (pp) | t | Holm p | n | Duplicate |",
           "|---|---:|---:|---:|---:|---:|---|"]
    for row in res.get("train_top", []):
        md.append(f"| {config(row)} | {number(row.get('mean_pp'), signed=True)} | {number(row.get('se_pp'))} | "
                  f"{number(row.get('t'))} | {number(row.get('p_holm_56'), 3)} | "
                  f"{row.get('n', '—')} | {'Yes' if row.get('duplicate_of_earlier_row') else 'No'} |")
    attrition = res.get("attrition_selected_config", {})
    if attrition:
        md += ["", "## Sample attrition", "",
               "Spacing is applied before returns are checked. The exit column counts exits missing or outside the stage.", "",
               "| Stage | Arm | Company-days | After spacing | Exit unavailable/outside stage | Nonfinite return | Used |",
               "|---|---|---:|---:|---:|---:|---:|"]
        for stage, rows in attrition.items():
            for row in rows:
                arm = "Clustered" if row["arm"] == "clustered" else "Below buyer threshold"
                md.append(f"| {stage} | {arm} | {row['company_days']} | {row['after_spacing']} | "
                          f"{row['dropped_exit_beyond_stage']} | {row['dropped_nonfinite_outcome']} | {row['used']} |")
    descriptives = res.get("descriptives_train_validation", {})
    if descriptives:
        md += ["", "## Train/validation descriptives", "",
               "Price is the median reported purchase price on the filing company-day, a proxy rather than an entry quote.", "",
               "| Stage | Reported price | n | Share of events | Mean (pp) | t | Status |",
               "|---|---|---:|---:|---:|---:|---|"]
        missing_prices = []
        for stage, data in descriptives.items():
            ps = data.get("price_screen", {})
            for floor in (1, 5):
                row = ps.get(f"price_ge_{floor}", {})
                share = row.get("share_of_events")
                pct = number(100 * share, 0) + "%" if share is not None else "—"
                md.append(f"| {stage} | ≥ ${floor} | {row.get('n', '—')} | {pct} | "
                          f"{number(row.get('mean_pp'), signed=True)} | {number(row.get('t'))} | {row.get('status', '—')} |")
            missing_prices.append(f"{stage}: missing reported price for {ps.get('missing_price', '—')} events.")
        md += [""] + missing_prices + [""]
        for stage, data in descriptives.items():
            row = data.get("prior_return_controlled", {})
            md += [f"- {stage}, clustered mean at comparison-arm prior-return levels: {estimate(row, 'intercept_pp')}."]
        md += ["", "Controlled SEs condition on the observed comparison-arm covariate means."]
    g = res.get("fragility")
    if g:
        md += ["", "## Fragility and dependence", "",
               "| Diagnostic | Saved result |", "|---|---|",
               f"| Issuer-pairs bootstrap, 95% interval | {interval(g['pairs_bootstrap'])} |",
               f"| Issuer × circular two-quarter product-weight sensitivity | {interval(g['block_bootstrap_sensitivity'])} |",
               f"| Leave-one-issuer-out mean range | [{number(g['leave_one_issuer_out_pp'][0], signed=True)}, {number(g['leave_one_issuer_out_pp'][1], signed=True)}] pp |",
               f"| Largest issuer share | {number(100 * g['largest_issuer_share'], 1)}% |", "",
               "Issuer-only resampling does not retain shared calendar shocks. The two resampling schemes are different dependence checks; neither interval is guaranteed to be wider.", "",
               "| Period | Events | Mean (pp) |", "|---|---:|---:|"]
        for period, row in g["halves"].items():
            md.append(f"| {period} | {row['n']} | {number(row['mean_pp'], signed=True)} |")
        labels = {"pairs_bootstrap_interval_includes_zero": "Pairs interval includes zero",
                  "leave_one_issuer_out_changes_sign": "Issuer omission changes the mean's sign",
                  "largest_issuer_share_above_10pct": "Largest issuer exceeds 10% of events",
                  "halves_have_opposite_signs": "Period halves have opposite signs"}
        md += ["", "| Flag | Triggered |", "|---|---|"]
        for key, value in g["flags"].items():
            md.append(f"| {labels.get(key, key)} | {'Unavailable' if value is None else 'Yes' if value else 'No'} |")
    secondary = res.get("secondaries", {})
    if secondary:
        md += ["", "## Descriptive secondaries", "",
               f"- Clustered minus below-threshold purchases: {estimate(secondary['contrast'], 'contrast_pp')}; comparison n={secondary['contrast'].get('n_solo', '—')}.",
               f"- Primary relative to XBI: {estimate(secondary['xbi_primary'])}.", "",
               "| Horizon (sessions) | Mean (pp) | SE (pp) | t | Two-sided p | n | Status |",
               "|---:|---:|---:|---:|---:|---:|---|"]
        for row in secondary.get("horizons_descriptive", []):
            md.append(f"| {row['h']} | {number(row.get('mean_pp'), signed=True)} | {number(row.get('se_pp'))} | "
                      f"{number(row.get('t'))} | {number(row.get('p_two'), 3)} | {row.get('n', '—')} | {row.get('status', '—')} |")
    if res.get("reference_c01_on_test"):
        md += ["", f"C01 descriptive reference ({config(res['reference_c01_on_test'])}): " + estimate(res["reference_c01_on_test"]) + "."]
    md += ["", "## Interpretation and inputs", "",
           "The maximum of four estimated SEs is an ad hoc safeguard; its finite-sample coverage is not established. "
           "The validation screen uses few time clusters. Calendar-quarter clustering does not cover every form of dependence across quarters.", "",
           "Returns are gross event-level excess returns. A constant additional per-event cost equal to the mean would erase that excess over a costless benchmark. Portfolio turnover and execution costs are not modeled.", ""]
    for key, value in res.get("inputs", {}).items():
        if key == "environment":
            md += ["", "Recorded analysis environment:", ""] + [f"- {name}: {version}" for name, version in value.items()]
        else:
            md.append(f"- {key}: `{value}`")
    return "\n".join(md) + "\n"


def validate_output(input_path, output_path):
    output = Path(output_path).resolve()
    if output.suffix.lower() != ".md":
        raise ValueError("Report output must be a Markdown (.md) file")
    if output in {p.resolve() for p in PROTECTED} | {Path(input_path).resolve()}:
        raise ValueError("Output must not overwrite an original artifact or the input")
    if output.exists() and output != HERE / "REPORT.md":
        raise FileExistsError("Choose a new report path; only the separate REPORT.md can be regenerated")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=HERE / "results.json")
    ap.add_argument("--output", type=Path, default=HERE / "REPORT.md")
    args = ap.parse_args()
    try:
        validate_output(args.input, args.output)
    except (ValueError, FileExistsError) as error:
        ap.error(str(error))
    data = args.input.read_bytes()
    text = render(json.loads(data), args.input.name, hashlib.sha256(data).hexdigest())
    args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
