"""run_split.py — train / validation / test search for one biotech multi-insider purchase signal.
Reads only the four frozen inputs in frozen/; computes no new returns. See README.md.
"""
import argparse, hashlib, importlib.util, json, platform
from pathlib import Path
import numpy as np, pandas as pd, scipy, statsmodels
from scipy.stats import t as student_t
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_cluster, cov_cluster_2groups

HERE = Path(__file__).resolve().parent
INPUTS = ["events.parquet", "purchases.parquet", "classification.py", "protocol.json"]   # the frozen inputs in frozen/
ARCHIVE_SHA256 = "7e7d284dff704aaf2b66eaa975df135c54d246b72940c77edef0f51da6ee51e4"   # historical archive hash; not read by this loader
STAGES = {"train": ("2009-01-01", "2016-12-31"), "validation": ("2017-01-01", "2019-12-31"),
          "test": ("2020-01-01", "2025-12-31")}
WINDOWS, MIN_BUYERS, HORIZONS, ESTIMATORS = (7, 14), (2, 3), (1, 2, 3, 5, 10, 21, 63), ("mean", "wins100")
TOP_K, VALID_ONE_SIDED_P, ALPHA, SEED, DRAWS = 3, 0.10, 0.05, 20260916, 1999
REFERENCE = (7, 2, 10, "mean")                       # frozen C01, shown descriptively when a configuration is selected
HALF_SPLIT = {"test": "2023-01-01"}                  # declared halves 2020-2022 / 2023-2025 (README, Selection and inference, step 4)
PRIOR_COLS = ["prior_return_21", "prior_return_252_21"]   # frozen protocol controls, descriptive only

def sha(path):
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

PINNED_SHA256 = {                                   # frozen inputs; the run refuses to proceed on any mismatch
    "events.parquet": "2b478839343afe5d527c996c2d1ea90eeb76a895a3236bde114a93c7f404d9df",
    "purchases.parquet": "97b762aa9264d6f16149eab10b78537545acd26b7bc8f5180a41f06f48b0b62f",
    "classification.py": "4297a1f2fb5d8febac8ed24d425bf60c21d7cc31a7073aa71f83b84cff902770",
    "protocol.json": "6cf61060cad0ce87cfa913cc1393263f05f9b3f5a6f751300e1bc91e2d9936a4",
}

def verify_inputs(folder=None):
    """Check that frozen/ holds the four inputs with their pinned SHA-256. Never regenerates or deletes them."""
    folder = Path(folder) if folder is not None else HERE / "frozen"
    for name in INPUTS:
        path = folder / name
        if not path.exists() or sha(path) != PINNED_SHA256[name]:
            raise RuntimeError(f"{folder.name}/{name} is missing or does not match its pinned SHA-256; "
                               "restore the four inputs from the author's backup, never regenerate them")
    return {name: sha(folder / name) for name in INPUTS} | {
        "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                        "statsmodels": statsmodels.__version__, "scipy": scipy.__version__}}

def load_classifier():
    """Load the frozen classification.py from frozen/; run_split uses only its cluster_company_days."""
    spec = importlib.util.spec_from_file_location("frozen_classification", HERE / "frozen/classification.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def build_rows(last_date="2025-12-31"):
    """Every eligible purchase company-day with outcomes at all horizons, plus cluster counts per window.
    `last_date` is the firewall: validation mode never loads rows with test-period signal dates."""
    E = pd.read_parquet(HERE / "frozen/events.parquet",
                        filters=[("signal_date", "<=", pd.Timestamp(last_date))])
    rows = E.loc[E.signal_family.eq("cluster") & E.event_status.eq("eligible")].copy()
    rows["signal_date"] = pd.to_datetime(rows.signal_date).dt.normalize()
    rows["issuer_id"] = rows.issuer_id.astype(str)
    rows = rows.loc[rows.signal_date.between("2009-01-01", last_date)]
    P = pd.read_parquet(HERE / "frozen/purchases.parquet",
                        filters=[("signal_date", "<=", pd.Timestamp(last_date))])
    selected = P.loc[P.sample_eligible & P.purchase_eligible]
    cls = load_classifier()
    for w in WINDOWS:
        cd = cls.cluster_company_days(selected, window_days=w)
        cd["signal_date"] = pd.to_datetime(cd.signal_date).dt.normalize()
        cd["issuer_id"] = cd.issuer_id.astype(str)
        cd = cd[["issuer_id", "signal_date", "cluster_count", "event_status"]].rename(
            columns={"cluster_count": f"buyers_w{w}", "event_status": f"status_w{w}"})
        rows = rows.merge(cd, on=["issuer_id", "signal_date"], how="left", validate="many_to_one")
    if rows.buyers_w7.isna().any() or not rows.buyers_w7.eq(rows.cluster_count).all():
        raise RuntimeError("7-day relabelling does not reproduce the frozen cluster counts")
    for w in WINDOWS:                                   # a multi-security window under either w excludes the day
        rows = rows.loc[rows[f"status_w{w}"].eq("eligible")]
    rows["quarter"] = rows.signal_date.dt.to_period("Q").astype(str)
    rows["qord"] = rows.signal_date.dt.to_period("Q").astype("int64")
    # Median reported purchase price per company-day: a price-level proxy for the tradeability descriptive.
    px = selected.assign(
        issuer_id=selected.issuer_id.astype(str),
        signal_date=pd.to_datetime(selected.signal_date).dt.normalize(),
        price=pd.to_numeric(selected.price, errors="coerce"),
    )
    prices = px.groupby(["issuer_id", "signal_date"]).price.median().rename("purchase_price_median").reset_index()
    return rows, prices

def spaced(rows, horizon, start, end, arm):
    """Frozen extension_sample rule: within issuer and arm, first event then next only after `horizon` sessions;
    outcome window must end inside the stage. Selection never looks at return values."""
    r = rows.loc[rows.horizon.eq(horizon) & rows.signal_date.between(start, end) & (rows.arm == arm)]
    keep = []
    for _, g in r.sort_values(["signal_date", "permno"]).groupby("issuer_id", sort=False):
        last = -1
        for i, e in zip(g.index, g.entry_pos):
            if int(e) >= last:
                keep.append(i)
                last = int(e) + horizon
    r = rows.loc[keep]
    r = r.loc[r.exit_date.notna() & (pd.to_datetime(r.exit_date) <= pd.Timestamp(end))]
    return r.loc[np.isfinite(r.excess_return)]

def attrition(rows, horizon, start, end, arm):
    """Outcome-blind sample-attrition counts for one stage/horizon/arm, mirroring spaced() step by step."""
    r = rows.loc[rows.horizon.eq(horizon) & rows.signal_date.between(start, end) & (rows.arm == arm)]
    keep = []
    for _, g in r.sort_values(["signal_date", "permno"]).groupby("issuer_id", sort=False):
        last = -1
        for i, e in zip(g.index, g.entry_pos):
            if int(e) >= last:
                keep.append(i)
                last = int(e) + horizon
    s = rows.loc[keep]
    inside = s.exit_date.notna() & (pd.to_datetime(s.exit_date) <= pd.Timestamp(end))
    nonfinite = int((~np.isfinite(s.loc[inside].excess_return)).sum())
    return dict(horizon=horizon, arm=arm, company_days=len(r), after_spacing=len(s),
                dropped_exit_beyond_stage=int((~inside).sum()), dropped_nonfinite_outcome=nonfinite,
                used=int(inside.sum()) - nonfinite)

def outcome(r, est, col="excess_return"):
    """Return the event outcomes, winsorized at +/-100 pp when the estimator is wins100."""
    y = r[col].to_numpy(float)
    return np.clip(y, -1.0, 1.0) if est == "wins100" else y

def floored_variance(fit, iss, q, col):
    """SE floor: the two-way CR1 estimator V_issuer + V_quarter - V_both has no lower bound and can fall far below
    every one-way and iid-robust estimate under fat-tailed outcomes. Use the largest of the four."""
    if len(np.unique(iss)) < 2 or len(np.unique(q)) < 2:
        return np.nan, "insufficient_clusters", {}
    parts = dict(two_way=float(cov_cluster_2groups(fit, iss, q, use_correction=True)[0][col, col]),
                 issuer=float(cov_cluster(fit, iss, use_correction=True)[col, col]),
                 quarter=float(cov_cluster(fit, q, use_correction=True)[col, col]),
                 hc1=float(fit.get_robustcov_results(cov_type="HC1").cov_params()[col, col]))
    if not all(np.isfinite(v) for v in parts.values()):      # fail closed: all four components must be finite
        return np.nan, "nonfinite_component", parts
    source = max(parts, key=parts.get)
    return parts[source], source, parts

def fit_mean(r, y):
    """Estimate a mean and floored clustered inference statistics."""
    if len(r) < 2:
        return dict(n=len(r), status="too_few")
    fit = OLS(y, np.ones((len(y), 1))).fit()
    iss = pd.factorize(r.issuer_id, sort=True)[0]
    q = pd.factorize(r.quarter, sort=True)[0]
    ni, nq = int(iss.max() + 1), int(q.max() + 1)
    var, source, parts = floored_variance(fit, iss, q, 0)
    df = min(ni, nq) - 1
    if not np.isfinite(var) or var <= 0 or df < 1:
        return dict(n=len(r), issuers=ni, quarters=nq, status=f"degenerate_variance:{source}")
    m, se = float(fit.params[0]), float(np.sqrt(var))
    tt = m / se
    return dict(
        n=len(r),
        issuers=ni,
        quarters=nq,
        mean_pp=m * 100,
        se_pp=se * 100,
        se_source=source,
        **{f"se_{k}_pp": float(np.sqrt(v)) * 100 if v > 0 else None for k, v in parts.items()},
        t=tt,
        df=df,
        p_two=float(2 * student_t.sf(abs(tt), df)),
        p_one=float(student_t.sf(tt, df)),
        ci_low_pp=(m - student_t.ppf(0.975, df) * se) * 100,
        ci_high_pp=(m + student_t.ppf(0.975, df) * se) * 100,
        inference="floored SE = max(two-way CR1, issuer CR1, quarter CR1, HC1); df = min(issuers, quarters) - 1",
        status="ok",
    )

def fit_xbi(r, est):
    """Estimate the mean excess return over XBI; report a status instead if any XBI return is missing."""
    bad = int((~np.isfinite(r.excess_xbi.to_numpy(float))).sum())
    if bad:
        return dict(n=len(r), missing_xbi=bad, status="incomplete_xbi_on_primary_sample")
    return fit_mean(r, outcome(r, est, "excess_xbi"))

def fit_contrast(rows, w, k, h, est, start, end):
    """Secondary: clustered minus non-clustered purchase days, both arms spaced within arm."""
    a = spaced(rows, h, start, end, "clustered")
    b = spaced(rows, h, start, end, "solo")
    df = min(a.issuer_id.nunique(), b.issuer_id.nunique(), a.quarter.nunique(), b.quarter.nunique()) - 1
    if len(a) < 2 or len(b) < 2 or df < 1:
        return dict(n_clustered=len(a), n_solo=len(b), status="insufficient_support")
    r = pd.concat([a, b])
    y = outcome(r, est)
    x = np.column_stack([np.ones(len(r)), r.arm.eq("clustered").to_numpy(float)])
    fit = OLS(y, x).fit()
    iss = pd.factorize(r.issuer_id, sort=True)[0]
    q = pd.factorize(r.quarter, sort=True)[0]
    var, source, _ = floored_variance(fit, iss, q, 1)
    if not np.isfinite(var) or var <= 0:
        return dict(n_clustered=len(a), n_solo=len(b), status=f"degenerate_variance:{source}")
    tt = float(fit.params[1]) / np.sqrt(var)
    return dict(
        n_clustered=len(a),
        n_solo=len(b),
        contrast_pp=float(fit.params[1]) * 100,
        se_pp=float(np.sqrt(var)) * 100,
        se_source=source,
        t=tt,
        df=df,
        p_two=float(2 * student_t.sf(abs(tt), df)),
        status="descriptive_secondary",
    )

def label(rows, w, k):
    """Mark each purchase day clustered (at least k buyers within w days) or solo."""
    rows = rows.copy()
    rows["arm"] = np.where(rows[f"buyers_w{w}"] >= k, "clustered", "solo")
    return rows

def price_screen(lab, prices, h, est, start, end):
    """Descriptive (train/validation only): the selected configuration restricted to higher-priced stocks."""
    r = spaced(lab, h, start, end, "clustered").merge(prices, on=["issuer_id", "signal_date"], how="left")
    out = dict(n_all=len(r), missing_price=int(r.purchase_price_median.isna().sum()))
    for floor in (1, 5):
        s = r.loc[r.purchase_price_median >= floor]
        f = fit_mean(s, outcome(s, est))
        out[f"price_ge_{floor}"] = dict(
            n=len(s),
            share_of_events=len(s) / len(r) if len(r) else None,
            **{
                c: f.get(c)
                for c in ("mean_pp", "se_pp", "se_source", "t", "p_two", "issuers", "quarters", "status")
            },
        )
    return out

def prior_return_controlled(lab, h, est, start, end):
    """Descriptive (train/validation only): clustered-arm mean excess at solo-arm prior-return levels.
    Uses the frozen protocol controls (21-session and 252-minus-21-session pre-filing returns), complete cases."""
    a = spaced(lab, h, start, end, "clustered")
    b = spaced(lab, h, start, end, "solo")
    a = a.loc[np.isfinite(a[PRIOR_COLS]).all(axis=1)]
    b = b.loc[np.isfinite(b[PRIOR_COLS]).all(axis=1)]
    if len(a) < 10 or len(b) < 10:
        return dict(status="insufficient_complete_cases")
    centre = b[PRIOR_COLS].mean()
    x = np.column_stack([np.ones(len(a)), (a[PRIOR_COLS] - centre).to_numpy(float)])
    fit = OLS(outcome(a, est), x).fit()
    iss = pd.factorize(a.issuer_id, sort=True)[0]
    q = pd.factorize(a.quarter, sort=True)[0]
    var, source, _ = floored_variance(fit, iss, q, 0)
    df = min(int(iss.max() + 1), int(q.max() + 1)) - 1
    m, se = float(fit.params[0]), float(np.sqrt(var))
    tt = m / se
    return dict(
        n=len(a),
        n_solo_reference=len(b),
        intercept_pp=m * 100,
        se_pp=se * 100,
        se_source=source,
        t=tt,
        df=df,
        p_two=float(2 * student_t.sf(abs(tt), df)),
        slopes=dict(zip(PRIOR_COLS, [float(v) for v in fit.params[1:]])),
        clustered_mean_priors={c: float(a[c].mean()) for c in PRIOR_COLS},
        solo_mean_priors={c: float(centre[c]) for c in PRIOR_COLS},
        status="descriptive",
    )

def holm(p):
    """Return Holm-adjusted p-values in the original order."""
    p = np.asarray(p, float)
    order = np.argsort(p, kind="mergesort")
    adj = np.empty_like(p)
    adj[order] = np.minimum(1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    return adj

def pairs_bootstrap(r, y):
    """Issuer-cluster pairs bootstrap of the mean: resample issuers with replacement, keep each issuer's events."""
    iss = pd.factorize(r.issuer_id, sort=True)[0]
    ni = int(iss.max() + 1)
    sums, cnt = np.bincount(iss, weights=y, minlength=ni), np.bincount(iss, minlength=ni)
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(DRAWS):
        w = np.bincount(rng.integers(ni, size=ni), minlength=ni)
        out.append((w * sums).sum() / (w * cnt).sum())
    lo, hi = np.quantile(out, [.025, .975]) * 100
    return dict(method="issuer pairs bootstrap", draws=DRAWS, ci_low_pp=float(lo), ci_high_pp=float(hi))

def block_bootstrap(r, y):
    """Frozen issuer x circular moving-two-quarter product-weight resampling; empty quarters stay in the calendar.
    Reported as a dependence sensitivity, not as the fragility flag."""
    iss = pd.factorize(r.issuer_id, sort=True)[0]
    q = (r.qord - r.qord.min()).to_numpy(int)
    cells = pd.DataFrame(dict(i=iss, q=q, y=y)).groupby(["i", "q"])["y"].agg(["sum", "size"]).reset_index()
    i, qq, s, c = cells.i.to_numpy(), cells.q.to_numpy(), cells["sum"].to_numpy(), cells["size"].to_numpy()
    ni, nq = int(iss.max() + 1), int(q.max() + 1)
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(DRAWS):
        wi = np.bincount(rng.integers(ni, size=ni), minlength=ni)
        starts = rng.integers(nq, size=(nq + 1) // 2)
        sampled = ((starts[:, None] + np.arange(2)) % nq).ravel()[:nq]
        wq = np.bincount(sampled, minlength=nq)
        wt = wi[i] * wq[qq]
        if (wt * c).sum() > 0:
            out.append((wt * s).sum() / (wt * c).sum())
    lo, hi = (np.quantile(out, [.025, .975]) * 100) if len(out) >= 0.9 * DRAWS else (np.nan, np.nan)
    return dict(method="issuer x moving-two-quarter product weights (wide by construction)", draws_valid=len(out),
                ci_low_pp=float(lo), ci_high_pp=float(hi))

def fragility(r, y, res, start, end, mid):
    """Fragility diagnostics: bootstrap intervals, leave-one-issuer-out range, largest issuer share, halves and flags."""
    loo = [float(np.mean(y[r.issuer_id.ne(i).to_numpy()])) for i in r.issuer_id.unique()]
    mid = pd.Timestamp(mid)                          # declared split date, not a computed midpoint
    halves = {}
    for name, lo, hi in (
        (f"{start[:4]}-{mid.year - 1}", start, mid - pd.Timedelta(days=1)),
        (f"{mid.year}-{end[:4]}", mid, end),
    ):
        m = r.signal_date.between(lo, hi).to_numpy()
        halves[name] = dict(n=int(m.sum()), mean_pp=float(np.mean(y[m]) * 100) if m.any() else None)
    pairs, block = pairs_bootstrap(r, y), block_bootstrap(r, y)
    share = float(r.issuer_id.value_counts(normalize=True).iloc[0])
    h1, h2 = list(halves.values())
    flags = dict(
        pairs_bootstrap_interval_includes_zero=not (pairs["ci_low_pp"] > 0 or pairs["ci_high_pp"] < 0),
        leave_one_issuer_out_changes_sign=bool(
            np.sign(min(loo)) != np.sign(res["mean_pp"]) or np.sign(max(loo)) != np.sign(res["mean_pp"])
        ),
        largest_issuer_share_above_10pct=share > 0.10,
        halves_have_opposite_signs=(
            None if not (h1["n"] and h2["n"]) else bool(np.sign(h1["mean_pp"]) != np.sign(h2["mean_pp"]))
        ),
    )
    return dict(
        pairs_bootstrap=pairs,
        block_bootstrap_sensitivity=block,
        leave_one_issuer_out_pp=[min(loo) * 100, max(loo) * 100],
        largest_issuer_share=share,
        halves=halves,
        flags=flags,
    )

def main():
    """Run the requested validation or test stage and write aggregate outputs."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["validation", "test"], default="validation")
    args = ap.parse_args()
    hashes = verify_inputs()
    rows, prices = build_rows(last_date=STAGES["validation"][1] if args.stage == "validation" else STAGES["test"][1])
    # ---- train: all 56 configurations, primary statistic, floored SE ----
    ledger = []
    for w in WINDOWS:
        for k in MIN_BUYERS:
            lab = label(rows, w, k)
            for h in HORIZONS:
                r = spaced(lab, h, *STAGES["train"], "clustered")
                for est in ESTIMATORS:
                    ledger.append(dict(w=w, k=k, h=h, est=est, **fit_mean(r, outcome(r, est))))
    L = pd.DataFrame(ledger)
    L["p_holm_56"] = holm(L.p_two.fillna(1))
    L = L.sort_values(["t", "est"], ascending=[False, True], kind="mergesort", na_position="last")
    key = (
        L[["w", "k", "h", "n"]].astype(str).agg("|".join, axis=1)
        + "|"
        + L.mean_pp.round(10).astype(str)
        + "|"
        + L.se_pp.round(10).astype(str)
    )
    L["duplicate_of_earlier_row"] = key.duplicated(keep="first")
    L.to_csv(HERE / "train_ledger.csv", index=False)
    top = (
        L.loc[~L.duplicate_of_earlier_row & L.status.eq("ok")]
        .head(TOP_K)[["w", "k", "h", "est"]]
        .to_records(index=False)
        .tolist()
    )
    # ---- validation: same floored SE, df = min(issuers, quarters) - 1; keep best t with t>0 and one-sided p<.10 ----
    val = []
    for w, k, h, est in top:
        r = spaced(label(rows, w, k), h, *STAGES["validation"], "clustered")
        val.append(dict(w=int(w), k=int(k), h=int(h), est=est, **fit_mean(r, outcome(r, est))))
    ok = [v for v in val if v.get("status") == "ok" and v["t"] > 0 and v["p_one"] < VALID_ONE_SIDED_P]
    result = dict(
        run="multi-insider purchases, temporal split",
        stage_executed=args.stage,
        inputs=hashes,
        stages=STAGES,
        grid=dict(windows=WINDOWS, min_buyers=MIN_BUYERS, horizons=HORIZONS, estimators=ESTIMATORS),
        train_top=L.head(10).to_dict("records"),
        validation=val,
        selected=None,
        test=None,
    )
    if not ok:
        result["decision"] = "no configuration survived validation; no signal claimed; test not run"
    else:
        sel = max(ok, key=lambda v: v["t"])
        w, k, h, est = sel["w"], sel["k"], sel["h"], sel["est"]
        result["selected"] = dict(w=w, k=k, h=h, est=est)
        lab_sel = label(rows, w, k)
        stages_seen = ["train", "validation"] + (["test"] if args.stage == "test" else [])
        result["attrition_selected_config"] = {
            s: [attrition(lab_sel, h, *STAGES[s], arm) for arm in ("clustered", "solo")] for s in stages_seen
        }
        result["descriptives_train_validation"] = {
            s: dict(
                price_screen=price_screen(lab_sel, prices, h, est, *STAGES[s]),
                prior_return_controlled=prior_return_controlled(lab_sel, h, est, *STAGES[s]),
            )
            for s in ("train", "validation")
        }
        if args.stage == "validation":
            result["decision"] = (
                "validation complete; selected configuration recorded; test NOT run (re-run with --stage test)"
            )
        else:
            lab = label(rows, w, k)
            r = spaced(lab, h, *STAGES["test"], "clustered")
            y = outcome(r, est)
            test = fit_mean(r, y)
            test["pass"] = bool(test.get("status") == "ok" and test["p_two"] < ALPHA and test["mean_pp"] > 0)
            result.update(
                test=test,
                fragility=(
                    fragility(r, y, test, *STAGES["test"], HALF_SPLIT["test"])
                    if test.get("status") == "ok"
                    else None
                ),
                secondaries=dict(
                    contrast=fit_contrast(lab, w, k, h, est, *STAGES["test"]),
                    xbi_primary=fit_xbi(r, est),
                    horizons_descriptive=[
                        dict(
                            h=hh,
                            **fit_mean(
                                spaced(lab, hh, *STAGES["test"], "clustered"),
                                outcome(spaced(lab, hh, *STAGES["test"], "clustered"), est),
                            ),
                        )
                        for hh in HORIZONS
                    ],
                ),
                decision=(
                    "PASS: selected configuration significant on 2020-2025"
                    if test["pass"]
                    else "FAIL: selected configuration not significant on 2020-2025"
                ),
            )
    if (
        args.stage == "test" and result["selected"] is not None
    ):  # stop rule: no selection -> no test-period statistic at all
        rw, rk, rh, rest = REFERENCE
        rr = spaced(label(rows, rw, rk), rh, *STAGES["test"], "clustered")
        result["reference_c01_on_test"] = dict(
            w=rw,
            k=rk,
            h=rh,
            est=rest,
            **fit_mean(rr, outcome(rr, rest)),
            status_note="descriptive reference, not a test",
        )
    (HERE / "results.json").write_text(json.dumps(result, indent=1, default=float))
    write_results_md(result)

def write_results_md(res):
    """Render stage results as a concise Markdown report."""
    f = lambda d: (
        f"{d.get('mean_pp', float('nan')):+.2f} pp (SE {d.get('se_pp', float('nan')):.2f} [{d.get('se_source', '-')}], t={d.get('t', float('nan')):.2f}, "
        f"p={d.get('p_two', float('nan')):.3f}, n={d.get('n')}, issuers={d.get('issuers')}, quarters={d.get('quarters')})"
    )
    md = [
        f"# Results — {res['run']}, stage executed: {res['stage_executed']}\n",
        f"Decision: **{res['decision']}**\n",
        "## Train: top 10 of 56 (floored SE; duplicates flagged)\n",
        "| w | k | h | est | mean | SE | SE source | t | p | Holm-56 p | n | dup |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res["train_top"]:
        md.append(
            f"| {r['w']} | {r['k']} | {r['h']} | {r['est']} | {r['mean_pp']:+.2f} | {r['se_pp']:.2f} | {r['se_source']} | {r['t']:.2f} | {r['p_two']:.4f} | {r['p_holm_56']:.3f} | {r['n']} | {r['duplicate_of_earlier_row']} |"
        )
    md += ["\n## Validation (floored SE, df = min(issuers, quarters) - 1; keep best t with t>0, one-sided p<.10)\n"]
    md += [
        f"- w={v['w']} k={v['k']} h={v['h']} {v['est']}: {f(v)} one-sided p={v.get('p_one', float('nan')):.3f}"
        for v in res["validation"]
    ]
    if res.get("selected"):
        s = res["selected"]
        md.append(f"\nSelected for the test: w={s['w']} k={s['k']} h={s['h']} {s['est']}")
        md += [
            "\n## Attrition for the selected configuration (outcome-blind counts)\n",
            "| stage | arm | company-days | after spacing | exit beyond stage | non-finite outcome | used |",
            "|---|---|---|---|---|---|---|",
        ]
        for st, rows_ in res["attrition_selected_config"].items():
            for a in rows_:
                md.append(
                    f"| {st} | {a['arm']} | {a['company_days']} | {a['after_spacing']} | {a['dropped_exit_beyond_stage']} | {a['dropped_nonfinite_outcome']} | {a['used']} |"
                )
        md += ["\n## Descriptives on train and validation only (selected configuration; no test-period rows)\n"]
        for st, d in res["descriptives_train_validation"].items():
            ps, pr = d["price_screen"], d["prior_return_controlled"]
            md.append(
                f"- {st}: price >= $1: n={ps['price_ge_1']['n']} mean={ps['price_ge_1']['mean_pp']:+.2f} pp (t={ps['price_ge_1']['t']:.2f}); price >= $5: n={ps['price_ge_5']['n']} "
                f"({ps['price_ge_5']['share_of_events']:.0%} of events) mean={ps['price_ge_5']['mean_pp']:+.2f} pp (t={ps['price_ge_5']['t']:.2f}); missing price {ps['missing_price']}"
            )
            if pr.get("status") == "descriptive":
                md.append(
                    f"- {st}: prior-return-controlled clustered mean at solo-arm prior levels: {pr['intercept_pp']:+.2f} pp (SE {pr['se_pp']:.2f} [{pr['se_source']}], t={pr['t']:.2f}, n={pr['n']}); "
                    f"clustered mean prior-21 {pr['clustered_mean_priors']['prior_return_21']:+.1%} vs solo {pr['solo_mean_priors']['prior_return_21']:+.1%}"
                )
        md.append(
            "- Break-even round-trip cost for the selected signal equals its point estimate in the stage shown; small-cap biotech spreads can be of that order."
        )
    if res.get("test"):
        s, t, g = res["selected"], res["test"], res["fragility"]
        md += [
            f"\n## Test 2020–2025 — w={s['w']} k={s['k']} h={s['h']} {s['est']}\n",
            f"**{f(t)}; 95% CI [{t['ci_low_pp']:+.2f}, {t['ci_high_pp']:+.2f}] pp; PASS={t['pass']}**\n",
            f"Fragility flags: {json.dumps(g['flags'])}; pairs-bootstrap CI [{g['pairs_bootstrap']['ci_low_pp']:+.2f}, {g['pairs_bootstrap']['ci_high_pp']:+.2f}]; "
            f"block-bootstrap sensitivity [{g['block_bootstrap_sensitivity']['ci_low_pp']:+.2f}, {g['block_bootstrap_sensitivity']['ci_high_pp']:+.2f}]; "
            f"leave-one-issuer-out range [{g['leave_one_issuer_out_pp'][0]:+.2f}, {g['leave_one_issuer_out_pp'][1]:+.2f}]; largest issuer share {g['largest_issuer_share']:.1%}; halves {g['halves']}\n",
            "### Secondaries (descriptive)\n",
            f"- Clustered minus solo: {res['secondaries']['contrast']}",
            f"- Primary vs XBI: {res['secondaries']['xbi_primary']}",
            "- Seven horizons: "
            + "; ".join(
                f"h={d['h']}: {d.get('mean_pp', float('nan')):+.2f} pp (t={d.get('t', float('nan')):.2f})"
                for d in res["secondaries"]["horizons_descriptive"]
            ),
        ]
    if res.get("reference_c01_on_test"):
        md.append(
            f"\n## Reference: frozen C01 (w=7,k=2,h=10,mean) on the test period — descriptive\n{f(res['reference_c01_on_test'])}\n"
        )
    md.append("## Inputs\n" + "\n".join(f"- {k}: `{v}`" for k, v in res["inputs"].items()))
    (HERE / "RESULTS.md").write_text("\n".join(md))

if __name__ == "__main__":
    main()
