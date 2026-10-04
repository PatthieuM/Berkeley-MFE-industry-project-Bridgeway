# Multi-insider purchases: results, readable version of RESULTS.md

Source: results.json. All estimates below are read from that saved result.

Recorded decision: **PASS: selected configuration significant on 2020-2025**

Source SHA-256: `e59118f6d5d4e847612c56c34b244e55ed66f324fc9fddabd286618892302c1a`.

Selected configuration: **w=7, k=2, h=3, mean**.

## Test result (2020-01-01 to 2025-12-31)

+0.90 pp; SE 0.39 (hc1); t=2.30; two-sided p=0.031; n=722

95% interval: [+0.09, +1.71] pp. 251 issuers; 24 quarters.

## Training and validation

Validation retains the largest positive t-statistic with one-sided p < .10 among the three selected training candidates.

| Validation candidate | Mean (pp) | SE (pp) | t | One-sided p | n | Status |
|---|---:|---:|---:|---:|---:|---|
| w=7, k=2, h=1, mean | -0.10 | 0.23 | -0.44 | 0.665 | 385 | ok |
| w=7, k=2, h=3, mean | +1.29 | 0.66 | 1.96 | 0.038 | 296 | ok |
| w=14, k=2, h=5, wins100 | +1.45 | 0.85 | 1.71 | 0.058 | 311 | ok |

Top 10 of 56 training configurations. Yes marks a wins100 row whose mean and SE equal its mean-estimator twin to 10 decimals; such rows were skipped when picking the top three.

| Configuration | Mean (pp) | SE (pp) | t | Holm p | n | Duplicate |
|---|---:|---:|---:|---:|---:|---|
| w=7, k=2, h=1, mean | +0.81 | 0.21 | 3.82 | 0.034 | 885 | No |
| w=7, k=2, h=1, wins100 | +0.81 | 0.21 | 3.82 | 0.034 | 885 | Yes |
| w=7, k=2, h=3, mean | +1.16 | 0.34 | 3.46 | 0.086 | 665 | No |
| w=7, k=2, h=3, wins100 | +1.16 | 0.34 | 3.46 | 0.086 | 665 | Yes |
| w=14, k=2, h=5, wins100 | +1.43 | 0.41 | 3.46 | 0.086 | 704 | No |
| w=7, k=3, h=3, mean | +1.92 | 0.57 | 3.38 | 0.100 | 282 | No |
| w=7, k=3, h=3, wins100 | +1.92 | 0.57 | 3.38 | 0.100 | 282 | Yes |
| w=14, k=2, h=1, mean | +0.70 | 0.21 | 3.34 | 0.107 | 1048 | No |
| w=14, k=2, h=1, wins100 | +0.70 | 0.21 | 3.34 | 0.107 | 1048 | Yes |
| w=14, k=3, h=3, mean | +1.59 | 0.48 | 3.33 | 0.107 | 362 | No |

## Sample attrition

Spacing is applied before returns are checked. The exit column counts exits missing or outside the stage.

| Stage | Arm | Company-days | After spacing | Exit unavailable/outside stage | Nonfinite return | Used |
|---|---|---:|---:|---:|---:|---:|
| train | Clustered | 890 | 670 | 0 | 5 | 665 |
| train | Below buyer threshold | 1888 | 1720 | 1 | 6 | 1713 |
| validation | Clustered | 385 | 297 | 1 | 0 | 296 |
| validation | Below buyer threshold | 770 | 717 | 1 | 1 | 715 |
| test | Clustered | 968 | 723 | 1 | 0 | 722 |
| test | Below buyer threshold | 2154 | 1976 | 6 | 2 | 1968 |

## Train/validation descriptives

Price is the median reported purchase price on the filing company-day, a proxy rather than an entry quote.

| Stage | Reported price | n | Share of events | Mean (pp) | t | Status |
|---|---|---:|---:|---:|---:|---|
| train | ≥ $1 | 566 | 85% | +0.95 | 2.56 | ok |
| train | ≥ $5 | 273 | 41% | +0.87 | 1.57 | ok |
| validation | ≥ $1 | 251 | 85% | +1.33 | 1.62 | ok |
| validation | ≥ $5 | 140 | 47% | +1.63 | 1.63 | ok |

train: missing reported price for 12 events.
validation: missing reported price for 2 events.

- train, clustered mean at comparison-arm prior-return levels: +1.00 pp; SE 0.37 (two_way); t=2.67; two-sided p=0.012; n=525.
- validation, clustered mean at comparison-arm prior-return levels: +1.48 pp; SE 0.58 (issuer); t=2.57; two-sided p=0.026; n=246.

Controlled SEs condition on the observed comparison-arm covariate means.

## Fragility and dependence

| Diagnostic | Saved result |
|---|---|
| Issuer-pairs bootstrap, 95% interval | [+0.14, +1.63] pp |
| Issuer × circular two-quarter product-weight sensitivity | [-0.41, +2.11] pp |
| Leave-one-issuer-out mean range | [+0.78, +1.01] pp |
| Largest issuer share | 3.5% |

Issuer-only resampling does not retain shared calendar shocks. The two resampling schemes are different dependence checks; neither interval is guaranteed to be wider.

| Period | Events | Mean (pp) |
|---|---:|---:|
| 2020-2022 | 385 | +0.48 |
| 2023-2025 | 337 | +1.38 |

| Flag | Triggered |
|---|---|
| Pairs interval includes zero | No |
| Issuer omission changes the mean's sign | No |
| Largest issuer exceeds 10% of events | No |
| Period halves have opposite signs | No |

## Descriptive secondaries

- Clustered minus below-threshold purchases: +0.49 pp; SE 0.46 (quarter); t=1.08; two-sided p=0.290; n=722; comparison n=1968.
- Primary relative to XBI: +0.96 pp; SE 0.40 (hc1); t=2.42; two-sided p=0.024; n=722.

| Horizon (sessions) | Mean (pp) | SE (pp) | t | Two-sided p | n | Status |
|---:|---:|---:|---:|---:|---:|---|
| 1 | +0.59 | 0.25 | 2.35 | 0.028 | 968 | ok |
| 2 | +0.92 | 0.32 | 2.82 | 0.010 | 816 | ok |
| 3 | +0.90 | 0.39 | 2.30 | 0.031 | 722 | ok |
| 5 | +1.03 | 0.52 | 1.97 | 0.061 | 633 | ok |
| 10 | +3.90 | 2.33 | 1.67 | 0.108 | 570 | ok |
| 21 | +1.17 | 1.53 | 0.76 | 0.454 | 533 | ok |
| 63 | +2.65 | 3.02 | 0.88 | 0.391 | 487 | ok |

C01 descriptive reference (w=7, k=2, h=10, mean): +3.90 pp; SE 2.33 (quarter); t=1.67; two-sided p=0.108; n=570.

## Interpretation and inputs

The maximum of four estimated SEs is an ad hoc safeguard; its finite-sample coverage is not established. The validation screen uses few time clusters. Calendar-quarter clustering does not cover every form of dependence across quarters.

Returns are gross event-level excess returns. A constant additional per-event cost equal to the mean would erase that excess over a costless benchmark. Portfolio turnover and execution costs are not modeled.

- events.parquet: `2b478839343afe5d527c996c2d1ea90eeb76a895a3236bde114a93c7f404d9df`
- purchases.parquet: `97b762aa9264d6f16149eab10b78537545acd26b7bc8f5180a41f06f48b0b62f`
- classification.py: `4297a1f2fb5d8febac8ed24d425bf60c21d7cc31a7073aa71f83b84cff902770`
- protocol.json: `6cf61060cad0ce87cfa913cc1393263f05f9b3f5a6f751300e1bc91e2d9936a4`
- archive: `7e7d284dff704aaf2b66eaa975df135c54d246b72940c77edef0f51da6ee51e4`

Recorded analysis environment:

- python: 3.12.2
- pandas: 2.2.2
- numpy: 1.26.4
- statsmodels: 0.14.2
- scipy: 1.13.1
