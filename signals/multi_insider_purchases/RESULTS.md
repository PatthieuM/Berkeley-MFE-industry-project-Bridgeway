# Results — multi-insider purchases, temporal split, stage executed: test

Decision: **PASS: selected configuration significant on 2020-2025**

## Train: top 10 of 56 (floored SE; duplicates flagged)

| w | k | h | est | mean | SE | SE source | t | p | Holm-56 p | n | dup |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 7 | 2 | 1 | mean | +0.81 | 0.21 | issuer | 3.82 | 0.0006 | 0.034 | 885 | False |
| 7 | 2 | 1 | wins100 | +0.81 | 0.21 | issuer | 3.82 | 0.0006 | 0.034 | 885 | True |
| 7 | 2 | 3 | mean | +1.16 | 0.34 | quarter | 3.46 | 0.0016 | 0.086 | 665 | False |
| 7 | 2 | 3 | wins100 | +1.16 | 0.34 | quarter | 3.46 | 0.0016 | 0.086 | 665 | True |
| 14 | 2 | 5 | wins100 | +1.43 | 0.41 | hc1 | 3.46 | 0.0016 | 0.086 | 704 | False |
| 7 | 3 | 3 | mean | +1.92 | 0.57 | hc1 | 3.38 | 0.0020 | 0.100 | 282 | False |
| 7 | 3 | 3 | wins100 | +1.92 | 0.57 | hc1 | 3.38 | 0.0020 | 0.100 | 282 | True |
| 14 | 2 | 1 | mean | +0.70 | 0.21 | issuer | 3.34 | 0.0022 | 0.107 | 1048 | False |
| 14 | 2 | 1 | wins100 | +0.70 | 0.21 | issuer | 3.34 | 0.0022 | 0.107 | 1048 | True |
| 14 | 3 | 3 | mean | +1.59 | 0.48 | hc1 | 3.33 | 0.0023 | 0.107 | 362 | False |

## Validation (floored SE, df = min(issuers, quarters) - 1; keep best t with t>0, one-sided p<.10)

- w=7 k=2 h=1 mean: -0.10 pp (SE 0.23 [hc1], t=-0.44, p=0.670, n=385, issuers=138, quarters=12) one-sided p=0.665
- w=7 k=2 h=3 mean: +1.29 pp (SE 0.66 [two_way], t=1.96, p=0.075, n=296, issuers=137, quarters=12) one-sided p=0.038
- w=14 k=2 h=5 wins100: +1.45 pp (SE 0.85 [two_way], t=1.71, p=0.116, n=311, issuers=142, quarters=12) one-sided p=0.058

Selected for the test: w=7 k=2 h=3 mean

## Attrition for the selected configuration (outcome-blind counts)

| stage | arm | company-days | after spacing | exit beyond stage | non-finite outcome | used |
|---|---|---|---|---|---|---|
| train | clustered | 890 | 670 | 0 | 5 | 665 |
| train | solo | 1888 | 1720 | 1 | 6 | 1713 |
| validation | clustered | 385 | 297 | 1 | 0 | 296 |
| validation | solo | 770 | 717 | 1 | 1 | 715 |
| test | clustered | 968 | 723 | 1 | 0 | 722 |
| test | solo | 2154 | 1976 | 6 | 2 | 1968 |

## Descriptives on train and validation only (selected configuration; no test-period rows)

- train: price >= $1: n=566 mean=+0.95 pp (t=2.56); price >= $5: n=273 (41% of events) mean=+0.87 pp (t=1.57); missing price 12
- train: prior-return-controlled clustered mean at solo-arm prior levels: +1.00 pp (SE 0.37 [two_way], t=2.67, n=525); clustered mean prior-21 -7.9% vs solo -1.7%
- validation: price >= $1: n=251 mean=+1.33 pp (t=1.62); price >= $5: n=140 (47% of events) mean=+1.63 pp (t=1.63); missing price 2
- validation: prior-return-controlled clustered mean at solo-arm prior levels: +1.48 pp (SE 0.58 [issuer], t=2.57, n=246); clustered mean prior-21 -11.1% vs solo -6.0%
- Break-even round-trip cost for the selected signal equals its point estimate in the stage shown; small-cap biotech spreads can be of that order.

## Test 2020–2025 — w=7 k=2 h=3 mean

**+0.90 pp (SE 0.39 [hc1], t=2.30, p=0.031, n=722, issuers=251, quarters=24); 95% CI [+0.09, +1.71] pp; PASS=True**

Fragility flags: {"pairs_bootstrap_interval_includes_zero": false, "leave_one_issuer_out_changes_sign": false, "largest_issuer_share_above_10pct": false, "halves_have_opposite_signs": false}; pairs-bootstrap CI [+0.14, +1.63]; block-bootstrap sensitivity [-0.41, +2.11]; leave-one-issuer-out range [+0.78, +1.01]; largest issuer share 3.5%; halves {'2020-2022': {'n': 385, 'mean_pp': 0.4823496717617632}, '2023-2025': {'n': 337, 'mean_pp': 1.3815285682410774}}

### Secondaries (descriptive)

- Clustered minus solo: {'n_clustered': 722, 'n_solo': 1968, 'contrast_pp': 0.4949948558380767, 'se_pp': 0.457388498484408, 'se_source': 'quarter', 't': 1.0822197267274545, 'df': 23, 'p_two': 0.2903712045713256, 'status': 'descriptive_secondary'}
- Primary vs XBI: {'n': 722, 'issuers': 251, 'quarters': 24, 'mean_pp': 0.959991501652335, 'se_pp': 0.3970996857286917, 'se_source': 'hc1', 'se_two_way_pp': 0.3704379146664935, 'se_issuer_pp': 0.38433028677032127, 'se_quarter_pp': 0.37174993598538064, 'se_hc1_pp': 0.3970996857286917, 't': 2.4175075834943494, 'df': 23, 'p_two': 0.02395577825713161, 'p_one': 0.011977889128565806, 'ci_low_pp': 0.13852821467466772, 'ci_high_pp': 1.7814547886300027, 'inference': 'floored SE = max(two-way CR1, issuer CR1, quarter CR1, HC1); df = min(issuers, quarters) - 1', 'status': 'ok'}
- Seven horizons: h=1: +0.59 pp (t=2.35); h=2: +0.92 pp (t=2.82); h=3: +0.90 pp (t=2.30); h=5: +1.03 pp (t=1.97); h=10: +3.90 pp (t=1.67); h=21: +1.17 pp (t=0.76); h=63: +2.65 pp (t=0.88)

## Reference: frozen C01 (w=7,k=2,h=10,mean) on the test period — descriptive
+3.90 pp (SE 2.33 [quarter], t=1.67, p=0.108, n=570, issuers=251, quarters=24)

## Inputs
- events.parquet: `2b478839343afe5d527c996c2d1ea90eeb76a895a3236bde114a93c7f404d9df`
- purchases.parquet: `97b762aa9264d6f16149eab10b78537545acd26b7bc8f5180a41f06f48b0b62f`
- classification.py: `4297a1f2fb5d8febac8ed24d425bf60c21d7cc31a7073aa71f83b84cff902770`
- protocol.json: `6cf61060cad0ce87cfa913cc1393263f05f9b3f5a6f751300e1bc91e2d9936a4`
- archive: `7e7d284dff704aaf2b66eaa975df135c54d246b72940c77edef0f51da6ee51e4`
- environment: `{'python': '3.12.2', 'pandas': '2.2.2', 'numpy': '1.26.4', 'statsmodels': '0.14.2', 'scipy': '1.13.1'}`