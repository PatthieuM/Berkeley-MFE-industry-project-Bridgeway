# Decision record

All decisions below were fixed on 2026-09-24 unless stated otherwise. They consolidate the v1–v6 pre-registrations and the research journal (local research archive, not in this repository) without changing the v6 signal.

| Date | Convention | Reason |
|---|---|---|
| 2026-09-24 | Use the team's effective-dated strict-biotech universe, including delisted firms. | Replicate the core signal on the independently built historical universe and avoid survivor-only membership. |
| 2026-09-24 | Use original Form 4 only, no amendments, non-derivative transaction codes P and S, and require a valid reporting-owner CIK. | Preserve the normalized SEC construction used by Step 1 and v6; prevent holdings/derivatives and amendments from changing the event definition. |
| 2026-09-24 | Deduplicate to issuer CIK × filing date × direction. | Multiple owners, accessions, or executions must not multiply the same public issuer-day return. |
| 2026-09-24 | Map by inclusive eligible universe interval; prefer point-in-time CIK, then resolve concurrent share classes only with the filing ticker. | Match v6 while limiting historical-identity and share-class ambiguity. Current-CIK fallback is retained and disclosed. |
| 2026-09-24 | Enter at the first exchange-session close strictly after filing. | The filing must be public before the assumed execution; same-day close is not used. |
| 2026-09-24 | Benchmark against XBI over exactly the stock's entry/end sessions. | XBI is the pre-registered biotech benchmark and aligned windows prevent calendar mismatch. |
| 2026-09-24 | Report every signal at 1, 2, 3, 5, 10, 21, 42, 63, and 126 sessions. | Use the client's common grid while retaining 5 sessions as the S1–S3 headline and 63/126 as the S4 headlines. |
| 2026-09-24 | S1 uses the lagged-cap lower tercile among turnover-eligible purchases; year Y breakpoints use only events filed before Y, beginning with 2006 history. | This is the exact v4/v6 walk-forward rule and prevents look-ahead. Equality is assigned to the lower bucket. |
| 2026-09-24 | S2 includes all purchase contexts. | The journal's L4 result found no reliable advantage from excluding financing participation. |
| 2026-09-24 | S3 includes any issuer-day with at least one transaction-code S row. | Provide the requested transparent sale baseline without adding post-outcome filters. |
| 2026-09-24 | Missing session, invalid total-return level, or delisting inside the path makes the event return missing; never fill with zero or shift entry. | Preserve engine-v2 semantics and avoid mechanically treating unavailable outcomes as flat returns. |
| 2026-09-24 | Report mean, median, hit rate, t-based 95% CI, t statistic, and two-sided p-value with issuer × filing-month two-way clustering. | Reflect dependence within issuers and calendar shocks while reporting direction-neutral inference. |
| 2026-09-24 | Separate 2009–2019 and 2020–2025 period summaries; keep 2026 untouched. | Show chronological stability and preserve the pre-registered 2026H1 holdout. |
| 2026-09-25 | `universe_full` is the official price source; `v6_cache` is regression-test only. | Full CRSP dsf_v2 coverage removes the v6 cache's missing-universe-price limitation while the cache proves exact backward compatibility. |
| 2026-09-25 | Write aggregates only. | Licensed WRDS/CRSP and normalized SEC row-level data remain in place and are never copied into reports. |

Unsuccessful prior tests remain part of the record: routine/opportunistic purchases, financing-versus-open-market separation, crash/rebound explanations, attention within size, and combination filters did not reliably improve the base small-cap purchase signal. Historical GICS is effective-dated rather than publication-vintage, and current-CIK fallback remains a disclosed limitation.
