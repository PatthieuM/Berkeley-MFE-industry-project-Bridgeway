# Data dictionary

## SEC Form 4 normalized tables

| Table | Fields used | Meaning |
|---|---|---|
| `submissions` | `accession`, `filing_date`, `form`, `issuer_cik`, `ticker`, `is_amendment` | Filing identity, public availability, issuer identity, and original-versus-amended status. Presentation code also uses `issuer_name`. |
| `transactions` | `accession`, `transaction_key`, `transaction_date`, `transaction_code` | Unique non-derivative execution and P/S direction. Sanity/reporting code may also use `shares`, `price`, `trade_value`, ownership direction, and quality flags. |
| `owners` | `accession`, `owner_cik` | Reporting owners attached to each filing. Presentation and sanity code may use `role` and `officer_title`. |
| `footnotes` | `accession`, `footnote_id`, `footnote_text` | Filing footnotes retained for audit; mention flags are descriptive, not verified classifications. |

One S1–S3 event is an issuer × filing date × direction row from an original Form 4 containing at least one eligible P or S transaction and at least one identified owner. SSN expands sale filings to reporting owners before constructing owner-month histories.

## Complete SEC Forms 3/4/5 archive (`data/sec/all_forms`, schema v1)

These lossless tables are separate from the frozen signal archive. SEC source column names are lower-cased; only `accession_number` is renamed `accession`. `source_quarter` is added to every table. Dates are Parquet timestamps; other source values remain strings so identifiers, exact reported decimals, blank values, and footnote references are not silently changed.

| Table | Columns and SEC meaning |
|---|---|
| `submissions` | `accession`: filing accession; `filing_date`: SEC filing date; `period_of_report`: reporting period; `date_of_orig_sub`: original-submission date for amendments; `no_securities_owned`: no-securities indicator; `not_subject_sec16`: Section 16 exemption indicator; `form3_holdings_reported`: holdings-present indicator; `form4_trans_reported`: transactions-present indicator; `document_type`: 3/3-A/4/4-A/5/5-A; `issuercik`: issuer CIK; `issuername`: legal name; `issuertradingsymbol`: ticker; `remarks`: remarks; `aff10b5one`: filing-level 10b5-1 indicator; `source_quarter`: source quarter. |
| `owners` | `accession`: accession; `rptownercik`: owner CIK; `rptownername`: owner name; `rptowner_relationship`: relationship; `rptowner_title`: officer title; `rptowner_txt`: other-role text; `rptowner_street1`, `rptowner_street2`, `rptowner_city`, `rptowner_state`, `rptowner_zipcode`, `rptowner_state_desc`: address; `file_number`: SEC file number; `source_quarter`: source quarter. |
| `nonderiv_transactions` | `accession`: accession; `nonderiv_trans_sk`: row key; `security_title`, `security_title_fn`: class and footnote; `trans_date`, `trans_date_fn`: transaction date and footnote; `deemed_execution_date`, `deemed_execution_date_fn`: deemed date and footnote; `trans_form_type`: row form; `trans_code`: all SEC transaction codes; `equity_swap_involved`, `equity_swap_trans_cd_fn`: swap flag and footnote; `trans_timeliness`, `trans_timeliness_fn`: timeliness and footnote; `trans_shares`, `trans_shares_fn`: shares and footnote; `trans_pricepershare`, `trans_pricepershare_fn`: price and footnote; `trans_acquired_disp_cd`, `trans_acquired_disp_cd_fn`: A/D and footnote; `shrs_ownd_folwng_trans`, `shrs_ownd_folwng_trans_fn`: post-transaction shares and footnote; `valu_ownd_folwng_trans`, `valu_ownd_folwng_trans_fn`: post-transaction value and footnote; `direct_indirect_ownership`, `direct_indirect_ownership_fn`: D/I and footnote; `nature_of_ownership`, `nature_of_ownership_fn`: ownership nature and footnote; `source_quarter`: source quarter. |
| `nonderiv_holdings` | `accession`: accession; `nonderiv_holding_sk`: row key; `security_title`, `security_title_fn`: class and footnote; `trans_form_type`, `trans_form_type_fn`: row form and footnote; `shrs_ownd_folwng_trans`, `shrs_ownd_folwng_trans_fn`: shares held and footnote; `valu_ownd_folwng_trans`, `valu_ownd_folwng_trans_fn`: value held and footnote; `direct_indirect_ownership`, `direct_indirect_ownership_fn`: D/I and footnote; `nature_of_ownership`, `nature_of_ownership_fn`: ownership nature and footnote; `source_quarter`: source quarter. |
| `deriv_transactions` | `accession`: accession; `deriv_trans_sk`: row key; `security_title`, `security_title_fn`: derivative and footnote; `conv_exercise_price`, `conv_exercise_price_fn`: conversion/exercise price and footnote; `trans_date`, `trans_date_fn`: date and footnote; `deemed_execution_date`, `deemed_execution_date_fn`: deemed date and footnote; `trans_form_type`: row form; `trans_code`: all SEC codes; `equity_swap_involved`, `equity_swap_trans_cd_fn`: swap flag and footnote; `trans_timeliness`, `trans_timeliness_fn`: timeliness and footnote; `trans_shares`, `trans_shares_fn`: units and footnote; `trans_total_value`, `trans_total_value_fn`: total value and footnote; `trans_pricepershare`, `trans_pricepershare_fn`: unit price and footnote; `trans_acquired_disp_cd`, `trans_acquired_disp_cd_fn`: A/D and footnote; `excercise_date`, `excercise_date_fn`: SEC-source spelling of exercise date and footnote; `expiration_date`, `expiration_date_fn`: expiration and footnote; `undlyng_sec_title`, `undlyng_sec_title_fn`: underlying class and footnote; `undlyng_sec_shares`, `undlyng_sec_shares_fn`: underlying shares and footnote; `undlyng_sec_value`, `undlyng_sec_value_fn`: underlying value and footnote; `shrs_ownd_folwng_trans`, `shrs_ownd_folwng_trans_fn`: units held and footnote; `valu_ownd_folwng_trans`, `valu_ownd_folwng_trans_fn`: value held and footnote; `direct_indirect_ownership`, `direct_indirect_ownership_fn`: D/I and footnote; `nature_of_ownership`, `nature_of_ownership_fn`: ownership nature and footnote; `source_quarter`: source quarter. |
| `deriv_holdings` | `accession`: accession; `deriv_holding_sk`: row key; `security_title`, `security_title_fn`: derivative and footnote; `conv_exercise_price`, `conv_exercise_price_fn`: conversion/exercise price and footnote; `trans_form_type`, `trans_form_type_fn`: row form and footnote; `exercise_date`, `exercise_date_fn`: exercise date and footnote; `expiration_date`, `expiration_date_fn`: expiration and footnote; `undlyng_sec_title`, `undlyng_sec_title_fn`: underlying class and footnote; `undlyng_sec_shares`, `undlyng_sec_shares_fn`: underlying shares and footnote; `undlyng_sec_value`, `undlyng_sec_value_fn`: underlying value and footnote; `shrs_ownd_folwng_trans`, `shrs_ownd_folwng_trans_fn`: units held and footnote; `valu_ownd_folwng_trans`, `valu_ownd_folwng_trans_fn`: value held and footnote; `direct_indirect_ownership`, `direct_indirect_ownership_fn`: D/I and footnote; `nature_of_ownership`, `nature_of_ownership_fn`: ownership nature and footnote; `source_quarter`: source quarter. |
| `footnotes` | `accession`: accession; `footnote_id`: ID referenced by `*_fn`; `footnote_txt`: exact text; `source_quarter`: source quarter. |

## Form 144 (`data/sec/form144`)

`form144.parquet` has one row per securities block: `filing_date`, `accession`, `form`, `issuer_cik`, `issuer_name`, `filer`, `relationship_to_issuer`, `sequence_in_filing`, `securities_class`, `shares_to_be_sold`, `aggregate_market_value`, `approximate_sale_date`, `broker`, `plan_10b5_1`, and `submission_path`. `discovered_filings.parquet` records CIK, company, form, filing date, submission path, and accession. Downloads require `SEC_USER_AGENT="Organization contact@example.com"` and run at no more than eight requests per second.

## Ownership continuity

`continuity_rows.parquet` retains every audited derivative and non-derivative transaction and holding with canonical arithmetic fields plus `status` (`consistent`, `first_observation`, `inconsistent`, or `not_comparable`) and `reason`. Groups are owner CIK × issuer CIK × table × normalized security title × direct/indirect nature (with derivative contract terms also in the key). CRSP `dlycumfacshr` changes supply intervening share multipliers.

## Event identifier exports (`data/outputs/events`)

Each of `s1_events.parquet` through `s4_events.parquet` contains `event_id`, `signal`, `permno`, `permco`, `cusip`, `ticker`, `issuer_cik`, `filing_date`, and `entry_date`. Identifiers are effective-dated; entry is the next benchmark trading session.

## CRSP daily market data

| Field | Use |
|---|---|
| `permno`, `permco` | Effective-dated security and company identifiers. |
| `dlycaldt` | Exchange session date and calendar alignment. |
| `dlyret`, `dlyretx` | Total and ex-distribution returns where required. |
| `dlyclose`, `dlyprc`, `dlyopen`, `dlyhigh`, `dlylow` | Price levels and sanity checks. |
| `dlyvol`, `shrout`, `dlycap` | Lagged turnover and market-cap features. |
| `dlydelflg`, delisting records | Terminal-path and delisting handling. |
| XBI `ticker`, `dlycaldt`, `dlyret`, `dlyclose` | Benchmark return/level and trading calendar. |

## Historical universe

`biotech_universe_intervals.parquet` supplies `interval_id`, `permno`, `permco`, `effective_start`, and `effective_end`. `identifier_crosswalk.parquet` adds dated `ticker` and `cik` plus provenance fields: `cik_source`, `cik_is_point_in_time`, `cik_match_status`, and `identifier_match_status`. The manifest must report validation status `passed` and rule version `crsp-us-common-10-11-v4`.
