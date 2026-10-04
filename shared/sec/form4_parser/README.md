# SEC ownership filing parser

Parse Forms 3, 4, 5, and 144 into a common row schema and pandas DataFrames.
Online requests require `SEC_USER_AGENT` containing an organization or requester
name and a contact email.

Run these examples from the repository root:

```bash
export SEC_USER_AGENT="Organization Name contact@example.com"
python -m shared.sec.form4_parser.scrape                         # AAPL Form 4, 50 filings
python -m shared.sec.form4_parser.scrape --form all
python -m shared.sec.form4_parser.scrape --ticker RNST --form 4 --limit 0
python -m shared.sec.form4_parser.scrape --cik 715072 --form 4
python -m shared.sec.form4_parser.scrape --ticker VKTX --form 144
```

```python
from shared.sec.form4_parser.engines import Form4Engine

df = Form4Engine(cache_dir=".cache/sec-form4").scrape("RNST")
```

Ticker resolution refreshes the current SEC map. Supply historical CIKs for
historical issuer discovery; current ticker identity does not establish past
universe membership. `EdgarClient(offline=True, cache_dir=...)` replays stored
responses, including discovery snapshots, and raises on a cache miss. Offline
snapshots may be older than the current SEC data.

## Modules and provenance

| Module | Purpose |
|---|---|
| `client.py` | Rate limiting, retries, raw response caching, offline replay |
| `discovery.py` | Ticker/CIK resolution, submission history, master indexes |
| `formats.py` | Format dispatch, canonical rows, identifiers, provenance |
| `legacy_form4.py` | Legacy text and rendered HTML tables |
| `engines.py` | Per-form discovery, parsing, validation flags, DataFrames |
| `scrape.py` | CLI and CSV reports |

The package originated in the team repository's `merged_engine/` at commit
`ff6f062`. Its engine architecture came from one team engine (`src/`); legacy parsing came
from a second (`form4_formats.py`), and Form 144/caching work from
`vktx_light.py`. The submission uses local helper imports, explicit SEC identity
configuration, and a standard-library XML fallback. Subsequent fixes preserve
parser diagnostics, distinguish holdings and empty filings, and harden error
handling. The legacy module retains its standalone parsing entry points.

## Formats and diagnostic status

| Input | Handler | Normal status |
|---|---|---|
| Ownership XML with table wrappers | Native XML parser | `ok` |
| Ownership XML X0101 with flat security elements | Native XML parser | `ok_legacy_xml` |
| Form 144 or 144/A XML | Native XML parser | `ok` |
| Supported fixed-width, pipe-delimited and wrapped Form 4 text | Legacy parser | `ok_legacy_table` |
| Supported SEC-rendered ownership tables | HTML parser | `ok_legacy_table` |
| Complete SGML submission | Dispatch to embedded document handler | Depends on embedded content |
| Recognized native filing without table rows | Metadata-only rows | `empty` |
| Recovered XML or detected partial legacy extraction | Rows plus diagnostics | `needs_review` |
| PDF | Placeholder; no text extraction or OCR | `unsupported_pdf` |
| Unrecognized content, parse failure or fetch failure | Error row | `error` |

XML is parsed strictly first. If lxml is installed, malformed XML can be
recovered, but every recovered result is marked `needs_review`: recovery may
lose or misplace rows. Without lxml, malformed XML produces an error row.
`parse_warnings` contains JSON diagnostics when present. Legacy layout detection
is heuristic; an `ok_legacy_table` status does not guarantee complete extraction
from every historical layout. Unresolved rendered joint-owner metadata and
missing referenced notes require review. When caching is enabled, source bytes
remain available in the raw cache and can be identified by their source hash.

Rows use `formats.CANON_FIELDS`, including `source_format`, `parse_status`,
filing notes, owner attribution and source provenance. Empty native filings
retain their owner and issuer metadata. Legacy balance-only lines are holdings.
Joint ownership XML associates each economic row with each reported owner
without asserting allocation: deduplicate `economic_row_id` before summing
transaction values. A calculated `derived_transaction_value` is not an
independently reported total.

## Quality reports

```bash
python -m shared.sec.form4_parser.scrape --ticker AAPL --form all --checks
python -m shared.sec.form4_parser.scrape --ticker AAPL --checks \
  --market-data /path/to/market.csv --split-adjustments /path/to/splits.csv
```

`--checks` calls the compatibility quality API and writes continuity and price
diagnostics. The existing continuity implementation is retained for comparison.
The stricter continuity implementation is an explicit programmatic opt-in:

```python
from shared.sec.form4_parser.quality_checks import check_ownership_continuity_v2

report = check_ownership_continuity_v2(df)
```

The v2 CLI checks an existing normalized ownership table and writes a new review
directory; it does not run acquisition or the full audit pipeline:

```bash
python -m shared.sec.form4_parser.src.quality_checks_v2 \
  --transactions /path/to/normalized-ownership.csv \
  --output-dir /path/to/new-v2-review-directory
```

Before promoting v2 results, run it through the full audit driver and compare
status counts with the existing implementation. Published continuity outputs
have not been regenerated by the parser tests.

Continuity compares the reported balance with
`expected_after = previous_position * split_factor + acquired - disposed`.
It operates within comparable owner/issuer/security/ownership groups. Diagnostic
output includes balances, differences and review reasons; input records are
retained. Check the selected implementation's documented skip rules before
interpreting its flags. A flag prompts review; it does not prove an error.

The price check compares reported per-share prices with exact-day, adjacent-day
and weekly market ranges. A separate value check compares `shares * price` with
`reported_transaction_value` when an independent reported total is available.
Supply `--market-data` or a `price_lookup(ticker, date)` callback returning
`low`, `high` and `close`; without market data, comparable rows return
`NO_MARKET_DATA`. Grants, gifts and other non-market transactions can be
`NOT_MARKET_COMPARABLE`.

The historical universe and market pipelines live in `shared/universe/` and
`shared/market/`. This CLI reads a market table only when supplied explicitly.

## Offline validation

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider shared/sec/form4_parser/tests
```

Tests cover XML/HTML agreement, legacy transactions and holdings, wrapped text,
notes, empty filings, amendments, recovered XML, failure reporting and offline
cache replay. The seven `rnst_*` source fixtures are preserved verbatim; tests
construct additional edge cases in memory.
