# Running without WRDS

## Available without WRDS

All SEC-facing layers can run without a WRDS connection when the public SEC files are present locally:

- quarterly Forms 3/4/5 acquisition and lossless normalization;
- the signal-oriented Form 4 normalization layer;
- Form 144 master-index discovery and XML parsing (the parser is sample-tested, but filings have not been collected);
- issuer-day event construction and aggregate event exports;
- ownership-continuity auditing from SEC holdings and transactions;
- an alternative universe built from EDGAR issuer identifiers and SEC SIC classifications.

The EDGAR/SIC universe is a practical non-WRDS alternative, but it is not equivalent to the official historical GICS universe: SIC is broader and SEC identifiers do not by themselves provide complete, effective-dated security histories.

## Requires WRDS-derived data

The official research outputs require locally prepared WRDS-derived inputs for:

- CRSP daily prices, total returns, volumes, market capitalisation, share factors, and delisting indicators;
- historical Compustat GICS membership;
- effective-dated CRSP–Compustat and security/issuer links.

Once those extracts already exist under the configured data root, report generation is offline and does not require a live WRDS session. The repository does not redistribute licensed rows.
