# Data

Raw data is not distributed (Wind and Compustat are licensed). Place two CSVs here:

**`TSLA_Merged_V3.csv`**: one row per trading day, 2016-03-31 to 2026-02-28.

| column | meaning |
|---|---|
| `date` | trading date (YYYY-MM-DD) |
| `returns` | TSLA daily simple return; the target is built as the sign of next day's `returns` |
| other numeric columns | candidate factors (macro, fundamentals aligned to first disclosure date, technical, AI narrative) |

Every numeric column except the target is treated as a candidate feature.

**`tsla_ai_narrative_daily_from_rpa.csv`**: the daily AI narrative factor file. Only its header is read, to tell the
pipeline which columns of the merged file are AI-narrative features (these get an extra
missing-rate / correlation filter).
