# Data contract

Supply one venue's BTCUSDT **perpetual L1 quotes**, not candles, trades or spot.
The project reads your CSV/Parquet; it does not download or collect history.

| Column | Units/meaning |
|---|---|
| timestamp | UTC ISO string, or Unix epoch with explicit --timestamp-unit |
| bid_price | Best bid, USDT per BTC |
| ask_price | Best ask, USDT per BTC |
| bid_qty | Displayed BTC at best bid |
| ask_qty | Displayed BTC at best ask |

Keep raw originals. Rename source-specific fields and verify quantities/instrument
in a small adapter before running. Column names cannot establish provenance.
`data/schema_example.csv` is a format example, too short for fitting.

Receipt time supports an 'available to my collector' study. Exchange timestamps
support an exchange-time offline study but do not establish live availability.
Document the clock; store both timestamps in raw data when possible. Offset-free
strings are interpreted as UTC. Conflicting books sharing a timestamp are rejected:
resolve using source sequence order; do not invent the order.

Exact duplicates are removed; unchanged books with distinct timestamps remain.
Bad quotes invalidate the state until the next valid update. Quotes older than
`--max-age` (default 2 seconds) are masked. This is only a staleness heuristic:
unchanged event-driven books can be valid longer, and brief disconnects may evade
it. Use collection health logs/sequence checks where available. Volatility requires
a complete trailing window, so it restarts after unavailable data.

Supply the tick size for the source/date range, not an inferred minimum observed
move. Split datasets across tick-size changes. Inspect the manifest's rejection
counts and quote-age plot before interpreting any model.

At least 100 usable rows per split and all three classes in train/calibration are
software guards, not statistical sufficiency. Seek multiple distinct market days.
The initial version loads one input file and its full second grid into memory.
Use a manageable contiguous interval, not months of raw ticks on a laptop.
A future chunked importer must preserve state across chunk boundaries.
