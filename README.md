# BTCUSDT L1 Microstructure Research Pipeline

A reproducible Python research pipeline for studying whether **Level-1 (best bid / best ask) order-book features contain short-horizon information about the next 1-second BTCUSDT perpetual-futures midprice move**.

The project focuses on research design, causal data alignment, simple interpretable baselines, and honest evaluation rather than on building a trading engine or claiming profitable alpha.

> **Status:** The repository includes a **synthetic demonstration dataset and synthetic example results** so the full pipeline can be run end to end. Those bundled scores are plumbing checks, **not empirical findings about BTCUSDT**. Real-market conclusions require running the frozen pipeline on documented historical L1 data.

## Research question

Can simple top-of-book variables - especially queue imbalance and microprice edge - help estimate the direction or size of the **next one-second midprice movement**?

The study is deliberately narrow. It uses only information available at time `t` and compares small, interpretable models against simple baselines.

## What the project does

The pipeline:

- reads BTCUSDT perpetual L1 quote data from CSV or Parquet;
- validates timestamps and best-book states;
- samples quotes on a 1-second UTC grid using a **backward as-of join** so future quotes cannot leak into the present;
- masks stale and invalid quotes rather than silently carrying bad state forward;
- constructs L1 microstructure features;
- creates an exact next-second target on the complete timeline;
- uses a chronological **70 / 15 / 15 train / calibration / test split**;
- evaluates regression and three-class directional models against explicit baselines;
- evaluates performance across train-defined volatility, spread, and depth regimes;
- writes reproducibility metadata, predictions, metrics, plots, and fitted models to a new output directory.

## Features

For each one-second observation, the study constructs:

| Feature | Definition / intuition |
|---|---|
| Midprice | `(best_bid + best_ask) / 2` |
| Spread | `best_ask - best_bid` |
| Depth | `bid_qty + ask_qty` at the best quotes |
| Queue imbalance | `(bid_qty - ask_qty) / (bid_qty + ask_qty)` |
| Microprice | L1 size-weighted price between bid and ask |
| Microprice edge | `microprice - midprice`, implemented as `spread * imbalance / 2` |
| Volatility | trailing standard deviation of 1-second log midprice returns |

The model feature set is intentionally small: `edge`, `spread`, `volatility`, and `depth`.

## Target and labels

The regression target is the exact next-second midprice move:

```text
move_t = midprice_(t+1) - midprice_t
```

The directional label is defined using the **current half-spread**:

- `UP` if the next move is greater than `+ spread_t / 2`
- `DOWN` if the next move is less than `- spread_t / 2`
- `NEUTRAL` otherwise

Targets are created on the full one-second grid **before unavailable rows are removed**. This prevents a missing observation from turning a "next row" target into a multi-second future target.

## Models and baselines

### Regression

- **Zero-move baseline** - predicts no change.
- **Raw-edge baseline** - uses current microprice edge directly as the predicted move.
- **Edge-only OLS** - maps microprice edge to a short-horizon move with an interpretable linear coefficient.
- **Expanded OLS** - linear model using edge, spread, volatility, and depth.

### Classification

- **Majority-class baseline**.
- Directional classes produced by thresholding the regression predictions.
- **Multinomial logistic regression** on the four features.
- **Sigmoid-calibrated logistic regression** fitted on a separate calibration block while the base classifier remains frozen.

No random train/test split is used.

## Evaluation

The test set reports:

- MAE measured in ticks;
- R-squared;
- accuracy;
- macro F1;
- per-class precision, recall, F1, and support;
- multiclass log loss;
- multiclass Brier score;
- regime-level results across volatility, spread, and depth.

The repository also produces confusion matrices, probability-calibration plots, class/signal diagnostics, market-quality plots, and regime comparisons.

## Causality and leakage safeguards

Several implementation choices are designed specifically to reduce accidental look-ahead bias:

1. **Backward quote lookup** - a quote timestamped after a grid point cannot populate that grid point.
2. **Exact next-second target** - the target is shifted on the complete timeline before missing rows are dropped.
3. **Chronological splitting** - training, calibration, and testing remain time ordered.
4. **Boundary purge** - labels that cross from one split into the next are removed.
5. **Train-only regime cutoffs** - regime boundaries are learned from training data, then applied to test data.
6. **Explicit feature list** - future target columns exist for evaluation but are never passed to the models.

The test suite checks several of these research-critical invariants.

## Repository structure

```text
btc_l1_project/
|-- pipeline.py                 # ingestion, cleaning, causal sampling, features, targets
|-- models.py                   # baselines, OLS, classifier, calibration, regime evaluation
|-- plots.py                    # interpretation figures
|-- run_study.py                # command-line experiment runner
|-- demo.py                     # synthetic end-to-end plumbing check
|-- tests/
|   `-- test_pipeline.py        # causality/alignment tests
|-- data/
|   `-- schema_example.csv      # expected input schema
|-- WALKTHROUGH.ipynb           # guided walkthrough
|-- DATA_GUIDE.md               # data contract and provenance guidance
|-- INTERPRETATION.md           # how to interpret each output figure
|-- RESEARCH_NOTES.md           # checklist for documenting a real-data run
`-- example_results/            # synthetic demo outputs only
```

## Quick start

### 1. Create an environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS / Linux:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

### 2. Run the synthetic demonstration

```bash
python run_study.py --demo --out demo_run
```

### 3. Run the tests

```bash
python -m unittest discover -s tests -v
```

### 4. Run on documented real L1 data

```bash
python run_study.py \
  --input data/quotes.csv \
  --tick-size 0.10 \
  --out real_run_01
```

`0.10` is only an example. Use the tick size documented by the venue/source for the dataset you actually analyze.

If timestamps are numeric Unix epochs, specify the unit explicitly, for example:

```bash
python run_study.py --input data/quotes.csv --timestamp-unit ms --tick-size 0.10 --out real_run_01
```

## Expected input schema

| Column | Meaning |
|---|---|
| `timestamp` | UTC ISO timestamp, or Unix epoch with explicit unit |
| `bid_price` | best bid price |
| `ask_price` | best ask price |
| `bid_qty` | displayed quantity at the best bid |
| `ask_qty` | displayed quantity at the best ask |

See `DATA_GUIDE.md` before using research data. In particular, document venue, instrument, timestamp semantics, source permissions, tick metadata, and any source-specific sequence handling.

## Outputs

Each experiment writes a self-contained output directory containing:

- `summary.txt` - headline model metrics and OLS edge parameters;
- `manifest.json` - input checksum, configuration, data-quality counts, split metadata, feature list, and package versions;
- `features_1s.parquet` - aligned research table;
- `test_predictions.csv` - predictions, labels, and class probabilities;
- `*_metrics.csv` - regression, classification, probability, and regime metrics;
- `per_class_reports.json` - detailed classification reports;
- `reliability_bins.csv` - probability calibration bins;
- `models.joblib` - fitted model bundle;
- `figures/` - six diagnostic / interpretation figures.

Output directories must be new or empty so previous experiments are not silently overwritten.

## What the synthetic demo proves - and what it does not

The bundled synthetic generator deliberately plants a weak relationship between prior L1 edge and the next move. Its purpose is to verify that the data pipeline, modeling, evaluation, plotting, and artifact-writing logic work together.

It does **not** establish that:

- BTCUSDT L1 features are predictive in live markets;
- the measured relationship is statistically significant;
- a signal survives fees, latency, slippage, or adverse selection;
- the model is profitable or suitable for execution.

Those questions require real, well-documented market data and stronger forward validation.

## Limitations and next steps

This is a deliberately small research project. Current limitations include:

- L1 only - no deeper order-book levels or trade-flow features;
- fixed 1-second sampling and prediction horizon;
- a single chronological holdout rather than repeated walk-forward evaluation;
- no dependence-aware confidence intervals;
- no transaction costs, latency, fills, inventory, or PnL model;
- no causal claims;
- no test-set-driven hyperparameter search by design.

Natural extensions are to run the frozen design on multiple market periods, add rolling / walk-forward evaluation, quantify uncertainty, compare horizons, and only then consider an execution study.

## Design philosophy

The project favors **simple models, explicit baselines, causal alignment, and reproducibility** over unnecessary model complexity. A weak or null result on real data is still a valid research outcome if the experiment is correctly specified and honestly reported.

## Tech stack

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- PyArrow / Parquet
- joblib
- `unittest`

## Development note

The research framework and experiment design were developed by the repository author. AI-assisted coding was used during implementation. The final code was reviewed, tested, and documented so that every major research decision can be explained and defended.

## Author

**Breal Ciceron**  
GitHub: https://github.com/Bral9  
LinkedIn: https://www.linkedin.com/in/breal-ciceron-1b94351a9/
