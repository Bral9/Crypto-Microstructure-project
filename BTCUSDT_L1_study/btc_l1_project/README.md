# BTCUSDT L1 microstructure study

Do simple best-quote features help estimate the next one-second midprice move?
This implements your framework with commented Python, pandas, NumPy,
scikit-learn and Matplotlib. No trading engine or deep learning.

**Included results are SYNTHETIC demonstrations, not market findings.** No real
quotes were supplied. The demo deliberately plants a weak signal to check the
pipeline. Do not present these scores as evidence on a CV or in a paper.

## Run (Windows PowerShell)
Install Python 3.11, extract the archive and open this directory in a terminal.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_study.py --demo --out my_demo
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

On macOS/Linux use `python3 -m venv .venv`, then `.venv/bin/python` instead of
`.\.venv\Scripts\python.exe`. No activation is required.

For real data (0.10 is an example: verify YOUR source's tick size):

```powershell
.\.venv\Scripts\python.exe run_study.py --input data/quotes.csv --tick-size 0.10 --out real_run_01
```

Numeric Unix timestamps additionally need `--timestamp-unit ms` (or s/us/ns).
Output folders must be new or empty to protect previous experiments.

## Read in this order
1. `pipeline.py`: quotes, quality checks, causal sampling, features and targets.
2. `models.py`: OLS, logistic regression, calibration, baselines, regime evaluation.
3. `plots.py`: six figures and probability-bin counts.
4. `run_study.py`: the command-line workflow connecting everything.
5. `demo.py`: synthetic test data, separate from research data.
6. `tests/test_pipeline.py`: research-critical alignment tests.

`WALKTHROUGH.ipynb` gives a guided route through the same functions.

## Framework coverage
- UTC second grid with backward quote lookup, invalid-book masking and quote-age limit.
- Midprice, spread, imbalance, depth, weighted L1 price, edge, trailing volatility.
- Exact next-second target, computed on the full grid before deleting missing rows.
- Chronological 70/15/15 split with boundary-crossing labels removed.
- Zero-move and raw-edge baselines, edge-only OLS, expanded OLS.
- Actual and predicted UP/NEUTRAL/DOWN labels using current half-spread.
- Majority-class and probability-prior baselines, scaled logistic regression.
- Separate sigmoid calibration, with the base model frozen.
- Test MAE in ticks, R², accuracy, macro F1, per-class reports, log loss and Brier score.
- Train-defined volatility/spread/depth regimes evaluated on test data.

Dynamic thresholds are deferred as in your staged framework. The half-spread rule
is fixed in this implementation; later tuning needs validation inside training,
not the final test or the probability-calibration block. Expanded OLS is a
predeclared comparison, not a winner selected using test scores.

## Outputs
| Output | Purpose |
|---|---|
| summary.txt | Model scores and edge OLS coefficients |
| manifest.json | Input checksum, configuration, quality counts, splits, versions |
| features_1s.parquet | Full timeline with unavailable rows and future outcomes |
| test_predictions.csv | Predictions, labels, probabilities and predicted center |
| *_metrics.csv | Regression, classification, probability and regime tables |
| per_class_reports.json | Precision, recall, F1 and support |
| reliability_bins.csv | Calibration-bin means, observed frequencies and counts |
| models.joblib | Fitted model bundle; only load trusted joblib files |
| figures/ | Six PNG interpretation figures |

The feature table contains future targets for evaluation: never send the whole
table to a model. The explicit FEATURES list selects only information known at t.

## Your work after delivery
Understand and manually verify timestamp/target pairs, then supply real quotes
and run the predeclared experiment. Report failures honestly. If you revise
settings after inspecting test results, obtain a fresh untouched test period.
A single holdout is a starting point, not robust evidence of generalization.
No confidence intervals, fills, fees, PnL or causal conclusions are implemented.

AI assisted this implementation from your framework. Explain your contributions
accurately and be able to defend the code. Internship value comes from a real,
reproducible investigation and your understanding; no code guarantees an offer.

See DATA_GUIDE.md, INTERPRETATION.md and RESEARCH_NOTES.md.

Notebook use is optional: install Jupyter with `python -m pip install notebook`
and open WALKTHROUGH.ipynb from this folder. The scripts do not require Jupyter.
