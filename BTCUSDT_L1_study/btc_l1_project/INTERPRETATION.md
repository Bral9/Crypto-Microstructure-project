# Interpret the six figures

Bundled graphs are synthetic. The generator plants an association; this is not alpha.

1. **Market/quality:** examine unavailable periods, quote age and split differences.
   Report missingness; cleaning may remove difficult market conditions.
2. **Classes/signal:** class fractions reveal imbalance and temporal changes.
   The training imbalance curve is descriptive association, not causation. Point
   labels are counts. No confidence intervals are implied.
3. **Model comparisons:** lower MAE is better; compare OLS to zero-move and raw edge.
   R² can be negative. OLS minimizes squared error, not MAE, so rankings can differ.
   Macro F1 weights the three classes equally; check per-class recall as well.
4. **Confusion matrices:** rows are actual, columns predicted. Look for an overly
   dominant neutral prediction. Raw edge always lies within half-spread, so its
   thresholded class is always neutral. OLS can rescale it beyond this boundary.
5. **Calibration:** closer to the diagonal is desirable. Inspect probability-bin
   counts/histograms: small bins are weak evidence. Sigmoid calibration may worsen
   logistic probabilities, especially after distribution shift. Each class is
   calibrated separately and outputs normalized; accuracy is not calibration.
6. **Regimes:** compare OLS and zero-move within each group; report n. Cutoffs come
   from training only. Tied/constant features can leave groups empty; do not force
   artificial equal-sized groups or infer results from empty/small groups.

Brier here is mean SUM over classes of (probability - one-hot truth)^2, range 0–2.
It is not half-scaled. Lower Brier and log loss are better, but neither measures
calibration alone: both also reflect discrimination and outcome uncertainty.

Actual labels use the half-spread known at t, never the future spread. OLS predicts
move size in USDT, then thresholds it; logistic regression predicts classes directly.
Predicted center = mid + edge-only OLS predicted move, not fundamental fair value.
No execution or profitability follows from this estimate.

Temporal dependence means thousands of adjacent seconds are not thousands of
independent experiments. Single-holdout differences do not establish significance.
Forward evaluations across more dates and dependence-aware uncertainty are future
extensions. If a model fails to beat its baseline, report that valid finding.
