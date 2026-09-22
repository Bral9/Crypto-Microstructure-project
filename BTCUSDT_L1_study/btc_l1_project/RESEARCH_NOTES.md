# Fill these notes with YOUR real-data findings

Before running: venue/instrument, dates, timestamp clock, tick metadata, source
permission, archive location, hypothesis, frozen settings and planned comparisons.
After running: cleaning losses, split counts/class balance, OLS alpha/beta,
baseline comparisons, calibration help/harm, weak regimes, limitations, next test.

Interview questions:
1. Why backward as-of rather than nearest sampling?
2. Why shift the full grid before dropping missing rows?
3. Why does normalized edge equal imbalance?
4. Why can raw edge never exceed half-spread?
5. What is the OLS coefficient, and why is it not confidence?
6. Why separate training, calibration and test?
7. Why can high accuracy be misleading?
8. Why can calibration hurt?
9. Why does prediction not imply profit?
10. What did you personally design, verify and change?

Source framework: BTCUSDT_L1_Microstructure_Framework_Polished(1).pdf supplied by you.
Official documentation checked for implementation:
- https://pandas.pydata.org/docs/reference/api/pandas.merge_asof.html
- https://scikit-learn.org/stable/modules/calibration.html
- https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html

The temporal split is explicit; automatic random cross-validation is not used.
Conceptual readings from your framework: Gould & Bonart on queue imbalance;
Stoikov (2018) on micro-price; Platt (1999) on sigmoid calibration. Read the papers
before making claims about their empirical conclusions.
