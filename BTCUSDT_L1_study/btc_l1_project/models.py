"""Small, fixed models. No test-driven tuning or random train/test splits."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (mean_absolute_error, r2_score, accuracy_score,
                             f1_score, log_loss, classification_report)
from pipeline import FEATURES, direction

CLASSES = np.array([-1, 0, 1])
NAMES = ['DOWN', 'NEUTRAL', 'UP']


def fit_models(train, calibration):
    edge = LinearRegression().fit(train[['edge']], train.move)
    expanded = make_pipeline(StandardScaler(), LinearRegression())
    expanded.fit(train[FEATURES], train.move)
    classifier = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000))
    classifier.fit(train[FEATURES], train.label)
    # FrozenEstimator prevents refitting the base model on calibration data.
    calibrated = CalibratedClassifierCV(FrozenEstimator(classifier), method='sigmoid')
    calibrated.fit(calibration[FEATURES], calibration.label)
    prior = train.label.value_counts(normalize=True).reindex(CLASSES, fill_value=0).to_numpy()
    return {'edge': edge, 'expanded': expanded, 'classifier': classifier,
            'calibrated': calibrated, 'prior': prior}


def evaluate(models, test, tick):
    out = test.copy()
    moves = {'zero_move': np.zeros(len(test)), 'raw_edge': test.edge.to_numpy(),
             'ols_edge': models['edge'].predict(test[['edge']]),
             'ols_expanded': models['expanded'].predict(test[FEATURES])}
    regression, classification, probability = [], [], []
    labels = {'majority': np.full(len(test), CLASSES[models['prior'].argmax()])}
    for name, pred in moves.items():
        out[f'prediction_{name}'] = pred
        regression.append({'model': name, 'mae_ticks': mean_absolute_error(test.move, pred) / tick,
                           'r2': r2_score(test.move, pred)})
        labels[name] = direction(pred, test.threshold)
    probabilities = {'class_prior': np.tile(models['prior'], (len(test), 1))}
    for name in ['classifier', 'calibrated']:
        model = models[name]
        # Explicit ordering protects the probability columns and Brier score.
        p = model.predict_proba(test[FEATURES])
        probabilities[name] = p[:, [list(model.classes_).index(c) for c in CLASSES]]
        labels[name] = CLASSES[probabilities[name].argmax(axis=1)]
    truth = (test.label.to_numpy()[:, None] == CLASSES).astype(float)
    for name, p in probabilities.items():
        probability.append({'model': name, 'log_loss': log_loss(test.label, p, labels=CLASSES),
                            'brier_multiclass': np.mean(np.sum((p - truth)**2, axis=1))})
        for i, c in enumerate(NAMES):
            out[f'p_{name}_{c.lower()}'] = p[:, i]
    reports = {}
    for name, pred in labels.items():
        out[f'class_{name}'] = pred
        classification.append({'model': name, 'accuracy': accuracy_score(test.label, pred),
            'macro_f1': f1_score(test.label, pred, labels=CLASSES, average='macro', zero_division=0),
            'directional_fraction': np.mean(pred != 0)})
        reports[name] = classification_report(test.label, pred, labels=CLASSES,
                           target_names=NAMES, output_dict=True, zero_division=0)
    out['predicted_center'] = out.mid + out.prediction_ols_edge
    return out, pd.DataFrame(regression), pd.DataFrame(classification), pd.DataFrame(probability), reports


def regime_scores(train, test, tick):
    rows, boundaries = [], {}
    for feature in ['volatility', 'spread', 'depth']:
        cuts = train[feature].quantile([1/3, 2/3]).to_numpy()
        boundaries[feature] = cuts.tolist()
        # Tied cut points legitimately produce empty regimes; do not force ranks.
        groups = np.where(test[feature] <= cuts[0], 'low',
                          np.where(test[feature] <= cuts[1], 'medium', 'high'))
        for group in ['low', 'medium', 'high']:
            sub = test.loc[groups == group]
            row = {'feature': feature, 'regime': group, 'n': len(sub)}
            if len(sub):
                row.update(ols_mae_ticks=mean_absolute_error(sub.move, sub.prediction_ols_edge)/tick,
                    zero_mae_ticks=np.mean(abs(sub.move))/tick,
                    calibrated_accuracy=accuracy_score(sub.label, sub['class_calibrated']),
                    majority_accuracy=accuracy_score(sub.label, sub['class_majority']),
                    macro_f1=f1_score(sub.label, sub['class_calibrated'], labels=CLASSES,
                                      average='macro', zero_division=0))
            rows.append(row)
    return pd.DataFrame(rows), boundaries
