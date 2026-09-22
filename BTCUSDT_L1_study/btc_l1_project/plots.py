"""Matplotlib figures; each figure states whether its data are synthetic."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from models import NAMES, CLASSES


def create_plots(grid, train, cal, test, reg, classification, regimes, output, demo):
    output.mkdir(exist_ok=True)
    plt.rcParams.update({'figure.dpi': 130, 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.size': 10})
    tag = 'SYNTHETIC DEMO — NOT MARKET EVIDENCE' if demo else 'BTCUSDT L1 — supplied data'
    def save(fig, name):
        fig.suptitle(tag, fontsize=11, color='#9a3412' if demo else '#123047')
        fig.tight_layout(rect=(0, 0, 1, .95))
        fig.savefig(output / name, bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(grid.timestamp, grid.mid, lw=.6)
    axes[0].set_ylabel('Midprice (USDT)')
    axes[1].plot(grid.timestamp, grid.spread, lw=.6)
    axes[1].set_ylabel('Spread (USDT)')
    axes[2].plot(grid.timestamp, grid.quote_age_seconds, lw=.6)
    axes[2].set_ylabel('Quote age (seconds)')
    for ax in axes:
        for part, color, name in [(train, '#dbeafe', 'train'), (cal, '#fef3c7', 'calibration'),
                                  (test, '#dcfce7', 'test')]:
            ax.axvspan(part.timestamp.min(), part.timestamp.max(), color=color, alpha=.4, label=name)
    axes[0].legend(ncol=3)
    axes[2].set_xlabel('UTC time; missing midprice segments indicate unusable quotes')
    save(fig, '01_market_and_quality.png')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts = pd.DataFrame({name: part.label.value_counts(normalize=True).reindex(CLASSES, fill_value=0)
                           for name, part in [('train', train), ('calibration', cal), ('test', test)]})
    counts.index = NAMES
    counts.plot.bar(ax=axes[0], rot=0)
    axes[0].set_ylabel('Class fraction')
    # Train-only descriptive bins. Count labels discourage overreading small bins.
    grouped = train.groupby(pd.cut(train.imbalance, np.linspace(-1, 1, 11)), observed=True)
    bins = grouped.agg(imbalance=('imbalance', 'mean'), move=('move', 'mean'), n=('move', 'size'))
    axes[1].plot(bins.imbalance, bins.move, 'o-')
    for row in bins.itertuples():
        axes[1].annotate(str(row.n), (row.imbalance, row.move), fontsize=7, xytext=(0, 5), textcoords='offset points')
    axes[1].axhline(0, color='grey', lw=.8)
    axes[1].set(xlabel='L1 imbalance (training only)', ylabel='Mean next-second move (USDT)',
                title='Descriptive association; labels are bin counts')
    save(fig, '02_classes_and_signal.png')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(reg.model, reg.mae_ticks, color='#2563eb')
    axes[0].set(xlabel='Test MAE (ticks); lower is better', title='Regression vs baselines')
    axes[1].barh(classification.model, classification.macro_f1, color='#0f766e')
    axes[1].set(xlabel='Test macro F1; higher is better', xlim=(0, 1), title='Directional classification')
    save(fig, '03_model_comparison.png')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, model in zip(axes, ['ols_edge', 'calibrated']):
        ConfusionMatrixDisplay.from_predictions(test.label, test[f'class_{model}'], labels=CLASSES,
            display_labels=NAMES, cmap='Blues', colorbar=False, ax=ax)
        ax.set_title(model + ' — test counts')
    save(fig, '04_confusion_matrices.png')

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    reliability = []
    for j, name in enumerate(NAMES):
        truth = (test.label == CLASSES[j]).to_numpy()
        axes[0, j].plot([0, 1], [0, 1], '--', color='grey')
        for model in ['classifier', 'calibrated']:
            p = test[f'p_{model}_{name.lower()}'].to_numpy()
            ids = np.minimum((p * 10).astype(int), 9)
            x, y = [], []
            for b in range(10):
                mask = ids == b
                if mask.any():
                    x.append(p[mask].mean()); y.append(truth[mask].mean())
                    reliability.append({'model': model, 'class': name, 'bin': b,
                        'n': int(mask.sum()), 'mean_probability': x[-1], 'observed_fraction': y[-1]})
            axes[0, j].plot(x, y, 'o-', label=model)
            axes[1, j].hist(p, bins=np.linspace(0, 1, 11), alpha=.45, label=model)
        axes[0, j].set(title=name, xlabel='Mean predicted probability', ylabel='Observed fraction',
                        xlim=(0, 1), ylim=(0, 1))
        axes[1, j].set(xlabel='Predicted probability', ylabel='Number of test observations')
    axes[0, 0].legend(fontsize=8)
    pd.DataFrame(reliability).to_csv(output.parent / 'reliability_bins.csv', index=False)
    save(fig, '05_probability_calibration.png')

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, feature in zip(axes, ['volatility', 'spread', 'depth']):
        sub = regimes[regimes.feature == feature]
        x = np.arange(3)
        ax.bar(x-.18, sub.ols_mae_ticks, .36, label='OLS edge')
        ax.bar(x+.18, sub.zero_mae_ticks, .36, label='Zero move')
        ax.set_xticks(x, [f'{r.regime}\nn={r.n}' for r in sub.itertuples()])
        ax.set(title=feature, ylabel='Test MAE (ticks)')
    axes[0].legend()
    save(fig, '06_regimes.png')
