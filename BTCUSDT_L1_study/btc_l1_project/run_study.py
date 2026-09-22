"""Run from this directory: python run_study.py --demo --out example_results"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import joblib
import numpy as np
from pipeline import read_quotes, sample_quotes, make_features, split_time, FEATURES
from models import fit_models, evaluate, regime_scores
from plots import create_plots
from demo import generate_demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path)
    source.add_argument('--demo', action='store_true')
    parser.add_argument('--out', type=Path, default=Path('results'))
    parser.add_argument('--tick-size', type=float, help='USDT price increment for YOUR dataset; required for real input')
    parser.add_argument('--timestamp-unit', choices=['s', 'ms', 'us', 'ns'])
    parser.add_argument('--max-age', type=float, default=2.0)
    parser.add_argument('--window', type=int, default=60, help='Trailing volatility window in seconds')
    args = parser.parse_args()
    if (args.tick_size is not None and (not np.isfinite(args.tick_size) or args.tick_size <= 0)) or args.window < 2 or not np.isfinite(args.max_age) or args.max_age < 0:
        parser.error('Tick must be positive, window >= 2, and max age finite and nonnegative.')
    if not args.demo and args.tick_size is None:
        parser.error('Supply --tick-size from your source metadata; do not guess.')
    tick = args.tick_size if args.tick_size is not None else .1
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Output directory must be empty or new, to protect earlier experiments.')
    args.out.mkdir(parents=True, exist_ok=True)
    if args.demo:
        args.input = args.out / 'synthetic_quotes.csv'
        generate_demo(args.input)
    raw = read_quotes(args.input, args.timestamp_unit)
    grid, audit = sample_quotes(raw, args.max_age)
    f = make_features(grid, args.window)
    train, cal, test = split_time(f)
    models = fit_models(train, cal)
    predictions, reg, cls, prob, reports = evaluate(models, test, tick)
    regimes, boundaries = regime_scores(train, predictions, tick)
    f.to_parquet(args.out / 'features_1s.parquet', index=False)
    predictions.to_csv(args.out / 'test_predictions.csv', index=False)
    for name, table in [('regression', reg), ('classification', cls), ('probability', prob), ('regimes', regimes)]:
        table.to_csv(args.out / f'{name}_metrics.csv', index=False)
    create_plots(f, train, cal, predictions, reg, cls, regimes, args.out / 'figures', args.demo)
    splits = {name: {'rows': len(block), 'start': str(block.timestamp.min()), 'end': str(block.timestamp.max()),
              'classes': {str(k): int(v) for k, v in block.label.value_counts().items()}}
              for name, block in [('train', train), ('calibration', cal), ('test', test)]}
    digest = hashlib.sha256()
    with args.input.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    manifest = {'synthetic_demo': args.demo, 'input': str(args.input), 'sha256': digest.hexdigest(),
        'config': {'tick_size': tick, 'max_age_seconds': args.max_age, 'volatility_window': args.window,
                   'horizon_seconds': 1, 'threshold': 'current half spread', 'features': FEATURES},
        'data_quality': audit, 'splits': splits, 'regime_cutoffs_train_only': boundaries,
        'ols_edge': {'intercept': float(models['edge'].intercept_), 'beta': float(models['edge'].coef_[0])},
        'python': platform.python_version(), 'versions': {p: importlib.metadata.version(p)
                     for p in ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'pyarrow', 'joblib']}}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (args.out / 'per_class_reports.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    joblib.dump(models, args.out / 'models.joblib')
    status = 'SYNTHETIC DEMO: these scores are NOT market research findings.' if args.demo else 'SUPPLIED DATA: verify provenance before research use.'
    summary = [status, '', 'Regression (test)', reg.to_string(index=False), '',
               'Classification (test)', cls.to_string(index=False), '', 'Probabilities (test)',
               prob.to_string(index=False), '', 'OLS edge parameters: '+str(manifest['ols_edge']), '',
               'Read INTERPRETATION.md before drawing conclusions. No profitability or causal claims.']
    (args.out / 'summary.txt').write_text('\n'.join(summary), encoding='utf-8')
    print('\n'.join(summary))
    print(f'\nSaved tables, figures, aligned data, models, and audit to {args.out.resolve()}')


if __name__ == '__main__':
    main()
