"""Synthetic plumbing check. The planted signal is NOT evidence of market alpha."""
import numpy as np
import pandas as pd


def generate_demo(path, n=18000, seed=42):
    rng = np.random.default_rng(seed)
    imbalance = rng.uniform(-.95, .95, n)
    spread = rng.choice([.2, .4, .6], n)
    depth = rng.lognormal(2, .5, n)
    mid = np.empty(n); mid[0] = 60000
    for i in range(1, n):
        # Deliberately planted weak association with the PREVIOUS quote's edge.
        noise = rng.normal(0, .22 if i < n//2 else .38)
        move = .8 * spread[i-1] * imbalance[i-1] / 2 + noise
        mid[i] = mid[i-1] + np.round(move / .1) * .1
    frame = pd.DataFrame({'timestamp': pd.date_range('2026-01-01', periods=n, freq='s', tz='UTC'),
        'bid_price': np.round(mid - spread/2, 8), 'ask_price': np.round(mid + spread/2, 8),
        'bid_qty': depth*(1+imbalance)/2, 'ask_qty': depth*(1-imbalance)/2})
    # One outage and one corrupt book exercise the cleaning path.
    frame = frame.drop(index=range(5000, min(5010, n)), errors='ignore')
    if n > 8000:
        frame.loc[8000, 'bid_qty'] = -1
    frame = pd.concat([frame, frame.iloc[[100]]], ignore_index=True)
    frame.to_csv(path, index=False)
