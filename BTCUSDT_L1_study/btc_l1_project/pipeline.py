"""Read quotes, sample causally, and create the framework's features/targets."""
from pathlib import Path
import numpy as np
import pandas as pd

BOOK = ['bid_price', 'ask_price', 'bid_qty', 'ask_qty']
FEATURES = ['edge', 'spread', 'volatility', 'depth']


def read_quotes(path, timestamp_unit=None):
    """Canonical CSV/Parquet; timestamps are UTC ISO strings or explicit epochs."""
    path = Path(path)
    raw = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
    missing = set(['timestamp'] + BOOK) - set(raw.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}. See DATA_GUIDE.md.')
    raw = raw[['timestamp'] + BOOK].copy()
    if pd.api.types.is_numeric_dtype(raw.timestamp) and timestamp_unit is None:
        raise ValueError('Numeric timestamps require --timestamp-unit s/ms/us/ns.')
    raw['timestamp'] = pd.to_datetime(raw.timestamp, unit=timestamp_unit,
                                     utc=True, errors='coerce')
    for col in BOOK:
        raw[col] = pd.to_numeric(raw[col], errors='coerce')
    return raw


def sample_quotes(raw, max_age=2.0):
    """Backward as-of join: a quote at 12:00:00.2 cannot inform 12:00:00."""
    audit = {'input_rows': len(raw), 'bad_timestamps': int(raw.timestamp.isna().sum())}
    q = raw.dropna(subset=['timestamp']).copy()
    audit['exact_duplicates'] = int(q.duplicated().sum())
    q = q.drop_duplicates().sort_values('timestamp', kind='stable')
    # Two different books with an identical timestamp need sequence information.
    # Silently choosing either could change targets. Ask the user to resolve it.
    if q.timestamp.duplicated().any():
        raise ValueError('Conflicting books at the same timestamp. Resolve using source sequence IDs.')
    valid = np.isfinite(q[BOOK]).all(axis=1) & (q[BOOK] > 0).all(axis=1)
    valid &= q.ask_price > q.bid_price
    audit['invalid_books'] = int((~valid).sum())
    # Invalid updates invalidate the state until another good update arrives.
    # Dropping them would incorrectly carry an older good book through bad data.
    q.loc[~valid, BOOK] = np.nan
    if len(q) < 2:
        raise ValueError('Need at least two timestamped quotes.')
    grid = pd.DataFrame({'timestamp': pd.date_range(q.timestamp.min().ceil('s'),
                                                   q.timestamp.max().floor('s'), freq='s')})
    if grid.empty:
        raise ValueError('Input does not span a full second.')
    q = q.rename(columns={'timestamp': 'quote_timestamp'})
    grid = pd.merge_asof(grid, q, left_on='timestamp', right_on='quote_timestamp', direction='backward')
    grid['quote_age_seconds'] = (grid.timestamp - grid.quote_timestamp).dt.total_seconds()
    stale = grid.quote_age_seconds > max_age
    grid.loc[stale, BOOK] = np.nan
    audit.update(grid_rows=len(grid), stale_seconds=int(stale.sum()),
                 unavailable_seconds=int(grid[BOOK].isna().any(axis=1).sum()))
    return grid, audit


def make_features(grid, window=60):
    f = grid.copy()
    f['mid'] = (f.ask_price + f.bid_price) / 2
    f['spread'] = f.ask_price - f.bid_price
    f['depth'] = f.bid_qty + f.ask_qty
    f['imbalance'] = (f.bid_qty - f.ask_qty) / f.depth
    f['microprice'] = (f.ask_price * f.bid_qty + f.bid_price * f.ask_qty) / f.depth
    # Equivalent to microprice - mid; avoids cancellation of large prices.
    f['edge'] = f.spread * f.imbalance / 2
    returns = np.log(f.mid / f.mid.shift(1))
    f['volatility'] = returns.rolling(window, min_periods=window).std()
    # Shift on the COMPLETE second grid before dropping unavailable rows.
    # Otherwise the 'next row' could actually be several seconds later.
    f['target_timestamp'] = f.timestamp + pd.Timedelta(seconds=1)
    f['future_mid'] = f.mid.shift(-1)
    f['move'] = f.future_mid - f.mid
    f['threshold'] = f.spread / 2
    f['label'] = direction(f.move, f.threshold)
    return f


def direction(move, threshold):
    # Tolerance keeps floating point roundoff at the boundary in NEUTRAL.
    return np.where(move > threshold + 1e-9, 1,
                    np.where(move < -threshold - 1e-9, -1, 0))


def split_time(f):
    """70/15/15 by elapsed grid rows, BEFORE missing-row removal; purge labels."""
    a, b = int(len(f) * .70), int(len(f) * .85)
    if a == 0 or b >= len(f):
        raise ValueError('Not enough time coverage.')
    cal_start, test_start = f.timestamp.iloc[a], f.timestamp.iloc[b]
    train = f[(f.timestamp < cal_start) & (f.target_timestamp < cal_start)]
    cal = f[(f.timestamp >= cal_start) & (f.timestamp < test_start)
            & (f.target_timestamp < test_start)]
    test = f[f.timestamp >= test_start]
    needed = FEATURES + ['move', 'future_mid']
    blocks = [x.replace([np.inf, -np.inf], np.nan).dropna(subset=needed).copy()
              for x in (train, cal, test)]
    for name, block in zip(['train', 'calibration', 'test'], blocks):
        if len(block) < 100:
            raise ValueError(f'{name} has only {len(block)} usable rows; collect more data.')
        if name != 'test' and set(block.label.unique()) != {-1, 0, 1}:
            raise ValueError(f'{name} lacks one of DOWN/NEUTRAL/UP. Collect a broader sample.')
    return blocks
