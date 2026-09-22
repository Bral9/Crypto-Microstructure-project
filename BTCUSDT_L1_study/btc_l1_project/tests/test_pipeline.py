import unittest
import numpy as np
import pandas as pd
from pipeline import sample_quotes, make_features, split_time, direction

class CausalityTests(unittest.TestCase):
    def quotes(self, times, bids):
        return pd.DataFrame({'timestamp': pd.to_datetime(times, unit='s', utc=True),
            'bid_price': bids, 'ask_price': np.array(bids)+2., 'bid_qty': 3., 'ask_qty': 1.})

    def test_backward_and_stale(self):
        g, _ = sample_quotes(self.quotes([0, .5, 4], [100, 200, 300]), max_age=1)
        self.assertEqual(g.bid_price.iloc[0], 100)
        self.assertEqual(g.bid_price.iloc[1], 200)
        self.assertTrue(np.isnan(g.bid_price.iloc[2]))

    def test_target_does_not_jump_gap(self):
        g, _ = sample_quotes(self.quotes([0, 1, 4, 5], [100, 101, 104, 105]), max_age=0)
        f = make_features(g, 2)
        self.assertTrue(np.isnan(f.move.iloc[1]))
        self.assertEqual(f.move.iloc[4], 1)

    def test_invalid_update_masks_old_book(self):
        q = self.quotes([0, 1, 3], [100, 101, 103])
        q.loc[1, 'bid_qty'] = -1
        g, _ = sample_quotes(q, 10)
        self.assertTrue(g.bid_price.iloc[1:3].isna().all())

    def test_edge_identity(self):
        g, _ = sample_quotes(self.quotes([0, 1, 2], [100, 101, 102]))
        f = make_features(g, 2)
        np.testing.assert_allclose(f.microprice-f.mid, f.edge)
        np.testing.assert_allclose(f.edge/(f.spread/2), f.imbalance)

    def test_neutral_boundary(self):
        np.testing.assert_array_equal(direction(np.array([-1., 1., 2.]), 1.), [0, 0, 1])

    def test_conflicting_time_rejected(self):
        with self.assertRaises(ValueError):
            sample_quotes(self.quotes([0, 0, 1], [100, 101, 102]))

    def test_purge_and_causal_features(self):
        rng = np.random.default_rng(1)
        bids = 100 + np.cumsum(rng.choice([-2, 0, 2], 3000))
        bids += 1000
        g, _ = sample_quotes(self.quotes(np.arange(3000), bids))
        f = make_features(g)
        tr, ca, te = split_time(f)
        self.assertLess(tr.target_timestamp.max(), ca.timestamp.min())
        self.assertLess(ca.target_timestamp.max(), te.timestamp.min())
        g.loc[1500:, ['bid_price', 'ask_price']] += 100
        new = make_features(g)
        cols = ['edge', 'mid', 'volatility', 'spread', 'depth']
        pd.testing.assert_frame_equal(f.loc[:1499, cols], new.loc[:1499, cols])

if __name__ == '__main__':
    unittest.main()
