import numpy as np
import pandas as pd

from config import ASSET_MAP_PATH, DROPPED_SYMBOLS, ROLL_FIXES

MAX_GAP_DAYS = 20  # AP, dropped for its holes, had a 159-day gap; the rest of the universe stays under 14
# contracts that genuinely stop before the end of the data, as found by scripts/audit_data.py
ENDS_EARLY = {"SS", "EC", "XX"}
# last bar of the retired big contract, where its mini takes over
JOINS = {"SP": "2021-09-17", "ND": "2015-06-10", "ZD": "2015-06-10"}


def test_universe_is_the_asset_map_minus_dropped(real_panel):
    asset_map = pd.read_csv(ASSET_MAP_PATH, index_col="ID", keep_default_na=False)
    assert set(real_panel.symbols) == set(asset_map.index) - DROPPED_SYMBOLS


def test_prices_are_positive_and_returns_finite(real_panel):
    assert np.nanmin(real_panel.close_df.to_numpy()) > 0
    assert not np.isinf(real_panel.returns.to_numpy()).any()


def test_no_long_gaps_in_the_universe(real_panel):
    gaps = {
        s: real_panel.close_df[s].dropna().index.to_series().diff().dt.days.max()
        for s in real_panel.symbols
    }
    assert {s: g for s, g in gaps.items() if g > MAX_GAP_DAYS} == {}


def test_only_the_expected_contracts_end_early(real_panel):
    last = real_panel.all_dates[-1]
    ended = {
        s
        for s, i in zip(real_panel.symbols, real_panel.last_valid_idx)
        if (last - real_panel.all_dates[i]).days > 10
    }
    assert ended == ENDS_EARLY


def test_big_index_contracts_run_to_the_end_of_the_data(real_panel):
    n = len(real_panel.all_dates)
    for big, join in JOINS.items():
        assert real_panel.last_valid_idx[real_panel.symbols.index(big)] == n - 1
        around = pd.Timestamp(join)
        window = real_panel.observed_returns[big].loc[
            around - pd.Timedelta(days=5) : around + pd.Timedelta(days=5)
        ]
        assert window.abs().max() < 0.10  # the join is an ordinary move


def test_roll_fixes_are_applied(real_panel):
    for symbol, dates in ROLL_FIXES.items():
        for d in dates:
            assert abs(real_panel.returns.loc[d, symbol]) < 1e-12
