import numpy as np
import pandas as pd
import pytest

from analysis import monthly_returns
from engine import run_backtest
from main import ASSET_MAP_PATH, BASELINE, DROPPED_SYMBOLS, SAMPLE_START, RAW_DIR, STRATEGIES, load_panel
from panel import Panel

pytestmark = pytest.mark.skipif(not RAW_DIR.exists(), reason="course data not in data/raw")


@pytest.fixture(scope="module")
def panel():
    return load_panel()


def sharpe(r: pd.Series) -> float:
    return r.mean() / r.std() * np.sqrt(12)


def test_baseline_results(panel):
    # known results: TSMOM Sharpe about 1.0, static benchmark about 0.47, low correlation
    tsmom = monthly_returns(run_backtest(panel, STRATEGIES[BASELINE]())).loc[SAMPLE_START:]
    static = monthly_returns(run_backtest(panel, STRATEGIES["Static benchmark"]())).loc[SAMPLE_START:]
    assert str(tsmom.index[0]) == SAMPLE_START
    assert sharpe(tsmom) == pytest.approx(1.01, abs=0.03)
    assert sharpe(static) == pytest.approx(0.47, abs=0.02)
    assert tsmom.corr(static) == pytest.approx(0.10, abs=0.04)


def test_no_lookahead(panel, tmp_path):
    # cut every file at a month end: weights decided up to that day must not change
    cutoff = pd.Timestamp("1999-12-31")
    for file in RAW_DIR.glob("*.csv"):
        prices = pd.read_csv(file, index_col=0, parse_dates=True).loc[:cutoff]
        if len(prices):  # Panel cannot read an empty file (its column becomes object dtype)
            prices.to_csv(tmp_path / file.name)
    short_panel = Panel(raw_dirs=tmp_path, asset_map_path=ASSET_MAP_PATH, dropped_symbols=DROPPED_SYMBOLS)

    full = run_backtest(panel, STRATEGIES["Vol target 11%"]()).weights.loc[:cutoff]
    short = run_backtest(short_panel, STRATEGIES["Vol target 11%"]()).weights.loc[:cutoff]
    later_symbols = full.columns.difference(short.columns)  # no data before the cutoff
    assert (full[later_symbols] == 0).all().all()
    pd.testing.assert_frame_equal(full[short.columns], short, check_freq=False)
