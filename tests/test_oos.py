"""Out-of-sample readiness: the join check, and a backtest that runs straight across 2014 -> 2015."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import monthly_returns
from baselines import BASELINE_STRATEGIES
from engine import FuturesCostModel, run_backtest
from oos import check_splice
from panel import Panel
from strategy import USER_STRATEGIES

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
ASSET_MAP_PATH = ROOT / "data" / "processed" / "AssetMapCsv.csv"
# keep in sync with main.py
DROPPED_SYMBOLS = {"EN", "ES", "SC", "BC", "BG", "ZH", "FF", "RL", "YM"}
IN_SAMPLE_END = pd.Timestamp("2014-12-31")
STRATEGIES = {**BASELINE_STRATEGIES, **USER_STRATEGIES}

needs_data = pytest.mark.skipif(not RAW_DIR.exists(), reason="course data not in data/raw")


def write_prices(folder: Path, symbol: str, close: pd.Series, volume=100.0, oi=1000.0):
    """One csv in the course format (Open/High/Low/Close/Volume/OpenInterest)."""
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": volume,
            "OpenInterest": oi,
        }
    ).to_csv(folder / f"{symbol}.csv")


def random_walk(days, start=100.0, vol=0.01, seed=0) -> pd.Series:
    steps = np.random.default_rng(seed).normal(0.0, vol, len(days))
    return pd.Series(start * np.cumprod(1.0 + steps), index=days)


# The join check on synthetic files

IN_DAYS = pd.bdate_range("2013-06-03", "2014-12-31")
OUT_DAYS = pd.bdate_range("2015-01-02", "2015-03-31")


@pytest.fixture
def folders(tmp_path):
    return tmp_path / "raw", tmp_path / "oos"


def test_smooth_continuation_is_ok(folders):
    ins, oos = folders
    history = random_walk(IN_DAYS)
    write_prices(ins, "A", history)
    write_prices(oos, "A", random_walk(OUT_DAYS, start=history.iloc[-1] * 1.005, seed=1))
    row = check_splice(ins, oos).loc["A"]
    assert row["status"] == "OK"
    assert row["gap_bdays"] == 2  # Wed 31-Dec -> Fri 2-Jan
    assert abs(row["z"]) < 6


def test_level_jump_at_the_join_is_flagged(folders):
    ins, oos = folders
    history = random_walk(IN_DAYS)
    write_prices(ins, "A", history)
    write_prices(oos, "A", random_walk(OUT_DAYS, start=history.iloc[-1] * 1.30, seed=1))
    row = check_splice(ins, oos).loc["A"]
    assert row["status"] == "JUMP"
    assert row["join_return"] == pytest.approx(0.30, abs=0.05)


def test_re_adjusted_overlap_is_flagged(folders):
    # the new file repeats the last week of 2014, but every price is 2% higher
    ins, oos = folders
    history = random_walk(IN_DAYS)
    write_prices(ins, "A", history)
    later = random_walk(OUT_DAYS, start=history.iloc[-1], seed=1)
    write_prices(oos, "A", pd.concat([history.iloc[-5:], later]) * 1.02)
    row = check_splice(ins, oos).loc["A"]
    assert row["status"] == "LEVEL_SHIFT"
    assert row["overlap_days"] == 5
    assert row["overlap_max_rel_diff"] == pytest.approx(0.02)


def test_identical_overlap_is_fine(folders):
    ins, oos = folders
    history = random_walk(IN_DAYS)
    write_prices(ins, "A", history)
    later = random_walk(OUT_DAYS, start=history.iloc[-1], seed=1)
    write_prices(oos, "A", pd.concat([history.iloc[-5:], later]))
    assert check_splice(ins, oos).loc["A", "status"] == "OK"


def test_missing_ended_new_gap_and_format(folders):
    ins, oos = folders
    write_prices(ins, "LIVE", random_walk(IN_DAYS))  # no new file -> MISSING
    write_prices(ins, "DEAD", random_walk(IN_DAYS[:200]))  # stopped in 2014 -> ENDED
    write_prices(oos, "FRESH", random_walk(OUT_DAYS))  # only out-of-sample -> NEW
    late = random_walk(IN_DAYS)
    write_prices(ins, "LATE", late)
    write_prices(oos, "LATE", random_walk(OUT_DAYS[30:], start=late.iloc[-1]))  # -> GAP
    write_prices(ins, "ODD", random_walk(IN_DAYS))
    random_walk(OUT_DAYS).rename("Close").to_frame().to_csv(oos / "ODD.csv")  # Close only
    write_prices(ins, "SKIP", random_walk(IN_DAYS))
    write_prices(oos, "SKIP", random_walk(OUT_DAYS) * 5)  # would JUMP, but is dropped

    status = check_splice(ins, oos, dropped_symbols={"SKIP"})["status"]
    assert status.to_dict() == {
        "DEAD": "ENDED",
        "FRESH": "NEW",
        "LATE": "GAP",
        "LIVE": "MISSING",
        "ODD": "BAD_FORMAT",
    }


# The real backtest running across the join, on made-up 2015 prices

FAKE_DAYS = pd.bdate_range("2015-01-02", "2015-06-30")


@pytest.fixture(scope="module")
def fake_oos(tmp_path_factory):
    """Six months of made-up prices that continue each live contract from its last 2014 close."""
    folder = tmp_path_factory.mktemp("oos")
    rng = np.random.default_rng(2015)
    for f in sorted(RAW_DIR.glob("*.csv")):
        hist = pd.read_csv(f, index_col=0, parse_dates=True)
        if hist.index[-1] < pd.Timestamp("2014-12-01"):
            continue  # contract had already stopped trading
        last = hist.iloc[-1]
        vol = hist["Close"].pct_change().iloc[-250:].std()
        close = last["Close"] * np.cumprod(1.0 + rng.normal(0.0, vol, len(FAKE_DAYS)))
        write_prices(
            folder,
            f.stem,
            pd.Series(close, index=FAKE_DAYS),
            volume=last["Volume"],
            oi=last["OpenInterest"],
        )
    return folder


@pytest.fixture(scope="module")
def panels(fake_oos):
    in_sample = Panel(RAW_DIR, ASSET_MAP_PATH, DROPPED_SYMBOLS)
    joined = Panel([RAW_DIR, fake_oos], ASSET_MAP_PATH, DROPPED_SYMBOLS)
    return in_sample, joined


@needs_data
def test_fake_oos_passes_the_join_check(fake_oos):
    status = check_splice(RAW_DIR, fake_oos, DROPPED_SYMBOLS)["status"]
    assert set(status) == {"OK"}


@needs_data
@pytest.mark.parametrize("name", ["TSMOM 12m (baseline)", *USER_STRATEGIES])
def test_backtest_continues_smoothly_across_the_join(panels, name):
    in_sample, joined = panels
    model = FuturesCostModel()
    short = run_backtest(in_sample, STRATEGIES[name](), cost_model=model, exec_lag=2)
    full = run_backtest(joined, STRATEGIES[name](), cost_model=model, exec_lag=2)

    # 1. adding 2015 data changes nothing up to 31-Dec-2014
    pd.testing.assert_frame_equal(
        full.weights.loc[:IN_SAMPLE_END], short.weights, check_freq=False
    )
    pd.testing.assert_series_equal(
        full.net_pnl.loc[:IN_SAMPLE_END], short.net_pnl, check_freq=False
    )

    # 2. the 2014 book is carried into 2015: no flat days, no restarted warm-up
    gross = full.positions.abs().sum(axis=1)
    last_in, first_out = gross.loc[:IN_SAMPLE_END].iloc[-1], gross.loc[FAKE_DAYS[0]]
    assert last_in > 0
    assert 0.5 < first_out / last_in < 2.0
    assert (gross.loc[FAKE_DAYS] > 0).all()

    # 3. the first new day's P&L is an ordinary day, not a jump
    pnl_2014 = full.net_pnl.loc["2014"]
    assert abs(full.net_pnl.loc[FAKE_DAYS[0]]) < 5 * pnl_2014.std()

    # 4. it keeps rebalancing and earning every month
    weight_changes = full.weights.diff().abs().sum(axis=1).loc[FAKE_DAYS]
    months_traded = set(weight_changes[weight_changes > 0].index.to_period("M"))
    assert months_traded == set(FAKE_DAYS.to_period("M"))
    oos_months = monthly_returns(full).loc["2015-01":]
    assert len(oos_months) == 6 and (oos_months != 0).all()
