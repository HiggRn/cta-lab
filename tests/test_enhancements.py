from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from common import Bar, PortfolioConstructionModel, RiskModel
from indicators import TrendTStat
from strategy import (
    EwmaVolRiskModel,
    NoShortAlpha,
    TrancheSchedule,
    TranchePCM,
    TrendStrengthAlpha,
    VolFloorRiskModel,
)


def cov_from(vols, corr):
    vols = np.asarray(vols, dtype=float)
    names = [chr(ord("a") + i) for i in range(len(vols))]
    return pd.DataFrame(np.asarray(corr) * np.outer(vols, vols), index=names, columns=names)


class FixedRisk(RiskModel):
    def __init__(self, cov):
        self.cov = cov

    def covariance(self, returns):
        return self.cov


class FixedPCM(PortfolioConstructionModel):
    """Returns whatever targets the test sets next."""

    def __init__(self):
        self.next = {}

    def create_targets(self, signals, cov):
        return dict(self.next)






def test_vol_floor_raises_quiet_contracts_and_keeps_correlation():
    cov = cov_from([0.005, 0.20, np.nan], [[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]])
    floored = VolFloorRiskModel(FixedRisk(cov), min_vol=0.03).covariance(None)
    vol = np.sqrt(np.diag(floored))
    assert vol[0] == pytest.approx(0.03)  # raised
    assert vol[1] == pytest.approx(0.20)  # untouched
    assert np.isnan(vol[2])  # unknown stays unknown
    assert floored.loc["a", "b"] / (vol[0] * vol[1]) == pytest.approx(0.5)














def flat_bar(day, close):
    return Bar(pd.Timestamp(day), close, close, close, close, 0.0, 0.0)








def flat_bars(closes, start="2000-01-03"):
    days = pd.bdate_range(start, periods=len(closes))
    return [flat_bar(day, close) for day, close in zip(days, closes)]


def random_closes(seed, n, drift=0.0):
    return 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(drift, 0.01, n)))














def test_no_short_alpha_flattens_equity_shorts_only():
    base = SimpleNamespace(
        update=lambda bars, context: {"a": -1.0, "b": 1.0, "c": -0.5, "d": -1.0}
    )
    alpha = NoShortAlpha(base, classes={"a": "EQ", "b": "EQ", "c": "FI"})
    assert alpha.update({}, None) == {"a": 0.0, "b": 1.0, "c": -0.5, "d": -1.0}


def test_trend_tstat_is_the_t_stat_of_daily_log_returns():
    closes = random_closes(7, 40, drift=0.002)
    ind = TrendTStat(lookback=30)
    for i, bar in enumerate(flat_bars(closes)):
        ind.update(bar)
        assert ind.is_ready == (i >= 30)  # 30 returns need 31 closes
    r = np.diff(np.log(closes[-31:]))
    assert ind.value == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(30))


def test_trend_strength_alpha_scales_by_the_cap_and_clips():
    days = pd.bdate_range("2000-01-03", periods=300)
    closes = {"up": random_closes(8, 300, drift=0.01), "flat": random_closes(9, 300)}
    alpha, tstat = TrendStrengthAlpha(lookback=252, cap=2.0), TrendTStat(252)
    for i, day in enumerate(days):
        bars = {s: flat_bar(day, x[i]) for s, x in closes.items()}
        signals = alpha.update(bars, None)
        tstat.update(bars["flat"])
    assert signals["up"] == 1.0  # t-stat of about 16
    assert signals["flat"] == pytest.approx(np.clip(tstat.value / 2.0, -1.0, 1.0))


def test_ewma_vol_risk_model_swaps_vols_and_keeps_correlations():
    rng = np.random.default_rng(10)
    a = np.concatenate([rng.normal(0, 0.005, 200), rng.normal(0, 0.02, 52)])
    rets = pd.DataFrame({"a": a, "b": 0.5 * a + rng.normal(0, 0.01, 252)})
    cov = EwmaVolRiskModel(com=60).covariance(rets)
    vol = np.sqrt(np.diag(cov))
    expected = rets.ewm(com=60).std().iloc[-1] * np.sqrt(252)
    assert vol == pytest.approx(expected.to_numpy())
    assert vol[0] > rets["a"].std() * np.sqrt(252)  # reacts to the rough end
    corr = cov.loc["a", "b"] / (vol[0] * vol[1])
    assert corr == pytest.approx(rets["a"].corr(rets["b"]))


def test_tranche_schedule_fires_on_the_same_days_of_every_cycle():
    schedule = TrancheSchedule(cycle=21, days=(0, 5, 10, 15))
    fired = [i for i in range(63) if schedule.should_rebalance({"a": None}, None)]
    assert fired == [0, 5, 10, 15, 21, 26, 31, 36, 42, 47, 52, 57]
    assert not schedule.should_rebalance({}, None)  # a day with no bars is not counted
    assert schedule.should_rebalance({"a": None}, None)  # day 63 starts a new cycle


def test_tranche_pcm_averages_the_last_targets():
    base = FixedPCM()
    pcm = TranchePCM(base, parts=4)
    outs = []
    for t in [{"a": 1.0}, {"a": 1.0, "b": -1.0}, {"a": 0.0}, {"a": 2.0}, {"b": 1.0}]:
        base.next = t
        outs.append(pcm.create_targets({}, None))
    assert outs[0] == pytest.approx({"a": 0.25})  # one tranche built, three in cash
    assert outs[3] == pytest.approx({"a": 1.0, "b": -0.25})
    assert outs[4] == pytest.approx({"a": 0.75, "b": 0.0})  # the first one rebuilt


