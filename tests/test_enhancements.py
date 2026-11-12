from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from common import Bar, InverseVolPCM, PortfolioConstructionModel, RiskModel, Strategy
from indicators import EwmaVolatility, TrendProfile, YangZhangVolatility
from strategy import (
    AssetClassBudgetPCM,
    CorrelationAdjustedPCM,
    LearnedProfileAlpha,
    PositionCapPCM,
    SleeveMixStrategy,
    TradeBufferPCM,
    TrendProfileAlpha,
    VolFloorRiskModel,
    YangZhangRiskModel,
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


def test_class_budget_equalises_sleeve_vols():
    cov = cov_from([0.2, 0.2, 0.2, 0.05], np.eye(4))
    classes = {"a": "COMM", "b": "COMM", "c": "COMM", "d": "FI"}
    base = FixedPCM()
    base.next = {s: 1.0 for s in "abcd"}
    w = AssetClassBudgetPCM(base, classes).create_targets({}, cov)
    comm = np.sqrt(sum((w[s] * 0.2) ** 2 for s in "abc"))
    fi = abs(w["d"]) * 0.05
    assert comm == pytest.approx(fi)
    assert w["a"] == w["b"] == w["c"]  # scaling within a class keeps its shape


def test_class_budget_keeps_signs_and_skips_empty_book():
    cov = cov_from([0.2, 0.1], np.eye(2))
    base = FixedPCM()
    base.next = {"a": -1.0, "b": 1.0}
    w = AssetClassBudgetPCM(base, {"a": "EQ", "b": "FX"}).create_targets({}, cov)
    assert w["a"] < 0 < w["b"]
    base.next = {}
    assert AssetClassBudgetPCM(base, {}).create_targets({}, cov) == {}


def test_vol_floor_raises_quiet_contracts_and_keeps_correlation():
    cov = cov_from([0.005, 0.20, np.nan], [[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]])
    floored = VolFloorRiskModel(FixedRisk(cov), min_vol=0.03).covariance(None)
    vol = np.sqrt(np.diag(floored))
    assert vol[0] == pytest.approx(0.03)  # raised
    assert vol[1] == pytest.approx(0.20)  # untouched
    assert np.isnan(vol[2])  # unknown stays unknown
    assert floored.loc["a", "b"] / (vol[0] * vol[1]) == pytest.approx(0.5)


def test_buffer_holds_inside_the_band_and_trades_to_its_edge():
    base = FixedPCM()
    pcm = TradeBufferPCM(base, buffer=0.20)
    base.next = {"a": 1.0}
    assert pcm.create_targets({}, None) == {"a": pytest.approx(0.8)}  # from flat: to the near edge
    base.next = {"a": 0.9}
    assert pcm.create_targets({}, None) == {"a": pytest.approx(0.8)}  # inside 0.72..1.08: hold
    base.next = {"a": 0.5}
    assert pcm.create_targets({}, None) == {"a": pytest.approx(0.6)}  # outside 0.4..0.6: to edge
    base.next = {"a": -0.5}
    assert pcm.create_targets({}, None) == {"a": pytest.approx(-0.4)}  # sign flip always trades
    base.next = {}
    assert pcm.create_targets({}, None) == {}  # exit is complete


class Scripted(Strategy):
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def on_bar(self, bars, context):
        return self.outputs.pop(0)


def test_sleeve_mix_holds_each_sleeve_and_rebalances_when_any_changes():
    monthly = Scripted([{"a": 1.0}, None, None, {"a": 0.0, "b": 1.0}])
    daily = Scripted([{"a": 0.5}, {"a": 0.4}, None, None])
    mix = SleeveMixStrategy([monthly, daily], [0.5, 0.5])
    outs = [mix.on_bar({"a": None}, None) for _ in range(4)]
    assert outs[0] == pytest.approx({"a": 0.75})
    assert outs[1] == pytest.approx({"a": 0.7})  # monthly sleeve held
    assert outs[2] is None  # nobody changed
    assert outs[3] == pytest.approx({"a": 0.2, "b": 0.5})


def test_corr_adjusted_matches_inverse_vol_when_uncorrelated():
    cov = cov_from([0.1, 0.2, 0.4], np.eye(3))
    signals = {"a": 1.0, "b": -1.0, "c": 1.0}
    w = CorrelationAdjustedPCM(1.5).create_targets(signals, cov)
    assert w == pytest.approx(InverseVolPCM(1.5).create_targets(signals, cov))
    assert sum(abs(x) for x in w.values()) == pytest.approx(1.5)


def test_corr_adjusted_cuts_crowded_contracts_only():
    corr = [[1, 0.8, 0], [0.8, 1, 0], [0, 0, 1]]
    w = CorrelationAdjustedPCM(1.0).create_targets(
        {"a": 1.0, "b": 1.0, "c": -1.0}, cov_from([0.2, 0.2, 0.2], corr)
    )
    assert w["a"] == pytest.approx(w["b"])
    assert w["a"] / abs(w["c"]) == pytest.approx(1 / 1.8)  # crowding 1.8 against 1
    assert w["c"] < 0  # signs kept
    # a negative correlation is not rewarded with extra weight
    hedge = [[1, -0.5], [-0.5, 1]]
    w = CorrelationAdjustedPCM(1.0).create_targets(
        {"a": 1.0, "b": 1.0}, cov_from([0.2, 0.2], hedge)
    )
    assert w == pytest.approx({"a": 0.5, "b": 0.5})


def test_corr_adjusted_skips_flat_and_unknown_symbols():
    cov = cov_from([0.2, 0.2, np.nan], np.eye(3))
    pcm = CorrelationAdjustedPCM(1.0)
    assert pcm.create_targets({"a": 1.0, "b": 0.0, "c": 1.0}, cov) == {"a": pytest.approx(1.0)}
    assert pcm.create_targets({"b": 0.0}, cov) == {}


def flat_bar(day, close):
    return Bar(pd.Timestamp(day), close, close, close, close, 0.0, 0.0)


def test_yang_zhang_equals_close_to_close_vol_without_intraday_range():
    days = pd.bdate_range("2000-01-03", periods=31)
    closes = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, len(days))))
    ind = YangZhangVolatility(lookback=60, min_periods=20)
    for i, (day, close) in enumerate(zip(days, closes)):
        ind.update(flat_bar(day, close))
        assert ind.is_ready == (i >= 20)  # 20 returns need 21 bars
    expected = np.std(np.diff(np.log(closes)), ddof=1) * np.sqrt(252)
    assert ind.value == pytest.approx(expected)


def test_yang_zhang_counts_intraday_range_and_ignores_bad_bars():
    days = pd.bdate_range("2000-01-03", periods=30)
    quiet, ranged = YangZhangVolatility(), YangZhangVolatility()
    for day in days:  # same closes; one series swings 2% around them intraday
        quiet.update(flat_bar(day, 100.0))
        ranged.update(Bar(day, 100.0, 102.0, 98.0, 100.0, 0.0, 0.0))
    assert not quiet.is_ready  # zero vol is never reported as ready
    assert ranged.is_ready and ranged.value > 0.10
    before = ranged.value
    ranged.update(flat_bar("2000-03-01", np.nan))
    assert ranged.value == before


def test_yang_zhang_risk_model_swaps_vols_and_keeps_correlations():
    days = pd.bdate_range("2000-01-03", periods=40)
    rng = np.random.default_rng(1)
    common = rng.normal(0, 0.01, len(days))
    rets = pd.DataFrame(
        {"a": common + rng.normal(0, 0.005, len(days)), "b": common, "c": common},
        index=days,
    )
    closes = 100 * np.exp(rets.cumsum())
    model = YangZhangRiskModel(lookback=60, min_periods=20)
    for day in days:  # "a" and "b" have wide intraday ranges, "c" has no bars at all
        model.update(
            {s: Bar(day, closes.at[day, s], closes.at[day, s] * 1.03,
                    closes.at[day, s] * 0.97, closes.at[day, s], 0.0, 0.0) for s in "ab"}
        )
    sample = rets.cov() * 252
    cov = model.covariance(rets)
    vol = np.sqrt(np.diag(cov))
    assert vol[0] == pytest.approx(model._yz["a"].value)
    assert vol[0] > np.sqrt(sample.loc["a", "a"])  # the range raises the estimate
    assert vol[2] == pytest.approx(np.sqrt(sample.loc["c", "c"]))  # no estimate: sample vol
    assert cov.loc["a", "b"] / (vol[0] * vol[1]) == pytest.approx(rets["a"].corr(rets["b"]))


def flat_bars(closes, start="2000-01-03"):
    days = pd.bdate_range(start, periods=len(closes))
    return [flat_bar(day, close) for day, close in zip(days, closes)]


def random_closes(seed, n, drift=0.0):
    return 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(drift, 0.01, n)))


def test_ewma_vol_matches_pandas():
    closes = random_closes(2, 80)
    expected = pd.Series(np.log(closes)).diff().ewm(span=60, min_periods=20).std()
    ind = EwmaVolatility(span=60, min_periods=20)
    for i, bar in enumerate(flat_bars(closes)):
        ind.update(bar)
        assert ind.is_ready == (i >= 20)  # 20 returns need 21 bars
        if ind.is_ready:
            assert ind.value == pytest.approx(expected[i])


def test_trend_profile_scores_each_window_and_fades_the_oldest():
    closes = random_closes(3, 60)
    ind = TrendProfile(follow=((0, 5), (5, 20)), fade=((20, 30),), vol_span=10, cap=100.0)
    for i, bar in enumerate(flat_bars(closes)):
        ind.update(bar)
        assert ind.is_ready == (i >= 30)  # the oldest window needs 31 closes
    logs = np.log(closes)
    vol = pd.Series(logs).diff().ewm(span=10, min_periods=20).std().iloc[-1]
    windows = [(0, 5), (5, 20), (20, 30)]
    z = [(logs[-1 - a] - logs[-1 - b]) / (vol * np.sqrt(b - a)) for a, b in windows]
    assert ind.scores() == pytest.approx(z)
    assert ind.value == pytest.approx((z[0] + z[1] - z[2]) / 3)


def test_trend_profile_caps_each_window():
    ind = TrendProfile(follow=((0, 5), (5, 20)), fade=((20, 30),), vol_span=10, cap=0.1)
    for bar in flat_bars(random_closes(3, 60)):
        ind.update(bar)
    assert max(abs(z) for z in ind.scores()) == pytest.approx(0.1)
    assert abs(ind.value) <= 0.1


def test_trend_profile_alpha_leans_long_only_where_told():
    classes = {"a": "EQ", "b": "COMM"}
    plain = TrendProfileAlpha(classes=classes)
    leaning = TrendProfileAlpha(premium={"EQ": 0.25}, classes=classes)
    first = None
    for i, bar in enumerate(flat_bars(random_closes(4, 330))):
        bars = {"a": bar, "b": bar, "c": bar}
        base, lean = plain.update(bars, None), leaning.update(bars, None)
        if base and first is None:
            first = i
    assert first == 315  # 316 closes cover the 15 months
    assert lean["a"] == pytest.approx(base["a"] + 0.25)
    assert lean["b"] == pytest.approx(base["b"])  # no lean for commodities
    assert lean["c"] == pytest.approx(base["c"])  # nor for a symbol the map does not know


def test_learned_profile_waits_for_its_burn_in_then_finds_the_drifting_class():
    days = pd.bdate_range("2000-01-03", periods=2200)
    closes = {"a": random_closes(5, len(days), drift=0.001), "b": random_closes(6, len(days))}
    alpha = LearnedProfileAlpha(classes={"a": "UP", "b": "FLAT"}, burn_in_years=3)
    first = None
    for i, day in enumerate(days):
        bars = {s: flat_bar(day, x[i]) for s, x in closes.items()}
        if alpha.update(bars, SimpleNamespace(current_date=day)) and first is None:
            first = day
    # scores start in 2001, so the third January with data behind it is 2004
    assert (first.year, first.month) == (2004, 1)
    lean = dict(zip(alpha.leans, alpha.weights[-2:]))
    assert lean["UP"] > 0.05 > lean["FLAT"]  # the drift is 0.1 daily standard deviations


def test_position_cap_clips_both_sides():
    base = FixedPCM()
    base.next = {"a": 2.5, "b": -1.7, "c": 0.4}
    capped = PositionCapPCM(base, cap=1.0).create_targets({}, None)
    assert capped == {"a": 1.0, "b": -1.0, "c": 0.4}
