import numpy as np
import pandas as pd
import pytest

from common import PortfolioConstructionModel, RiskModel
from strategy import HrpPCM, LeverageCapPCM, TradeBufferPCM, VolFloorRiskModel


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


# Vol floor


def test_vol_floor_raises_quiet_contracts_and_keeps_correlation():
    cov = cov_from([0.005, 0.20, np.nan], [[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]])
    floored = VolFloorRiskModel(FixedRisk(cov), min_vol=0.03).covariance(None)
    vol = np.sqrt(np.diag(floored))
    assert vol[0] == pytest.approx(0.03)  # raised
    assert vol[1] == pytest.approx(0.20)  # untouched
    assert np.isnan(vol[2])  # unknown stays unknown
    assert floored.loc["a", "b"] / (vol[0] * vol[1]) == pytest.approx(0.5)


# Leverage caps


def test_caps_clip_each_weight_then_total():
    base = FixedPCM()
    pcm = LeverageCapPCM(base, max_weight=1.0, max_gross=2.0)
    base.next = {"a": 3.0, "b": -0.5}
    assert pcm.create_targets({}, None) == pytest.approx({"a": 1.0, "b": -0.5})
    base.next = {"a": 1.5, "b": -1.5, "c": 1.0}  # clipped to 1, -1, 1 -> gross 3 -> scaled to 2
    assert pcm.create_targets({}, None) == pytest.approx({"a": 2 / 3, "b": -2 / 3, "c": 2 / 3})


# Trade buffer


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


# HRP


def test_hrp_weights_sum_to_one_and_follow_signal_signs():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 6))
    cov = pd.DataFrame(np.cov(x.T), index=list("abcdef"), columns=list("abcdef"))
    signals = {"a": 1.0, "b": -1.0, "c": 1.0, "d": -1.0, "e": 1.0, "f": 0.5}
    w = HrpPCM(min_assets=3).create_targets(signals, cov)
    unit = {s: v / signals[s] for s, v in w.items()}
    assert sum(unit.values()) == pytest.approx(1.0)
    assert all(np.sign(w[s]) == np.sign(signals[s]) for s in w)


def test_hrp_equal_for_identical_uncorrelated_assets():
    cov = cov_from([0.1] * 4, np.eye(4))
    w = HrpPCM(min_assets=2).create_targets({s: 1.0 for s in "abcd"}, cov)
    assert w == pytest.approx({s: 0.25 for s in "abcd"})


def test_hrp_gives_less_to_riskier_and_to_crowded_assets():
    # a and b are near-copies; c is on its own with the same vol; d is twice as volatile
    corr = np.eye(4)
    corr[0, 1] = corr[1, 0] = 0.95
    w = HrpPCM(min_assets=2).create_targets({s: 1.0 for s in "abcd"}, cov_from([0.1, 0.1, 0.1, 0.2], corr))
    assert w["a"] + w["b"] < 0.5  # the crowded pair shares less than half
    assert w["d"] < w["c"]


def test_hrp_treats_a_short_as_its_own_asset():
    # a and b move together; long a + short b is a hedge, so their signed correlation is negative
    corr = np.eye(3)
    corr[0, 1] = corr[1, 0] = 0.9
    cov = cov_from([0.1, 0.1, 0.1], corr)
    long_both = HrpPCM(min_assets=2).create_targets({"a": 1, "b": 1, "c": 1}, cov)
    hedged = HrpPCM(min_assets=2).create_targets({"a": 1, "b": -1, "c": 1}, cov)
    assert hedged["a"] + abs(hedged["b"]) > long_both["a"] + long_both["b"]


def test_hrp_skips_flat_unknown_and_too_few():
    cov = cov_from([0.1, 0.1, np.nan], np.eye(3))
    w = HrpPCM(min_assets=1).create_targets({"a": 1.0, "b": 0.0, "c": 1.0, "z": 1.0}, cov)
    assert set(w) == {"a"} and w["a"] == pytest.approx(1.0)
    assert HrpPCM(min_assets=10).create_targets({"a": 1.0}, cov) == {}


# Round 2: risk caps, asset-class budgets, continuous signal, strategy mix

from common import Strategy  # noqa: E402
from strategy import (  # noqa: E402
    AssetClassBudgetPCM,
    ContinuousTsmomAlpha,
    RiskCapPCM,
    SleeveMixStrategy,
)


def risk_shares(w: dict, cov: pd.DataFrame) -> dict:
    names = list(w)
    x = np.array([w[s] for s in names])
    c = cov.loc[names, names].to_numpy()
    return dict(zip(names, x * (c @ x) / (x @ c @ x)))


def test_risk_cap_limits_each_share_and_keeps_total_vol():
    vols = [0.40] + [0.10] * 19  # one contract dominates risk; 20 contracts so the 10% cap applies
    cov = cov_from(vols, np.eye(20))
    base = FixedPCM()
    base.next = {s: 1.0 for s in cov.index}
    capped = RiskCapPCM(base, max_share=0.10).create_targets({}, cov)
    shares = risk_shares(capped, cov)
    assert max(shares.values()) <= 0.10 + 1e-6
    before = np.sqrt(len(vols) - 1 + 16) * 0.10  # vol of the uncapped book
    x = np.array(list(capped.values()))
    assert np.sqrt(x @ cov.to_numpy() @ x) == pytest.approx(before)
    assert np.sign(capped["a"]) == 1.0


def test_risk_cap_loosens_when_too_few_contracts():
    cov = cov_from([0.4, 0.1, 0.1], np.eye(3))
    base = FixedPCM()
    base.next = {"a": 1.0, "b": 1.0, "c": 1.0}
    shares = risk_shares(RiskCapPCM(base, max_share=0.10).create_targets({}, cov), cov)
    assert max(shares.values()) <= 0.5 + 1e-6  # cap becomes 1.5 / 3


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


def test_continuous_signal_scales_with_trend_strength():
    alpha = ContinuousTsmomAlpha((12,))
    noisy = [0.04, -0.05, 0.03, -0.02, 0.05, -0.04, 0.01, -0.03, 0.02, 0.0, -0.01, 0.02]
    strong = [0.02] * 11 + [0.03]
    assert alpha.direction(strong) == 1.0  # t-stat far above 1: capped
    assert alpha.direction([-x for x in strong]) == -1.0
    weak = alpha.direction(noisy)
    assert -1.0 < weak < 1.0
    r = np.array(noisy)
    assert weak == pytest.approx(np.clip(r.mean() / (r.std(ddof=1) / np.sqrt(12)), -1, 1))


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
