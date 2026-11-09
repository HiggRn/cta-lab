import numpy as np
import pandas as pd
import pytest

from common import PortfolioConstructionModel, RiskModel, Strategy
from strategy import AssetClassBudgetPCM, SleeveMixStrategy, TradeBufferPCM, VolFloorRiskModel


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
