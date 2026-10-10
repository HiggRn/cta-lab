import numpy as np
import pandas as pd
import pytest

from common import Bar
from indicators.monthly_returns import MonthlyReturn
from strategy import (
    EqualRiskPCM,
    EwmaMonthlyRiskModel,
    MonthlyTsmomAlpha,
    StaticLongAlpha,
    VolTargetPCM,
    exante_vol,
    monthly_from_daily,
)


def month_end_bar(day: str, close: float) -> Bar:
    return Bar(pd.Timestamp(day), close, close, close, close, 0.0, 0.0, is_month_end=True, is_calendar_month_end=True)


def test_monthly_return_uses_month_end_closes():
    indicator = MonthlyReturn(lookback=2)
    for day, close in [("2000-01-31", 100), ("2000-02-29", 110), ("2000-03-31", 99)]:
        indicator.update(month_end_bar(day, close))
    assert indicator.is_ready
    assert list(indicator.returns) == pytest.approx([0.10, -0.10])


def test_monthly_return_ignores_bars_inside_the_month():
    indicator = MonthlyReturn(lookback=1)
    indicator.update(month_end_bar("2000-01-31", 100))
    indicator.update(Bar(pd.Timestamp("2000-02-15"), 1, 1, 1, 500, 0, 0))
    indicator.update(month_end_bar("2000-02-29", 105))
    assert indicator.value == pytest.approx(0.05)


def test_tsmom_signal_and_blend():
    assert MonthlyTsmomAlpha((12,)).direction([0.01] * 11 + [-0.5]) == -1.0
    # 1m down, 3m/6m/12m up -> (-1 + 1 + 1 + 1) / 4
    assert MonthlyTsmomAlpha((1, 3, 6, 12)).direction([0.02] * 11 + [-0.01]) == 0.5
    assert StaticLongAlpha().direction([-0.1] * 12) == 1.0


def test_equal_risk_sizing():
    cov = pd.DataFrame(np.diag([0.2**2, 0.4**2]), index=["a", "b"], columns=["a", "b"])
    weights = EqualRiskPCM(position_vol=0.40, min_assets=2).create_targets({"a": 1.0, "b": -1.0}, cov)
    assert weights == pytest.approx({"a": 0.40 / 0.2 / 2, "b": -0.40 / 0.4 / 2})
    assert EqualRiskPCM(min_assets=3).create_targets({"a": 1.0, "b": -1.0}, cov) == {}


def test_vol_target_hits_target():
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["a", "b"], columns=["a", "b"])
    pcm = VolTargetPCM(EqualRiskPCM(min_assets=1), target_vol=0.11)
    weights = pcm.create_targets({"a": 1.0, "b": 1.0}, cov)
    assert exante_vol(weights, cov) == pytest.approx(0.11)


def test_exante_vol_formula():
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["a", "b"], columns=["a", "b"])
    w = {"a": 0.5, "b": -0.25}
    x = np.array([0.5, -0.25])
    assert exante_vol(w, cov) == pytest.approx(np.sqrt(x @ cov.to_numpy() @ x))


def test_monthly_from_daily_compounds_and_keeps_gaps():
    days = pd.to_datetime(["2000-01-03", "2000-01-04", "2000-02-01"])
    daily = pd.DataFrame({"a": [0.10, 0.10, np.nan]}, index=days)
    monthly = monthly_from_daily(daily)
    assert monthly.loc["2000-01", "a"] == pytest.approx(0.21)
    assert np.isnan(monthly.loc["2000-02", "a"])


def test_ewma_risk_model_needs_enough_history():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2000-01-03", periods=60 * 21)
    daily = pd.DataFrame(rng.normal(0, 0.01, (len(days), 2)), index=days, columns=["a", "b"])
    daily.loc[: days[-200], "b"] = np.nan  # b has under 36 months of data
    cov = EwmaMonthlyRiskModel(min_months=36).covariance(daily)
    assert cov.loc["a", "a"] > 0
    assert np.isnan(cov.loc["b", "b"])
