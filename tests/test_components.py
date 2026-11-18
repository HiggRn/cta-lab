"""Unit tests for small building blocks: indicators, schedules, PCMs, risk models, costs, metrics."""

import numpy as np
import pandas as pd
import pytest

from analysis import max_drawdown, monthly_metrics
from baselines import ExpandingMonthlyRiskModel, MonthlyTsmomAlpha, StaticLongAlpha
from common import (
    Bar,
    DailySchedule,
    EqualRiskPCM,
    HistoryContext,
    InverseVolPCM,
    SampleCovarianceRiskModel,
    VolTargetPCM,
    WeeklySchedule,
    exante_vol,
)
from engine import CostBreakdown, FuturesCostModel, ZeroCostModel
from indicators import MonthlyReturn, TrendTStat
from strategy import TrancheSchedule, TranchePCM


def bar(date, close):
    return Bar(pd.Timestamp(date), close, close, close, close, 1.0, 1.0)


def context(dates, symbols=("A",), idx=0):
    dates = pd.DatetimeIndex(dates)
    arr = np.full((len(dates), len(symbols)), 0.01)
    ctx = HistoryContext(arr, arr + 1, dates, list(symbols), np.zeros(len(symbols), int), np.full(len(symbols), len(dates) - 1))
    ctx._advance_to(idx)
    return ctx


def cov_of(*vols):
    names = [chr(65 + i) for i in range(len(vols))]
    return pd.DataFrame(np.diag(np.square(vols)), index=names, columns=names)


# Indicators


def test_monthly_return_completes_months():
    ind = MonthlyReturn(lookback=2)
    for d, c in [("2020-01-30", 100), ("2020-01-31", 100), ("2020-02-28", 110), ("2020-03-02", 110), ("2020-03-31", 121)]:
        ind.update(bar(d, c))
    # Jan has no start price, Feb: 100 -> 110, March still open
    assert ind.completed_returns[0] == pytest.approx(0.10)
    assert not ind.is_ready
    ind.update(bar("2020-04-01", 121))
    assert ind.just_closed_month
    assert ind.is_ready
    assert ind.value == pytest.approx(0.20)


def test_trend_tstat_needs_full_window_and_sign_follows_trend():
    up, down = TrendTStat(lookback=5), TrendTStat(lookback=5)
    for i in range(6):
        assert not up.is_ready
        up.update(bar("2020-01-01", 100 * 1.01 ** i * (1 + 0.002 * (-1) ** i)))
        down.update(bar("2020-01-01", 100 * 0.99 ** i * (1 + 0.002 * (-1) ** i)))
    assert up.is_ready and down.is_ready
    assert up.value > 0 > down.value


def test_trend_tstat_flat_prices_is_zero_and_bad_close_ignored():
    ind = TrendTStat(lookback=3)
    for _ in range(4):
        ind.update(bar("2020-01-01", 50.0))
    ind.update(bar("2020-01-01", -1.0))
    assert ind.is_ready and ind.value == 0.0


# HistoryContext


def test_history_context_exposes_only_data_up_to_today():
    ctx = context(pd.bdate_range("2020-01-01", periods=5), idx=2)
    assert ctx.current_date == pd.Timestamp("2020-01-03")
    assert ctx.prev_date == pd.Timestamp("2020-01-02")
    assert len(ctx.get_returns()) == 3
    assert len(ctx.get_closes(2)) == 2
    with pytest.raises(ValueError):
        ctx.get_returns(0)


def test_history_context_first_day_has_no_prev_date():
    assert context(pd.bdate_range("2020-01-01", periods=2)).prev_date is None


# Schedules


def test_daily_schedule_needs_a_bar():
    ctx = context(pd.bdate_range("2020-01-01", periods=2))
    assert DailySchedule().should_rebalance({"A": bar("2020-01-01", 1)}, ctx)
    assert not DailySchedule().should_rebalance({}, ctx)


def test_weekly_schedule_fires_on_first_day_of_week():
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    sched = WeeklySchedule()
    fired = [sched.should_rebalance({"A": bar(d, 1)}, context(dates, idx=i)) for i, d in enumerate(dates)]
    assert fired == [False, False, True, False]


def test_tranche_schedule_fires_on_chosen_cycle_days():
    sched = TrancheSchedule(cycle=4, days=(0, 2))
    ctx = context(pd.bdate_range("2020-01-01", periods=2))
    fired = [sched.should_rebalance({"A": bar("2020-01-01", 1)}, ctx) for _ in range(8)]
    assert fired == [True, False, True, False] * 2
    assert not sched.should_rebalance({}, ctx)


# Alpha, risk and portfolio construction


def test_tsmom_direction_blends_lookbacks():
    assert MonthlyTsmomAlpha((1, 2)).direction([0.05, -0.01]) == 0.0
    assert MonthlyTsmomAlpha((2,)).direction([0.05, -0.01]) == 1.0
    assert StaticLongAlpha((12,)).direction([-0.5]) == 1.0


def test_exante_vol_independent_assets():
    assert exante_vol({"A": 1.0, "B": 1.0}, cov_of(0.2, 0.3)) == pytest.approx(np.hypot(0.2, 0.3))
    assert exante_vol({}, cov_of(0.2)) == 0.0


def test_sample_covariance_is_annualised():
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(0, 0.01, (200, 2)), columns=["A", "B"])
    cov = SampleCovarianceRiskModel().covariance(returns)
    assert cov.loc["A", "A"] == pytest.approx(returns["A"].var() * 252)


def test_expanding_monthly_risk_model_needs_min_months():
    idx = pd.bdate_range("2020-01-01", "2021-12-31")
    returns = pd.DataFrame({"A": 0.001, "B": -0.001}, index=idx) + np.random.default_rng(1).normal(0, 0.01, (len(idx), 2))
    short = ExpandingMonthlyRiskModel(min_months=36).covariance(returns)
    assert short.isna().all().all()
    assert ExpandingMonthlyRiskModel(min_months=12).covariance(returns).notna().all().all()


def test_equal_risk_pcm_sizes_by_vol_and_requires_min_assets():
    cov = cov_of(0.1, 0.2)
    pcm = EqualRiskPCM(position_vol=0.4, min_assets=2)
    out = pcm.create_targets({"A": 1.0, "B": -1.0}, cov)
    assert out == {"A": pytest.approx(2.0), "B": pytest.approx(-1.0)}
    assert pcm.create_targets({"A": 1.0, "B": 0.0}, cov) == {}


def test_inverse_vol_pcm_hits_gross_leverage():
    out = InverseVolPCM(target_gross_leverage=2.0).create_targets({"A": 1.0, "B": -1.0}, cov_of(0.1, 0.2))
    assert sum(abs(w) for w in out.values()) == pytest.approx(2.0)
    assert out["A"] > 0 > out["B"] and abs(out["A"]) == pytest.approx(2 * abs(out["B"]))
    assert InverseVolPCM().create_targets({"A": 0.0}, cov_of(0.1)) == {}


def test_vol_target_pcm_scales_to_target():
    cov = cov_of(0.1, 0.2)
    out = VolTargetPCM(InverseVolPCM(), target_vol=0.11).create_targets({"A": 1.0, "B": 1.0}, cov)
    assert exante_vol(out, cov) == pytest.approx(0.11)
    assert VolTargetPCM(InverseVolPCM()).create_targets({"A": 0.0}, cov) == {}


def test_tranche_pcm_averages_last_books():
    class Fixed:
        def __init__(self):
            self.books = iter([{"A": 4.0}, {"A": 0.0, "B": 4.0}])

        def create_targets(self, signals, cov):
            return next(self.books)

    pcm = TranchePCM(Fixed(), parts=2)
    assert pcm.create_targets({}, None) == {"A": 2.0}
    assert pcm.create_targets({}, None) == {"A": 2.0, "B": 2.0}


# Costs and metrics


def test_futures_cost_model_components():
    model = FuturesCostModel(trading_cost_bps=2, roll_cost_bps=2, rolls_per_year=4, financing_cost_bps=50, free_leverage_threshold=1.0)
    assert model.trading_cost(np.array([0.5, -0.5]), {}, ["A", "B"]) == pytest.approx(2e-4)
    roll, fin = model.holding_costs(np.array([1.5, -1.5]))
    assert roll == pytest.approx(3.0 * 2e-4 * 4 / 252)
    assert fin == pytest.approx(2.0 * 50e-4 / 252)
    assert model.holding_costs(np.array([0.5]))[1] == 0.0


def test_zero_cost_model_and_breakdown():
    zero = ZeroCostModel()
    assert zero.trading_cost(np.array([1.0]), {}, ["A"]) == 0.0
    assert zero.holding_costs(np.array([1.0])) == (0.0, 0.0)
    assert CostBreakdown(trading=0.1, roll=0.2, financing=0.3).total == pytest.approx(0.6)


def test_max_drawdown_and_monthly_metrics():
    assert max_drawdown(pd.Series([0.1, -0.2, 0.05])) == pytest.approx(-0.2)
    assert max_drawdown(pd.Series([0.1, 0.1])) == 0.0
    m = monthly_metrics(pd.Series([0.01, 0.02, -0.01, 0.03]))
    assert m["Avg"] == pytest.approx(0.15) and m["Hit rate"] == 0.75
    assert np.isnan(monthly_metrics(pd.Series([0.01, 0.01]))["SR"])
