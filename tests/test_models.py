import numpy as np
import pandas as pd
import pytest

from baselines import (
    ExpandingMonthlyRiskModel,
    MonthlyTsmomAlpha,
    StaticLongAlpha,
    _completed_monthly_returns,
)
from common import (
    AlphaModel,
    Bar,
    DailySchedule,
    EqualRiskPCM,
    FirstAllTradableDayMonthlySchedule,
    HistoryContext,
    InverseVolPCM,
    ModularStrategy,
    PortfolioConstructionModel,
    RiskModel,
    VolTargetPCM,
    WeeklySchedule,
    exante_vol,
)
from indicators.monthly_returns import MonthlyReturn


def bar(day, close=100.0) -> Bar:
    return Bar(pd.Timestamp(day), close, close, close, close, 0.0, 0.0)


def make_context(n_days=10, symbols=("a", "b"), start="2000-01-31", first=None, last=None):
    dates = pd.bdate_range(start, periods=n_days)
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.01, (n_days, len(symbols)))
    closes = 100.0 * np.cumprod(1.0 + returns, axis=0)
    first = np.zeros(len(symbols), dtype=int) if first is None else np.array(first)
    last = np.full(len(symbols), n_days - 1) if last is None else np.array(last)
    return HistoryContext(returns, closes, dates, list(symbols), first, last)


def daily_with_monthly_returns(monthly: pd.DataFrame) -> pd.DataFrame:
    """Daily returns that compound to the given monthly returns (PeriodIndex rows).

    Each month's return lands on its first business day, all other days are 0.
    """
    days = pd.bdate_range(
        monthly.index[0].start_time, monthly.index[-1].end_time.normalize()
    )
    daily = pd.DataFrame(0.0, index=days, columns=monthly.columns)
    first_days = days.to_series().groupby(days.to_period("M")).first()
    for period, row in monthly.iterrows():
        daily.loc[first_days[period]] = row.to_numpy()
    return daily


# Indicators


def test_monthly_return_uses_completed_months_only():
    ind = MonthlyReturn(lookback=2)
    bars = [
        ("2000-01-31", 100.0),  # the first month is partial and never counts
        ("2000-02-15", 105.0),
        ("2000-02-29", 110.0),
        ("2000-03-15", 120.0),  # Feb is now complete
        ("2000-03-31", 99.0),
    ]
    for day, close in bars:
        ind.update(bar(day, close))
    assert list(ind.completed_returns) == pytest.approx([0.10])
    assert not ind.is_ready
    assert ind.returns[-1] == pytest.approx(99.0 / 110.0 - 1.0)  # in-progress month

    ind.update(bar("2000-04-03", 100.0))
    assert ind.just_closed_month
    assert list(ind.completed_returns) == pytest.approx([0.10, 99.0 / 110.0 - 1.0])
    assert ind.is_ready
    assert ind.value == pytest.approx(0.10 + 99.0 / 110.0 - 1.0)


def test_monthly_return_flag_only_on_first_bar_of_new_month():
    ind = MonthlyReturn(lookback=1)
    flags = []
    for day, close in [
        ("2000-01-03", 100.0),
        ("2000-02-01", 101.0),
        ("2000-02-15", 102.0),
        ("2000-03-01", 103.0),
        ("2000-03-15", 104.0),
    ]:
        ind.update(bar(day, close))
        flags.append(ind.just_closed_month)
    assert flags == [False, False, False, True, False]




# Alpha models


def test_tsmom_signal_and_blend():
    assert MonthlyTsmomAlpha((12,)).direction([0.01] * 11 + [-0.5]) == -1.0
    # 1m down, 3m/6m/12m up -> (-1 + 1 + 1 + 1) / 4
    assert MonthlyTsmomAlpha((1, 3, 6, 12)).direction([0.02] * 11 + [-0.01]) == 0.5
    assert StaticLongAlpha().direction([-0.1] * 12) == 1.0


def month_bars(n_months, growth):
    """Two bars per month (1st and 15th) with a constant per-bar growth factor."""
    out, close = [], 100.0
    for k in range(n_months):
        year, month = 2000 + k // 12, k % 12 + 1
        for day in (1, 15):
            close *= growth
            out.append(bar(pd.Timestamp(year, month, day), close))
    return out


@pytest.mark.parametrize("alpha_cls, growth, expected", [
    (MonthlyTsmomAlpha, 1.02, 1.0),
    (MonthlyTsmomAlpha, 0.98, -1.0),
    (StaticLongAlpha, 0.98, 1.0),
])
def test_monthly_alpha_emits_once_the_history_is_complete(alpha_cls, growth, expected):
    # 12 completed months need 14 months of bars: the first month is partial and
    # a month only completes on the first bar of the next one
    alpha = alpha_cls(lookbacks=(12,))
    emitted = []
    for b in month_bars(14, growth):
        signals = alpha.update({"a": b}, None)
        if signals:
            emitted.append((b.date, signals["a"]))
    assert emitted == [(pd.Timestamp("2001-02-01"), expected)]


# Risk models


def test_completed_monthly_returns_compounds_and_drops_the_current_month():
    days = pd.to_datetime(["2000-01-03", "2000-01-04", "2000-02-01", "2000-02-03"])
    daily = pd.DataFrame({"a": [0.10, 0.10, 0.05, 0.02]}, index=days)
    monthly = _completed_monthly_returns(daily)  # last day is the 3rd: Feb is incomplete
    assert list(monthly.index) == [200001]
    assert monthly.loc[200001, "a"] == pytest.approx(0.21)


def test_completed_monthly_returns_keeps_gaps_as_nan():
    days = pd.to_datetime(["2000-01-03", "2000-01-04", "2000-02-01", "2000-02-15"])
    daily = pd.DataFrame(
        {"a": [0.10, 0.10, 0.05, 0.02], "b": [0.10, 0.10, np.nan, np.nan]}, index=days
    )
    monthly = _completed_monthly_returns(daily)  # last day is the 15th: Feb is kept
    assert monthly.loc[200002, "a"] == pytest.approx(1.05 * 1.02 - 1.0)
    assert np.isnan(monthly.loc[200002, "b"])


def test_expanding_monthly_risk_model_matches_sample_covariance():
    rng = np.random.default_rng(1)
    months = pd.period_range("2000-01", periods=42, freq="M")
    monthly = pd.DataFrame(rng.normal(0, 0.04, (42, 2)), index=months, columns=["a", "b"])
    cov = ExpandingMonthlyRiskModel(min_months=36).covariance(
        daily_with_monthly_returns(monthly)
    )
    np.testing.assert_allclose(cov.to_numpy(), (monthly.cov() * 12.0).to_numpy())


def test_monthly_risk_models_need_enough_history():
    rng = np.random.default_rng(0)
    months = pd.period_range("2000-01", periods=60, freq="M")
    monthly = pd.DataFrame(rng.normal(0, 0.04, (60, 2)), index=months, columns=["a", "b"])
    daily = daily_with_monthly_returns(monthly)
    daily.loc[: daily.index[-200], "b"] = np.nan  # b has under 36 months of data
    for model in (ExpandingMonthlyRiskModel(min_months=36),):
        cov = model.covariance(daily)
        assert cov.loc["a", "a"] > 0
        assert np.isnan(cov.loc["b", "b"])
        assert np.isnan(cov.loc["a", "b"])






# Portfolio construction


def test_equal_risk_sizing():
    cov = pd.DataFrame(np.diag([0.2**2, 0.4**2]), index=["a", "b"], columns=["a", "b"])
    weights = EqualRiskPCM(position_vol=0.40, min_assets=2).create_targets(
        {"a": 1.0, "b": -1.0}, cov
    )
    assert weights == pytest.approx({"a": 0.40 / 0.2 / 2, "b": -0.40 / 0.4 / 2})
    assert EqualRiskPCM(min_assets=3).create_targets({"a": 1.0, "b": -1.0}, cov) == {}


def test_equal_risk_skips_flat_signals_and_unknown_vol():
    cov = pd.DataFrame(
        np.diag([0.2**2, 0.2**2, np.nan]), index=list("abc"), columns=list("abc")
    )
    weights = EqualRiskPCM(min_assets=1).create_targets(
        {"a": 1.0, "b": 0.0, "c": 1.0, "d": 1.0}, cov  # b flat, c NaN vol, d not in cov
    )
    assert set(weights) == {"a"}
    assert weights["a"] == pytest.approx(0.40 / 0.2)


def test_vol_target_hits_target():
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["a", "b"], columns=["a", "b"])
    pcm = VolTargetPCM(EqualRiskPCM(min_assets=1), target_vol=0.11)
    weights = pcm.create_targets({"a": 1.0, "b": 1.0}, cov)
    assert exante_vol(weights, cov) == pytest.approx(0.11)
    assert pcm.create_targets({}, cov) == {}


def test_exante_vol_formula():
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["a", "b"], columns=["a", "b"])
    w = {"a": 0.5, "b": -0.25}
    x = np.array([0.5, -0.25])
    assert exante_vol(w, cov) == pytest.approx(np.sqrt(x @ cov.to_numpy() @ x))


def test_inverse_vol_scales_to_target_gross_leverage():
    cov = pd.DataFrame(np.diag([0.1**2, 0.2**2]), index=["a", "b"], columns=["a", "b"])
    weights = InverseVolPCM(target_gross_leverage=1.5).create_targets(
        {"a": 1.0, "b": -1.0}, cov
    )
    assert sum(abs(w) for w in weights.values()) == pytest.approx(1.5)
    assert weights["a"] == pytest.approx(2.0 * -weights["b"])  # half the vol, twice the weight
    assert weights["b"] < 0


# HistoryContext


def test_context_only_exposes_data_up_to_today():
    ctx = make_context(n_days=6)
    ctx._advance_to(3)
    assert len(ctx.get_returns()) == 4
    assert ctx.get_returns().index[-1] == ctx.current_date
    assert len(ctx.get_closes(lookback=2)) == 2
    with pytest.raises(ValueError):
        ctx.get_returns(0)


def test_context_returns_a_copy():
    ctx = make_context(n_days=4)
    ctx._advance_to(2)
    frame = ctx.get_returns()
    frame.iloc[:] = 123.0
    assert not (ctx.get_returns() == 123.0).any().any()


def test_context_prev_date_and_active_universe():
    ctx = make_context(n_days=6, first=[0, 3], last=[5, 4])
    assert ctx.prev_date is None
    assert ctx.active_universe == {"a"}
    ctx._advance_to(3)
    assert ctx.prev_date == ctx.get_returns().index[-2]
    assert ctx.active_universe == {"a", "b"}
    ctx._advance_to(5)
    assert ctx.active_universe == {"a"}  # b is past its last bar


# Rebalance schedules


def run_schedule(schedule, ctx, traded):
    """`traded[i]` is the set of symbols with a bar on day i. Returns rebalance days."""
    fired = []
    for i, symbols in enumerate(traded):
        ctx._advance_to(i)
        bars = {s: bar(ctx.current_date) for s in symbols}
        if schedule.should_rebalance(bars, ctx):
            fired.append(i)
    return fired


def test_monthly_schedule_fires_on_first_day_everything_trades():
    ctx = make_context(n_days=6)  # Mon 2000-01-31 ... Mon 2000-02-07
    both = {"a", "b"}
    assert run_schedule(FirstAllTradableDayMonthlySchedule(), ctx, [both] * 6) == [1]


def test_monthly_schedule_waits_for_missing_symbols_then_gives_up():
    ctx = make_context(n_days=6)
    both, only_a = {"a", "b"}, {"a"}
    traded = [both, only_a, both, both, both, both]
    assert run_schedule(FirstAllTradableDayMonthlySchedule(), ctx, traded) == [2]

    traded = [both, only_a, only_a, only_a, both, both]
    fired = run_schedule(FirstAllTradableDayMonthlySchedule(max_wait_days=2), ctx, traded)
    assert fired == [2]  # second bar of the month, b still missing


def test_monthly_schedule_fires_once_per_month():
    ctx = make_context(n_days=40)
    traded = [{"a", "b"}] * 40
    fired = run_schedule(FirstAllTradableDayMonthlySchedule(), ctx, traded)
    dates = [ctx._dates[i] for i in fired]
    assert dates == [pd.Timestamp("2000-02-01"), pd.Timestamp("2000-03-01")]


def test_daily_and_weekly_schedules():
    ctx = make_context(n_days=8)  # Mon 2000-01-31 ... Wed 2000-02-09
    traded = [{"a"}] * 8
    assert run_schedule(DailySchedule(), ctx, traded) == list(range(8))
    assert run_schedule(WeeklySchedule(), ctx, traded) == [5]  # Mon 2000-02-07
    assert run_schedule(DailySchedule(), ctx, [set()] * 8) == []


# ModularStrategy wiring


class StubAlpha(AlphaModel):
    def __init__(self):
        self.calls = 0

    def update(self, bars, context):
        self.calls += 1
        return {"a": 1.0} if self.calls == 1 else {}  # one signal, then silence


class StubRisk(RiskModel):
    def __init__(self):
        self.lengths = []

    def covariance(self, returns):
        self.lengths.append(len(returns))
        eye = 0.04 * np.eye(len(returns.columns))
        return pd.DataFrame(eye, index=returns.columns, columns=returns.columns)


class StubPCM(PortfolioConstructionModel):
    def __init__(self):
        self.seen = []

    def create_targets(self, signals, cov):
        self.seen.append(dict(signals))
        return {s: 0.1 * x for s, x in signals.items()}


def drive(strategy, ctx, traded):
    out = []
    for i, symbols in enumerate(traded):
        ctx._advance_to(i)
        bars = {s: bar(ctx.current_date) for s in symbols}
        out.append(strategy.on_bar(bars, ctx))
    return out


def test_modular_strategy_runs_alpha_every_bar_and_models_on_rebalance():
    alpha, risk, pcm = StubAlpha(), StubRisk(), StubPCM()
    strategy = ModularStrategy(
        alpha, risk, pcm, schedule=DailySchedule(), risk_lookback=3
    )
    out = drive(strategy, make_context(n_days=5), [{"a", "b"}] * 5)
    assert alpha.calls == 5
    assert risk.lengths == [1, 2, 3, 3, 3]  # risk_lookback bounds the history
    assert all(o == {"a": 0.1} for o in out)
    assert all(seen == {"a": 1.0} for seen in pcm.seen)  # old signal is carried forward


def test_modular_strategy_holds_between_rebalances_and_ignores_empty_days():
    alpha, risk, pcm = StubAlpha(), StubRisk(), StubPCM()
    strategy = ModularStrategy(alpha, risk, pcm)  # monthly by default
    ctx = make_context(n_days=6)
    out = drive(strategy, ctx, [{"a", "b"}] * 3 + [set()] + [{"a", "b"}] * 2)
    assert out[0] is None and out[2] is None  # no rebalance
    assert out[1] == {"a": 0.1}  # first trading day of February
    assert alpha.calls == 5  # the empty day never reaches the alpha
    assert len(risk.lengths) == 1
