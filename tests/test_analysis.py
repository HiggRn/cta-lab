from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from analysis import (
    correlation_structure,
    lag_profile,
    max_drawdown,
    monthly_metrics,
    monthly_returns,
    ols,
    plot_strategies,
    regime_returns,
    risk_model_bias,
    sg_trend_comparison,
    sharpe_diff_test,
    smile_regression,
    strategy_summary_table,
    to_quarterly,
)
from common import RiskModel
from engine import BacktestResult


def make_result(pnl: pd.Series, position, turnover=None, trading_costs=None):
    zeros = pd.Series(0.0, index=pnl.index)
    positions = pd.DataFrame({"a": position}, index=pnl.index)
    return BacktestResult(
        weights=positions,
        positions=positions,
        gross_pnl=pnl,
        trading_costs=zeros if trading_costs is None else trading_costs,
        roll_costs=zeros,
        financing_costs=zeros,
        turnover=zeros if turnover is None else turnover,
    )


# Monthly returns and statistics


def test_monthly_returns_compound_and_skip_the_first_trade_month():
    days = pd.bdate_range("2000-01-03", "2000-04-28")
    pnl = pd.Series(0.001, index=days)
    position = pd.Series(np.where(days >= "2000-01-10", 1.0, 0.0), index=days)
    trading_costs = pd.Series(0.0, index=days)
    trading_costs[pd.Timestamp("2000-03-01")] = 0.01
    res = make_result(pnl - trading_costs, position, trading_costs=trading_costs)
    res.gross_pnl = pnl  # gross ignores the cost

    net = monthly_returns(res)
    gross = monthly_returns(res, gross=True)
    assert list(net.index) == list(pd.period_range("2000-02", "2000-04", freq="M"))
    n_feb = int((days.month == 2).sum())
    assert net["2000-02"] == pytest.approx(1.001**n_feb - 1.0)
    assert gross["2000-03"] > net["2000-03"]


def test_monthly_metrics():
    r = pd.Series([0.01, 0.03] * 12)
    m = monthly_metrics(r)
    assert m["Avg"] == pytest.approx(0.02 * 12)
    assert m["Std"] == pytest.approx(r.std() * np.sqrt(12))
    assert m["SR"] == pytest.approx(0.02 / r.std() * np.sqrt(12))
    assert m["Hit rate"] == 1.0
    assert m["MaxDD"] == pytest.approx(0.0)
    assert np.isnan(m["Calmar"])  # no drawdown, no ratio
    assert np.isnan(monthly_metrics(pd.Series([0.01] * 5))["SR"])  # zero variance


def test_max_drawdown():
    assert max_drawdown(pd.Series([0.1, -0.5, 0.0])) == pytest.approx(-0.5)


def test_calmar_is_annual_return_over_max_drawdown():
    r = pd.Series([0.05, -0.10, 0.02, 0.03] * 6)
    m = monthly_metrics(r)
    assert m["Calmar"] == pytest.approx(r.mean() * 12 / -max_drawdown(r))
    assert m["Calmar"] == pytest.approx(m["Avg"] / abs(m["MaxDD"]))


def test_sharpe_diff_test():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0.02, 0.03, 600))
    b = pd.Series(rng.normal(0.0, 0.03, 600))
    assert sharpe_diff_test(a, b)["p"] < 0.01
    assert sharpe_diff_test(a, b)["dSR"] > 0
    same = sharpe_diff_test(a, a)
    assert same["dSR"] == 0.0 and np.isnan(same["p"])


def test_summary_table_reports_executed_turnover_and_costs():
    days = pd.bdate_range(
        "2000-01-03", "2001-12-31"
    )  # 23 monthly rows after the first trade month
    rng = np.random.default_rng(1)
    trade_days = days[[100, 200, 300, 400]]
    turnover = pd.Series(0.0, index=days)
    turnover[trade_days] = 1.0
    trading_costs = turnover * 1e-4

    def result(seed):
        pnl = pd.Series(
            np.random.default_rng(seed).normal(0.0005, 0.005, len(days)), index=days
        )
        return make_result(pnl, 1.0, turnover=turnover, trading_costs=trading_costs)

    table = strategy_summary_table({"base": result(1), "other": result(2)}, "base")
    years = 23 / 12
    assert table.loc["base", "Turnover/yr"] == pytest.approx(4.0 / years)
    assert table.loc["base", "TradeCost(bps/yr)"] == pytest.approx(4.0 / years)
    assert table.loc["base", "Corr(base)"] == pytest.approx(1.0)
    assert table.loc["other", "RollCost(bps/yr)"] == 0.0


# Regressions


def test_ols_recovers_alpha_and_beta():
    rng = np.random.default_rng(0)
    months = pd.period_range("1990-01", periods=240, freq="M")
    x = pd.Series(rng.normal(0, 0.04, 240), index=months)
    y = 0.004 + 1.5 * x + rng.normal(0, 0.002, 240)
    fit = ols(y, x.to_frame("x"))
    assert fit["alpha"] == pytest.approx(0.048, abs=0.005)  # annualised
    assert fit["beta_x"] == pytest.approx(1.5, abs=0.05)
    assert fit["t_alpha"] > 3 and fit["n"] == 240
    y.iloc[0] = np.nan
    assert ols(y, x.to_frame("x"))["n"] == 239


def test_to_quarterly_keeps_complete_quarters_only():
    months = pd.period_range("2000-01", periods=7, freq="M")  # Jul-2000 is alone in Q3
    q = to_quarterly(pd.Series([0.01] * 7, index=months))
    assert len(q) == 2
    assert q.tolist() == pytest.approx([0.03, 0.03])


def test_smile_regression_finds_convexity():
    rng = np.random.default_rng(0)
    months = pd.period_range("1990-01", periods=240, freq="M")
    market = pd.Series(rng.normal(0, 0.04, 240), index=months)
    quarterly = to_quarterly(market)
    per_month = (2.0 * quarterly**2 / 3.0).reindex(months.asfreq("Q")).to_numpy()
    strategy = pd.Series(per_month + rng.normal(0, 0.0005, 240), index=months)
    fit = smile_regression(strategy, market)
    assert fit["beta_Mkt2"] == pytest.approx(2.0, abs=0.2)
    assert fit["beta_Mkt"] == pytest.approx(0.0, abs=0.2)


def test_regime_returns_buckets_by_market_month():
    months = pd.period_range("2000-01", periods=10, freq="M")
    market = pd.Series(np.linspace(-0.05, 0.05, 10), index=months)
    table = regime_returns({"s": market * 2.0}, market)
    assert table.loc["Bear", "s"] == pytest.approx(2.0 * market.iloc[:2].mean() * 12)
    assert table.loc["Normal", "s"] == pytest.approx(2.0 * market.iloc[2:8].mean() * 12)
    assert table.loc["Bull", "s"] == pytest.approx(2.0 * market.iloc[8:].mean() * 12)


def test_sg_trend_comparison_regresses_strategy_on_sg(tmp_path):
    pytest.importorskip("openpyxl")
    rng = np.random.default_rng(0)
    months = pd.period_range("1990-01", periods=120, freq="M")
    sg_excess = pd.Series(rng.normal(0.004, 0.03, 120), index=months)
    strategy = 0.004 + 1.5 * sg_excess + rng.normal(0, 0.002, 120)
    sg_file = pd.DataFrame(
        {"SG Trend": sg_excess.to_numpy() + 0.002, "RF": 0.002},
        index=months.to_timestamp(),
    )
    path = tmp_path / "sg.xlsx"
    sg_file.to_excel(path)

    out = sg_trend_comparison(strategy, path)
    assert out["beta"] == pytest.approx(1.5, abs=0.05)  # strategy per unit of SG
    assert out["alpha"] == pytest.approx(0.048, abs=0.01)  # strategy alpha over SG
    assert out["corr"] > 0.95


# Correlation structure


def test_correlation_structure_flips_with_short_positions(make_panel):
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2000-01-03", periods=780)

    def prices(common):
        ret = common + rng.normal(0, 0.002, len(days))
        return pd.Series(100.0 * np.cumprod(1.0 + ret), index=days)

    fx, eq = rng.normal(0, 0.01, len(days)), rng.normal(0, 0.01, len(days))
    closes = {"f1": prices(fx), "f2": prices(fx), "e1": prices(eq), "e2": prices(eq)}
    classes = {"f1": "FX", "f2": "FX", "e1": "EQ", "e2": "EQ"}
    panel = make_panel(closes, asset_classes=classes)

    positions = pd.DataFrame(1.0, index=panel.all_dates, columns=panel.symbols)
    positions["f2"] = -1.0
    table = correlation_structure(panel, SimpleNamespace(positions=positions))
    assert table.loc["Passive Buy & Hold", "Within FX"] > 0.5
    assert table.loc["Strategy Positions", "Within FX"] < -0.5  # one leg is short
    assert table.loc["Strategy Positions", "Within EQ"] == pytest.approx(
        table.loc["Passive Buy & Hold", "Within EQ"]
    )


# Risk model bias


class FlatVol(RiskModel):
    """20% annual vol for the only asset, whatever the history."""

    def covariance(self, returns):
        return pd.DataFrame([[0.04]], index=["a"], columns=["a"])


def test_risk_model_bias_uses_the_weights_that_earn_the_month():
    days = pd.bdate_range("2000-01-03", "2001-12-31")
    month = days.to_period("M")
    weight = pd.Series(
        np.where(days < "2001-01-01", 1.0, 4.0), index=days
    )  # re-sized on 1 Jan 2001
    weights = pd.DataFrame({"a": weight})

    # each month's return is exactly +/- 1 predicted monthly stdev of the weights held that month
    pnl = pd.Series(0.0, index=days)
    first_days = days.to_series().groupby(month).first()
    for k, (period, first_day) in enumerate(first_days.items()):
        sign = 1.0 if k % 2 == 0 else -1.0
        pnl[first_day] = sign * abs(weight[first_day]) * 0.20 / np.sqrt(12.0)

    res = SimpleNamespace(
        weights=weights, positions=weights, gross_pnl=pnl, net_pnl=pnl
    )
    table = risk_model_bias(
        pd.DataFrame(0.0, index=days, columns=["a"]),
        {"s": res},
        {"flat": (FlatVol(), False, None)},
        window=2,
    )
    # with the weights of the previous month the Jan-2001 point would be off by 4x
    assert table.loc[("s", "flat"), "Bias stat"] == pytest.approx(1.0, abs=0.06)


def test_lag_profile_sign_follows_the_autocorrelation(make_panel):
    days = pd.bdate_range("2000-01-03", periods=1500)
    rng = np.random.default_rng(6)

    def ar1(phi):
        r = np.zeros(len(days))
        for i, shock in enumerate(rng.normal(0, 0.01, len(days))):
            r[i] = phi * r[i - 1] + shock
        return pd.Series(100 * np.exp(np.cumsum(r)), index=days)

    panel = make_panel(
        {"trend": ar1(0.3), "revert": ar1(-0.3)}, {"trend": "FX", "revert": "EQ"}
    )
    out = lag_profile(panel, [(0, 1)], lag=1)
    assert list(out.columns) == ["All", "EQ", "FX"]
    assert out.loc["0-1", "FX"] > 1.0  # yesterday's move continues
    assert out.loc["0-1", "EQ"] < -1.0  # yesterday's move reverses


def test_correlation_structure_slices_by_start_and_end(make_panel):
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2000-01-03", periods=780)
    common = rng.normal(0, 0.01, len(days))
    own = rng.normal(0, 0.01, (len(days), 2))
    first_half = (np.arange(len(days)) < 390)[:, None]
    fx = np.where(
        first_half, common[:, None] + 0.1 * own, own
    )  # one shared driver, first half only
    eq = rng.normal(0, 0.01, (len(days), 2))

    def prices(ret):
        return pd.Series(100.0 * np.cumprod(1.0 + ret), index=days)

    closes = {
        "f1": prices(fx[:, 0]),
        "f2": prices(fx[:, 1]),
        "e1": prices(eq[:, 0]),
        "e2": prices(eq[:, 1]),
    }
    classes = {"f1": "FX", "f2": "FX", "e1": "EQ", "e2": "EQ"}
    panel = make_panel(closes, asset_classes=classes)
    res = SimpleNamespace(
        positions=pd.DataFrame(1.0, index=panel.all_dates, columns=panel.symbols)
    )

    early = correlation_structure(panel, res, end=str(days[389].date()))
    late = correlation_structure(panel, res, start=str(days[390].date()))
    fx_early = early.loc["Passive Buy & Hold", "Within FX"]
    fx_late = late.loc["Passive Buy & Hold", "Within FX"]
    assert fx_early > fx_late + 0.5


def test_plot_strategies_saves_figure_and_data_without_showing(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    monkeypatch.setattr(
        plt, "show", lambda *a, **k: pytest.fail("plt.show() must not be called")
    )
    days = pd.bdate_range("2000-01-03", periods=60)
    pnl = pd.Series(np.where(np.arange(60) % 2 == 0, 0.01, -0.005), index=days)

    plot_strategies(
        {"s": make_result(pnl, 1.0)},
        tmp_path,
        "t",
        sample_start="2000-01-10",
        sample_end="2000-03-01",
    )

    window = pnl.loc["2000-01-10":"2000-03-01"]
    nav = pd.read_csv(tmp_path / "plot_nav_t.csv", index_col=0, parse_dates=True)
    dd = pd.read_csv(tmp_path / "plot_drawdown_t.csv", index_col=0, parse_dates=True)
    assert (tmp_path / "equity_drawdown_t.png").stat().st_size > 0
    np.testing.assert_allclose(nav["s"], (1.0 + window).cumprod())
    assert nav.index[0] == window.index[0] and nav.index[-1] == window.index[-1]
    assert (dd["s"] <= 0).all()
