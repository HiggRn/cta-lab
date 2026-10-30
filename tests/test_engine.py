import numpy as np
import pandas as pd
import pytest

from common import Strategy
from engine import FuturesCostModel, run_backtest

DATES = pd.bdate_range("2000-03-01", periods=12)
RET_A = np.array([0, 0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.0, 0.02, -0.01, 0.01, 0.02])
RET_B = RET_A * 0.5


def closes_from(returns) -> pd.Series:
    return pd.Series(100.0 * np.cumprod(1.0 + returns), index=DATES)


class OneShot(Strategy):
    """Sends `target` on day `on_idx` and nothing else."""

    def __init__(self, on_idx, target):
        self.on_date = DATES[on_idx]
        self.target = target

    def on_bar(self, bars, context):
        return dict(self.target) if context.current_date == self.on_date else None


class Spy(Strategy):
    """Records what the context shows on every call."""

    def __init__(self):
        self.rows = []
        self.mutated_copy_leaked = False

    def on_bar(self, bars, context):
        returns, closes = context.get_returns(), context.get_closes()
        self.rows.append(
            dict(
                today=context.current_date,
                last_return_date=returns.index[-1],
                last_close_date=closes.index[-1],
                n_rows=len(returns),
                traded=sorted(bars),
                b_return=returns["b"].iloc[-1] if "b" in returns else np.nan,
            )
        )
        returns.iloc[:] = 123.0
        if (context.get_returns() == 123.0).any().any():
            self.mutated_copy_leaked = True
        return None


# Panel


def test_panel_drops_symbols_and_sorts(make_panel):
    panel = make_panel(
        {"b": closes_from(RET_B), "a": closes_from(RET_A), "x": closes_from(RET_A)},
        dropped=("x",),
    )
    assert panel.symbols == ["a", "b"]


def test_panel_gap_handling(make_panel):
    b = closes_from(RET_B).drop(DATES[3])
    panel = make_panel({"a": closes_from(RET_A), "b": b})
    close = b.reindex(DATES).ffill()
    assert panel.returns.loc[DATES[3], "b"] == 0.0  # holiday: P&L earns nothing
    assert panel.returns.loc[DATES[4], "b"] == pytest.approx(close[DATES[4]] / close[DATES[2]] - 1.0)
    assert np.isnan(panel.observed_returns.loc[DATES[3], "b"])  # estimation: not a return
    assert panel.observed_returns.loc[DATES[4], "b"] == panel.returns.loc[DATES[4], "b"]
    assert "b" not in panel.timeline[3][1]


def test_panel_first_and_last_valid_idx(make_panel):
    late = closes_from(RET_B).iloc[2:8]
    panel = make_panel({"a": closes_from(RET_A), "late": late})
    j = panel.symbols.index("late")
    assert panel.first_valid_idx[j] == 2
    assert panel.last_valid_idx[j] == 7
    assert panel.first_valid_idx[panel.symbols.index("a")] == 0
    assert np.isnan(panel.observed_returns.iloc[2]["late"])  # first bar has no return
    assert panel.returns.iloc[9]["late"] == 0.0  # after its last bar


# Execution timing


def test_exec_lag_one_trades_at_the_signal_close(make_panel):
    panel = make_panel({"a": closes_from(RET_A)})
    res = run_backtest(panel, OneShot(2, {"a": 0.5}), exec_lag=1)
    assert (res.positions["a"].iloc[:3] == 0).all()
    assert res.positions["a"].iloc[3] == pytest.approx(0.5)
    assert (res.gross_pnl.iloc[:3] == 0).all()
    assert res.gross_pnl.iloc[3] == pytest.approx(0.5 * RET_A[3])  # first earned return is t+1


def test_exec_lag_two_trades_one_bar_later(make_panel):
    panel = make_panel({"a": closes_from(RET_A)})
    res = run_backtest(panel, OneShot(2, {"a": 0.5}), exec_lag=2)
    assert res.positions["a"].iloc[3] == 0.0
    assert res.gross_pnl.iloc[3] == 0.0
    assert res.positions["a"].iloc[4] == pytest.approx(0.5)
    assert res.gross_pnl.iloc[4] == pytest.approx(0.5 * RET_A[4])


def test_exec_lag_must_be_positive(make_panel):
    panel = make_panel({"a": closes_from(RET_A)})
    with pytest.raises(ValueError):
        run_backtest(panel, OneShot(2, {"a": 0.5}), exec_lag=0)


def test_weights_record_the_decision_and_positions_the_fill(make_panel):
    panel = make_panel({"a": closes_from(RET_A)})
    res = run_backtest(panel, OneShot(2, {"a": 0.5}), exec_lag=1)
    assert res.weights["a"].iloc[1] == 0.0
    assert (res.weights["a"].iloc[2:] == 0.5).all()  # target, carried forward


def test_positions_drift_with_prices_between_rebalances(make_panel):
    panel = make_panel({"a": closes_from(RET_A)})
    res = run_backtest(panel, OneShot(2, {"a": 0.5}), exec_lag=1)
    held = 0.5 * (1.0 + RET_A[3]) / (1.0 + 0.5 * RET_A[3])  # no cost: net = gross
    assert res.positions["a"].iloc[4] == pytest.approx(held)
    assert res.turnover.sum() == pytest.approx(0.5)  # drift is not turnover


def test_pending_order_waits_for_a_bar(make_panel):
    b = closes_from(RET_B).drop(DATES[3])  # b does not trade on the fill day
    panel = make_panel({"a": closes_from(RET_A), "b": b})
    res = run_backtest(panel, OneShot(2, {"a": 0.5, "b": 0.5}), exec_lag=1)
    assert res.positions["a"].iloc[3] == pytest.approx(0.5)
    assert res.positions["b"].iloc[3] == 0.0
    assert res.positions["b"].iloc[4] == pytest.approx(0.5)
    assert res.turnover.iloc[3] == pytest.approx(0.5)
    assert res.turnover.iloc[4] == pytest.approx(0.5)


# Costs


def test_futures_cost_model_accounting(make_panel):
    panel = make_panel({"a": closes_from(RET_A)})
    model = FuturesCostModel(
        trading_cost_bps=1.5,
        roll_cost_bps=1.5,
        rolls_per_year=4.0,
        financing_cost_bps=50.0,
        free_leverage_threshold=1.0,
    )
    res = run_backtest(panel, OneShot(2, {"a": 2.0}), cost_model=model, exec_lag=1)

    assert res.turnover.iloc[3] == pytest.approx(2.0)
    assert res.trading_costs.iloc[3] == pytest.approx(2.0 * 1.5e-4)
    assert res.trading_costs.drop(DATES[3]).sum() == 0.0

    gross_leverage = res.positions.abs().sum(axis=1)
    np.testing.assert_allclose(
        res.roll_costs, gross_leverage * 1.5e-4 * 4.0 / 252
    )
    np.testing.assert_allclose(
        res.financing_costs, (gross_leverage - 1.0).clip(lower=0.0) * 50e-4 / 252
    )
    np.testing.assert_allclose(res.net_pnl, res.gross_pnl - res.costs)
    assert res.nav.iloc[-1] == pytest.approx((1.0 + res.net_pnl).prod())


def test_no_financing_below_the_free_leverage_threshold(make_panel):
    panel = make_panel({"a": closes_from(RET_A)})
    model = FuturesCostModel(free_leverage_threshold=5.0)
    res = run_backtest(panel, OneShot(2, {"a": 2.0}), cost_model=model, exec_lag=1)
    assert res.financing_costs.sum() == 0.0
    assert res.roll_costs.sum() > 0.0


def test_cost_bps_shortcut_charges_trading_cost_only(make_panel):
    panel = make_panel({"a": closes_from(RET_A)})
    res = run_backtest(panel, OneShot(2, {"a": 0.5}), cost_bps=10.0, exec_lag=1)
    assert res.trading_costs.iloc[3] == pytest.approx(0.5 * 10e-4)
    assert res.roll_costs.sum() == 0.0 and res.financing_costs.sum() == 0.0


def test_default_is_frictionless(make_panel):
    panel = make_panel({"a": closes_from(RET_A)})
    res = run_backtest(panel, OneShot(2, {"a": 0.5}))
    assert res.costs.sum() == 0.0
    np.testing.assert_allclose(res.net_pnl, res.gross_pnl)


# No lookahead through the context


def test_strategy_only_sees_data_up_to_today(make_panel):
    b = closes_from(RET_B).drop(DATES[3])
    panel = make_panel({"a": closes_from(RET_A), "b": b})
    spy = Spy()
    run_backtest(panel, spy)
    assert len(spy.rows) == len(DATES)
    for i, row in enumerate(spy.rows):
        assert row["today"] == DATES[i]
        assert row["last_return_date"] == DATES[i]
        assert row["last_close_date"] == DATES[i]
        assert row["n_rows"] == i + 1
    assert spy.rows[3]["traded"] == ["a"]
    assert np.isnan(spy.rows[3]["b_return"])  # no bar, no observed return
    assert not spy.mutated_copy_leaked
