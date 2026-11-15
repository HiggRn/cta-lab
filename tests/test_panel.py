import numpy as np
import pandas as pd
import pytest

DAYS = pd.bdate_range("2000-03-01", periods=12)
RET = np.array([0, 0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.0, 0.02, -0.01, 0.01, 0.02])


def closes_from(returns, scale=100.0):
    return pd.Series(
        scale * np.cumprod(1.0 + np.asarray(returns)), index=DAYS[: len(returns)]
    )


# Loading and cleaning


def test_unmapped_symbols_are_ignored(make_panel):
    panel = make_panel(
        {"a": closes_from(RET), "junk": closes_from(RET)}, unmapped=("junk",)
    )
    assert panel.symbols == ["a"]


def test_missing_data_file_fails_loudly(make_panel):
    with pytest.raises(FileNotFoundError, match="b"):
        make_panel({"a": closes_from(RET), "b": closes_from(RET)}, no_file=("b",))


def test_non_positive_close_fails_loudly(make_panel):
    bad = closes_from(RET)
    bad.iloc[5] = -1.0
    with pytest.raises(ValueError, match="a"):
        make_panel({"a": bad})


def test_placeholder_rows_do_not_count_as_trading_days(make_panel):
    late = closes_from(RET).iloc[2:8]
    panel = make_panel(
        {"a": closes_from(RET), "late": late},
        zero_days={"late": DAYS[[0, 1, 8, 9, 10, 11]]},
    )
    j = panel.symbols.index("late")
    assert panel.first_valid_idx[j] == 2
    assert panel.last_valid_idx[j] == 7
    assert not np.isinf(panel.returns.to_numpy()).any()
    assert (panel.returns["late"].dropna() > -0.5).all()
    assert "late" not in panel.timeline[0][1]


# Known contract-switch gap


def test_roll_fix_zeroes_the_switch_day_and_leaves_the_rest(make_panel):
    ret = RET.copy()
    ret[6] = 0.6  # old -> new contract
    raw = closes_from(ret)
    panel = make_panel({"a": raw}, roll_fixes={"a": [str(DAYS[6].date())]})

    expected = ret.copy()
    expected[6] = 0.0
    np.testing.assert_allclose(panel.returns["a"].iloc[1:], expected[1:], atol=1e-12)
    np.testing.assert_allclose(
        panel.close_df["a"].iloc[:6], raw.iloc[:6]
    )  # earlier prices untouched
    # the whole bar is rescaled, not just the close
    assert panel.timeline[7][1]["a"].open == pytest.approx(panel.close_df["a"].iloc[7])


def test_roll_fix_on_a_missing_day_fails_loudly(make_panel):
    with pytest.raises(KeyError):
        make_panel(
            {"a": closes_from(RET).drop(DAYS[6])},
            roll_fixes={"a": [str(DAYS[6].date())]},
        )


# Big contract continued by its mini


def test_mini_continues_the_retired_big_contract(make_panel):
    big = closes_from(RET).iloc[:7]  # last bar is DAYS[6]
    mini = closes_from(RET, scale=20.0)  # same index at a different price level
    panel = make_panel(
        {"big": big, "mini": mini}, dropped=("mini",), successors={"big": "mini"}
    )

    assert panel.symbols == ["big"]
    assert panel.last_valid_idx[0] == 11
    np.testing.assert_allclose(panel.close_df["big"].iloc[:7], big)
    np.testing.assert_allclose(
        panel.returns["big"].iloc[1:], RET[1:], atol=1e-12
    )  # one continuous series
    assert panel.close_df["big"].iloc[11] == pytest.approx(
        big.iloc[6] * mini.iloc[11] / mini.iloc[6]
    )


def test_chain_anchors_on_the_last_mini_close_before_the_join(make_panel):
    big = closes_from(RET).iloc[:7]
    mini = closes_from(RET, scale=20.0).drop(DAYS[6])  # no mini bar on the join day
    panel = make_panel(
        {"big": big, "mini": mini}, dropped=("mini",), successors={"big": "mini"}
    )
    expected = big.iloc[6] * mini[DAYS[7]] / mini[DAYS[5]]
    assert panel.close_df["big"].loc[DAYS[7]] == pytest.approx(expected)


def test_chain_changes_nothing_when_the_big_contract_outlives_the_mini(make_panel):
    big = closes_from(RET)
    mini = closes_from(RET, scale=20.0).iloc[:9]
    panel = make_panel(
        {"big": big, "mini": mini}, dropped=("mini",), successors={"big": "mini"}
    )
    np.testing.assert_allclose(panel.close_df["big"], big)
    assert panel.last_valid_idx[0] == 11
