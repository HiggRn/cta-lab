import numpy as np
import pandas as pd
import pytest

from analysis import bias_statistic, sharpe_by_period


def test_bias_statistic_is_one_when_risk_is_predicted_right():
    months = pd.period_range("1980-01", periods=600, freq="M")
    predicted_vol = pd.Series(np.linspace(0.05, 0.30, 600), index=months)  # annual vol, changing over time
    realised = pd.Series(np.random.default_rng(1).normal(0, 1, 600), index=months) * predicted_vol / np.sqrt(12)
    assert bias_statistic(realised, predicted_vol)["bias"] == pytest.approx(1.0, abs=0.1)
    # predicting half the true risk gives a bias statistic of about 2
    assert bias_statistic(realised, predicted_vol / 2)["bias"] == pytest.approx(2.0, abs=0.2)


def test_sharpe_by_period():
    months = pd.period_range("1990-01", periods=24, freq="M")
    r = pd.Series([0.01, 0.03] * 12, index=months)
    table = sharpe_by_period({"x": r}, [("1990", "1990")])
    assert table.loc["x", "Full"] == pytest.approx(0.02 / r.std() * np.sqrt(12))
