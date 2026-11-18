"""Run the docstring examples of the library modules as tests."""

import doctest
import importlib

import pytest

MODULES = ["analysis", "baselines", "common", "engine", "indicators.monthly_returns", "indicators.trend_tstat"]


@pytest.mark.parametrize("name", MODULES)
def test_doctests(name):
    result = doctest.testmod(importlib.import_module(name))
    assert result.failed == 0
