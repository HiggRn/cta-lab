from pathlib import Path

import numpy as np
import pandas as pd

from analysis import monthly_returns
from common import RiskModel
from engine import BacktestResult, run_backtest
from main import BASELINE, SAMPLE_START, STRATEGIES, load_panel, monthly_tsmom
from panel import Panel
from strategy import (
    EwmaDailyRiskModel,
    EwmaMonthlyRiskModel,
    ExpandingMonthlyRiskModel,
    MonthlyTsmomAlpha,
    SampleCovarianceRiskModel,
    exante_vol,
)

# Testing risk models: is realised risk equal to predicted risk?
# Each month end we predict the portfolio's vol from data up to that day, then divide the next
# month's return by it. The std of these ratios (the bias statistic) should be 1, and a 36-month
# window should land inside 1 +/- sqrt(2/36).

WINDOW = 36
RISK_MODELS = {
    "Expanding monthly, no correlations (baseline sizing)": (ExpandingMonthlyRiskModel(), True),
    "Expanding monthly covariance": (ExpandingMonthlyRiskModel(), False),
    "EWMA monthly covariance (12m half-life)": (EwmaMonthlyRiskModel(), False),
    "EWMA daily vol, no correlations (MOP)": (EwmaDailyRiskModel(), True),
    "Sample covariance, last 60 days": (SampleCovarianceRiskModel(), False),
}


class RandomSignAlpha(MonthlyTsmomAlpha):
    """Random long/short positions with the baseline's sizing: a control portfolio with no trend."""

    def __init__(self, seed: int):
        super().__init__()
        self.rng = np.random.default_rng(seed)

    def direction(self, monthly_returns: list[float]) -> float:
        return float(self.rng.choice([-1.0, 1.0]))


def month_ends(panel: Panel) -> pd.DatetimeIndex:
    dates = panel.all_dates.to_series()
    return pd.DatetimeIndex(dates.groupby(dates.index.to_period("M")).max())


def forecasts(panel: Panel, model: RiskModel, lookback: int | None) -> dict:
    """The model's covariance at every month end, using only returns up to that day."""
    covs = {}
    for day in month_ends(panel):
        history = panel.observed_returns.loc[:day]
        if lookback:
            history = history.iloc[-lookback:]
        covs[day] = model.covariance(history)
    return covs


def predicted_vols(result: BacktestResult, covs: dict, diagonal: bool) -> pd.Series:
    """Ex-ante vol of the portfolio chosen at each month end, labelled by the month it is held."""
    vols = {}
    for day, cov in covs.items():
        weights = result.weights.loc[day]
        weights = weights[weights != 0].to_dict()
        if not weights:
            continue
        if diagonal:
            cov = pd.DataFrame(np.diag(np.diag(cov)), index=cov.index, columns=cov.columns)
        vols[day.to_period("M") + 1] = exante_vol(weights, cov)
    return pd.Series(vols)


def bias_summary(z: pd.Series) -> dict:
    rolling = z.rolling(WINDOW).std().dropna()
    band = np.sqrt(2 / WINDOW)
    return {
        "bias": z.std(),
        "rolling min": rolling.min(),
        "rolling max": rolling.max(),
        "share in band": ((rolling > 1 - band) & (rolling < 1 + band)).mean(),
    }


def scorecard(panel: Panel, portfolios: dict[str, BacktestResult]) -> pd.DataFrame:
    rows = []
    for model_name, (model, diagonal) in RISK_MODELS.items():
        lookback = 60 if isinstance(model, SampleCovarianceRiskModel) else None
        covs = forecasts(panel, model, lookback)
        for portfolio_name, result in portfolios.items():
            realised = monthly_returns(result).loc[SAMPLE_START:]
            predicted = predicted_vols(result, covs, diagonal) / np.sqrt(12)
            z = (realised / predicted).dropna()
            rows.append({"Risk model": model_name, "Portfolio": portfolio_name, **bias_summary(z)})
    return pd.DataFrame(rows)


def main():
    panel = load_panel()
    portfolios = {
        "TSMOM 12m": run_backtest(panel, STRATEGIES[BASELINE]()),
        "Static": run_backtest(panel, STRATEGIES["Static benchmark"]()),
    }
    for seed in range(10):
        portfolios[f"Random {seed}"] = run_backtest(panel, monthly_tsmom(alpha=RandomSignAlpha(seed)))

    table = scorecard(panel, portfolios)
    is_random = table["Portfolio"].str.startswith("Random")
    random_median = table[is_random].groupby("Risk model").median(numeric_only=True)
    random_median["Portfolio"] = "10 random (median)"
    table = pd.concat([table[~is_random], random_median.reset_index()], ignore_index=True)

    out = Path("./outputs")
    out.mkdir(exist_ok=True)
    table.to_csv(out / "risk_model_tests.csv", index=False)
    print("bias statistic (1 = unbiased) and share of 36-month windows inside 1 +/- 0.24:")
    for portfolio, rows in table.groupby("Portfolio", sort=False):
        print(f"\n{portfolio}")
        print(rows.set_index("Risk model")[["bias", "share in band"]].round(2).to_string())


if __name__ == "__main__":
    main()
