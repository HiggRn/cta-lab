import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from engine import BacktestResult


def first_trade_date(result: BacktestResult) -> pd.Timestamp:
    return result.positions.abs().sum(axis=1).gt(0).idxmax()


def performance_metrics(
    result: BacktestResult,
    periods_per_year: int = 252,
) -> dict[str, float]:
    # skip the years before the first trade, otherwise they count as zero-return days
    returns = result.net_pnl.loc[first_trade_date(result):].dropna()

    annual_return = returns.mean() * periods_per_year
    annual_volatility = returns.std() * periods_per_year**0.5

    if annual_volatility > 0:
        sharpe = annual_return / annual_volatility
    else:
        sharpe = float("nan")

    nav = (1.0 + returns).cumprod()

    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0
    max_drawdown = drawdown.min()

    total_cost = result.costs.sum()
    final_nav = nav.iloc[-1]

    return {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "final_nav": final_nav,
        "total_cost": total_cost,
    }


def print_performance_summary(
    result: BacktestResult,
    periods_per_year: int = 252,
) -> None:
    metrics = performance_metrics(result, periods_per_year)

    print("\nPerformance Summary")
    print("-------------------")
    print(f"Annual return:     {metrics['annual_return']:.2%}")
    print(f"Annual volatility: {metrics['annual_volatility']:.2%}")
    print(f"Sharpe ratio:      {metrics['sharpe']:.2f}")
    print(f"Max drawdown:      {metrics['max_drawdown']:.2%}")
    print(f"Final NAV:         {metrics['final_nav']:.3f}")
    print(f"Total cost:        {metrics['total_cost']:.4f}")


def plot_equity_curve(result: BacktestResult) -> None:
    nav = result.nav.dropna()

    plt.figure(figsize=(10, 5))
    plt.plot(nav.index, nav.values)
    plt.title("TSMOM Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.tight_layout()
    plt.show()


def plot_drawdown(result: BacktestResult) -> None:
    nav = result.nav.dropna()

    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0

    plt.figure(figsize=(10, 4))
    plt.plot(drawdown.index, drawdown.values)
    plt.title("TSMOM Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.show()


def analyze_backtest(result: BacktestResult) -> dict[str, float]:
    metrics = performance_metrics(result)

    print_performance_summary(result)
    plot_equity_curve(result)
    plot_drawdown(result)

    return metrics


# ---- monthly statistics -------------------------------------------------------------


def monthly_returns(result: BacktestResult) -> pd.Series:
    """Net returns compounded to calendar months, from the first full month after the first trade."""
    pnl = result.net_pnl
    monthly = (1 + pnl).groupby(pnl.index.to_period("M")).prod() - 1
    return monthly.loc[monthly.index > first_trade_date(result).to_period("M")]


def max_drawdown(r: pd.Series) -> float:
    wealth = (1 + r.fillna(0)).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def monthly_metrics(r: pd.Series) -> dict:
    r = r.dropna()
    return {
        "Avg": r.mean() * 12,
        "Std": r.std() * np.sqrt(12),
        "SR": r.mean() / r.std() * np.sqrt(12),
        "t(mean)": r.mean() / r.std() * np.sqrt(len(r)),
        "Skew": stats.skew(r),
        "ExKurt": stats.kurtosis(r),
        "MaxDD": max_drawdown(r),
        "Hit rate": (r > 0).mean(),
        "Start": str(r.index[0]),
        "Months": len(r),
    }


def performance_table(series: dict[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame({name: monthly_metrics(r) for name, r in series.items()}).T


def ols(y: pd.Series, X: pd.DataFrame, nw_lags: int = 6, periods: int = 12) -> dict:
    """OLS with Newey-West t-stats. Alpha is annualised with `periods` per year."""
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    fit = sm.OLS(df["y"], sm.add_constant(df.drop(columns="y"))).fit(
        cov_type="HAC", cov_kwds={"maxlags": nw_lags}
    )
    out = {"alpha": fit.params["const"] * periods, "t_alpha": fit.tvalues["const"]}
    for name in X.columns:
        out[f"beta_{name}"] = fit.params[name]
        out[f"t_{name}"] = fit.tvalues[name]
    out["R2"] = fit.rsquared
    out["n"] = int(fit.nobs)
    return out


def sharpe_ci(r: pd.Series, level: float = 0.95) -> tuple[float, float]:
    """Lo (2002) confidence interval for the annual Sharpe ratio, assuming iid returns."""
    r = r.dropna()
    s = r.mean() / r.std()
    se = np.sqrt((1 + 0.5 * s**2) / len(r))
    z = stats.norm.ppf(0.5 + level / 2)
    return (s - z * se) * np.sqrt(12), (s + z * se) * np.sqrt(12)


def sharpe_diff_test(a: pd.Series, b: pd.Series) -> dict:
    """Jobson-Korkie test, with Memmel's correction, that two Sharpe ratios are equal."""
    df = pd.concat([a, b], axis=1).dropna()
    x, y = df.iloc[:, 0], df.iloc[:, 1]
    sa, sb, rho = x.mean() / x.std(), y.mean() / y.std(), x.corr(y)
    var = (2 - 2 * rho + 0.5 * (sa**2 + sb**2 - 2 * sa * sb * rho**2)) / len(df)
    if var <= 0:  # identical series
        return {"dSR": 0.0, "p": np.nan}
    z = (sa - sb) / np.sqrt(var)
    return {"dSR": (sa - sb) * np.sqrt(12), "p": 2 * (1 - stats.norm.cdf(abs(z)))}


def deflated_sharpe(r: pd.Series, trial_sharpes: list[float]) -> float:
    """Probability the true Sharpe is above zero after picking the best of several trials
    (Bailey and Lopez de Prado, 2014)."""
    r = r.dropna()
    n = len(trial_sharpes)
    s = r.mean() / r.std()
    sd_trials = np.std(np.array(trial_sharpes) / np.sqrt(12), ddof=1)
    gamma = 0.5772  # Euler-Mascheroni constant
    expected_max = sd_trials * ((1 - gamma) * stats.norm.ppf(1 - 1 / n) + gamma * stats.norm.ppf(1 - 1 / (n * np.e)))
    skew, kurt = stats.skew(r), stats.kurtosis(r) + 3
    z = (s - expected_max) * np.sqrt(len(r) - 1) / np.sqrt(1 - skew * s + (kurt - 1) / 4 * s**2)
    return float(stats.norm.cdf(z))
