import io
import urllib.request
import zipfile
from pathlib import Path

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


# ---- data: asset classes and autocorrelation --------------------------------------------


def asset_class_stats(returns: pd.DataFrame, classes: pd.Series) -> pd.DataFrame:
    """Two readings of "equally weighted average across instruments":
    A = average of each symbol's statistics, B = statistics of the equal-weighted class portfolio."""
    per_symbol = pd.DataFrame({"Avg": returns.mean() * 12, "Std": returns.std() * np.sqrt(12)})
    per_symbol["SR"] = per_symbol["Avg"] / per_symbol["Std"]
    a = per_symbol.groupby(classes).mean().add_prefix("A: ")
    class_returns = returns.T.groupby(classes).mean().T
    b = pd.DataFrame({"B: Avg": class_returns.mean() * 12, "B: Std": class_returns.std() * np.sqrt(12)})
    b["B: SR"] = b["B: Avg"] / b["B: Std"]
    return a.join(b)


def autocorrelation_tstats(returns: pd.DataFrame, classes: pd.Series, max_lag: int = 12) -> pd.DataFrame:
    """Rank autocorrelation of each symbol at lags 1..max_lag, averaged by class.
    Returns the t-stat of each average: mean / (std / sqrt(n)) across symbols."""
    rho = pd.DataFrame(
        {lag: {s: returns[s].corr(returns[s].shift(lag), method="spearman") for s in returns} for lag in range(1, max_lag + 1)}
    )
    groups = rho.groupby(classes.reindex(rho.index))
    tstats = (groups.mean() / (groups.std() / np.sqrt(groups.count()))).T
    tstats.insert(0, "ALL", rho.mean() / (rho.std() / np.sqrt(len(rho))))
    tstats.index.name = "lag"
    return tstats


def pooled_predictability(returns: pd.DataFrame, max_lag: int = 36) -> pd.Series:
    """Moskowitz, Ooi and Pedersen (2012), Figure 1: regress r(t)/vol on r(t-h)/vol over all symbols
    at once. vol is the expanding std up to t-1; t-stats are clustered by month."""
    scaled = returns / returns.expanding(min_periods=36).std().shift(1)
    y = scaled.stack()
    tstats = {}
    for h in range(1, max_lag + 1):
        df = pd.concat([y.rename("y"), scaled.shift(h).stack().rename("x")], axis=1).dropna()
        month = pd.factorize(df.index.get_level_values(0))[0]
        fit = sm.OLS(df["y"], sm.add_constant(df["x"])).fit(cov_type="cluster", cov_kwds={"groups": month})
        tstats[h] = fit.tvalues["x"]
    return pd.Series(tstats, name="t-stat")


# ---- benchmarks -------------------------------------------------------------------------

FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_FILES = ["F-F_Research_Data_Factors", "F-F_Momentum_Factor", "F-F_Research_Data_5_Factors_2x3"]


def read_french_csv(path: Path) -> pd.DataFrame:
    """Monthly rows of a Ken French factor file, converted from percent to decimals."""
    lines = path.read_text().splitlines()
    header_row = next(i for i, line in enumerate(lines) if line.startswith(","))
    df = pd.read_csv(path, skiprows=header_row, index_col=0)
    df.index = df.index.astype(str).str.strip()
    df = df[df.index.str.len() == 6]  # yyyymm rows only, which drops the annual block
    df.index = pd.PeriodIndex(df.index, freq="M")
    df.columns = df.columns.str.strip()
    return df.astype(float) / 100


def load_factors(folder: Path) -> pd.DataFrame:
    """Fama-French factors plus momentum (UMD). Downloaded from the Ken French library the first time."""
    folder.mkdir(parents=True, exist_ok=True)
    for name in FRENCH_FILES:
        if not (folder / f"{name}.csv").exists():
            with urllib.request.urlopen(FRENCH_URL + f"{name}_CSV.zip") as response:
                zipfile.ZipFile(io.BytesIO(response.read())).extractall(folder)
    ff3 = read_french_csv(folder / "F-F_Research_Data_Factors.csv")
    umd = read_french_csv(folder / "F-F_Momentum_Factor.csv").rename(columns={"Mom": "UMD"})
    ff5 = read_french_csv(folder / "F-F_Research_Data_5_Factors_2x3.csv")[["RMW", "CMA"]]
    return ff3.join(umd).join(ff5)


def load_sg_trend(path: Path) -> pd.DataFrame:
    """Course file with the SG Trend index, RF and other series, monthly."""
    df = pd.read_excel(path, index_col=0)
    df.index = pd.PeriodIndex(df.index, freq="M")
    return df


def alpha_table(strategy: pd.Series, benchmarks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One Newey-West regression of the strategy on each benchmark."""
    return pd.DataFrame({name: ols(strategy, X) for name, X in benchmarks.items()}).T


def to_quarterly(r: pd.Series) -> pd.Series:
    """Sum of monthly returns in each complete calendar quarter."""
    r = r.dropna()
    quarters = r.index.asfreq("Q")
    count = r.groupby(quarters).count()
    return r.groupby(quarters).sum()[count == 3]


def smile(strategy: pd.Series, market: pd.Series) -> dict:
    """Quarterly strategy return on the market and the market squared. A positive
    Mkt^2 beta means the strategy does well in big up and big down markets."""
    q = pd.DataFrame({"s": to_quarterly(strategy), "m": to_quarterly(market)}).dropna()
    return ols(q["s"], pd.DataFrame({"Mkt": q["m"], "Mkt2": q["m"] ** 2}), nw_lags=2, periods=4)


def regime_returns(series: pd.DataFrame, market: pd.Series, q: float = 0.2) -> pd.DataFrame:
    """Annualised average return in the market's worst 20%, middle 60% and best 20% of months."""
    m = market.dropna()
    regime = pd.Series("Normal", index=m.index)
    regime[m <= m.quantile(q)] = "Bear"
    regime[m >= m.quantile(1 - q)] = "Bull"
    return (series.reindex(m.index).groupby(regime).mean() * 12).loc[["Bear", "Normal", "Bull"]]


def sg_trend_comparison(strategy: pd.Series, sg: pd.DataFrame) -> dict:
    """Compare with the SG Trend index of live CTAs. SG is net of fees and earns interest on
    cash, so we compare our excess returns with SG minus RF."""
    sg_excess = (sg["SG Trend"] - sg["RF"]).rename("SG")
    both = pd.concat([strategy.rename("TSMOM"), sg_excess], axis=1).dropna()
    fit = ols(both["SG"], both[["TSMOM"]])
    return {
        "period": f"{both.index[0]} to {both.index[-1]}",
        "corr": both.corr().iloc[0, 1],
        "beta": fit["beta_TSMOM"],
        "alpha": fit["alpha"],
        "t_alpha": fit["t_alpha"],
        "SR TSMOM": monthly_metrics(both["TSMOM"])["SR"],
        "SR SG - RF": monthly_metrics(both["SG"])["SR"],
    }


# ---- risk model check -------------------------------------------------------------------
# Each month end a risk model predicts the portfolio's volatility for the next month.
# We divide the realised next-month return by that prediction. If the model is right these
# ratios have a std of 1 (the "bias statistic"). Above 1 means risk was under-predicted.


def forecast_covariances(daily_returns: pd.DataFrame, dates: pd.DatetimeIndex, model, lookback: int | None = None) -> dict:
    """The model's covariance on each date, using only returns up to that date."""
    covs = {}
    for day in dates:
        history = daily_returns.loc[:day]
        if lookback:
            history = history.iloc[-lookback:]
        covs[day] = model.covariance(history)
    return covs


def predicted_vols(result: BacktestResult, covs: dict, diagonal: bool = False) -> pd.Series:
    """Predicted vol of the weights chosen on each date, labelled by the month they are held.
    diagonal=True ignores correlations, which is what 40% / vol sizing implicitly assumes."""
    vols = {}
    for day, cov in covs.items():
        w = result.weights.loc[day]
        w = w[w != 0]
        if w.empty:
            continue
        q = cov.reindex(index=w.index, columns=w.index).fillna(0).to_numpy()
        if diagonal:
            q = np.diag(np.diag(q))
        vols[day.to_period("M") + 1] = np.sqrt(max(w.to_numpy() @ q @ w.to_numpy(), 0.0))
    return pd.Series(vols)


def bias_statistic(realised: pd.Series, predicted_vol: pd.Series, window: int = 36) -> dict:
    """std(realised / predicted) over the full sample, and the share of `window`-month
    windows that fall inside 1 +/- sqrt(2 / window)."""
    z = (realised / (predicted_vol / np.sqrt(12))).dropna()
    rolling = z.rolling(window).std().dropna()
    band = np.sqrt(2 / window)
    return {"bias": z.std(), "share in band": ((rolling > 1 - band) & (rolling < 1 + band)).mean()}


# ---- lookbacks --------------------------------------------------------------------------


def sharpe_by_period(returns: dict, periods: list[tuple[str, str]]) -> pd.DataFrame:
    """Sharpe ratio of each series over the full sample and each (start, end) sub-period."""
    rows = {}
    for name, r in returns.items():
        rows[name] = {"Full": monthly_metrics(r)["SR"]}
        for start, end in periods:
            rows[name][f"{start}-{end}"] = monthly_metrics(r.loc[start:end])["SR"]
    return pd.DataFrame(rows).T

