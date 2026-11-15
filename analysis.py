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
from panel import Panel

# Strategy Performance Table


def first_trade_date(result: BacktestResult) -> pd.Timestamp:
    active = result.positions.abs().sum(axis=1).gt(0)
    return active.idxmax() if active.any() else result.positions.index[0]


def monthly_returns(result: BacktestResult, gross: bool = False) -> pd.Series:
    """Compound daily net (or gross) returns into calendar months after the first trade."""
    pnl = result.gross_pnl if gross else result.net_pnl
    monthly = (1.0 + pnl).groupby(pnl.index.to_period("M")).prod() - 1.0
    start_period = first_trade_date(result).to_period("M")
    return monthly.loc[monthly.index > start_period]


def max_drawdown(r: pd.Series) -> float:
    wealth = (1.0 + r.fillna(0.0)).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def monthly_metrics(r: pd.Series) -> dict[str, float]:
    r = r.dropna()
    if r.empty or r.std() == 0:
        return {
            k: np.nan
            for k in (
                "Avg",
                "Std",
                "SR",
                "t(mean)",
                "Skew",
                "ExKurt",
                "MaxDD",
                "Calmar",
                "Hit rate",
            )
        }
    mdd = max_drawdown(r)
    return {
        "Avg": float(r.mean() * 12),
        "Std": float(r.std() * np.sqrt(12)),
        "SR": float(r.mean() / r.std() * np.sqrt(12)),
        "t(mean)": float(r.mean() / r.std() * np.sqrt(len(r))),
        "Skew": float(stats.skew(r)),
        "ExKurt": float(stats.kurtosis(r)),
        "MaxDD": mdd,
        # annualised average return over the worst peak-to-trough loss, full sample
        "Calmar": float(r.mean() * 12 / -mdd) if mdd < 0 else np.nan,
        "Hit rate": float((r > 0).mean()),
    }


def sharpe_diff_test(a: pd.Series, b: pd.Series) -> dict[str, float]:
    """Jobson-Korkie test with Memmel's correction that two Sharpe ratios are equal."""
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < 3:
        return {"dSR": np.nan, "p": np.nan}
    x, y = df.iloc[:, 0], df.iloc[:, 1]
    sa, sb, rho = x.mean() / x.std(), y.mean() / y.std(), x.corr(y)
    var = (2.0 - 2.0 * rho + 0.5 * (sa**2 + sb**2 - 2.0 * sa * sb * rho**2)) / len(df)
    if var <= 1e-12:
        return {"dSR": 0.0, "p": np.nan}
    z = (sa - sb) / np.sqrt(var)
    return {
        "dSR": float((sa - sb) * np.sqrt(12)),
        "p": float(2.0 * (1.0 - stats.norm.cdf(abs(z)))),
    }


def strategy_summary_table(
    results: dict[str, BacktestResult],
    baseline_name: str,
    sample_start: str | None = None,
    sample_end: str | None = None,
) -> pd.DataFrame:
    """Comprehensive comparison table of net/gross performance, turnover, cost drag (bps/yr),
    and Sharpe difference p-value against `baseline_name`, over months sample_start..sample_end.
    """
    net_m = {
        name: monthly_returns(r).loc[sample_start:sample_end]
        for name, r in results.items()
    }
    gross_m = {
        name: monthly_returns(r, gross=True).loc[sample_start:sample_end]
        for name, r in results.items()
    }

    rows = {}
    base_series = net_m[baseline_name]

    for name, res in results.items():
        r_net = net_m[name]
        r_gross = gross_m[name]
        m = monthly_metrics(r_net)

        # Slice daily series to match the monthly evaluation window
        start_ts = (
            r_net.index[0].start_time if not r_net.empty else res.net_pnl.index[0]
        )
        end_ts = r_net.index[-1].end_time if not r_net.empty else None
        years = max(len(r_net) / 12.0, 1e-6)

        def per_year(daily: pd.Series) -> float:
            return float(daily.loc[start_ts:end_ts].sum() / years)

        turnover_yr = per_year(res.turnover)
        trade_bps_yr = per_year(res.trading_costs) * 1e4
        roll_bps_yr = per_year(res.roll_costs) * 1e4
        fin_bps_yr = per_year(res.financing_costs) * 1e4

        rows[name] = {
            "Net Avg": m["Avg"],
            "Net Std": m["Std"],
            "Net SR": m["SR"],
            "Gross SR": monthly_metrics(r_gross)["SR"],
            "MaxDD": m["MaxDD"],
            "Calmar": m["Calmar"],
            "Skew": m["Skew"],
            "Hit rate": m["Hit rate"],
            "Turnover/yr": turnover_yr,
            "TradeCost(bps/yr)": trade_bps_yr,
            "RollCost(bps/yr)": roll_bps_yr,
            "FinCost(bps/yr)": fin_bps_yr,
            "Corr(base)": float(r_net.corr(base_series)),
            "p(SR=base)": sharpe_diff_test(r_net, base_series)["p"],
        }

    return pd.DataFrame(rows).T


# Benchmarks & Factor Alpha Regressions

FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_FILES = [
    "F-F_Research_Data_Factors",
    "F-F_Momentum_Factor",
    "F-F_Research_Data_5_Factors_2x3",
]


def _read_french_csv(path: Path) -> pd.DataFrame:
    lines = path.read_text().splitlines()
    header_row = next(i for i, line in enumerate(lines) if line.startswith(","))
    df = pd.read_csv(path, skiprows=header_row, index_col=0)
    df.index = df.index.astype(str).str.strip()
    df = df[df.index.str.len() == 6]
    df.index = pd.PeriodIndex(df.index, freq="M")
    df.columns = df.columns.str.strip()
    return df.astype(float) / 100.0


def load_factors(folder: Path) -> pd.DataFrame:
    """Download/load Fama-French 3/5 factors plus Momentum (UMD)."""
    folder.mkdir(parents=True, exist_ok=True)
    for name in FRENCH_FILES:
        if not (folder / f"{name}.csv").exists():
            with urllib.request.urlopen(FRENCH_URL + f"{name}_CSV.zip") as response:
                zipfile.ZipFile(io.BytesIO(response.read())).extractall(folder)
    ff3 = _read_french_csv(folder / "F-F_Research_Data_Factors.csv")
    umd = _read_french_csv(folder / "F-F_Momentum_Factor.csv").rename(
        columns={"Mom": "UMD"}
    )
    ff5 = _read_french_csv(folder / "F-F_Research_Data_5_Factors_2x3.csv")[
        ["RMW", "CMA"]
    ]
    return ff3.join(umd).join(ff5)


def load_benchmarks(
    panel: Panel,
    static_ret: pd.Series,
    factor_dir: Path,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Build standard Lecture 3 benchmark suite and return (benchmarks_dict, factors_df)."""
    factors = load_factors(factor_dir).reindex(static_ret.index)
    if factors["Mkt-RF"].isna().any():
        raise ValueError(
            f"Fama-French data does not cover {static_ret.index[0]} to {static_ret.index[-1]}; "
            f"delete the cached files in {factor_dir} to download them again"
        )

    # US 30Y T-Bond futures monthly return for the 60/40 benchmark
    us_px = panel.close_df["US"].dropna()
    t_bond = (
        us_px.groupby(us_px.index.to_period("M"))
        .last()
        .pct_change()
        .reindex(static_ret.index)
    )

    benchmarks = {
        "Static benchmark": static_ret.to_frame("Static"),
        "US equity (Mkt-RF)": factors[["Mkt-RF"]],
        "60/40 (Mkt-RF / T-bond)": (0.6 * factors["Mkt-RF"] + 0.4 * t_bond).to_frame(
            "60/40"
        ),
        "Carhart 4-factor": factors[["Mkt-RF", "SMB", "HML", "UMD"]],
        "FF 5-factor + UMD": factors[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]],
    }
    return benchmarks, factors


def ols(
    y: pd.Series, X: pd.DataFrame, nw_lags: int = 6, periods: int = 12
) -> dict[str, float]:
    """OLS with Newey-West HAC standard errors. Alpha is annualized with `periods` per year."""
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    fit = sm.OLS(df["y"], sm.add_constant(df.drop(columns="y"))).fit(
        cov_type="HAC", cov_kwds={"maxlags": nw_lags}
    )
    out = {
        "alpha": float(fit.params["const"] * periods),
        "t_alpha": float(fit.tvalues["const"]),
    }
    for name in X.columns:
        out[f"beta_{name}"] = float(fit.params[name])
        out[f"t_{name}"] = float(fit.tvalues[name])
    out["R2"] = float(fit.rsquared)
    out["n"] = int(fit.nobs)
    return out


def alpha_table(
    strategy_ret: pd.Series, benchmarks: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Regress a strategy's monthly return series against each benchmark."""
    return pd.DataFrame(
        {name: ols(strategy_ret, X) for name, X in benchmarks.items()}
    ).T


def sg_trend_comparison(
    strategy_ret: pd.Series, sg_excel_path: Path
) -> dict[str, float | str]:
    """Regress strategy excess returns on the live CTA index (SG Trend - RF).

    alpha is the strategy's annualized alpha over SG Trend.
    """
    sg = pd.read_excel(sg_excel_path, index_col=0)
    sg.index = pd.PeriodIndex(sg.index, freq="M")
    sg_excess = (sg["SG Trend"] - sg["RF"]).rename("SG")

    both = pd.concat([strategy_ret.rename("Strategy"), sg_excess], axis=1).dropna()
    fit = ols(both["Strategy"], both[["SG"]])
    return {
        "period": f"{both.index[0]} to {both.index[-1]}",
        "corr": float(both.corr().iloc[0, 1]),
        "beta": fit["beta_SG"],
        "alpha": fit["alpha"],
        "t_alpha": fit["t_alpha"],
        "SR Strategy": monthly_metrics(both["Strategy"])["SR"],
        "SR SG-RF": monthly_metrics(both["SG"])["SR"],
    }


# Crisis Alpha: Market Regimes & Smile Regression


def to_quarterly(r: pd.Series) -> pd.Series:
    """Sum of monthly returns in each complete 3-month calendar quarter."""
    r = r.dropna()
    quarters = r.index.asfreq("Q")
    count = r.groupby(quarters).count()
    return r.groupby(quarters).sum()[count == 3]


def smile_regression(
    strategy_ret: pd.Series, market_ret: pd.Series
) -> dict[str, float]:
    """Quarterly regression on Mkt and Mkt^2 to test positive tail convexity."""
    q = pd.DataFrame(
        {"s": to_quarterly(strategy_ret), "m": to_quarterly(market_ret)}
    ).dropna()
    return ols(
        q["s"], pd.DataFrame({"Mkt": q["m"], "Mkt2": q["m"] ** 2}), nw_lags=2, periods=4
    )


def regime_returns(
    strategies_monthly: dict[str, pd.Series],
    market_ret: pd.Series,
    q: float = 0.2,
) -> pd.DataFrame:
    """Annualized average return during Bear (worst 20%), Normal (middle 60%), and Bull (best 20%) equity months."""
    m = market_ret.dropna()
    regime = pd.Series("Normal", index=m.index)
    regime[m <= m.quantile(q)] = "Bear"
    regime[m >= m.quantile(1.0 - q)] = "Bull"

    df = pd.DataFrame(strategies_monthly)
    df["Mkt-RF"] = m
    return (df.reindex(m.index).groupby(regime).mean() * 12.0).loc[
        ["Bear", "Normal", "Bull"]
    ]


# Correlation Structure Decomposition


def _avg_within_across_corr(
    ret_df: pd.DataFrame, asset_classes: pd.Series
) -> dict[str, float]:
    corr = ret_df.corr(min_periods=12)
    classes = asset_classes.reindex(corr.index)

    within_vals = []
    across_vals = []
    symbols = list(corr.index)

    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            val = corr.iloc[i, j]
            if pd.isna(val):
                continue
            if classes.iloc[i] == classes.iloc[j]:
                within_vals.append(val)
            else:
                across_vals.append(val)

    out = {
        "Within Asset Class (All)": float(np.mean(within_vals))
        if within_vals
        else np.nan,
        "Across Asset Classes": float(np.mean(across_vals)) if across_vals else np.nan,
    }
    for cls in sorted(classes.dropna().unique()):
        cls_syms = [s for s in symbols if classes.get(s) == cls]
        sub = corr.loc[cls_syms, cls_syms]
        vals = [
            sub.iloc[i, j]
            for i in range(len(cls_syms))
            for j in range(i + 1, len(cls_syms))
            if pd.notna(sub.iloc[i, j])
        ]
        out[f"Within {cls}"] = float(np.mean(vals)) if vals else np.nan
    return out


def correlation_structure(
    panel: Panel,
    result: BacktestResult,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Compare average pairwise correlation within and across asset classes for
    Passive Buy-and-Hold vs. Strategy active position returns (Lecture 3), using data from `start` to `end`.
    """
    classes = panel.asset_map["AssetClass"]

    # Monthly passive asset returns vs strategy component returns
    passive_daily = panel.observed_returns.loc[start:end]
    strat_comp_daily = (
        np.sign(result.positions.loc[start:end]).replace(0.0, np.nan) * passive_daily
    )

    passive_m = (1.0 + passive_daily).groupby(passive_daily.index.to_period("M")).prod(
        min_count=1
    ) - 1.0
    strat_m = (1.0 + strat_comp_daily).groupby(
        strat_comp_daily.index.to_period("M")
    ).prod(min_count=1) - 1.0

    return pd.DataFrame(
        {
            "Passive Buy & Hold": _avg_within_across_corr(passive_m, classes),
            "Strategy Positions": _avg_within_across_corr(strat_m, classes),
        }
    ).T


# Lag Profile


def lag_profile(
    panel: Panel,
    windows: list[tuple[int, int]],
    start: str | None = None,
    end: str | None = None,
    lag: int = 2,
    cap: float = 2.0,
    vol_span: int = 60,
) -> pd.DataFrame:
    """Sharpe ratio, before costs, of trading each lookback window on its own.

    For a window (a, b) every contract is held in proportion to its move from b to a bars
    ago (in volatility units, capped at +-`cap`), at equal risk, from `lag` days later.
    Positive means the window predicts continuation, negative a reversal.
    """
    close = panel.close_df.loc[:end]
    logs = {s: np.log(close[s].dropna()) for s in close}
    vol = {
        s: x.diff().ewm(span=vol_span, min_periods=20).std() for s, x in logs.items()
    }

    def on_panel(per_symbol: dict[str, pd.Series]) -> pd.DataFrame:
        # each contract's own-bar series, carried over the days it did not trade
        return pd.DataFrame(per_symbol).reindex(close.index).ffill()

    unit_return = panel.returns.loc[:end].shift(-lag) / on_panel(vol)
    classes = panel.asset_map["AssetClass"].reindex(close.columns)
    groups = {"All": list(close.columns)}
    for cls in sorted(classes.dropna().unique()):
        groups[cls] = list(classes.index[classes == cls])

    rows = {}
    for a, b in windows:
        score = on_panel(
            {
                s: ((x.shift(a) - x.shift(b)) / (vol[s] * np.sqrt(b - a))).clip(
                    -cap, cap
                )
                for s, x in logs.items()
            }
        )
        pnl = (score * unit_return).loc[start:]
        row = {}
        for name, members in groups.items():
            daily = pnl[members].mean(axis=1)
            monthly = daily.groupby(daily.index.to_period("M")).sum()
            row[name] = float(monthly.mean() / monthly.std() * np.sqrt(12))
        rows[f"{a}-{b}"] = row
    return pd.DataFrame(rows).T


# Risk Model Bias Check


def risk_model_bias(
    daily_returns: pd.DataFrame,
    results: dict[str, BacktestResult],
    risk_models: dict[str, tuple[object, bool, int | None]],
    sample_start: str | None = None,
    window: int = 36,
) -> pd.DataFrame:
    """Evaluate ex-ante vs. realized volatility bias statistic: std(realized / predicted).
    A bias of 1.0 means accurate risk forecasts; >1.0 means risk was under-predicted.
    """
    monthly = {
        name: (
            monthly_returns(r).loc[sample_start:]
            if sample_start
            else monthly_returns(r)
        )
        for name, r in results.items()
    }
    ref_weights = next(iter(results.values())).weights
    month_ends = (
        ref_weights.index.to_series().groupby(ref_weights.index.to_period("M")).max()
    )
    first_dt = first_trade_date(next(iter(results.values())))
    eval_dates = pd.DatetimeIndex(month_ends[month_ends >= first_dt])

    rows = []
    band = np.sqrt(2.0 / window)

    for rm_name, (model, diagonal, lookback) in risk_models.items():
        covs = {}
        for day in eval_dates:
            hist = daily_returns.loc[:day]
            if lookback:
                hist = hist.iloc[-lookback:]
            covs[day] = model.covariance(hist)

        for strat_name, res in results.items():
            vols = {}
            for day, cov in covs.items():
                # forecast made at the end of month P (cov) for the weights set at
                # the start of month P+1, which are what earn month P+1's return
                nxt = day.to_period("M") + 1
                if nxt not in month_ends.index:
                    continue
                w = res.weights.loc[month_ends[nxt]]
                w = w[w != 0.0]
                if w.empty:
                    continue
                q = cov.reindex(index=w.index, columns=w.index).fillna(0.0).to_numpy()
                if diagonal:
                    q = np.diag(np.diag(q))
                vols[nxt] = np.sqrt(max(float(w.to_numpy() @ q @ w.to_numpy()), 0.0))

            pred_vol = pd.Series(vols)
            z = (monthly[strat_name] / (pred_vol / np.sqrt(12.0))).dropna()
            rolling = z.rolling(window).std().dropna()
            rows.append(
                {
                    "Strategy": strat_name,
                    "Risk model": rm_name,
                    "Bias stat": float(z.std()),
                    "Share in band": float(
                        ((rolling > 1.0 - band) & (rolling < 1.0 + band)).mean()
                    ),
                }
            )

    return pd.DataFrame(rows).set_index(["Strategy", "Risk model"])


def plot_strategies(
    results: dict[str, BacktestResult],
    out_dir: Path,
    tag: str,
    sample_start: str | None = None,
    sample_end: str | None = None,
) -> None:
    """Plot cumulative net wealth and drawdowns; save the figure and the daily series behind it."""
    navs, dds = {}, {}
    for name, res in results.items():
        pnl = (
            res.net_pnl.loc[sample_start:sample_end]
            if sample_start
            else res.net_pnl.loc[first_trade_date(res) : sample_end]
        ).fillna(0.0)
        nav = (1.0 + pnl).cumprod()
        navs[name] = nav
        dds[name] = nav / nav.cummax() - 1.0
    nav_df = pd.DataFrame(navs)
    dd_df = pd.DataFrame(dds)

    # the data behind the figure, so it can be redrawn later without rerunning the backtests
    out_dir.mkdir(parents=True, exist_ok=True)
    nav_df.to_csv(out_dir / f"plot_nav_{tag}.csv")
    dd_df.to_csv(out_dir / f"plot_drawdown_{tag}.csv")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for name in nav_df.columns:
        ax1.plot(nav_df.index, nav_df[name], label=name)
        ax2.plot(dd_df.index, dd_df[name], label=name)

    ax1.set_yscale("log")
    ax1.set_title("Cumulative Net Wealth (Log Scale)")
    ax1.set_ylabel("NAV")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Drawdown")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / f"equity_drawdown_{tag}.png", dpi=150)
    plt.close(fig)
