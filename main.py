from pathlib import Path

import numpy as np
import pandas as pd

from analysis import (
    alpha_table,
    analyze_backtest,
    asset_class_stats,
    autocorrelation_tstats,
    bias_statistic,
    deflated_sharpe,
    forecast_covariances,
    load_factors,
    load_sg_trend,
    monthly_returns,
    performance_table,
    pooled_predictability,
    predicted_vols,
    regime_returns,
    sg_trend_comparison,
    sharpe_by_period,
    sharpe_diff_test,
    smile,
)
from engine import run_backtest
from panel import Panel
from strategy import (
    ALL_HISTORY,
    EqualRiskPCM,
    EwmaDailyRiskModel,
    EwmaMonthlyRiskModel,
    ExpandingMonthlyRiskModel,
    InverseVolPCM,
    ModularStrategy,
    MonthlyTsmomAlpha,
    SampleCovarianceRiskModel,
    StaticLongAlpha,
    TsmomAlpha,
    VolTargetPCM,
)

RAW_DIR = Path("./data/raw")
ASSET_MAP_PATH = Path("./data/processed/AssetMapCsv.csv")
MONTHLY_CSV_PATH = Path("./data/processed/MonthlyReturns.csv")
SG_TREND_PATH = Path("./data/processed/Lecture3_livedata.xlsx")
FACTOR_DIR = Path("./data/external")
OUTPUT_DIR = Path("./outputs")

# duplicates: ES, SC -> SP and EN -> ND; YM duplicates ZD; bad data: BC, BG, ZH, FF.
# RL stops in 2008, so we keep its mini contract ER instead (dropping both left no Russell 2000 at all).
DROPPED_SYMBOLS = {"EN", "ES", "SC", "BC", "BG", "ZH", "FF", "RL", "YM"}
SAMPLE_START = "1976-02"   # first month the baseline trades


def load_panel() -> Panel:
    return Panel(raw_dirs=RAW_DIR, asset_map_path=ASSET_MAP_PATH, dropped_symbols=DROPPED_SYMBOLS)


def monthly_tsmom(alpha=None, risk_model=None, pcm=None) -> ModularStrategy:
    """Baseline set-up: monthly signal and rebalance, 40% vol per position, 1/N, expanding risk model."""
    return ModularStrategy(
        alpha_model=alpha or MonthlyTsmomAlpha(),
        risk_model=risk_model or ExpandingMonthlyRiskModel(),
        pcm=pcm or EqualRiskPCM(),
        risk_lookback=ALL_HISTORY,
        rebalance="monthly",
    )


# models keep state, so each entry builds a fresh strategy
STRATEGIES = {
    "TSMOM 12m (baseline)": lambda: monthly_tsmom(),
    "Static benchmark": lambda: monthly_tsmom(alpha=StaticLongAlpha()),
    "Vol target 11%": lambda: monthly_tsmom(pcm=VolTargetPCM(EqualRiskPCM(), 0.11)),
    "Vol target 11%, EWMA risk": lambda: monthly_tsmom(risk_model=EwmaMonthlyRiskModel(), pcm=VolTargetPCM(EqualRiskPCM(), 0.11)),
    "Blend 1/3/6/12m": lambda: monthly_tsmom(alpha=MonthlyTsmomAlpha((1, 3, 6, 12))),
    "Blend + EWMA vol target": lambda: monthly_tsmom(alpha=MonthlyTsmomAlpha((1, 3, 6, 12)), risk_model=EwmaMonthlyRiskModel(),
                                                pcm=VolTargetPCM(EqualRiskPCM(), 0.11)),
    "Daily EWMA sizing (MOP 2012)": lambda: monthly_tsmom(risk_model=EwmaDailyRiskModel()),
    "Daily 252d TSMOM, inverse vol": lambda: ModularStrategy(TsmomAlpha(252), SampleCovarianceRiskModel(),
                                                             InverseVolPCM(1.0), risk_lookback=60),
}
BASELINE = "TSMOM 12m (baseline)"
STATIC = "Static benchmark"

# risk models to check against realised risk: (model, ignore correlations?, days of history or None for all)
RISK_MODEL_CHECKS = {
    "Expanding monthly, no correlations (baseline sizing)": (ExpandingMonthlyRiskModel(), True, None),
    "Expanding monthly covariance": (ExpandingMonthlyRiskModel(), False, None),
    "EWMA monthly covariance": (EwmaMonthlyRiskModel(), False, None),
    "EWMA daily vol, no correlations": (EwmaDailyRiskModel(), True, None),
    "Sample covariance, last 60 days": (SampleCovarianceRiskModel(), False, 60),
}


def compare_monthly_returns(
    calculated: pd.DataFrame,
    ground_truth: pd.DataFrame,
    tolerance: float = 1e-4,
) -> None:
    calc = calculated.copy()
    gt = ground_truth.copy()

    # collapse to Period('M') so month-end date labels don't have to line up exactly
    calc.index = calc.index.to_period("M")
    gt.index = gt.index.to_period("M")
    calc = calc[~calc.index.duplicated(keep="last")]
    gt = gt[~gt.index.duplicated(keep="last")]

    calc.columns = [str(c).strip().upper() for c in calc.columns]
    gt.columns = [str(c).strip().upper() for c in gt.columns]

    periods = calc.index.intersection(gt.index)
    symbols = sorted(set(calc.columns) & set(gt.columns))

    print(f"calc: {len(calc)} months x {len(calc.columns)} symbols")
    print(f"gt:   {len(gt)} months x {len(gt.columns)} symbols")
    print(f"overlap: {len(periods)} months x {len(symbols)} symbols")

    if not len(periods) or not symbols:
        print("no overlap between calculated and ground truth")
        return

    if len(gt) > len(calc):
        print("missing month: ", gt.index.difference(calc.index))

    sub_calc = calc.loc[periods, symbols].to_numpy(dtype=np.float64)
    sub_gt = gt.loc[periods, symbols].to_numpy(dtype=np.float64)
    mask = ~np.isnan(sub_calc) & ~np.isnan(sub_gt)
    n = int(mask.sum())
    if n == 0:
        print("no overlapping non-null values")
        return

    diff = np.abs(sub_calc - sub_gt)
    max_diff = float(np.nanmax(diff[mask]))
    mean_diff = float(np.nanmean(diff[mask]))
    corr = float(np.corrcoef(sub_calc[mask], sub_gt[mask])[0, 1])
    matches = int((diff[mask] <= tolerance).sum())

    print(f"match rate (tol={tolerance}): {matches / n:.2%} ({matches}/{n})")
    print(f"max abs diff:  {max_diff:.6e}")
    print(f"mean abs diff: {mean_diff:.6e}")
    print(f"corr: {corr:.6f}")

    if max_diff <= tolerance:
        print("matches MonthlyReturns.csv")
        return

    diff_df = pd.DataFrame(diff, index=periods, columns=symbols)
    top = diff_df.stack().sort_values(ascending=False).head(5)
    print("worst offenders:")
    for (p, s), d in top.items():
        print(
            f"  {p} {s}: calc={calc.loc[p, s]:.6f} gt={gt.loc[p, s]:.6f} diff={d:.6e}"
        )


def monthly_returns_from_bars(panel: Panel) -> pd.DataFrame:
    # uses each symbol's own month-end flag, so this checks Panel's calendar logic
    closes: dict[str, dict[pd.Period, float]] = {}
    for t, bars in panel.timeline:
        for s, bar in bars.items():
            if bar.is_month_end:
                closes.setdefault(s, {})[t.to_period("M")] = bar.close

    prices = pd.DataFrame(closes).sort_index()
    returns = prices / prices.shift(1) - 1.0
    returns.index = returns.index.to_timestamp()
    return returns

def data_analysis(panel: Panel) -> None:
    returns = monthly_returns_from_bars(panel)
    returns.index = returns.index.to_period("M")
    classes = panel.asset_map["AssetClass"].reindex(returns.columns)
    print("\nAsset classes, two definitions (A = average of symbols, B = equal-weighted portfolio):")
    print(asset_class_stats(returns, classes).round(3).to_string())
    print("\nRank autocorrelation t-stats by lag and class:")
    print(autocorrelation_tstats(returns, classes).round(1).to_string())
    print("\nPooled regression t-stats by lag (Moskowitz, Ooi and Pedersen 2012, Figure 1):")
    print(pooled_predictability(returns).round(1).to_frame().T.to_string())


def strategy_table(results: dict, monthly: dict) -> pd.DataFrame:
    """Every strategy from SAMPLE_START, with its Sharpe tested against the baseline."""
    table = performance_table(monthly)
    years = pd.Series({name: len(r) / 12 for name, r in monthly.items()})
    table["Turnover/yr"] = pd.Series({n: r.weights.diff().abs().sum(axis=1).sum() for n, r in results.items()}) / years
    table["Corr(baseline)"] = pd.DataFrame(monthly).corr()[BASELINE]
    table["p(SR = baseline)"] = [sharpe_diff_test(r, monthly[BASELINE])["p"] for r in monthly.values()]
    return table


def benchmark_analysis(panel: Panel, tsmom: pd.Series, static: pd.Series) -> None:
    factors = load_factors(FACTOR_DIR).reindex(tsmom.index)
    t_bond = monthly_returns_from_bars(panel)["US"]
    t_bond.index = t_bond.index.to_period("M")
    benchmarks = {
        "Static benchmark": static.to_frame("Static"),
        "US equity (Mkt-RF)": factors[["Mkt-RF"]],
        "60/40 (Mkt-RF / T-bond fut.)": (0.6 * factors["Mkt-RF"] + 0.4 * t_bond.reindex(tsmom.index)).to_frame("60/40"),
        "Carhart 4-factor": factors[["Mkt-RF", "SMB", "HML", "UMD"]],
        "FF 5-factor + UMD": factors[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]],
    }
    print("\nAlpha of the baseline against each benchmark (monthly, Newey-West):")
    print(alpha_table(tsmom, benchmarks)[["alpha", "t_alpha", "R2"]].round(3).to_string())

    fit = smile(tsmom, factors["Mkt-RF"])
    print(f"\nQuarterly smile: beta on Mkt^2 = {fit['beta_Mkt2']:.2f} (t = {fit['t_Mkt2']:.1f})")
    print("Annualised return by equity market regime:")
    regimes = pd.DataFrame({"TSMOM": tsmom, "Static": static, "Mkt-RF": factors["Mkt-RF"]})
    print(regime_returns(regimes, factors["Mkt-RF"]).round(3).to_string())

    sg = sg_trend_comparison(tsmom, load_sg_trend(SG_TREND_PATH))
    print(f"\nSG Trend, {sg['period']}: corr {sg['corr']:.2f}, SG = {sg['beta']:.2f} x TSMOM, "
          f"alpha {sg['alpha']:.1%} (t = {sg['t_alpha']:.1f}); Sharpe TSMOM {sg['SR TSMOM']:.2f}, SG - RF {sg['SR SG - RF']:.2f}")


def risk_model_check(panel: Panel, results: dict, monthly: dict) -> pd.DataFrame:
    """Predicted vs realised risk of the baseline and the static benchmark under each risk model."""
    weights = results[BASELINE].weights
    month_ends = weights.index.to_series().groupby(weights.index.to_period("M")).max()
    first_decision = weights.abs().sum(axis=1).gt(0).idxmax()
    month_ends = pd.DatetimeIndex(month_ends[month_ends >= first_decision])
    rows = []
    for name, (model, diagonal, lookback) in RISK_MODEL_CHECKS.items():
        covs = forecast_covariances(panel.observed_returns, month_ends, model, lookback)
        for portfolio in (BASELINE, STATIC):
            check = bias_statistic(monthly[portfolio], predicted_vols(results[portfolio], covs, diagonal))
            rows.append({"Risk model": name, "Portfolio": portfolio, **check})
    return pd.DataFrame(rows).set_index(["Portfolio", "Risk model"])


def lookback_study(panel: Panel, monthly: dict) -> None:
    returns = {}
    for k in range(1, 25):
        result = run_backtest(panel, monthly_tsmom(alpha=MonthlyTsmomAlpha((k,))))
        returns[f"{k}m"] = monthly_returns(result).loc["1977-01":]  # common start for every lookback
    returns["Blend 1/3/6/12"] = monthly["Blend 1/3/6/12m"].loc["1977-01":]
    sweep = sharpe_by_period(returns, [("1977", "1989"), ("1990", "1999"), ("2000", "2014")])
    sweep.to_csv(OUTPUT_DIR / "lookback_sweep.csv")
    print("\nSharpe ratio by lookback:")
    print(sweep.round(2).to_string())
    singles = sweep.drop(index="Blend 1/3/6/12")
    print(f"P(true Sharpe > 0) for 12m after trying 24 lookbacks (deflated Sharpe): "
          f"{deflated_sharpe(returns['12m'], singles['Full'].tolist()):.3f}")
    chosen = ["1m", "3m", "6m", "12m", "Blend 1/3/6/12"]
    print("Correlation between lookbacks:")
    print(pd.DataFrame(returns)[chosen].corr().round(2).to_string())


def main():
    panel = load_panel()
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("== Data check against MonthlyReturns.csv")
    gt_df = pd.read_csv(MONTHLY_CSV_PATH, index_col=0, parse_dates=True)
    compare_monthly_returns(monthly_returns_from_bars(panel), gt_df)

    print("\n== Data analysis")
    data_analysis(panel)

    print(f"\n== Strategies, monthly returns from {SAMPLE_START}, no costs")
    results = {name: run_backtest(panel=panel, strategy=make(), cost_bps=0.0, exec_lag=1) for name, make in STRATEGIES.items()}
    monthly = {name: monthly_returns(r).loc[SAMPLE_START:] for name, r in results.items()}
    table = strategy_table(results, monthly)
    table.to_csv(OUTPUT_DIR / "strategy_comparison.csv")
    print(table[["Avg", "Std", "SR", "MaxDD", "Turnover/yr", "Corr(baseline)", "p(SR = baseline)"]].round(3).to_string())

    print("\n== Baseline vs benchmarks")
    benchmark_analysis(panel, monthly[BASELINE], monthly[STATIC])

    print("\n== Risk model check: realised / predicted risk (1 = right, above 1 = risk under-predicted)")
    check = risk_model_check(panel, results, monthly)
    check.to_csv(OUTPUT_DIR / "risk_model_check.csv")
    print(check.round(2).to_string())

    print("\n== Lookbacks")
    lookback_study(panel, monthly)

    analyze_backtest(results[BASELINE])


if __name__ == "__main__":
    main()
