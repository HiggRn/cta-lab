from pathlib import Path

import numpy as np
import pandas as pd

from analysis import analyze_backtest, monthly_returns, performance_table, sharpe_diff_test
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

def compare_strategies(panel: Panel, cost_bps: float = 0.0) -> pd.DataFrame:
    """Every strategy in STRATEGIES from SAMPLE_START, with its Sharpe tested against the baseline."""
    monthly, turnover = {}, {}
    for name, make in STRATEGIES.items():
        result = run_backtest(panel=panel, strategy=make(), cost_bps=cost_bps, exec_lag=1)
        monthly[name] = monthly_returns(result).loc[SAMPLE_START:]
        years = len(monthly[name]) / 12
        turnover[name] = result.weights.diff().abs().sum(axis=1).sum() / years
    table = performance_table(monthly)
    table["Turnover/yr"] = pd.Series(turnover)
    table["Corr(baseline)"] = pd.DataFrame(monthly).corr()[BASELINE]
    table["p(SR = baseline)"] = [sharpe_diff_test(r, monthly[BASELINE])["p"] for r in monthly.values()]
    return table


def main():
    panel = load_panel()

    gt_df = pd.read_csv(MONTHLY_CSV_PATH, index_col=0, parse_dates=True)
    compare_monthly_returns(monthly_returns_from_bars(panel), gt_df)

    table = compare_strategies(panel)
    OUTPUT_DIR.mkdir(exist_ok=True)
    table.to_csv(OUTPUT_DIR / "strategy_comparison.csv")
    print(f"\nStrategies, monthly returns from {SAMPLE_START} (no costs):")
    print(table[["Avg", "Std", "SR", "MaxDD", "Turnover/yr", "Corr(baseline)", "p(SR = baseline)"]].round(3).to_string())

    result = run_backtest(panel=panel, strategy=STRATEGIES[BASELINE](), cost_bps=0.0, exec_lag=1)
    analyze_backtest(result)


if __name__ == "__main__":
    main()
