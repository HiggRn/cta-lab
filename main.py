import pandas as pd

from analysis import (
    alpha_table,
    correlation_structure,
    lag_profile,
    load_benchmarks,
    monthly_returns,
    plot_strategies,
    regime_returns,
    sg_trend_comparison,
    smile_regression,
    strategy_summary_table,
)
from baselines import BASELINE_STRATEGIES
from config import (
    ASSET_MAP_PATH,
    DROPPED_SYMBOLS,
    FACTOR_DIR,
    FULL_RAW_DIR,
    OUTPUT_DIR,
    ROLL_FIXES,
    SAMPLE_END,
    SAMPLE_START,
    SG_TREND_PATH,
    SUCCESSORS,
)
from engine import FuturesCostModel, run_backtest
from panel import Panel
from strategy import USER_STRATEGIES

BASELINE_NAME = "TSMOM 12m (baseline)"
STATIC_NAME = "Static benchmark"

OOS_START = "2015-01"
MIN_REG_MONTHS = 24  # fewer months than this make the factor regressions meaningless


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    panel = Panel(
        raw_dir=FULL_RAW_DIR,
        asset_map_path=ASSET_MAP_PATH,
        dropped_symbols=DROPPED_SYMBOLS,
        successors=SUCCESSORS,
        roll_fixes=ROLL_FIXES,
    )

    cost_model = FuturesCostModel(
        trading_cost_bps=2,
        roll_cost_bps=2,
        rolls_per_year=4.0,
        financing_cost_bps=50.0,
        free_leverage_threshold=1.0,
    )

    # Combine course baselines with user custom strategies
    all_strategies = {**BASELINE_STRATEGIES, **USER_STRATEGIES}
    results = {
        name: run_backtest(
            panel=panel, strategy=cls(), cost_model=cost_model, exec_lag=2
        )
        for name, cls in all_strategies.items()
    }

    # Common comparison window: the latest first full month across all strategies
    common_start = max(
        [pd.Period(SAMPLE_START, "M")]
        + [monthly_returns(res).index[0] for res in results.values()]
    )
    START = str(common_start)

    # In-sample and out-of-sample come from the same full-sample run: signals, risk models and
    # positions carry straight across 2014/2015. Only the evaluation window differs.
    periods = {
        "In-Sample": ("is", START, SAMPLE_END, True),
        "Out-of-Sample": ("oos", OOS_START, None, False),
    }
    target_name = "Trend profile"
    months = [(a, a + 21) for a in range(0, 378, 21)]

    for label, (tag, start, end, run_sg) in periods.items():
        monthly = {
            name: monthly_returns(res).loc[start:end] for name, res in results.items()
        }
        stop = end or str(monthly[BASELINE_NAME].index[-1])
        print(
            f"\n==================== {label} ({start} to {stop}) ===================="
        )

        # Strategy Performance & Cost Breakdown Table
        summary = strategy_summary_table(
            results, baseline_name=BASELINE_NAME, sample_start=start, sample_end=end
        )
        summary.to_csv(OUTPUT_DIR / f"strategy_comparison_{tag}.csv")
        print(summary.round(3).to_string())

        # Which past months predict returns
        profile = lag_profile(panel, months, start=start, end=end)
        profile.to_csv(OUTPUT_DIR / f"lag_profile_{tag}.csv")
        print(
            "\n== Lag Profile: gross Sharpe of trading each past month alone (bars ago) =="
        )
        print(profile.round(2).to_string())

        selected = {
            BASELINE_NAME: monthly[BASELINE_NAME],
            STATIC_NAME: monthly[STATIC_NAME],
            **{k: monthly[k] for k in USER_STRATEGIES},
        }
        target_ret = monthly[target_name]

        if len(target_ret) < MIN_REG_MONTHS:
            print(f"\nOnly {len(target_ret)} months: skipping the factor regressions")
        else:
            # Factor & Benchmark Alpha Attribution
            benchmarks, factors = load_benchmarks(
                panel, monthly[STATIC_NAME], FACTOR_DIR
            )
            print(f"\n== Alpha Attribution for [{target_name}] ==")
            print(
                alpha_table(target_ret, benchmarks)[["alpha", "t_alpha", "R2"]]
                .round(3)
                .to_string()
            )

            # Crisis Alpha: Market Smile & Regime Returns
            fit_smile = smile_regression(target_ret, factors["Mkt-RF"])
            print(
                f"\n== Market Smile for [{target_name}]: beta(Mkt^2) = {fit_smile['beta_Mkt2']:.2f} "
                f"(t = {fit_smile['t_Mkt2']:.2f}) =="
            )
            print("\n== Annualized Return by Equity Market Regime ==")
            print(regime_returns(selected, factors["Mkt-RF"]).round(3).to_string())

        # Within vs. Across Asset Class Correlation Structure
        print(f"\n== Correlation Structure [{target_name} vs. Passive] ==")
        print(
            correlation_structure(panel, results[target_name], start=start, end=end)
            .round(3)
            .to_string()
        )

        # Live SG Trend Index Comparison (its data ends in 2014, so in-sample only)
        if run_sg:
            sg = sg_trend_comparison(target_ret, SG_TREND_PATH)
            print(
                f"\n== SG Trend Comparison ({sg['period']}) ==\n"
                f"Corr: {sg['corr']:.2f} | Beta: {sg['beta']:.2f} | Alpha: {sg['alpha']:.1%} (t={sg['t_alpha']:.1f}) | "
                f"SR({target_name}): {sg['SR Strategy']:.2f} vs SR(SG-RF): {sg['SR SG-RF']:.2f}"
            )

        # Plot Equity & Drawdown Curves
        plot_strategies(
            {k: results[k] for k in selected},
            OUTPUT_DIR,
            tag,
            sample_start=start,
            sample_end=end,
        )


if __name__ == "__main__":
    main()
