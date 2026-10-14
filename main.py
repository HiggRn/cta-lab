from pathlib import Path

import pandas as pd

from analysis import (
    analyze_backtest,
    monthly_returns,
    performance_table,
    risk_model_check,
    sharpe_diff_test,
)
from engine import run_backtest
from panel import Panel
from strategy import (
    ALL_HISTORY,
    EqualRiskPCM,
    ExpandingMonthlyRiskModel,
    ModularStrategy,
    MonthlyTsmomAlpha,
    StaticLongAlpha,
    TrendAlpha,
    TwoTierHrpPCM,
    YangZhangRiskModel,
)

RAW_DIR = Path("./data/raw")
ASSET_MAP_PATH = Path("./data/processed/AssetMapCsv.csv")
OUTPUT_DIR = Path("./outputs")

DROPPED_SYMBOLS = {"EN", "ES", "SC", "BC", "BG", "ZH", "FF", "RL", "YM"}
SAMPLE_START = "1976-02"
BASELINE = "TSMOM 12m (baseline)"
STATIC = "Static benchmark"


def load_panel() -> Panel:
    return Panel(
        raw_dirs=RAW_DIR, asset_map_path=ASSET_MAP_PATH, dropped_symbols=DROPPED_SYMBOLS
    )


def build_strategies(panel: Panel) -> dict[str, ModularStrategy]:
    classes = panel.asset_map["AssetClass"]
    return {
        BASELINE: ModularStrategy(
            alpha_model=MonthlyTsmomAlpha((12,)),
            risk_model=ExpandingMonthlyRiskModel(),
            pcm=EqualRiskPCM(),
            risk_lookback=ALL_HISTORY,
            rebalance="monthly",
        ),
        STATIC: ModularStrategy(
            alpha_model=StaticLongAlpha(),
            risk_model=ExpandingMonthlyRiskModel(),
            pcm=EqualRiskPCM(),
            risk_lookback=ALL_HISTORY,
            rebalance="monthly",
        ),
        "Daily Trend + YZ-NW + HRP": ModularStrategy(
            alpha_model=TrendAlpha(),
            risk_model=YangZhangRiskModel(asset_classes=classes),
            pcm=TwoTierHrpPCM(asset_classes=classes, target_vol=0.11),
            risk_lookback=252,
            rebalance="daily",
        ),
    }


def strategy_table(results: dict, monthly: dict) -> pd.DataFrame:
    table = performance_table(monthly)
    years = pd.Series({name: len(r) / 12 for name, r in monthly.items()})
    table["Turnover/yr"] = (
        pd.Series(
            {n: r.weights.diff().abs().sum(axis=1).sum() for n, r in results.items()}
        )
        / years
    )
    table["Corr(baseline)"] = pd.DataFrame(monthly).corr()[BASELINE]
    table["p(SR = baseline)"] = [
        sharpe_diff_test(r, monthly[BASELINE])["p"] for r in monthly.values()
    ]
    return table


def main() -> None:
    panel = load_panel()
    OUTPUT_DIR.mkdir(exist_ok=True)
    classes = panel.asset_map["AssetClass"]

    strategies = build_strategies(panel)
    results = {
        name: run_backtest(panel=panel, strategy=strat, cost_bps=0.0, exec_lag=2)
        for name, strat in strategies.items()
    }
    monthly = {
        name: monthly_returns(r).loc[SAMPLE_START:] for name, r in results.items()
    }

    print(f"\n== Strategy Comparison (from {SAMPLE_START}, no costs)")
    table = strategy_table(results, monthly)
    table.to_csv(OUTPUT_DIR / "strategy_comparison.csv")
    print(
        table[
            [
                "Avg",
                "Std",
                "SR",
                "MaxDD",
                "Turnover/yr",
                "Corr(baseline)",
                "p(SR = baseline)",
            ]
        ]
        .round(3)
        .to_string()
    )

    print("\n== Risk Model Check: realised / predicted risk (1 = ideal)")
    risk_models = {
        "Expanding monthly (baseline)": (ExpandingMonthlyRiskModel(), False, None),
        "Yang-Zhang + NW Shrinkage (252d)": (
            YangZhangRiskModel(asset_classes=classes),
            False,
            252,
        ),
    }
    check = risk_model_check(panel.observed_returns, results, monthly, risk_models)
    check.to_csv(OUTPUT_DIR / "risk_model_check.csv")
    print(check.round(2).to_string())

    analyze_backtest(results["Daily Trend + YZ-NW + HRP"])


if __name__ == "__main__":
    main()
