from pathlib import Path

import numpy as np

from engine import run_backtest
from panel import Panel
from strategy import TimeSeriesMomentumStrategy


def summarize_performance(res) -> None:
    pnl = res.net_pnl.dropna()
    if pnl.empty:
        print("No PnL recorded.")
        return

    ann_ret = pnl.mean() * 261.0
    ann_vol = pnl.std() * np.sqrt(261.0)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-6 else np.nan

    nav = res.nav
    max_dd = float(((nav - nav.cummax()) / nav.cummax()).min())

    print("\n=== Time Series Momentum (TSMOM) Performance ===")
    print(f"Period:            {pnl.index[0].date()} to {pnl.index[-1].date()}")
    print(f"Annualized Return: {ann_ret:.2%}")
    print(f"Annualized Vol:    {ann_vol:.2%}")
    print(f"Sharpe Ratio:      {sharpe:.2f}")
    print(f"Max Drawdown:      {max_dd:.2%}")
    print("================================================\n")


def main():
    raw_dir = Path("./data/raw")
    asset_map_path = Path("./data/processed/AssetMapCsv.csv")

    panel = Panel(
        raw_dirs=raw_dir,
        asset_map_path=asset_map_path,
        dropped_symbols={"EN", "ES", "ER", "SC", "BC", "BG", "ZH", "FF"},
    )

    strategy = TimeSeriesMomentumStrategy(target_vol=0.40, normalize_by_count=True)
    res = run_backtest(panel=panel, strategy=strategy, cost_bps=0.0, exec_lag=1)

    summarize_performance(res)


if __name__ == "__main__":
    main()
