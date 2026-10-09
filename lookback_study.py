from pathlib import Path

import pandas as pd

from analysis import deflated_sharpe, monthly_returns, performance_table
from engine import run_backtest
from main import load_panel, monthly_tsmom
from strategy import MonthlyTsmomAlpha

# Was 12 months cherry-picked, and does combining lookbacks help?
# Every lookback from 1 to 24 months, over the full sample and three sub-periods, from a common
# start in 1977. The deflated Sharpe ratio asks whether the 12-month result survives the fact
# that we looked at 24 lookbacks.

START = "1977-01"
PERIODS = [("1977", "1989"), ("1990", "1999"), ("2000", "2014")]


def sharpe(r: pd.Series) -> float:
    return r.mean() / r.std() * 12**0.5


def main():
    panel = load_panel()
    returns = {}
    for k in range(1, 25):
        result = run_backtest(panel, monthly_tsmom(alpha=MonthlyTsmomAlpha((k,))))
        returns[k] = monthly_returns(result).loc[START:]
    blend = run_backtest(panel, monthly_tsmom(alpha=MonthlyTsmomAlpha((1, 3, 6, 12))))
    returns["Blend 1/3/6/12"] = monthly_returns(blend).loc[START:]

    sweep = pd.DataFrame({k: {"Full": sharpe(r), **{f"{a}-{b}": sharpe(r.loc[a:b]) for a, b in PERIODS}}
                          for k, r in returns.items()}).T
    singles = sweep.drop(index="Blend 1/3/6/12")
    print("Sharpe ratio by lookback (months):")
    print(sweep.round(2).to_string())
    print(f"\nbest lookback: {singles['Full'].idxmax()} months; lowest Sharpe in any sub-period: {singles.iloc[:, 1:].min().min():.2f}")
    print(f"deflated Sharpe of 12m, best of 24: P(true Sharpe > 0) = {deflated_sharpe(returns[12], singles['Full'].tolist()):.4f}")

    chosen = {f"{k}m": returns[k] for k in (1, 3, 6, 12)}
    chosen["Blend 1/3/6/12"] = returns["Blend 1/3/6/12"]
    print("\ncorrelation between lookbacks:")
    print(pd.DataFrame(chosen).corr().round(2).to_string())
    print()
    print(performance_table(chosen)[["Avg", "Std", "SR", "MaxDD"]].round(3).to_string())

    out = Path("./outputs")
    out.mkdir(exist_ok=True)
    sweep.to_csv(out / "lookback_sweep.csv")


if __name__ == "__main__":
    main()
