import matplotlib.pyplot as plt
import pandas as pd

from engine import BacktestResult


def performance_metrics(
    result: BacktestResult,
    periods_per_year: int = 252,
) -> dict[str, float]:
    returns = result.net_pnl.dropna()

    annual_return = returns.mean() * periods_per_year
    annual_volatility = returns.std() * periods_per_year**0.5

    if annual_volatility > 0:
        sharpe = annual_return / annual_volatility
    else:
        sharpe = float("nan")

    nav = result.nav.dropna()

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
