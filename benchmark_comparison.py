from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Lecture 5 baseline: TSMOM vs static benchmark on monthly returns.
# Vol-adjust each instrument to 40% annualized using full-sample vol (as in the lecture), then compare:
#   TSMOM  = sign of past 12-month return x next month's return
#   STATIC = always long
# both equal-weighted across instruments.

TARGET_VOL = 0.40
LOOKBACK = 12
MIN_INSTRUMENTS = 10


def load_monthly_returns(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = df.index.to_period("M")
    return df


def build_strategy_returns(returns: pd.DataFrame) -> pd.DataFrame:
    scale = TARGET_VOL / (returns.std() * 12**0.5)
    scaled = returns * scale

    # signal known at end of month t-1, earns month t
    past = returns.rolling(LOOKBACK, min_periods=LOOKBACK).sum()
    position = np.sign(past).shift(1)

    # TSMOM: only instruments with both a position and a return
    live = position.notna() & scaled.notna()
    out = pd.DataFrame(
        {
            # "STATIC": scaled.where(live).mean(axis=1), # only instruments with a signal
            "STATIC": scaled.mean(axis=1),
            "TSMOM": (position * scaled).where(live).mean(axis=1),
        }
    )
    return out[live.sum(axis=1) >= MIN_INSTRUMENTS]


def summarize(rets: pd.DataFrame) -> pd.DataFrame:
    ann_ret = rets.mean() * 12
    ann_vol = rets.std() * 12**0.5
    return pd.DataFrame(
        {
            "Avg": ann_ret,
            "Std": ann_vol,
            "SR": ann_ret / ann_vol,
            "Corr(Static)": rets.corr()["STATIC"],
        }
    )


def alpha_regression(rets: pd.DataFrame) -> dict[str, float]:
    # OLS of TSMOM on STATIC: TSMOM_t = alpha + beta * STATIC_t + e_t
    x = rets["STATIC"].to_numpy()
    y = rets["TSMOM"].to_numpy()
    n = len(x)

    # manually computed; can use scipy or statsmodel for conducting regression
    x_dm = x - x.mean()
    beta = (x_dm * (y - y.mean())).sum() / (x_dm**2).sum()
    alpha = y.mean() - beta * x.mean()
    resid = y - alpha - beta * x

    sigma2 = (resid**2).sum() / (n - 2)
    se_beta = (sigma2 / (x_dm**2).sum()) ** 0.5
    se_alpha = (sigma2 * (1.0 / n + x.mean() ** 2 / (x_dm**2).sum())) ** 0.5
    return {
        "alpha (ann.)": alpha * 12,
        "alpha t-stat": alpha / se_alpha,
        "beta": beta,
        "beta t-stat": beta / se_beta,
        "R^2": 1.0 - (resid**2).sum() / ((y - y.mean()) ** 2).sum(),
        "resid vol (ann.)": resid.std(ddof=2) * 12**0.5,
    }


def plot_cumulative_returns(cumulative: pd.DataFrame, out_path: Path):
    colors = {"TSMOM": "#5b9bd5", "STATIC": "#ed7d31"}
    ts = cumulative.to_timestamp(how="end")

    fig, ax = plt.subplots(figsize=(10, 6))
    for name in ts.columns:
        ax.plot(ts.index, ts[name], color=colors.get(name), linewidth=2, label=name)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Cumulative Return of TSMOM vs Static Bmk", fontweight="bold")
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=len(ts.columns),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    returns = load_monthly_returns(Path("./data/processed/MonthlyReturns.csv"))
    rets = build_strategy_returns(returns)

    print(f"sample: {rets.index[0]} to {rets.index[-1]} ({len(rets)} months)")
    print(
        summarize(rets).to_string(
            formatters={
                "Avg": "{:.2%}".format,
                "Std": "{:.2%}".format,
                "SR": "{:.2f}".format,
                "Corr(Static)": "{:.2f}".format,
            }
        )
    )

    reg = alpha_regression(rets)
    print("\nregression: TSMOM = alpha + beta * STATIC + e (monthly)")
    print(f"  alpha (ann.):  {reg['alpha (ann.)']:.2%}  (t = {reg['alpha t-stat']:.2f})")
    print(f"  beta:          {reg['beta']:.3f}  (t = {reg['beta t-stat']:.2f})")
    print(f"  R^2:           {reg['R^2']:.3f}")
    print(f"  resid vol (ann.): {reg['resid vol (ann.)']:.2%}")

    cumulative = (1 + rets[["TSMOM", "STATIC"]]).cumprod() - 1
    # cumulative = rets[["TSMOM", "STATIC"]].cumsum() # additive cumulation, as in the lecture
    plot_path = Path("./plots/tsmom_vs_static.png")
    plot_cumulative_returns(cumulative, plot_path)
    print(f"\nsaved cumulative return plot to {plot_path}")


if __name__ == "__main__":
    main()
