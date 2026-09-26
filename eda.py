from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

### Load monthly returns data ###
def load_monthly_returns(path: Path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = df.index.to_period("M")
    return df


### 1. Outlier detection and visualization of monthly returns ###

# report statistics and flag outliers in monthly returns, for sanity check
def summarize(returns: pd.DataFrame, asset_map: pd.DataFrame):
    stats = pd.DataFrame(
        {
            "asset_class": asset_map["AssetClass"].reindex(returns.columns),
            "n": returns.count(),
            "mean": returns.mean(),
            "std": returns.std(),
            "min": returns.min(),
            "max": returns.max(),
            "skew": returns.skew(),
            "kurt": returns.kurt(),
        }
    )
    return stats

# Can be refined further, but I think this is good enough for EDA
def flag_outliers(returns: pd.DataFrame, z_thresh: float = 3.0) -> pd.Series:
    z = (returns - returns.mean()) / returns.std()
    flagged = returns[z.abs() > z_thresh].stack()
    flagged.index.names = ["month", "symbol"]
    return flagged.sort_values(key=np.abs, ascending=False)


# Box plots of monthly returns by asset class
def plot_returns_by_asset_class(
    returns: pd.DataFrame, asset_map: pd.DataFrame, out_path: Path
):
    # pool every symbol-month return within each asset class
    classes = asset_map["AssetClass"].reindex(returns.columns)
    groups = {
        cls: returns.loc[:, classes == cls].stack().to_numpy()
        for cls in sorted(classes.dropna().unique())
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(
        groups.values(),
        tick_labels=[f"{cls}\n(n={len(v)})" for cls, v in groups.items()],
        widths=0.5,
        patch_artist=True,
        boxprops={"facecolor": "#c6d7ee", "edgecolor": "#2b5c9e", "linewidth": 1.2},
        medianprops={"color": "#1b3a66", "linewidth": 2},
        whiskerprops={"color": "#2b5c9e", "linewidth": 1.2},
        capprops={"color": "#2b5c9e", "linewidth": 1.2},
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": "none",
            "markeredgecolor": "#8a8a8a",
            "alpha": 0.6,
        },
    )
    ax.axhline(0, color="#8a8a8a", linewidth=0.8, zorder=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_ylabel("Monthly return")
    ax.set_title("Monthly return distribution by asset class", loc="left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)



### 2. Compute and visualize cumulative returns by asset class over time ###

def cumulative_returns_by_asset_class(returns: pd.DataFrame, asset_map: pd.DataFrame, min_instruments: int = 5):
    # equal-weight the symbols available in each class each month, then cumulate additively
    # (as in the lecture); each class starts once it first has min_instruments with data
    classes = asset_map["AssetClass"].reindex(returns.columns)
    class_returns = returns.T.groupby(classes).mean().T
    counts = returns.notna().T.groupby(classes).sum().T
    started = (counts >= min_instruments).cummax()
    return class_returns.where(started).cumsum()


def plot_cumulative_returns(cumulative: pd.DataFrame, out_path: Path):
    colors = {"COMM": "#5b9bd5", "EQ": "#ed7d31", "FI": "#a5a5a5", "FX": "#ffc000"}
    ts = cumulative.to_timestamp(how="end")

    fig, ax = plt.subplots(figsize=(10, 6))
    for cls in ts.columns:
        series = ts[cls].dropna()
        ax.plot(series.index, series, color=colors.get(cls), linewidth=2, label=cls)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Cumulative Asset Class Returns", fontweight="bold")
    ax.legend(
        frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(ts.columns)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)



### 3. Autocorrelation across horizons ###

def rank_acf(x: pd.Series, lag: int):
    # Spearman correlation of r_t with r_{t-lag}, on months where both are present
    pair = pd.concat([x, x.shift(lag)], axis=1).dropna()
    return pair.rank().corr().iloc[0, 1]


def acf_by_asset_class(returns: pd.DataFrame, asset_map: pd.DataFrame, max_lag: int = 12):
    # per-symbol rank autocorrelation at each lag, then equal-weight average within class
    lags = range(1, max_lag + 1)
    per_symbol = pd.DataFrame(
        {lag: {s: rank_acf(returns[s], lag) for s in returns.columns} for lag in lags}
    )
    classes = asset_map["AssetClass"].reindex(per_symbol.index)
    by_class = per_symbol.groupby(classes).mean().T
    by_class.insert(0, "ALL", per_symbol.mean())
    by_class.index.name = "lag"
    return by_class


def plot_acf(acf: pd.DataFrame, out_path: Path):
    colors = {
        "ALL": "#5b9bd5",
        "COMM": "#ed7d31",
        "EQ": "#a5a5a5",
        "FI": "#ffc000",
        "FX": "#4472c4",
    }
    n = len(acf.columns)
    width = 0.8 / n
    x = np.arange(len(acf.index))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, col in enumerate(acf.columns):
        ax.bar(x + (i - (n - 1) / 2) * width, acf[col], width, color=colors.get(col), label=col)
    ax.set_xticks(x, acf.index)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.4f}"))
    ax.axhline(0, color="#d9d9d9", linewidth=0.8)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel("Lag (months)")
    ax.set_title(
        "Avg Autocorrelation (by Asset Class) for different horizons", fontweight="bold"
    )
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=n)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)




def main():
    returns = load_monthly_returns(Path("./data/processed/MonthlyReturns.csv"))
    asset_map = pd.read_csv(Path("./data/processed/AssetMapCsv.csv"), index_col="ID")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)

    stats = summarize(returns, asset_map)
    print("per-symbol summary (sorted by kurtosis):")
    print(stats.sort_values("kurt", ascending=False).round(4))

    flagged = flag_outliers(returns)
    print(f"\nflagged observations: {len(flagged)} of {int(returns.count().sum())}")
    print(flagged.head(20).round(4))

    print("\nflagged count by symbol:")
    print(flagged.groupby(level="symbol").size().sort_values(ascending=False).head(10))

    plot_path = Path("./plots/returns_by_asset_class.png")
    plot_returns_by_asset_class(returns, asset_map, plot_path)
    print(f"\nsaved boxplot to {plot_path}")

    cumulative = cumulative_returns_by_asset_class(returns, asset_map)
    print("\ncumulative return by asset class (final month):")
    print(cumulative.iloc[-1].map("{:.0%}".format))

    plot_path = Path("./plots/cumulative_returns_by_asset_class.png")
    plot_cumulative_returns(cumulative, plot_path)
    print(f"\nsaved cumulative return plot to {plot_path}")

    acf = acf_by_asset_class(returns, asset_map)
    print("\navg rank autocorrelation by asset class:")
    print(acf.round(4))

    plot_path = Path("./plots/autocorr_by_asset_class.png")
    plot_acf(acf, plot_path)
    print(f"\nsaved autocorrelation plot to {plot_path}")

if __name__ == "__main__":
    main()