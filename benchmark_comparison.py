import io
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import monthly_returns, ols
from engine import run_backtest
from main import BASELINE, SAMPLE_START, STRATEGIES, load_panel, monthly_returns_from_bars

# Baseline TSMOM vs the static benchmark, both run through the backtest engine.
#   TSMOM  = sign of the past 12 monthly returns x 40% / vol, 1/N
#   STATIC = always long, same instruments, dates and sizing
# Volatility only uses data up to each month end. (The first version of this script used
# full-sample volatility, which looks ahead.) Then: alpha against market and factor
# benchmarks, crisis behaviour, and the SG Trend index of live CTAs.

FACTOR_DIR = Path("./data/external")
SG_TREND_PATH = Path("./data/processed/Lecture3_livedata.xlsx")
FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_FILES = ["F-F_Research_Data_Factors", "F-F_Momentum_Factor", "F-F_Research_Data_5_Factors_2x3"]


def read_french_csv(path: Path) -> pd.DataFrame:
    """Monthly rows of a Ken French factor file, converted from percent to decimals."""
    lines = path.read_text().splitlines()
    header_row = next(i for i, line in enumerate(lines) if line.startswith(","))
    df = pd.read_csv(path, skiprows=header_row, index_col=0)
    df.index = df.index.astype(str).str.strip()
    df = df[df.index.str.len() == 6]  # yyyymm rows only, which drops the annual block
    df.index = pd.PeriodIndex(df.index, freq="M")
    df.columns = df.columns.str.strip()
    return df.astype(float) / 100


def load_factors() -> pd.DataFrame:
    """Fama-French factors plus momentum (UMD). Downloaded from the Ken French library the first time."""
    FACTOR_DIR.mkdir(parents=True, exist_ok=True)
    for name in FRENCH_FILES:
        if not (FACTOR_DIR / f"{name}.csv").exists():
            with urllib.request.urlopen(FRENCH_URL + f"{name}_CSV.zip") as response:
                zipfile.ZipFile(io.BytesIO(response.read())).extractall(FACTOR_DIR)
    ff3 = read_french_csv(FACTOR_DIR / "F-F_Research_Data_Factors.csv")
    umd = read_french_csv(FACTOR_DIR / "F-F_Momentum_Factor.csv").rename(columns={"Mom": "UMD"})
    ff5 = read_french_csv(FACTOR_DIR / "F-F_Research_Data_5_Factors_2x3.csv")[["RMW", "CMA"]]
    return ff3.join(umd).join(ff5)


def load_sg_trend() -> pd.DataFrame:
    """Course file: SG Trend index, Fama-French factors, RF and GSCI, 2000-2014."""
    df = pd.read_excel(SG_TREND_PATH, index_col=0)
    df.index = pd.PeriodIndex(df.index, freq="M")
    return df


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


def to_quarterly(r: pd.Series) -> pd.Series:
    """Sum of monthly returns in each complete calendar quarter."""
    r = r.dropna()
    quarters = r.index.asfreq("Q")
    count = r.groupby(quarters).count()
    return r.groupby(quarters).sum()[count == 3]


def regime_returns(series: pd.DataFrame, market: pd.Series, q: float = 0.2) -> pd.DataFrame:
    """Annualised average return in the market's worst 20%, middle 60% and best 20% of months."""
    m = market.dropna()
    regime = pd.Series("Normal", index=m.index)
    regime[m <= m.quantile(q)] = "Bear"
    regime[m >= m.quantile(1 - q)] = "Bull"
    return (series.reindex(m.index).groupby(regime).mean() * 12).loc[["Bear", "Normal", "Bull"]]


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
    panel = load_panel()
    tsmom = monthly_returns(run_backtest(panel, STRATEGIES[BASELINE]())).loc[SAMPLE_START:]
    static = monthly_returns(run_backtest(panel, STRATEGIES["Static benchmark"]())).loc[SAMPLE_START:]
    rets = pd.DataFrame({"STATIC": static, "TSMOM": tsmom}).dropna()

    print(f"sample: {rets.index[0]} to {rets.index[-1]} ({len(rets)} months)")
    print(summarize(rets).round(3).to_string())

    # alpha against each benchmark, monthly, Newey-West t-stats
    factors = load_factors().reindex(rets.index)
    t_bond = monthly_returns_from_bars(panel)["US"]
    t_bond.index = t_bond.index.to_period("M")
    benchmarks = {
        "Static benchmark": rets[["STATIC"]],
        "US equity (Mkt-RF)": factors[["Mkt-RF"]],
        "60/40 (Mkt-RF / T-bond fut.)": (0.6 * factors["Mkt-RF"] + 0.4 * t_bond.reindex(rets.index)).to_frame("60/40"),
        "Carhart 4-factor": factors[["Mkt-RF", "SMB", "HML", "UMD"]],
        "FF 5-factor + UMD": factors[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]],
    }
    alphas = pd.DataFrame({name: ols(rets["TSMOM"], X) for name, X in benchmarks.items()}).T
    print("\nregression: TSMOM = alpha + beta' benchmark + e (monthly, Newey-West)")
    print(alphas[["alpha", "t_alpha", "R2"]].round(3).to_string())

    # crisis behaviour: quarterly smile (MOP 2012, Table 3) and returns by market regime
    q = pd.DataFrame({"s": to_quarterly(rets["TSMOM"]), "m": to_quarterly(factors["Mkt-RF"])}).dropna()
    smile = ols(q["s"], pd.DataFrame({"Mkt": q["m"], "Mkt2": q["m"] ** 2}), nw_lags=2, periods=4)
    print(f"\nquarterly smile: beta(Mkt^2) = {smile['beta_Mkt2']:.2f} (t = {smile['t_Mkt2']:.1f})")
    print(regime_returns(rets.join(factors["Mkt-RF"]), factors["Mkt-RF"]).round(3).to_string())

    # SG Trend index, 2000-2014. SG is net of fees and earns cash interest, so compare SG - RF.
    sg = load_sg_trend()
    sg_excess = (sg["SG Trend"] - sg["RF"]).rename("SG Trend - RF")
    both = pd.concat([rets["TSMOM"], sg_excess], axis=1).dropna()
    fit = ols(both["SG Trend - RF"], both[["TSMOM"]])
    print(f"\nSG Trend {both.index[0]} to {both.index[-1]}: corr with TSMOM {both.corr().iloc[0, 1]:.2f}, "
          f"SG = {fit['beta_TSMOM']:.2f} x TSMOM, alpha {fit['alpha']:.1%} (t = {fit['t_alpha']:.1f})")
    print(f"Sharpe: TSMOM {both['TSMOM'].mean() / both['TSMOM'].std() * np.sqrt(12):.2f}, "
          f"SG - RF {sg_excess.mean() / sg_excess.std() * np.sqrt(12):.2f}")

    cumulative = (1 + rets[["TSMOM", "STATIC"]]).cumprod() - 1
    # cumulative = rets[["TSMOM", "STATIC"]].cumsum() # additive cumulation, as in the lecture
    plot_path = Path("./plots/tsmom_vs_static.png")
    plot_cumulative_returns(cumulative, plot_path)
    print(f"\nsaved cumulative return plot to {plot_path}")


if __name__ == "__main__":
    main()
