from pathlib import Path

import numpy as np
import pandas as pd

from config import ASSET_MAP_PATH, DROPPED_SYMBOLS, FULL_RAW_DIR, ROLL_FIXES, SUCCESSORS
from panel import Panel

XLSX_PATH = Path("./data/processed/MonthlyReturns_FullSample.xlsx")

sheets = pd.read_excel(XLSX_PATH, sheet_name=None, index_col=0)
print("sheets:", {name: df.shape for name, df in sheets.items()})
returns = next(iter(sheets.values()))
returns.index = pd.to_datetime(returns.index).to_period("M")
returns = returns.apply(pd.to_numeric, errors="coerce")
print(returns.iloc[:3, :6])
print(returns.iloc[-3:, :6])
print(f"{returns.shape[1]} columns | {returns.index[0]} to {returns.index[-1]}")
print(
    f"max |value| = {returns.abs().max().max():.3f}  (far above 1 means percent, not decimal)"
)

asset_map = pd.read_csv(ASSET_MAP_PATH, index_col="ID", keep_default_na=False)
cols = set(returns.columns)
print("in file but not in asset map:", sorted(cols - set(asset_map.index)))
print("in asset map but not in file:", sorted(set(asset_map.index) - cols))
print("dropped by us but in file:", sorted(cols & DROPPED_SYMBOLS))
span = pd.DataFrame(
    {
        "first": returns.apply(pd.Series.first_valid_index),
        "last": returns.apply(pd.Series.last_valid_index),
    }
)
print("columns that end before the last month:")
print(span[span["last"] < returns.index[-1]].to_string())

panel = Panel(FULL_RAW_DIR, ASSET_MAP_PATH, DROPPED_SYMBOLS, SUCCESSORS, ROLL_FIXES)
px = panel.close_df.groupby(panel.close_df.index.to_period("M")).last()
ours = px.pct_change(fill_method=None).reindex(returns.index)

common = [s for s in ours.columns if s in returns.columns]
diff = (returns[common] - ours[common]).abs()
check = pd.DataFrame(
    {
        "corr": pd.Series({s: returns[s].corr(ours[s]) for s in common}),
        "max_abs_diff": diff.max(),
        "months_diff_gt_1e-3": (diff > 1e-3).sum(),
    }
)
print(
    f"\n== Teacher's monthly returns vs ours ({len(common)} common symbols), worst 15 =="
)
print(check.sort_values("max_abs_diff", ascending=False).head(15).round(4).to_string())

universe = sorted((set(asset_map.index) - DROPPED_SYMBOLS) & cols)
minis = sorted(
    (
        ((set(asset_map.index) - DROPPED_SYMBOLS) - set(SUCCESSORS))
        | set(SUCCESSORS.values())
    )
    & cols
)
variants = {
    "all columns": list(returns.columns),
    "our universe": universe,
    "our universe, minis instead of bigs": minis,
}
windows = {
    "IS": (None, "2014-12"),
    "OOS": ("2015-01", None),
    "15-19": ("2015-01", "2019-12"),
    "15-20": ("2015-01", "2020-12"),
    "15-22": ("2015-01", "2022-12"),
    "15-24": ("2015-01", "2024-12"),
    "20-22": ("2020-01", "2022-12"),
}

table = {}
for vname, variant_cols in variants.items():
    r = returns[variant_cols]
    position = np.sign(r.rolling(12, min_periods=12).sum()).shift(1)
    print(
        f"{vname}: {len(variant_cols)} instruments, avg live in OOS = "
        f"{(position.notna() & r.notna()).loc['2015-01':].sum(axis=1).mean():.1f}"
    )
    vols = {
        "full-sample vol (lecture, look-ahead)": r.std(),
        "IS-only vol": r.loc[:"2014-12"].std(),
        "expanding vol (causal)": r.expanding(36).std().shift(1),
    }
    for volname, vol in vols.items():
        scaled = r * (0.40 / (vol * 12**0.5))
        live = position.notna() & scaled.notna()
        keep = live.sum(axis=1) >= 10
        series = {
            "TSMOM": (position * scaled).where(live).mean(axis=1)[keep],
            "STATIC": scaled.mean(axis=1)[keep],
        }
        for sname, s in series.items():
            table[(vname, volname, sname)] = {
                w: s.loc[a:b].mean() / s.loc[a:b].std() * np.sqrt(12)
                for w, (a, b) in windows.items()
            }

print("\n== Monthly gross Sharpe, lecture method ==")
print(pd.DataFrame(table).T.round(2).to_string())
