import re

import numpy as np
import pandas as pd

from analysis import monthly_returns
from baselines import BaselineTsmomStrategy
from config import (
    ASSET_MAP_PATH,
    DROPPED_SYMBOLS,
    FULL_RAW_DIR,
    OUTPUT_DIR,
    ROLL_FIXES,
    SUCCESSORS,
)
from engine import ZeroCostModel, run_backtest
from panel import Panel

OUTPUT_DIR.mkdir(exist_ok=True)

# 1. panels that differ in one assumption each
asset_map = pd.read_csv(ASSET_MAP_PATH, index_col="ID", keep_default_na=False)
files = {
    re.sub("_RAD$", "", f.stem, flags=re.I)
    for f in FULL_RAW_DIR.iterdir()
    if f.suffix.lower() == ".csv"
}
extra = sorted(files - set(asset_map.index))
extended = pd.concat(
    [
        asset_map,
        pd.DataFrame(
            {c: ("OTHER" if c == "AssetClass" else "") for c in asset_map.columns},
            index=pd.Index(extra, name="ID"),
        ),
    ]
)
extended_path = OUTPUT_DIR / "asset_map_extended.csv"
extended.to_csv(extended_path)

panels = {
    "current": Panel(
        FULL_RAW_DIR, ASSET_MAP_PATH, DROPPED_SYMBOLS, SUCCESSORS, ROLL_FIXES
    ),
    "no chain": Panel(FULL_RAW_DIR, ASSET_MAP_PATH, DROPPED_SYMBOLS, None, ROLL_FIXES),
    "with AP": Panel(
        FULL_RAW_DIR, ASSET_MAP_PATH, DROPPED_SYMBOLS - {"AP"}, SUCCESSORS, ROLL_FIXES
    ),
    "no LB fix": Panel(FULL_RAW_DIR, ASSET_MAP_PATH, DROPPED_SYMBOLS, SUCCESSORS, None),
    "+30 unmapped": Panel(
        FULL_RAW_DIR, extended_path, DROPPED_SYMBOLS, SUCCESSORS, ROLL_FIXES
    ),
}
runs = {
    "current, lag 2 (as main.py)": (panels["current"], 2),
    "current, lag 1": (panels["current"], 1),
    "no chain, lag 1": (panels["no chain"], 1),
    "with AP, lag 1": (panels["with AP"], 1),
    "no LB fix, lag 1": (panels["no LB fix"], 1),
    "+30 unmapped, lag 1": (panels["+30 unmapped"], 1),
}

# 2. gross Sharpe of the course baseline over different windows
windows = {
    "IS": ("1976-06", "2014-12"),
    "OOS": ("2015-01", None),
    "15-19": ("2015-01", "2019-12"),
    "15-20": ("2015-01", "2020-12"),
    "15-22": ("2015-01", "2022-12"),
    "15-24": ("2015-01", "2024-12"),
    "20-22": ("2020-01", "2022-12"),
}
results, table = {}, {}
for name, (panel, lag) in runs.items():
    res = run_backtest(
        panel, BaselineTsmomStrategy(), cost_model=ZeroCostModel(), exec_lag=lag
    )
    results[name] = res
    monthly = monthly_returns(res, gross=True)
    table[name] = {
        w: monthly.loc[a:b].mean() / monthly.loc[a:b].std() * np.sqrt(12)
        for w, (a, b) in windows.items()
    }
print("== Baseline TSMOM gross Sharpe ==")
print(pd.DataFrame(table).T.round(2).to_string())

# 3. where the OOS return comes from (current panel, lag 1): by asset class and by contract
panel, res = panels["current"], results["current, lag 1"]
pnl = (res.positions * panel.returns.fillna(0.0)).loc["2015-01-01":]
classes = panel.asset_map["AssetClass"].reindex(panel.symbols)
print("\n== OOS annualised gross return contribution by asset class ==")
print((pnl.T.groupby(classes).sum().T.mean() * 252).round(4).to_string())
per_symbol = (pnl.mean() * 252).sort_values()
print("\n== worst 8 and best 8 contracts ==")
print(pd.concat([per_symbol.head(8), per_symbol.tail(8)]).round(4).to_string())

# 4. the biggest single-contract days, to spot roll artifacts that survived cleaning
daily = pnl.stack()
print("\n== 10 largest single-contract daily P&L days in OOS ==")
print(daily.reindex(daily.abs().nlargest(10).index).round(4).to_string())
