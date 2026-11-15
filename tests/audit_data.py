import re
from pathlib import Path

import numpy as np
import pandas as pd

from main import ASSET_MAP_PATH, DROPPED_SYMBOLS

FULL_RAW_DIR = Path("./data/full_raw")
IS_RAW_DIR = Path("./data/raw")  # used only for the IS price comparison
OUT_PATH = Path("./outputs/data_audit.csv")
IS_END = pd.Timestamp("2014-12-31")

COLUMNS = ["date", "open", "high", "low", "close", "volume", "oi"]
RET_ABS_LIMIT = 0.25  # |daily return| above this is flagged
RET_Z_LIMIT = 8.0  # |return / lagged 60d EWM vol| above this is flagged
GAP_DAYS = 10  # calendar-day gap between consecutive valid bars
STALE_RUN = 5  # consecutive bars with an unchanged close
EARLY_END_DAYS = 10  # last bar this many days before the global last date

# keep_default_na=False so an ID like "NA" is not silently read as NaN
asset_map = pd.read_csv(ASSET_MAP_PATH, index_col="ID", keep_default_na=False)
files = {
    re.sub("_RAD$", "", f.stem, flags=re.I): f
    for f in sorted(FULL_RAW_DIR.iterdir())
    if f.suffix.lower() == ".csv"
}

mapped = set(asset_map.index)
unmapped = sorted(set(files) - mapped)
no_file = sorted(mapped - set(files))
dropped_present = sorted(set(files) & DROPPED_SYMBOLS)
universe = sorted((set(files) & mapped) - DROPPED_SYMBOLS)

print(f"files in {FULL_RAW_DIR}: {len(files)} | asset map ids: {len(mapped)}")
print(f"in files but not in asset map ({len(unmapped)}): {unmapped}")
print(f"in asset map but no file ({len(no_file)}): {no_file}")
print(
    f"dropped via DROPPED_SYMBOLS and present ({len(dropped_present)}): {dropped_present}"
)
print(f"tradable universe ({len(universe)} symbols)")

rows = {}
for sym, path in files.items():
    row = {
        "status": "universe"
        if sym in universe
        else "unmapped"
        if sym in unmapped
        else "dropped"
    }
    raw = pd.read_csv(path, header=None)
    row["n_cols"] = raw.shape[1]
    if raw.shape[1] != len(COLUMNS):
        rows[sym] = row
        continue
    raw.columns = COLUMNS

    # file-level sanity: bad dates, duplicates, ordering (checked before any cleaning)
    raw["date"] = pd.to_datetime(raw["date"], format="%m/%d/%Y", errors="coerce")
    row["bad_dates"] = int(raw["date"].isna().sum())
    raw = raw.dropna(subset=["date"])
    row["dup_dates"] = int(raw["date"].duplicated().sum())
    row["unsorted"] = not raw["date"].is_monotonic_increasing
    df = raw.drop_duplicates("date", keep="last").sort_values("date").set_index("date")

    # placeholder rows: all of OHLC == 0 (and partial zeros, negatives, NaN closes)
    ohlc = df[["open", "high", "low", "close"]]
    zero_all = (ohlc == 0).all(axis=1).to_numpy()
    valid = (df["close"].notna() & (df["close"] != 0)).to_numpy()
    row["rows"] = len(df)
    row["zero_rows"] = int(zero_all.sum())
    row["close_zero_partial"] = int(((df["close"] == 0) & ~zero_all).sum())
    row["nan_close"] = int(df["close"].isna().sum())
    row["open_hl_zero"] = int(
        (((ohlc[["open", "high", "low"]] == 0).any(axis=1)) & (df["close"] > 0)).sum()
    )
    row["neg_price_rows"] = int((ohlc < 0).any(axis=1).sum())

    # where the zero rows sit decides whether first/last_valid_idx would be broken
    v = np.flatnonzero(valid)
    if len(v) == 0:
        rows[sym] = row
        continue
    row["zero_leading"] = int(zero_all[: v[0]].sum())
    row["zero_trailing"] = int(zero_all[v[-1] + 1 :].sum())
    row["zero_interior"] = row["zero_rows"] - row["zero_leading"] - row["zero_trailing"]

    clean = df[valid]
    close = clean["close"]
    row["valid_rows"] = len(clean)
    row["first_date"] = close.index[0]
    row["last_date"] = close.index[-1]
    row["oos_rows"] = int((close.index > IS_END).sum())
    row["min_close"] = float(close.min())

    # calendar gaps between consecutive valid bars
    gap = close.index.to_series().diff().dt.days
    row["max_gap_days"] = float(gap.max()) if len(gap) > 1 else np.nan
    row["n_gaps"] = int((gap > GAP_DAYS).sum())

    # extreme moves across the whole file (not only at the IS/OOS join)
    ret = close.pct_change()
    z = ret / ret.ewm(span=60, min_periods=20).std().shift(1)
    row["n_extreme"] = int(
        ((ret.abs() > RET_ABS_LIMIT) | (z.abs() > RET_Z_LIMIT)).sum()
    )
    row["max_abs_ret"] = float(ret.abs().max()) if len(ret) > 1 else np.nan
    row["max_abs_ret_date"] = ret.abs().idxmax() if len(ret) > 1 else pd.NaT

    # frozen prices: longest run of an unchanged close
    run_id = (close.diff() != 0).cumsum()
    row["max_stale_run"] = int(close.groupby(run_id).size().max())

    # OHLC consistency, only on rows where all four prices are positive
    full = clean[(clean[["open", "high", "low", "close"]] > 0).all(axis=1)]
    bad = (
        (full["low"] > full[["open", "close"]].min(axis=1))
        | (full["high"] < full[["open", "close"]].max(axis=1))
        | (full["low"] > full["high"])
    )
    row["ohlc_inconsistent"] = int(bad.sum())
    row["zero_volume"] = int((clean["volume"] == 0).sum())
    row["zero_oi"] = int((clean["oi"] == 0).sum())

    # IS comparison against the original course file, on closes up to IS_END
    is_path = IS_RAW_DIR / f"{sym}.csv"
    if is_path.exists():
        is_close = pd.read_csv(is_path, index_col=0, parse_dates=True)["Close"].dropna()
        full_is = close.loc[:IS_END]
        common = is_close.index.intersection(full_is.index)
        row["is_missing_in_full"] = int(len(is_close.index.difference(full_is.index)))
        row["is_extra_in_full"] = int(len(full_is.index.difference(is_close.index)))
        row["is_max_rel_diff"] = (
            float((full_is[common] / is_close[common] - 1.0).abs().max())
            if len(common)
            else np.nan
        )

    rows[sym] = row

audit = pd.DataFrame.from_dict(rows, orient="index")

uni = audit.loc[audit["status"] == "universe"]
global_last = uni["last_date"].max()
print(f"\nuniverse sample: {uni['first_date'].min().date()} to {global_last.date()}")
print(
    f"symbols with no rows after {IS_END.date()}: {sorted(uni.index[uni['oos_rows'] == 0])}"
)

audit["early_end"] = (global_last - audit["last_date"]).dt.days > EARLY_END_DAYS
flags = pd.DataFrame(
    {
        "BAD_FORMAT": audit["n_cols"] != len(COLUMNS),
        "BAD_DATES": audit["bad_dates"] > 0,
        "DUP_DATES": audit["dup_dates"] > 0,
        "UNSORTED": audit["unsorted"] == True,
        "ZERO_ROWS": audit["zero_rows"] > 0,
        "ZERO_INTERIOR": audit["zero_interior"] > 0,
        "CLOSE_ZERO_PARTIAL": audit["close_zero_partial"] > 0,
        "NAN_CLOSE": audit["nan_close"] > 0,
        "NEG_PRICE": audit["neg_price_rows"] > 0,
        "EARLY_END": audit["early_end"],
        "GAPS": audit["n_gaps"] > 0,
        "EXTREME_RET": audit["n_extreme"] > 0,
        "STALE": audit["max_stale_run"] >= STALE_RUN,
        "OHLC_BAD": audit["ohlc_inconsistent"] > 0,
        "IS_MISMATCH": (audit["is_max_rel_diff"] > 1e-6)
        | (audit["is_missing_in_full"] > 0)
        | (audit["is_extra_in_full"] > 0),
    }
).fillna(False)
audit["flags"] = flags.apply(lambda r: ",".join(r.index[r]), axis=1)

pd.set_option("display.width", 250, "display.max_columns", 50, "display.max_rows", 200)
show = [
    "status",
    "first_date",
    "last_date",
    "valid_rows",
    "zero_rows",
    "zero_leading",
    "zero_trailing",
    "zero_interior",
    "min_close",
    "oos_rows",
    "flags",
]

print("\n== Flag counts (all files) ==")
print(flags.sum().loc[lambda s: s > 0].to_string())

print("\n== Symbols with flags ==")
print(audit.loc[audit["flags"] != "", show].to_string())

print("\n== Extreme returns (top 15 by max |ret|) ==")
print(
    audit.loc[audit["n_extreme"] > 0, ["n_extreme", "max_abs_ret", "max_abs_ret_date"]]
    .sort_values("max_abs_ret", ascending=False)
    .head(15)
    .to_string()
)

print("\n== IS comparison against data/raw (flagged only) ==")
print(
    audit.loc[
        flags["IS_MISMATCH"],
        ["is_missing_in_full", "is_extra_in_full", "is_max_rel_diff"],
    ].to_string()
)

OUT_PATH.parent.mkdir(exist_ok=True)
audit.to_csv(OUT_PATH)
print(f"\nfull audit saved to {OUT_PATH}")
