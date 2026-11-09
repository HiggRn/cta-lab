"""Checks that out-of-sample price files continue smoothly from the in-sample files.

Run before backtesting on newly released data. A vendor that re-adjusted its rolled series,
a missing file or a bad first print all show up as a jump at the join, which the backtest
would otherwise book as a genuine one-day return.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# What each status in `check_splice` means
STATUS = {
    "OK": "first new close is a normal move from the last in-sample close",
    "JUMP": "the move across the join is larger than z_limit daily standard deviations",
    "LEVEL_SHIFT": "dates in both folders have different closes (series was probably re-adjusted)",
    "GAP": "more than max_gap_days calendar days between the last in-sample and first new close",
    "MISSING": "symbol traded at the end of the in-sample period but has no new closes",
    "ENDED": "symbol had already stopped trading in-sample and has no new closes (expected)",
    "NEW": "symbol only exists in the out-of-sample folder",
    "BAD_FORMAT": "out-of-sample file has different columns from the in-sample file",
}


def _load(folder: Path, dropped: set[str]) -> dict[str, pd.DataFrame]:
    return {
        f.stem: pd.read_csv(f, index_col=0, parse_dates=True).sort_index()
        for f in sorted(Path(folder).glob("*.csv"))
        if f.stem not in dropped
    }


def check_splice(
    in_sample_dir: Path,
    oos_dir: Path,
    dropped_symbols: set[str] = frozenset(),
    z_limit: float = 6.0,
    vol_days: int = 250,
    max_gap_days: int = 10,
    price_tol: float = 1e-6,
) -> pd.DataFrame:
    """One row per symbol comparing the last in-sample close with the first later close.

    `z` is the join return divided by the symbol's recent daily volatility, scaled by
    sqrt(business days across the join). See STATUS for the meaning of each status.
    """
    ins = _load(in_sample_dir, set(dropped_symbols))
    oos = _load(oos_dir, set(dropped_symbols))
    in_end = max(df["Close"].dropna().index[-1] for df in ins.values())

    rows = {}
    for s in sorted(set(ins) | set(oos)):
        a, b = ins.get(s), oos.get(s)
        row: dict = {"status": "OK"}
        if a is None:
            row["status"] = "NEW"
            rows[s] = row
            continue

        a_close = a["Close"].dropna()
        last_date, last_close = a_close.index[-1], float(a_close.iloc[-1])
        row.update(last_in_date=last_date.date(), last_in_close=last_close)
        ended = (in_end - last_date).days > max_gap_days

        if b is None:
            row["status"] = "ENDED" if ended else "MISSING"
            rows[s] = row
            continue
        if set(b.columns) != set(a.columns):
            row["status"] = "BAD_FORMAT"
            rows[s] = row
            continue

        b_close = b["Close"].dropna()
        overlap = a_close.index.intersection(b_close.index)
        rel_diff = (b_close[overlap] / a_close[overlap] - 1.0).abs()
        row["overlap_days"] = len(overlap)
        row["overlap_max_rel_diff"] = float(rel_diff.max()) if len(overlap) else np.nan

        after = b_close[b_close.index > last_date]
        if after.empty:
            row["status"] = "ENDED" if ended else "MISSING"
            rows[s] = row
            continue

        first_date, first_close = after.index[0], float(after.iloc[0])
        bdays = max(int(np.busday_count(last_date.date(), first_date.date())), 1)
        vol = float(a_close.pct_change().iloc[-vol_days:].std())
        join_ret = first_close / last_close - 1.0
        z = join_ret / (vol * np.sqrt(bdays)) if vol > 0 else np.nan
        row.update(
            first_oos_date=first_date.date(),
            first_oos_close=first_close,
            gap_bdays=bdays,
            join_return=join_ret,
            daily_vol=vol,
            z=z,
        )

        if len(overlap) and rel_diff.max() > price_tol:
            row["status"] = "LEVEL_SHIFT"
        elif abs(z) > z_limit:
            row["status"] = "JUMP"
        elif (first_date - last_date).days > max_gap_days:
            row["status"] = "GAP"
        rows[s] = row

    columns = [
        "status", "last_in_date", "last_in_close", "first_oos_date", "first_oos_close",
        "gap_bdays", "join_return", "daily_vol", "z", "overlap_days", "overlap_max_rel_diff",
    ]
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=columns)
