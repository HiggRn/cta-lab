from pathlib import Path

import numpy as np
import pandas as pd

from common import Bar


class Panel:
    def __init__(
        self,
        raw_dirs: Path | list[Path],
        asset_map_path: Path,
        dropped_symbols: set[str] = set(),
    ) -> None:
        if isinstance(raw_dirs, (str, Path)):
            raw_dirs = [raw_dirs]

        self.asset_map = pd.read_csv(asset_map_path, index_col="ID")

        raw = {}
        for d in raw_dirs:
            for f in sorted(Path(d).glob("*.csv")):
                symbol = f.stem
                if symbol in dropped_symbols:
                    continue
                df = pd.read_csv(f, index_col=0, parse_dates=True).rename(
                    columns={
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                        "OpenInterest": "oi",
                    }
                )
                raw.setdefault(symbol, []).append(df)

        frames = {s: pd.concat(dfs).sort_index() for s, dfs in raw.items()}
        self.symbols = sorted(frames.keys())
        self.all_dates = pd.DatetimeIndex(
            sorted(set().union(*[df.index for df in frames.values()]))
        )

        # find calendar month end (panel-wide, for cross-symbol rebalance timing)
        month_period = self.all_dates.to_period("M")
        is_cal_month_end_arr = np.zeros(len(self.all_dates), dtype=bool)
        is_cal_month_end_arr[:-1] = month_period[:-1] != month_period[1:]
        is_cal_month_end_arr[-1] = True
        cal_month_end_map = dict(zip(self.all_dates, is_cal_month_end_arr))

        # construct close prices df and returns df
        close_dict = {}
        for s in self.symbols:
            if "close" in frames[s].columns:
                close_dict[s] = frames[s]["close"]
        self.close_df = pd.DataFrame(close_dict, index=self.all_dates)

        # P&L returns: a gap day earns 0, the first day back earns the whole gap
        px = self.close_df.ffill()
        self.returns = px / px.shift(1) - 1.0
        # estimation returns: NaN on days a symbol has no bar
        self.observed_returns = self.returns.where(self.close_df.notna())

        # pre-materialization
        self.timeline: list[tuple[pd.Timestamp, dict[str, Bar]]] = []

        daily_records: dict[pd.Timestamp, dict[str, Bar]] = {
            t: {} for t in self.all_dates
        }
        for s, df in frames.items():
            clean_df = df.dropna(subset=["close"])
            if clean_df.empty:
                continue

            # month end w.r.t. THIS symbol's own trading calendar, not the panel union
            sym_period = clean_df.index.to_period("M")
            is_sym_month_end = ~sym_period.duplicated(keep="last")

            for row, sym_end in zip(clean_df.itertuples(), is_sym_month_end):
                t = row.Index
                if t in daily_records:
                    daily_records[t][s] = Bar(
                        date=t,
                        open=float(getattr(row, "open", np.nan)),
                        high=float(getattr(row, "high", np.nan)),
                        low=float(getattr(row, "low", np.nan)),
                        close=float(getattr(row, "close", np.nan)),
                        volume=float(getattr(row, "volume", 0.0)),
                        open_interest=float(getattr(row, "oi", 0.0)),
                        is_month_end=bool(sym_end),
                        is_calendar_month_end=cal_month_end_map[t],
                    )

        self.timeline = [(t, daily_records[t]) for t in self.all_dates]
