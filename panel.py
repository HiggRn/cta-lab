from pathlib import Path

import numpy as np
import pandas as pd

from common import Bar


class Panel:
    """Loads raw OHLCV+OI futures CSVs and aligns them on a unified chronological timeline
    without injecting any future calendar/lookahead flags.
    """

    def __init__(
        self,
        raw_dirs: Path | list[Path],
        asset_map_path: Path,
        dropped_symbols: set[str] = set(),
    ) -> None:
        if isinstance(raw_dirs, (str, Path)):
            raw_dirs = [raw_dirs]

        self.asset_map = pd.read_csv(asset_map_path, index_col="ID")

        raw: dict[str, list[pd.DataFrame]] = {}
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

        frames = {
            s: pd.concat(dfs)
            .sort_index()
            .loc[lambda x: ~x.index.duplicated(keep="last")]
            for s, dfs in raw.items()
        }
        self.symbols = sorted(frames.keys())
        self.all_dates = pd.DatetimeIndex(
            sorted(set().union(*[df.index for df in frames.values()]))
        )

        # Construct close prices and volume DataFrames aligned to all_dates
        close_dict = {
            s: frames[s]["close"] for s in self.symbols if "close" in frames[s].columns
        }
        vol_dict = {
            s: frames[s]["volume"]
            for s in self.symbols
            if "volume" in frames[s].columns
        }

        self.close_df = pd.DataFrame(
            close_dict, index=self.all_dates, columns=self.symbols
        )
        self.volume_df = pd.DataFrame(
            vol_dict, index=self.all_dates, columns=self.symbols
        )

        # P&L returns: a holiday/gap day earns 0, the first day back earns the gap return
        px = self.close_df.ffill()
        self.returns = px / px.shift(1) - 1.0

        # Estimation returns: NaN on days a symbol had no bar (did not trade)
        self.observed_returns = self.returns.where(self.close_df.notna())

        # Track each symbol's first and last valid trading index to define active_universe at any t
        n_days = len(self.all_dates)
        self.first_valid_idx = np.full(len(self.symbols), n_days, dtype=int)
        self.last_valid_idx = np.full(len(self.symbols), -1, dtype=int)

        notna_mat = self.close_df.notna().to_numpy()
        for j in range(len(self.symbols)):
            valid_rows = np.flatnonzero(notna_mat[:, j])
            if len(valid_rows) > 0:
                self.first_valid_idx[j] = int(valid_rows[0])
                self.last_valid_idx[j] = int(valid_rows[-1])

        # Pre-materialize daily Bar slices
        daily_records: dict[pd.Timestamp, dict[str, Bar]] = {
            t: {} for t in self.all_dates
        }
        for s, df in frames.items():
            clean_df = df.dropna(subset=["close"])
            if clean_df.empty:
                continue

            for row in clean_df.itertuples():
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
                    )

        self.timeline = [(t, daily_records[t]) for t in self.all_dates]
