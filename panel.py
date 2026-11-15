import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import Bar

COLUMNS = ["date", "open", "high", "low", "close", "volume", "oi"]
PRICE_COLUMNS = ["open", "high", "low", "close"]


class Panel:
    """Loads raw OHLCV+OI futures CSVs and aligns them on a unified chronological timeline
    without injecting any future calendar/lookahead flags.
    """

    def __init__(
        self,
        raw_dir: Path,
        asset_map_path: Path,
        dropped_symbols: frozenset[str] = frozenset(),
        successors: dict[str, str] | None = None,
        roll_fixes: dict[str, list[str]] | None = None,
    ) -> None:
        successors = successors or {}
        roll_fixes = roll_fixes or {}

        # keep_default_na=False: ID "NA" is not read as NaN
        self.asset_map = pd.read_csv(
            asset_map_path, index_col="ID", keep_default_na=False
        )
        tradable = set(self.asset_map.index) - set(dropped_symbols)
        wanted = tradable | set(successors.values())

        frames: dict[str, pd.DataFrame] = {}
        for f in sorted(Path(raw_dir).iterdir()):
            symbol = re.sub("_RAD$", "", f.stem, flags=re.I)
            if f.suffix.lower() != ".csv" or symbol not in wanted:
                continue
            df = pd.read_csv(f, header=None, names=COLUMNS, index_col="date")
            df.index = pd.to_datetime(df.index, format="%m/%d/%Y")
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]
            # all-zero OHLC rows are placeholders for days the symbol did not exist or trade
            df = df[df["close"].notna() & ~(df[PRICE_COLUMNS] == 0).all(axis=1)]
            if not (df["close"] > 0).all():
                raise ValueError(
                    f"{symbol}: non-positive close after dropping placeholder rows"
                )
            frames[symbol] = df

        missing = wanted - set(frames)
        if missing:
            raise FileNotFoundError(f"no data file in {raw_dir} for {sorted(missing)}")

        # Neutralise known contract-switch gaps: rescale every bar from the switch day on, so that day's
        # close-to-close return is zero and every later return is unchanged. Raises if the date is missing.
        for symbol, dates in roll_fixes.items():
            for d in pd.to_datetime(dates):
                prev_close = (
                    frames[symbol]["close"].loc[: d - pd.Timedelta(days=1)].iloc[-1]
                )
                frames[symbol].loc[d:, PRICE_COLUMNS] *= (
                    prev_close / frames[symbol].loc[d, "close"]
                )

        # Chain each retired big contract with its mini. Volume and open interest are appended unscaled.
        for big, mini in successors.items():
            last = frames[big].index[-1]
            after = frames[mini].loc[frames[mini].index > last].copy()
            if after.empty:
                continue
            anchor = frames[mini]["close"].loc[:last].iloc[-1]
            after[PRICE_COLUMNS] *= frames[big]["close"].iloc[-1] / anchor
            frames[big] = pd.concat([frames[big], after])

        frames = {s: frames[s] for s in sorted(tradable)}
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
