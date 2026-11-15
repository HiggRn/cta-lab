import pandas as pd
import pytest

from config import ASSET_MAP_PATH, DROPPED_SYMBOLS, FULL_RAW_DIR, ROLL_FIXES, SUCCESSORS
from panel import Panel

RAD_COLUMNS = ["date", "open", "high", "low", "close", "volume", "oi"]


@pytest.fixture
def write_rad():
    """Write a frame (columns open..oi, DatetimeIndex) as a headerless vendor file {symbol}_RAD.CSV."""

    def write(folder, symbol, frame):
        out = frame[RAD_COLUMNS[1:]].copy()
        out.insert(0, "date", out.index.strftime("%m/%d/%Y"))
        out.to_csv(folder / f"{symbol}_RAD.CSV", header=False, index=False)

    return write


@pytest.fixture
def read_rad():
    def read(path):
        df = pd.read_csv(path, header=None, names=RAD_COLUMNS, index_col="date")
        df.index = pd.to_datetime(df.index, format="%m/%d/%Y")
        return df

    return read


@pytest.fixture
def make_panel(tmp_path, write_rad):
    """Factory: write one vendor-format file per symbol plus an asset map, and return a Panel.

    `closes` maps symbol -> close Series. A day missing from a series is a day that symbol did not
    trade. `zero_days` maps symbol -> days written as all-zero placeholder rows, `unmapped` symbols
    have a file but no asset map entry, `no_file` symbols have an asset map entry but no file.
    """

    def build(
        closes,
        asset_classes=None,
        dropped=(),
        unmapped=(),
        no_file=(),
        successors=None,
        roll_fixes=None,
        zero_days=None,
    ):
        raw = tmp_path / "raw"
        raw.mkdir(exist_ok=True)
        for symbol, close in closes.items():
            if symbol in no_file:
                continue
            frame = pd.DataFrame(
                {
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 1.0,
                    "oi": 1.0,
                }
            )
            placeholders = (zero_days or {}).get(symbol, [])
            if len(placeholders):
                zeros = pd.DataFrame(
                    0.0, index=pd.DatetimeIndex(placeholders), columns=frame.columns
                )
                frame = pd.concat([frame, zeros]).sort_index()
            write_rad(raw, symbol, frame)

        classes = asset_classes or {s: "EQ" for s in closes}
        mapped = [s for s in closes if s not in unmapped]
        asset_map = pd.DataFrame(
            {"Name": mapped, "Ccy": "USD", "AssetClass": [classes[s] for s in mapped]},
            index=pd.Index(mapped, name="ID"),
        )
        asset_map_path = tmp_path / "AssetMapCsv.csv"
        asset_map.to_csv(asset_map_path)
        return Panel(
            raw_dir=raw,
            asset_map_path=asset_map_path,
            dropped_symbols=set(dropped),
            successors=successors,
            roll_fixes=roll_fixes,
        )

    return build


@pytest.fixture(scope="session")
def full_raw_dir():
    if not FULL_RAW_DIR.exists():
        pytest.skip(f"{FULL_RAW_DIR} not present")
    return FULL_RAW_DIR


@pytest.fixture(scope="session")
def real_panel(full_raw_dir):
    return Panel(full_raw_dir, ASSET_MAP_PATH, DROPPED_SYMBOLS, SUCCESSORS, ROLL_FIXES)
