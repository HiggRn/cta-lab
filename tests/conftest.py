import pandas as pd
import pytest

from panel import Panel


@pytest.fixture
def make_panel(tmp_path):
    """Factory: write one OHLCV csv per symbol plus an asset map, and return a Panel.

    `closes` maps symbol -> close Series. A day missing from a series is a day that
    symbol did not trade.
    """

    def build(closes, asset_classes=None, dropped=()):
        raw = tmp_path / "raw"
        raw.mkdir(exist_ok=True)
        for symbol, close in closes.items():
            frame = pd.DataFrame(
                {
                    "Open": close,
                    "High": close,
                    "Low": close,
                    "Close": close,
                    "Volume": 1.0,
                    "OpenInterest": 1.0,
                }
            )
            frame.index.name = "Date"
            frame.to_csv(raw / f"{symbol}.csv")

        classes = asset_classes or {s: "EQ" for s in closes}
        asset_map = pd.DataFrame(
            {
                "Name": list(closes),
                "Ccy": "USD",
                "AssetClass": [classes[s] for s in closes],
            },
            index=pd.Index(list(closes), name="ID"),
        )
        asset_map_path = tmp_path / "AssetMapCsv.csv"
        asset_map.to_csv(asset_map_path)
        return Panel(
            raw_dirs=raw, asset_map_path=asset_map_path, dropped_symbols=set(dropped)
        )

    return build
