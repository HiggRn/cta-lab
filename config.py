from pathlib import Path

FULL_RAW_DIR = Path("./data/full_raw")
ASSET_MAP_PATH = Path("./data/processed/AssetMapCsv.csv")
SG_TREND_PATH = Path("./data/processed/Lecture3_livedata.xlsx")
FACTOR_DIR = Path("./data/external")
OUTPUT_DIR = Path("./outputs")

# Never traded. Duplicates: ES, SC -> SP; EN -> ND; YM -> ZD. Bad data: BC, BG, ZH, FF (data stops at
# various points and has holes), AP (no data for most of 2022-2024). RL stops in 2008, so its mini ER is kept.
# ES, EN and YM are still read, as price sources for SUCCESSORS.
DROPPED_SYMBOLS = frozenset(
    {"EN", "ES", "SC", "BC", "BG", "ZH", "FF", "RL", "YM", "AP"}
)

# Retired big contract -> mini contract on the same index. From the big contract's last bar on, the mini's
# prices (rescaled to match the big contract's last close) continue the series under the big contract's symbol.
SUCCESSORS = {"SP": "ES", "ND": "EN", "ZD": "YM"}

# Known contract-switch gaps in a rolled series. All prices from that day on are rescaled so that day's
# close-to-close return is zero. LB: ~500 -> ~800 between 2023-04-28 and 2023-05-03 (old -> new lumber contract).
ROLL_FIXES = {"LB": ["2023-05-03"]}
