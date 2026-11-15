
from pathlib import Path
import pandas as pd

SOURCE = Path("data/full_sample/FuturesUnderlyingData_FullSample")
RAW = Path("data/raw")
OOS = Path("data/oos")

COLUMNS = [
    "Date", "Open", "High", "Low",
    "Close", "Volume", "OpenInterest"
]

RAW.mkdir(parents=True, exist_ok=True)
OOS.mkdir(parents=True, exist_ok=True)

for file in sorted(SOURCE.glob("*_RAD.CSV")):
    symbol = file.stem.removesuffix("_RAD")

    df = pd.read_csv(
        file,
        header=None,
        names=COLUMNS,
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        format="%m/%d/%Y",
        errors="raise",
    )

    df = df.sort_values("Date")
    df = df.drop_duplicates("Date", keep="last")

    insample = df[df["Date"] < "2015-01-01"]
    oos = df[df["Date"] >= "2015-01-01"]

    if not insample.empty:
        insample.to_csv(RAW / f"{symbol}.csv", index=False)

    if not oos.empty:
        oos.to_csv(OOS / f"{symbol}.csv", index=False)

    print(
        f"{symbol}: "
        f"IS={len(insample)}, "
        f"OOS={len(oos)}"
    )

print("Data preparation complete.")
