import numpy as np
import pandas as pd

from src.util.path import DATA_DIR


def main() -> None:
    df = pd.read_csv(DATA_DIR / "macroeconomic" / "WTI_USD Historical Data.csv")

    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.sort_values("Date").set_index("Date")
    df["WTI_Index"] = np.log(df["Price"])

    weekly = df["WTI_Index"].resample("W-MON").last()
    weekly = weekly.rename("WTI_Index").reset_index()
    weekly["Date"] = weekly["Date"].dt.strftime("%Y-%m-%d")
    weekly.to_csv(DATA_DIR / "macroeconomic" / "WTI_index.csv", index=False)


if __name__ == "__main__":
    main()
