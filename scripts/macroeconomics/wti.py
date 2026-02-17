import numpy as np
import pandas as pd

from src.util.path import DATA_DIR


def main() -> None:
    wti_df = pd.read_csv(DATA_DIR / "macroeconomic" / "WTI_USD Historical Data.csv")

    wti_df["Date"] = pd.to_datetime(wti_df["Date"], format="%m/%d/%Y")
    wti_df["Price"] = pd.to_numeric(wti_df["Price"], errors="coerce")
    wti_df = wti_df.sort_values("Date").set_index("Date")
    wti_df["WTI_Index"] = np.log(wti_df["Price"])

    weekly = wti_df["WTI_Index"].resample("W-MON").last()
    weekly = weekly.rename("WTI_Index").reset_index()
    weekly["Date"] = weekly["Date"].dt.strftime("%Y-%m-%d")
    weekly.to_csv(DATA_DIR / "macroeconomic" / "WTI_index.csv", index=False)


if __name__ == "__main__":
    main()
