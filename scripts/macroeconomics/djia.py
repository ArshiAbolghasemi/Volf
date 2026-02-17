import numpy as np
import pandas as pd

from src.util.path import DATA_DIR


def main() -> None:
    djia_df = pd.read_csv(
        DATA_DIR / "macroeconomic" / "Dow Jones Industrial Average Historical Data.csv"
    )

    djia_df["Date"] = pd.to_datetime(djia_df["Date"], format="%m/%d/%Y")

    djia_df["Price"] = (
        djia_df["Price"].astype(str).str.replace(",", "", regex=False).astype(float)
    )

    djia_df = djia_df.sort_values("Date").set_index("Date")

    djia_df["DJIA_Index"] = np.log(djia_df["Price"])

    weekly = djia_df["DJIA_Index"].resample("W-MON").last()

    weekly = weekly.rename("DJIA_Index").reset_index()
    weekly["Date"] = weekly["Date"].dt.strftime("%Y-%m-%d")

    weekly.to_csv(
        DATA_DIR / "macroeconomic" / "DJIA_index.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
