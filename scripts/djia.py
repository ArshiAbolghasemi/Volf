import numpy as np
import pandas as pd

from src.util.path import DATA_DIR


def main() -> None:
    df = pd.read_csv(
        DATA_DIR / "macroeconomic" / "Dow Jones Industrial Average Historical Data.csv"
    )

    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")

    df["Price"] = df["Price"].astype(str).str.replace(",", "", regex=False).astype(float)

    df = df.sort_values("Date").set_index("Date")

    df["DJIA_Index"] = np.log(df["Price"])

    weekly = df["DJIA_Index"].resample("W-MON").last()

    weekly = weekly.rename("DJIA_Index").reset_index()
    weekly["Date"] = weekly["Date"].dt.strftime("%Y-%m-%d")

    weekly.to_csv(
        DATA_DIR / "macroeconomic" / "DJIA_index.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
