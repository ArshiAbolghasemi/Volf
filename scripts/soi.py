import pandas as pd

from src.util.path import DATA_DIR

MONTHS = [
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
]
MONTH_NUM = {m: i + 1 for i, m in enumerate(MONTHS)}


def main() -> None:
    df = pd.read_csv(DATA_DIR / "climate" / "soi.csv")

    long_df = df.melt(
        id_vars=["YEAR"],
        value_vars=MONTHS,
        var_name="MONTH",
        value_name="SOI_index",
    )

    long_df["MONTH"] = long_df["MONTH"].map(lambda m: MONTH_NUM.get(m))

    long_df["Date"] = pd.to_datetime(
        long_df[["YEAR", "MONTH"]].assign(day=1),
        errors="raise",
    )

    monthly = long_df.sort_values("Date")[["Date", "SOI_index"]].set_index("Date")

    daily_idx = pd.date_range(monthly.index.min(), monthly.index.max(), freq="D")
    daily = monthly.reindex(daily_idx)
    daily["SOI_index"] = daily["SOI_index"].interpolate(method="time")

    weekly = daily.resample("W-MON").mean(numeric_only=True)

    weekly = weekly.reset_index().rename(columns={"index": "Date"})
    weekly["Date"] = weekly["Date"].dt.strftime("%Y-%m-%d")
    weekly = weekly[["Date", "SOI_index"]]

    weekly.to_csv(DATA_DIR / "climate" / "SOI_index.csv", index=False)


if __name__ == "__main__":
    main()
