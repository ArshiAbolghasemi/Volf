import pandas as pd

from src.util.path import DATA_DIR

MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
MONTH_NUM = {m: i + 1 for i, m in enumerate(MONTHS)}


def main() -> None:
    df = pd.read_csv(DATA_DIR / "climate" / "nao.csv")

    long_df = df.melt(
        id_vars=["Year"],
        value_vars=MONTHS,
        var_name="Month",
        value_name="NAO_index",
    )

    long_df["Month"] = long_df["Month"].map(lambda m: MONTH_NUM.get(m))

    long_df["Date"] = pd.to_datetime(
        long_df[["Year", "Month"]].assign(day=1),
        errors="raise",
    )

    monthly = long_df.sort_values("Date")[["Date", "NAO_index"]].set_index("Date")

    daily_idx = pd.date_range(monthly.index.min(), monthly.index.max(), freq="D")
    daily = monthly.reindex(daily_idx)
    daily["NAO_index"] = daily["NAO_index"].interpolate(method="time")

    weekly = daily.resample("W-MON").mean(numeric_only=True)

    weekly = weekly.reset_index().rename(columns={"index": "Date"})
    weekly["Date"] = weekly["Date"].dt.strftime("%Y-%m-%d")
    weekly = weekly[["Date", "NAO_index"]]

    weekly.to_csv(DATA_DIR / "climate" / "NAO_index.csv", index=False)


if __name__ == "__main__":
    main()
