import pandas as pd

from src.util.path import DATA_DIR


def main() -> None:
    anomaly_df = pd.read_csv(DATA_DIR / "climate" / "global_temperature_anomaly.csv")

    anomaly_df["Date"] = pd.to_datetime(anomaly_df["Year"].astype(str) + "-01-01")
    anomaly_df = anomaly_df.set_index("Date")[["Anomaly"]]

    weekly_index = pd.date_range(
        start=anomaly_df.index.min(), end=anomaly_df.index.max(), freq="W-MON"
    )

    weekly_df = (
        anomaly_df.reindex(anomaly_df.index.union(weekly_index))
        .sort_index()
        .interpolate(method="time")
        .reindex(weekly_index)
    )

    weekly_df = weekly_df.reset_index()
    weekly_df.columns = ["Date", "TA_Index"]
    weekly_df["Date"] = weekly_df["Date"].dt.strftime("%Y-%m-%d")

    weekly_df.to_csv(DATA_DIR / "climate" / "TA_Index.csv", index=False)


if __name__ == "__main__":
    main()
