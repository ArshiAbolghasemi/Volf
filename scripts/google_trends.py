from src.dataset.google_trend.climate_change import get_text_climate_anomaly_w_mon
from src.util.path import DATA_DIR


def main() -> None:
    weekly_climate_anomaly = get_text_climate_anomaly_w_mon(
        start_year=2009,
        start_mon=1,
        stop_year=2025,
        stop_mon=12,
    )
    outputpath = DATA_DIR / "google_trend" / "climate_change_anomaly.csv"
    outputpath.parent.mkdir(parents=True, exist_ok=True)
    weekly_climate_anomaly.to_csv(outputpath, index=False)


if __name__ == "__main__":
    main()
