from dynaconf import Dynaconf


class Config:
    def __init__(self) -> None:
        self._settings = Dynaconf(
            envvar_prefix="CLIMATE",
            load_dotenv=True,
            environments=True,
            envvar_prefix_for_dynaconf=False,
        )

    @property
    def start_date(self) -> str:
        return self._settings.get("START_DATE", "2009-04-13")

    @property
    def end_date(self) -> str:
        return self._settings.get("END_DATE", "2025-03-17")

    @property
    def region(self) -> str:
        return self._settings.get("REGION", "us")

    @property
    def resolution(self) -> str:
        return self._settings.get("RESOLUTION", "4km")

    @property
    def format(self) -> str:
        return self._settings.get("FORMAT", "asc")

    @property
    def elements(self) -> list[str]:
        elements = self._settings.get("ELEMENTS")

        if isinstance(elements, str):
            return [e.strip() for e in elements.split(",") if e.strip()]
        if isinstance(elements, list):
            return elements
        return ["ppt", "tmax", "tmin", "tmean", "tdmean", "vpdmin", "vpdmax"]

    @property
    def max_retries(self) -> int:
        return int(self._settings.get("MAX_RETRIES", 3))

    @property
    def timeout(self) -> int:
        return int(self._settings.get("TIMEOUT", 30))

    @property
    def delay_between_downloads(self) -> float:
        return float(self._settings.get("DELAY_BETWEEN_DOWNLOADS", 0.5))

    @property
    def chunk_size(self) -> int:
        return int(self._settings.get("CHUNK_SIZE", 8192))

    @property
    def progress_interval(self) -> int:
        return int(self._settings.get("PROGRESS_INTERVAL", 100))

    @property
    def base_url(self) -> str:
        return self._settings.get("BASE_URL", "https://services.nacse.org/prism/data/get/")

    @property
    def filename_template(self) -> str:
        return self._settings.get(
            "FILENAME_TEMPLATE", "prism_{region}_{resolution}_{time_period}.zip"
        )

    @property
    def url_template(self) -> str:
        return self._settings.get(
            "URL_TEMPLATE",
            "{base_url}{region}/{resolution}/{element}/{date}?format={format}",
        )


config = Config()
