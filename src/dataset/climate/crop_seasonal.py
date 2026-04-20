# ============================================================
# U.S. Crop Seasonal Calendar (Week-of-Year Based)
# ============================================================
# Week-of-year reference (approximate mapping):
#
# Week 01-04   -> January
# Week 05-08   -> February
# Week 09-13   -> March
# Week 14-17   -> April
# Week 18-22   -> May
# Week 23-26   -> June
# Week 27-30   -> July
# Week 31-35   -> August
# Week 36-39   -> September
# Week 40-43   -> October
# Week 44-47   -> November
# Week 48-52   -> December
# ============================================================


# --------------------------------------------------------
# CORN (Maize) # noqa: ERA001
# Planting: Mar-Jun  -> Weeks ~10-22
# Harvest:  Sep-Nov  -> Weeks ~36-48
# --------------------------------------------------------
# SOYBEAN
# Planting: Apr-Jun  -> Weeks ~14-26
# Harvest:  Sep-Nov  -> Weeks ~36-48
# --------------------------------------------------------
# WINTER WHEAT
# Planting: Sep-Nov  -> Weeks ~36-48
# Harvest:  May-Jul  -> Weeks ~20-28
# --------------------------------------------------------
# SPRING WHEAT
# Planting: Apr-May -> Weeks ~14-22
# Harvest:  Aug-Sep -> Weeks ~32-38
# --------------------------------------------------------
CROP_CALENDAR = {
    "corn": {
        "planting": [(10, 22)],  # Mar-May/early Jun
        "harvesting": [(36, 48)],  # Sep-Nov
    },
    "soybean": {
        "planting": [(14, 26)],  # Apr-Jun
        "harvesting": [(36, 48)],  # Sep-Nov
    },
    "wheat": {
        "planting": [(14, 22), (36, 48)],  # Apr-May
        "harvesting": [(32, 38), (20, 28)],  # Aug-Sep
    },
}


def in_ranges(week: int, ranges: list[tuple]) -> bool:
    return any(start <= week <= end for start, end in ranges)


def crop_season_flag(week_of_year: int, commodity: str) -> dict[str, int]:
    commodity = commodity.lower()

    if commodity not in CROP_CALENDAR:
        msg = f"Unknown commodity: {commodity}"
        raise ValueError(msg)

    calendar = CROP_CALENDAR[commodity]

    is_planting = in_ranges(week_of_year, calendar["planting"])
    is_harvesting = in_ranges(week_of_year, calendar["harvesting"])

    return {
        "is_planting_week": int(is_planting),
        "is_harvesting_week": int(is_harvesting),
        "is_active_season": int(is_planting or is_harvesting),
    }
