# ============================================================
# U.S. Crop Seasonal Calendar (Week-of-Year Based)
# ============================================================
# Week-of-year reference (approximate):
#   01-04  Jan | 05-08  Feb | 09-13  Mar | 14-17  Apr
#   18-22  May | 23-26  Jun | 27-30  Jul | 31-35  Aug
#   36-39  Sep | 40-43  Oct | 44-47  Nov | 48-52  Dec
#
# CORN:         Planting Mar-Jun (10-22),  Harvest Sep-Nov (36-48)
# SOYBEAN:      Planting Apr-Jun (14-26),  Harvest Sep-Nov (36-48)
# WHEAT:        Planting Apr-May (14-22) + Sep-Nov (36-48)
#               Harvest  Aug-Sep (32-38) + May-Jul (20-28)
# ============================================================

WeekRanges = list[tuple[int, int]]

CROP_CALENDAR: dict[str, dict[str, WeekRanges]] = {
    "corn": {"planting": [(10, 22)], "harvesting": [(36, 48)]},
    "soybean": {"planting": [(14, 26)], "harvesting": [(36, 48)]},
    "wheat": {"planting": [(14, 22), (36, 48)], "harvesting": [(32, 38), (20, 28)]},
}


def _in_ranges(week: int, ranges: WeekRanges) -> bool:
    return any(start <= week <= end for start, end in ranges)


def crop_season_flag(week_of_year: int, commodity: str) -> dict[str, int]:
    key = commodity.lower()
    if key not in CROP_CALENDAR:
        msg = f"Unknown commodity: {commodity!r}. Valid options: {sorted(CROP_CALENDAR)}"
        raise ValueError(msg)

    calendar = CROP_CALENDAR[key]
    planting = _in_ranges(week_of_year, calendar["planting"])
    harvesting = _in_ranges(week_of_year, calendar["harvesting"])

    return {
        "is_planting_week": int(planting),
        "is_harvesting_week": int(harvesting),
        "is_active_season": int(planting or harvesting),
    }
