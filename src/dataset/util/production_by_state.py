import re
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.dataset.util.path import require_path

STATE_ABBREV_TO_NAME: dict[str, str] = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
}


def load_production_weights(crop: str, production_dir: Path) -> pd.DataFrame:
    path = require_path(production_dir / f"{crop}.csv", f"{crop} production by state")
    production_df = pd.read_csv(path)
    if "state" not in production_df.columns:
        msg = f"Missing 'state' column in production file: {path}"
        raise ValueError(msg)
    pattern = re.compile(rf"^{crop.capitalize()}ProductionByState_(\d{{4}})$")
    year_cols = [col for col in production_df.columns if pattern.match(col)]
    if not year_cols:
        msg = f"No production-by-year columns found for {crop} in: {path}"
        raise ValueError(msg)
    long = production_df[["state", *year_cols]].melt(
        id_vars="state",
        value_vars=year_cols,
        var_name="year_col",
        value_name="production",
    )
    long["production"] = pd.to_numeric(long["production"], errors="coerce")
    long["state"] = long["state"].astype(str).str.strip()
    long = (
        long[long["state"] != "United States"]
        .dropna(subset=["production"])
        .pipe(lambda d: d[d["production"] > 0])
    )
    state_avg = (
        long.groupby("state", as_index=False)["production"]
        .mean()
        .rename(columns={"production": "avg_production"})
    )
    total = state_avg["avg_production"].sum()
    state_avg["state_weight"] = np.where(
        total > 0, state_avg["avg_production"] / total, 0.0
    )
    return cast("pd.DataFrame", state_avg[["state", "state_weight"]])
