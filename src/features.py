"""Data cleaning and feature engineering for freight-rate prediction.

Every transform here works the same way for training rows, validation rows and the
December chart rows, which carry only pickup, delivery, distance, equipment, weight
and date. Nothing in this module reads the target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL = ["pickup", "delivery", "equipment"]
NUMERIC = [
    "log_distance",
    "weight",
    "weight_missing",
    "market_index_day",
    "day_of_week",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
]
FEATURES = CATEGORICAL + NUMERIC


def build_reference_tables(*frames: pd.DataFrame) -> dict:
    """Lookups learned from feature columns only (no target).

    - market_index_day: the daily mean market index. market_index is a market-wide
      daily level plus small per-load noise, so the daily mean is the useful signal.
      It is also the only way to give the December chart rows (which have no
      market_index column) a market level for each date.
    - city coordinates, so chart rows that lack lat/lon can still be featurised.
    - median weight per equipment type, for imputing missing weights.
    """
    allrows = pd.concat(frames, ignore_index=True)
    allrows["date"] = pd.to_datetime(allrows["date"])

    mi_day = allrows.groupby("date")["market_index"].mean()

    pickups = allrows[["pickup", "pickup_lat", "pickup_lon"]].drop_duplicates("pickup")
    coords = {r.pickup: (r.pickup_lat, r.pickup_lon) for r in pickups.itertuples()}
    deliveries = allrows[["delivery", "delivery_lat", "delivery_lon"]].drop_duplicates("delivery")
    for r in deliveries.itertuples():
        coords.setdefault(r.delivery, (r.delivery_lat, r.delivery_lon))

    weight = allrows["weight"].abs()
    weight_median = weight.groupby(allrows["equipment"]).median().to_dict()

    return {"mi_day": mi_day, "coords": coords, "weight_median": weight_median}


def make_features(frame: pd.DataFrame, ref: dict) -> pd.DataFrame:
    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Coordinates: fill from the city lookup when absent (December chart rows).
    for side in ("pickup", "delivery"):
        lat, lon = f"{side}_lat", f"{side}_lon"
        if lat not in df:
            df[lat] = df[side].map(lambda c: ref["coords"][c][0])
            df[lon] = df[side].map(lambda c: ref["coords"][c][1])

    # Weight: negative values are sign errors (|w| matches the normal distribution);
    # missing values are imputed with the equipment median and flagged.
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").abs()
    df["weight_missing"] = df["weight"].isna().astype(int)
    df["weight"] = df["weight"].fillna(df["equipment"].map(ref["weight_median"]))

    # Market level for the load's date (robust to a missing per-row market_index).
    df["market_index_day"] = df["date"].map(ref["mi_day"])
    if df["market_index_day"].isna().any():
        missing = df.loc[df["market_index_day"].isna(), "date"].unique()
        raise ValueError(f"No market index available for dates: {missing[:5]}")

    df["log_distance"] = np.log(df["distance"].astype(float))
    df["day_of_week"] = df["date"].dt.dayofweek

    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    return df


def flag_rate_outliers(train: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
    """Flag corrupted labels (~1.4% of rows).

    These sit at ~0.2-0.4x or ~2.5-4.5x the price implied by distance and equipment,
    which looks like entry/unit errors rather than market moves. We fit a simple
    log-log baseline per equipment type and flag rows whose log residual exceeds
    `threshold` (a factor of ~1.65x). Normal residuals have a std of ~0.06.
    """
    log_rate = np.log(train["posted_rate"])
    log_dist = np.log(train["distance"])
    residual = pd.Series(index=train.index, dtype=float)
    for _, idx in train.groupby("equipment", observed=True).groups.items():
        slope, intercept = np.polyfit(log_dist[idx], log_rate[idx], 1)
        residual[idx] = log_rate[idx] - (intercept + slope * log_dist[idx])
    return residual.abs() > threshold
