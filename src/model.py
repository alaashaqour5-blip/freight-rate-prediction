"""Two-stage freight-rate model.

Stage 1 - linear backbone on log(rate): lane/equipment effects, log distance, weight,
log daily market index, a linear time trend, a quarter-end ramp and day of week.
It carries everything that must *extrapolate* into Nov/Dec, which trees cannot do
(they flat-line outside the training range): the upward trend (~+8%/yr) and the
market-index elasticity.

Stage 2 - gradient-boosted trees on the stage-1 residuals, using only features that
are stationary (no date index), to pick up non-linear structure such as distance x
equipment curvature and lane-specific quirks.

Prediction = exp(stage1 + stage2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

T0 = pd.Timestamp("2025-01-01")
QUARTER_ENDS = pd.to_datetime(
    ["2024-12-31", "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31"]
)

TREE_FEATURES = [
    "pickup", "delivery", "equipment", "log_distance", "weight", "weight_missing",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "day_of_week", "days_to_quarter_end", "market_index_day",
]


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dates = df["date"].values.astype("datetime64[D]")
    qe = QUARTER_ENDS.values.astype("datetime64[D]")
    nxt = qe[np.searchsorted(qe, dates, side="left")]
    prev = qe[np.searchsorted(qe, dates, side="left") - 1]
    df["days_to_quarter_end"] = (nxt - dates).astype(int)
    df["days_since_quarter_end"] = (dates - prev).astype(int)
    df["t_years"] = (df["date"] - T0).dt.days / 365.25
    return df


def _linear_design(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["log_distance"] = df["log_distance"]
    X["log_distance_sq"] = df["log_distance"] ** 2
    X["weight_k"] = df["weight"] / 10_000
    X["weight_missing"] = df["weight_missing"]
    X["log_mi"] = np.log(df["market_index_day"])
    X["t_years"] = df["t_years"]
    # Quarter-end ramp: one indicator per bucket of days remaining in the quarter.
    dq = df["days_to_quarter_end"]
    for lo, hi in [(0, 2), (3, 6), (7, 10), (11, 14)]:
        X[f"qe_{lo}_{hi}"] = dq.between(lo, hi).astype(float)
    X["qs_0_2"] = df["days_since_quarter_end"].between(1, 3).astype(float)
    cats = pd.get_dummies(
        df[["pickup", "delivery", "equipment"]].astype(str), drop_first=False
    ).astype(float)
    dow = pd.get_dummies(df["day_of_week"].astype(str), prefix="dow").astype(float)
    eq_dist = pd.get_dummies(df["equipment"].astype(str), prefix="ld_x").astype(float).mul(
        df["log_distance"], axis=0
    )
    X = pd.concat([X, cats, dow, eq_dist], axis=1)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X


class FreightRateModel:
    def __init__(self, use_trees: bool = True, random_state: int = 0):
        self.use_trees = use_trees
        self.random_state = random_state

    def fit(self, df: pd.DataFrame, y_rate: pd.Series) -> "FreightRateModel":
        df = add_calendar(df)
        y = np.log(y_rate.values)
        X = _linear_design(df)
        self.columns_ = list(X.columns)
        self.linear_ = Ridge(alpha=1.0).fit(X, y)
        if self.use_trees:
            resid = y - self.linear_.predict(X)
            self.trees_ = HistGradientBoostingRegressor(
                learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=60,
                l2_regularization=1.0, categorical_features="from_dtype",
                random_state=self.random_state,
            ).fit(df[TREE_FEATURES], resid)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        df = add_calendar(df)
        log_pred = self.linear_.predict(_linear_design(df, self.columns_))
        if self.use_trees:
            log_pred = log_pred + self.trees_.predict(df[TREE_FEATURES])
        return np.exp(log_pred)

    def linear_coefficients(self) -> pd.Series:
        return pd.Series(self.linear_.coef_, index=self.columns_)
