"""Model comparison on time-based folds. Prints the tables used in the report.

Folds mirror the real task (train on the past, predict the next two months):
  fold A: train Jan-Jun, test Jul-Aug
  fold B: train Jan-Aug, test Sep-Oct
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

sys.path.insert(0, "src")
from features import FEATURES, build_reference_tables, flag_rate_outliers, make_features  # noqa: E402
from model import FreightRateModel  # noqa: E402

train_raw = pd.read_csv("data/train_test.csv")
valid_raw = pd.read_csv("data/validation.csv")
ref = build_reference_tables(train_raw, valid_raw)
data = make_features(train_raw, ref)
data["outlier"] = flag_rate_outliers(data)
data["month"] = data["date"].dt.month
print(f"Flagged outliers: {data.outlier.sum()} ({data.outlier.mean():.2%})")


def metrics(y, p):
    err = p - y
    return {
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err**2)),
        "MAPE%": 100 * np.mean(np.abs(err) / y),
    }


def hgb(features=FEATURES, loss="squared_error"):
    return HistGradientBoostingRegressor(
        loss=loss, learning_rate=0.05, max_iter=800, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0, categorical_features="from_dtype",
        random_state=0,
    ), features


def fit_predict(name, tr, te):
    tr = tr[~tr.outlier]
    y = np.log(tr.posted_rate)
    if name == "Lane median $/mile":
        rpm = (tr.posted_rate / tr.distance).groupby([tr.pickup, tr.delivery, tr.equipment], observed=True).median()
        eq = (tr.posted_rate / tr.distance).groupby(tr.equipment, observed=True).median()
        key = pd.MultiIndex.from_frame(te[["pickup", "delivery", "equipment"]].astype(str))
        rpm.index = rpm.index.set_levels([lv.astype(str) for lv in rpm.index.levels])
        r = pd.Series(rpm.reindex(key).values, index=te.index)
        r = r.fillna(te.equipment.map(eq).astype(float))
        return r * te.distance
    if name == "Ridge (log-linear)":
        cols = ["log_distance", "weight", "market_index_day", "weight_missing"]
        X = lambda d: pd.get_dummies(d[cols + ["pickup", "delivery", "equipment"]], drop_first=True).astype(float)
        Xtr = X(tr); Xte = X(te).reindex(columns=Xtr.columns, fill_value=0)
        m = Ridge(alpha=1.0).fit(Xtr, y)
        return np.exp(m.predict(Xte))
    if name == "Linear backbone (trend+QE)":
        return FreightRateModel(use_trees=False).fit(tr, tr.posted_rate).predict(te)
    if name == "Two-stage (linear + trees)":
        return FreightRateModel(use_trees=True).fit(tr, tr.posted_rate).predict(te)
    if name.startswith("HGB"):
        feats = FEATURES + (["quote_signal"] if "quote_signal" in name else [])
        m, f = hgb(feats)
        m.fit(tr[f], y)
        return np.exp(m.predict(te[f]))
    raise ValueError(name)


models = ["Lane median $/mile", "Ridge (log-linear)", "HGB", "HGB + quote_signal",
          "Linear backbone (trend+QE)", "Two-stage (linear + trees)"]
folds = {"A: Jan-Jun -> Jul-Aug": ([1, 2, 3, 4, 5, 6], [7, 8]),
         "B: Jan-Aug -> Sep-Oct": ([1, 2, 3, 4, 5, 6, 7, 8], [9, 10])}
rows = []
for fold, (tm, vm) in folds.items():
    tr = data[data.month.isin(tm)]; te = data[data.month.isin(vm)]
    for name in models:
        p = fit_predict(name, tr, te)
        clean = ~te.outlier
        rows.append({"fold": fold, "model": name,
                     **{f"{k} (clean)": v for k, v in metrics(te.posted_rate[clean], p[clean]).items()},
                     "MAE (all rows)": metrics(te.posted_rate, p)["MAE"]})
res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(res.round(2).to_string(index=False))

# Why a random split is misleading here: quote_signal looks great on a random split.
tr, te = train_test_split(data, test_size=0.2, random_state=0)
for name in ["HGB", "HGB + quote_signal"]:
    p = fit_predict(name, tr, te)
    c = ~te.outlier
    print(f"Random 80/20 split | {name:20s} MAPE (clean) = {metrics(te.posted_rate[c], p[c])['MAPE%']:.2f}%")
