"""Train the final model on all labelled data and write both prediction files.

Usage:
    python train.py

Outputs:
    validation_predictions.csv          (load_id,predicted_rate for the 12,000 validation loads)
    data/december_chart_inputs.csv      (predicted_rate column filled in)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from features import build_reference_tables, flag_rate_outliers, make_features  # noqa: E402
from model import FreightRateModel  # noqa: E402

DATA = ROOT / "data"


def main() -> None:
    train_raw = pd.read_csv(DATA / "train_test.csv")
    valid_raw = pd.read_csv(DATA / "validation.csv")
    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    december = pd.read_csv(DATA / "december_chart_inputs.csv")

    # Reference tables use feature columns only (daily market index, city coordinates,
    # weight medians). Validation features are included so Nov/Dec dates have a market level.
    ref = build_reference_tables(train_raw, valid_raw)

    train = make_features(train_raw, ref)
    outliers = flag_rate_outliers(train)
    train = train[~outliers]
    print(f"Training on {len(train):,} loads ({outliers.sum()} corrupted rates excluded)")

    model = FreightRateModel().fit(train, train["posted_rate"])
    coef = model.linear_coefficients()
    print(f"Market-index elasticity: {coef['log_mi']:.3f} | trend: {np.expm1(coef['t_years']):+.1%} per year | "
          f"last-3-days-of-quarter uplift: {np.expm1(coef['qe_0_2']):+.1%}")

    # Final validation predictions, in template order.
    valid = make_features(valid_raw, ref)
    preds = pd.Series(model.predict(valid), index=valid_raw["load_id"]).round(2)
    out = template[["load_id"]].copy()
    out["predicted_rate"] = out["load_id"].map(preds)
    assert out["predicted_rate"].notna().all() and (out["predicted_rate"] > 0).all()
    out.to_csv(ROOT / "validation_predictions.csv", index=False)
    print(f"Wrote validation_predictions.csv ({len(out):,} rows)")

    # December chart: same fixed load, only the date changes.
    dec = make_features(december.drop(columns="predicted_rate"), ref)
    december["predicted_rate"] = model.predict(dec).round(2)
    december.to_csv(DATA / "december_chart_inputs.csv", index=False)
    print("Wrote data/december_chart_inputs.csv")
    print(december[["date", "predicted_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
