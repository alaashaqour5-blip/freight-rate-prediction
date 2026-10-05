# Freight Rate Prediction

Predicts the posted rate for a truckload from its lane, distance, equipment, weight, date and market index.

**Holdout result (train Jan–Aug, test Sep–Oct):** MAPE 1.66%, MAE $39.49 on clean loads — versus 3.74% for a lane-median baseline and 4.96% for a standard gradient-boosting model.

## Quick start

Requires Python 3.10+.

```bash
python -m pip install -r requirements.txt

# 1. Train on data/train_test.csv and write both prediction files
python train.py

# 2. Validate outputs and draw the December chart (provided scorer)
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv

# Optional: reproduce the model comparison on time-based folds
python experiments.py
```

`train.py` writes `validation_predictions.csv` and fills `predicted_rate` in `data/december_chart_inputs.csv`. The scorer saves the chart to `scorer_results/candidate_december.png`. A full run takes well under a minute on a laptop.

## Repository layout

| Path | Purpose |
|---|---|
| `src/features.py` | Cleaning, feature engineering, corrupted-rate detection |
| `src/model.py` | Two-stage model: linear backbone + gradient-boosted residuals |
| `train.py` | Final training on all labelled data; writes predictions |
| `experiments.py` | Time-based model comparison used in the report |
| `score.py` | Scorer provided by Spotter (unchanged) |
| `data/` | Input data as provided |

## What the data showed

1. **Time split, not random.** Training covers Jan–Oct 2025; validation is Nov–Dec 2025 with no date overlap. This is a forecasting problem, so all validation uses time-based folds.
2. **`quote_signal` is a trap.** On about half the training days it equals the true rate-per-mile to within 2% (a leak). On the other days it is weakly *negatively* related. In Nov/Dec it shows neither pattern — no correlation with distance or equipment — so it carries no information there. It is excluded.
3. **`market_index` is a daily market level.** Its daily mean explains the daily price level (elasticity ≈ 0.16); within a day it is mostly noise. The model uses the daily mean, which also supplies a market level for the December chart rows (which have no market_index column).
4. **Prices drift upward ≈ 7.6% per year** beyond what the market index explains.
5. **Quarter-end surges.** Rates rise over the last two weeks of each quarter, peaking about +4% in the final three days (seen in Mar, Jun and Sep).
6. **Data quality.** 677 rates (1.4%) are corrupted at ~0.2–0.4× or ~2.5–4.5× the expected price and are excluded from training. Negative weights (292 train, 145 validation) are sign errors and are made positive. Missing weights (300 / 165) are imputed with the equipment median and flagged. Missing market_index values are covered by using the daily mean.

## Model

- **Stage 1 — Ridge regression on log(rate):** pickup, delivery and equipment effects; log distance (with a per-equipment slope); weight; log daily market index; linear time trend; quarter-end and quarter-start indicators; day of week. A linear model is used here because these effects must *extrapolate* into Nov/Dec — tree models flat-line beyond the training date range.
- **Stage 2 — HistGradientBoosting on Stage-1 residuals:** captures non-linear structure (distance curvature, lane quirks, interactions) using only stationary features (no date index).
- Prediction = exp(stage 1 + stage 2).

## Results (time-based holdout, clean loads)

| Model | Fold A MAPE (Jan–Jun → Jul–Aug) | Fold B MAPE (Jan–Aug → Sep–Oct) |
|---|---|---|
| Lane median $/mile | 4.28% | 3.74% |
| Ridge (no trend / quarter-end) | 2.30% | 4.43% |
| Gradient boosting | 2.43% | 4.96% |
| Gradient boosting + quote_signal | 2.76% | 4.16% |
| **Two-stage (final)** | 3.03% | **1.66%** |

Fold A is weaker for the final model because in Jan–Jun the market index rises almost in step with time (correlation 0.86), so the trend cannot be separated from the index. With the full Jan–Oct data that correlation is −0.11 and both effects are estimated cleanly; fold B is the closer analogue of the real Nov–Dec task.

## Limitations

- No November or December history exists, so holiday effects (e.g. Christmas) are not learned beyond the quarter-end pattern.
- The time trend is assumed linear and is extrapolated two months ahead.
- Corrupted-rate rows can't be predicted; if the final metric includes such rows, all models' errors are higher.
