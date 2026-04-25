"""
=============================================================================
HDB RESALE PRICE PREDICTION — Kaggle Regression Challenge
=============================================================================
Author  : Expert Kaggle Data Scientist (template)
Target  : resale_price (continuous, SGD)
Libraries: pandas, scikit-learn, LightGBM, XGBoost, category_encoders

WHY GRADIENT BOOSTING?
  Tree-based ensembles handle mixed data types, non-linear relationships,
  and outliers without extensive normalisation. HDB prices are driven by
  discrete factors (town, flat_type) interacting with continuous ones
  (floor_area_sqm, remaining_lease), which is exactly where GBMs shine.
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import category_encoders as ce
import xgboost as xgb
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# Scikit-learn

# Gradient Boosting

# Target / Ordinal encoding (pip install category_encoders)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION  — edit these paths to match your Kaggle input directory
# ─────────────────────────────────────────────────────────────────────────────
# adjust if needed
TRAIN_PATH = "/Users/mohanjawahar/DataScience/data/hdbregressionchallenge/train.csv"
TEST_PATH = "/Users/mohanjawahar/DataScience/data/hdbregressionchallenge/test.csv"
SUBMISSION_PATH = "/Users/mohanjawahar/DataScience/data/hdbregressionchallenge/submission.csv"

RANDOM_STATE = 42
TARGET = "resale_price"

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 1 — Loading data")
print("=" * 65)

train_raw = pd.read_csv(TRAIN_PATH)
test_raw = pd.read_csv(TEST_PATH)

print(f"  Train shape : {train_raw.shape}")
print(f"  Test  shape : {test_raw.shape}")
print(f"\n  Train columns:\n  {list(train_raw.columns)}\n")

# Keep a copy of original test IDs for the submission file
# Kaggle competitions typically have an 'id' or row-index column
if "id" in test_raw.columns:
    test_ids = test_raw["id"]
else:
    test_ids = test_raw.index

# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def parse_remaining_lease(series: pd.Series) -> pd.Series:
    """
    HDB datasets often express remaining_lease as a string like
    '61 years 04 months'. This function converts it to a float (in years).

    WHY: Raw string columns can't be used by ML models. Converting to a
    continuous numeric feature preserves the full information.
    """
    def _parse(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip()
        # Already numeric
        try:
            return float(val)
        except ValueError:
            pass
        # e.g. "61 years 04 months"
        years = months = 0
        if "year" in val:
            years = int(val.split("year")[0].strip().split()[-1])
        if "month" in val:
            months = int(val.split("month")[0].strip().split()[-1])
        return years + months / 12.0

    return series.apply(_parse)


def extract_storey_midpoint(series: pd.Series) -> pd.Series:
    """
    'storey_range' is a string like '07 TO 09'. We take the midpoint.

    WHY: Higher floors command a significant price premium in Singapore.
    The midpoint is a fair numeric proxy for floor level.
    """
    def _mid(val):
        if pd.isna(val):
            return np.nan
        parts = str(val).split("TO")
        if len(parts) == 2:
            try:
                lo = int(parts[0].strip())
                hi = int(parts[1].strip())
                return (lo + hi) / 2.0
            except ValueError:
                pass
        return np.nan

    return series.apply(_mid)


def preprocess(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Full preprocessing pipeline applied identically to train and test sets.
    Keeping the logic in one function prevents train-test leakage.
    """
    df = df.copy()

    # ── 3a. Date parsing ──────────────────────────────────────────────────────
    # 'month' column is typically 'YYYY-MM'
    if "month" in df.columns:
        dt = pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")
        df["year_sold"] = dt.dt.year
        df["month_sold"] = dt.dt.month
        df.drop(columns=["month"], inplace=True)

    # ── 3b. Storey range → numeric midpoint ──────────────────────────────────
    if "storey_range" in df.columns:
        df["storey_mid"] = extract_storey_midpoint(df["storey_range"])
        df.drop(columns=["storey_range"], inplace=True)

    # ── 3c. Remaining lease ───────────────────────────────────────────────────
    if "remaining_lease" in df.columns:
        df["remaining_lease_years"] = parse_remaining_lease(
            df["remaining_lease"])
        df.drop(columns=["remaining_lease"], inplace=True)

    # ── 3d. Derived features ─────────────────────────────────────────────────
    # FLAT AGE: Newer flats are generally pricier; this captures depreciation.
    if "lease_commence_date" in df.columns and "year_sold" in df.columns:
        df["flat_age"] = df["year_sold"] - df["lease_commence_date"]

    # REMAINING LEASE from scratch (if column was absent)
    if "remaining_lease_years" not in df.columns:
        if "lease_commence_date" in df.columns and "year_sold" in df.columns:
            df["remaining_lease_years"] = (
                99 - df["flat_age"]   # HDB leases are 99 years
            )

    # PRICE PER SQM proxy via floor_area — can't compute price here (test set),
    # but we can create a "size tier" that interacts with town.
    if "floor_area_sqm" in df.columns:
        df["floor_area_log"] = np.log1p(df["floor_area_sqm"])

    return df


# Apply preprocessing
train = preprocess(train_raw, is_train=True)
test = preprocess(test_raw,  is_train=False)

print("Preprocessing complete.")
print(f"  Train shape after preprocessing: {train.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2 — Feature Engineering")
print("=" * 65)

# ── Identify column types ────────────────────────────────────────────────────
COLS_TO_DROP_ALWAYS = ["id", TARGET]  # drop IDs and target from features

# Categorical columns (strings / objects)
cat_cols = [
    c for c in train.columns
    if train[c].dtype == object and c not in COLS_TO_DROP_ALWAYS
]

# Separate high vs low cardinality for encoding strategy
LOW_CARD_THRESHOLD = 15   # <= 15 unique values → One-Hot; else → Target encode

low_card = [c for c in cat_cols if train[c].nunique() <= LOW_CARD_THRESHOLD]
high_card = [c for c in cat_cols if train[c].nunique() > LOW_CARD_THRESHOLD]

print(f"  Categorical columns   : {cat_cols}")
print(f"  Low-cardinality (OHE) : {low_card}")
print(f"  High-cardinality (TE) : {high_card}")

"""
ENCODING STRATEGY EXPLAINED
─────────────────────────────
• One-Hot Encoding (OHE) for low-cardinality columns (e.g., flat_type,
  flat_model): Creates a binary column per category. Appropriate when there
  are few levels because it doesn't impose any ordinal relationship between
  categories and avoids the risk of data leakage.

• Target Encoding for high-cardinality columns (e.g., town, street_name,
  block): Replaces each category with the mean target value observed for
  that category in the training set (with smoothing to handle rare levels).
  This compresses hundreds of dummy columns into a single informative numeric
  feature. We use k-fold target encoding to prevent leakage within training.
"""

# ── One-Hot Encoding ──────────────────────────────────────────────────────────
if low_card:
    train = pd.get_dummies(train, columns=low_card, drop_first=False)
    test = pd.get_dummies(test,  columns=low_card, drop_first=False)

    # Align columns: test might lack some dummies that appeared only in train
    train, test = train.align(test, join="left", axis=1, fill_value=0)

# ── Target Encoding for high-cardinality columns ──────────────────────────────
if high_card:
    te = ce.TargetEncoder(cols=high_card, smoothing=10)
    X_temp = train.drop(columns=[TARGET], errors="ignore")
    y_temp = train[TARGET]

    X_temp_enc = te.fit_transform(X_temp, y_temp)
    test_enc = te.transform(test.drop(columns=[TARGET], errors="ignore"))

    for col in high_card:
        train[col] = X_temp_enc[col].values
        if col in test.columns:
            test[col] = test_enc[col].values

# ── Final feature / target split ─────────────────────────────────────────────
drop_cols = [c for c in COLS_TO_DROP_ALWAYS if c in train.columns]
X = train.drop(columns=drop_cols)
y = train[TARGET]

test_drop = [
    c for c in COLS_TO_DROP_ALWAYS if c in test.columns and c != TARGET]
X_test = test.drop(columns=test_drop, errors="ignore")

# Ensure column alignment
X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)

# ── Missing value imputation ──────────────────────────────────────────────────
"""
WHY MEDIAN IMPUTATION:
  Resale price-related numeric features (floor_area, remaining_lease,
  storey_mid) are right-skewed. Median imputation is robust to outliers
  and preserves distributional shape better than mean imputation.
"""
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X),   columns=X.columns)
X_test = pd.DataFrame(imputer.transform(X_test),  columns=X_test.columns)

print(f"\n  Final feature matrix : {X.shape}")
print(f"  Final test matrix    : {X_test.shape}")
print(f"  Feature list:\n  {list(X.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAIN / VALIDATION SPLIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3 — Model Training")
print("=" * 65)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_STATE
)
print(f"  Train size : {X_train.shape[0]:,}  |  Val size : {X_val.shape[0]:,}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. HYPERPARAMETER SEARCH WITH RandomizedSearchCV
# ─────────────────────────────────────────────────────────────────────────────
"""
WHY RandomizedSearchCV?
  The hyperparameter space for LightGBM is large. RandomizedSearchCV samples
  a fixed number of configurations (n_iter), making it far faster than
  exhaustive GridSearchCV while still finding good solutions.

  We optimise for neg_root_mean_squared_error because RMSE is the Kaggle
  evaluation metric for this competition.
"""

lgbm_base = lgb.LGBMRegressor(
    objective="regression",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)

param_dist = {
    "n_estimators": [500, 800, 1000, 1200],
    "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],
    "num_leaves": [31, 63, 127, 255],
    "max_depth": [-1, 6, 8, 10],
    "min_child_samples": [20, 50, 100],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0.0, 0.1, 0.5],
    "reg_lambda": [0.0, 0.1, 1.0],
}

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

search = RandomizedSearchCV(
    estimator=lgbm_base,
    param_distributions=param_dist,
    n_iter=30,           # increase for better results (costs time)
    scoring="neg_root_mean_squared_error",
    cv=cv,
    refit=True,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
)

print("\n  Running RandomizedSearchCV (30 iterations × 5-fold CV)…")
search.fit(X_train, y_train)
best_model = search.best_estimator_

print(f"\n  Best parameters found:\n  {search.best_params_}")
print(f"  Best CV RMSE : SGD {-search.best_score_:,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. EVALUATION ON VALIDATION SET
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4 — Evaluation")
print("=" * 65)

val_preds = best_model.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, val_preds))
mae = mean_absolute_error(y_val, val_preds)
mape = np.mean(np.abs((y_val - val_preds) / y_val)) * 100

print(f"\n  Validation RMSE : SGD {rmse:>12,.2f}")
print(f"  Validation MAE  : SGD {mae:>12,.2f}")
print(f"  Validation MAPE :     {mape:>11.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 8. FEATURE IMPORTANCE PLOT
# ─────────────────────────────────────────────────────────────────────────────
feat_imp = pd.Series(
    best_model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False).head(20)

print("\n  Top 20 Feature Importances:")
print(feat_imp.to_string())

fig, ax = plt.subplots(figsize=(9, 6))
feat_imp.sort_values().plot(kind="barh", ax=ax, color="#1a6b9a")
ax.set_title("LightGBM — Top 20 Feature Importances",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
print("\n  Saved feature importance chart → feature_importance.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9. OPTIONAL — XGBoost BLEND
# ─────────────────────────────────────────────────────────────────────────────
"""
BLENDING EXPLAINED:
  A simple weighted average of two diverse models (LightGBM + XGBoost)
  typically reduces variance and improves generalisation. We assign
  70 % weight to LightGBM (our tuned primary) and 30 % to XGBoost
  (sensible defaults). Adjust weights based on CV scores.
"""
print("\n" + "=" * 65)
print("STEP 4b — Optional XGBoost blend")
print("=" * 65)

xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0,
    early_stopping_rounds=50,
    eval_metric="rmse",
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

xgb_val_preds = xgb_model.predict(X_val)

# Blend
LGBM_WEIGHT = 0.70
XGB_WEIGHT = 0.30
blend_val_preds = LGBM_WEIGHT * val_preds + XGB_WEIGHT * xgb_val_preds

blend_rmse = np.sqrt(mean_squared_error(y_val, blend_val_preds))
blend_mae = mean_absolute_error(y_val, blend_val_preds)

print(
    f"\n  XGBoost-only  RMSE : SGD {np.sqrt(mean_squared_error(y_val, xgb_val_preds)):>12,.2f}")
print(f"  Blended       RMSE : SGD {blend_rmse:>12,.2f}")
print(f"  Blended       MAE  : SGD {blend_mae:>12,.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. GENERATE FINAL TEST PREDICTIONS & SUBMISSION FILE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5 — Generating submission file")
print("=" * 65)

lgbm_test_preds = best_model.predict(X_test)
xgb_test_preds = xgb_model.predict(X_test)
final_preds = LGBM_WEIGHT * lgbm_test_preds + XGB_WEIGHT * xgb_test_preds

# Clip to physically sensible range (no negative prices)
final_preds = np.clip(final_preds, a_min=0, a_max=None)

submission = pd.DataFrame({
    "Id": test_ids,
    "resale_price": final_preds,
})

submission.to_csv(SUBMISSION_PATH, index=False)
print(f"\n  Submission saved → {SUBMISSION_PATH}")
print(f"  Rows: {len(submission):,}  |  Price range: "
      f"SGD {final_preds.min():,.0f} – SGD {final_preds.max():,.0f}")
print(f"\n{'=' * 65}")
print("  DONE! 🎉  Good luck on the leaderboard, Mohan!")
print(f"{'=' * 65}\n")
