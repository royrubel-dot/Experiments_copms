import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

# =============================================================
# 1. LOAD BOTH SHEETS
# =============================================================

FILE = r"C:\Users\rubro\OneDrive\Desktop\Azimuth\coms_tests\tests-_4_12\synthetic_merged_with_features.xlsx"

df_syn = pd.read_excel(FILE, sheet_name="SYNTHETIC_1")
df_real = pd.read_excel(FILE, sheet_name="REAL")

print("Synthetic:", df_syn.shape)
print("Real:", df_real.shape)

# =============================================================
# 2. SETUP TARGET + BASELINE
# =============================================================

TARGET_PRICE = "rebased_valuation_amount"
BASELINE = "anchor_weighted_price"

# Add valuation_base if missing
if "valuation_base" not in df_syn.columns:
    df_syn["valuation_base"] = df_syn["valuation_id"].str.split("_").str[0]
    print("⚠️ ORIGINAL ID NOT FOUND — using valuation_base as origin.")

df_real = df_real.dropna(subset=[TARGET_PRICE, BASELINE]).copy()
df_syn = df_syn.dropna(subset=[TARGET_PRICE, BASELINE]).copy()

# =============================================================
# 3. TRAIN/TEST SPLIT (REAL DATA ONLY)
# =============================================================

df_real_train, df_real_test = train_test_split(df_real, test_size=0.2, random_state=42)

train_ids = set(df_real_train["valuation_id"])

# =============================================================
# 4. FILTER SYNTHETIC (LEAKAGE SAFE)
# =============================================================

df_syn["origin"] = df_syn["valuation_base"]
df_syn_filtered = df_syn[df_syn["origin"].isin(train_ids)].copy()

print("Synthetic kept:", df_syn_filtered.shape)

# =============================================================
# 5. BUILD TRAINING SET
# =============================================================

df_train_all = pd.concat([df_real_train, df_syn_filtered], ignore_index=True)

# =============================================================
# 6. LOG TRANSFORMS & RESIDUAL TARGET
# =============================================================

df_train_all["log_price"] = np.log(df_train_all[TARGET_PRICE])
df_train_all["log_anchor"] = np.log(df_train_all[BASELINE])
df_train_all["log_residual"] = df_train_all["log_price"] - df_train_all["log_anchor"]

df_real_test["log_price"] = np.log(df_real_test[TARGET_PRICE])
df_real_test["log_anchor"] = np.log(df_real_test[BASELINE])
df_real_test["log_residual"] = df_real_test["log_price"] - df_real_test["log_anchor"]

TARGET = "log_residual"

# =============================================================
# 7. FEATURES
# =============================================================

exclude_cols = [
    TARGET_PRICE, "log_price", "log_anchor", "log_residual",
    "valuation_id", "source_folder", "mj_valuation_date",
    "valuation_base", "origin"
]

FEATURES = [c for c in df_train_all.columns if c not in exclude_cols]

# Only keep features present in BOTH train & test
FEATURES = [c for c in FEATURES if c in df_real_test.columns]

print("Final feature count:", len(FEATURES))

# =============================================================
# 8. CATEGORICAL HANDLING
# =============================================================

cat_cols = df_train_all[FEATURES].select_dtypes(include=["object"]).columns.tolist()

for col in cat_cols:
    df_train_all[col] = df_train_all[col].fillna("Unknown").astype(str)
    df_real_test[col] = df_real_test[col].fillna("Unknown").astype(str)

cat_idx = [FEATURES.index(c) for c in cat_cols]

train_pool = Pool(df_train_all[FEATURES], df_train_all[TARGET], cat_features=cat_idx)
test_pool  = Pool(df_real_test[FEATURES], df_real_test[TARGET], cat_features=cat_idx)

# =============================================================
# 9. OPTUNA OBJECTIVE FUNCTION
# =============================================================

def objective(trial):

    params = {
        "loss_function": "RMSE",
        "eval_metric": "MAE",
        "depth": trial.suggest_int("depth", 4, 9),
        "iterations": trial.suggest_int("iterations", 500, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 20),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 5),
        "random_strength": trial.suggest_float("random_strength", 0, 10),
        "rsm": trial.suggest_float("rsm", 0.7, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 20),
        "verbose": False,
        "random_seed": 42,
        "boosting_type": "Plain",
        "grow_policy": "Depthwise"
    }

    model = CatBoostRegressor(**params)

    model.fit(
        train_pool,
        eval_set=test_pool,
        early_stopping_rounds=100,
        verbose=False
    )

    # Predict log residual
    log_resid_pred = model.predict(test_pool)

    # Reconstruct price
    log_price_pred = df_real_test["log_anchor"] + log_resid_pred
    pred_price = np.exp(log_price_pred)
    true_price = df_real_test[TARGET_PRICE].values

    MAPE = mean_absolute_percentage_error(true_price, pred_price) * 100
    return MAPE

# =============================================================
# 10. RUN OPTUNA
# =============================================================

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("\n==== BEST HYPERPARAMETERS ====\n")
print(study.best_params)
print("\nBest validation MAPE:", study.best_value)

# =============================================================
# 11. TRAIN FINAL MODEL
# =============================================================

best_params = study.best_params
best_params.update({
    "loss_function": "RMSE",
    "eval_metric": "MAE",
    "verbose": False,
    "random_seed": 42,
    "boosting_type": "Plain",
    "grow_policy": "Depthwise"
})

final_model = CatBoostRegressor(**best_params)
final_model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=100, verbose=False)

# =============================================================
# 12. FINAL PREDICTIONS
# =============================================================

log_resid_pred = final_model.predict(test_pool)
log_price_pred = df_real_test["log_anchor"] + log_resid_pred

pred_price = np.exp(log_price_pred)
true_price = df_real_test[TARGET_PRICE].values

final_mape = mean_absolute_percentage_error(true_price, pred_price) * 100

print("\n=================== FINAL MODEL MAPE ===================")
print(f"FINAL MAPE: {final_mape:.3f}%")
print("========================================================\n")

# =============================================================
# 13. SAVE RESULTS
# =============================================================

df_out = df_real_test.copy()
df_out["predicted_price"] = pred_price
df_out["abs_error"] = abs(true_price - pred_price)
df_out["pct_error"] = df_out["abs_error"] / true_price * 100

out_file = r"C:\Users\rubro\OneDrive\Desktop\Azimuth\AVM_residual_optuna_catboost.xlsx"
df_out.to_excel(out_file, index=False)

print("💾 Saved predictions to:", out_file)
