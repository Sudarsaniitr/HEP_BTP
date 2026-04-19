import numpy as np
import pandas as pd
import statsmodels.api as sm

from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score,
    f1_score, roc_curve
)

# ============================================================
# 1. LOAD DATA
# ============================================================

hrv_df = pd.read_csv("sleep_hrv_dataset_paper_faithful_old.csv")
hep_df = pd.read_csv("hep_dataset_FINAL_CLEAN.csv")

# ============================================================
# 2. CLEAN COLUMN NAMES (STRONG CLEANING)
# ============================================================

def clean_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("/", "_")       # FIX LF/HF
        .str.replace("-", "_")
        .str.replace(r"[^\w_]", "", regex=True)
        .str.lower()
    )
    return df

hrv_df = clean_columns(hrv_df)
hep_df = clean_columns(hep_df)

# ============================================================
# 3. MERGE (ONLY COMMON DATA)
# ============================================================

df = pd.merge(
    hrv_df,
    hep_df,
    on=["subject_id", "window_id"],
    how="inner"
)

print("Merged shape:", df.shape)

# ============================================================
# 4. REMOVE DUPLICATE COLUMNS
# ============================================================

df = df.loc[:, ~df.columns.duplicated()]

# ============================================================
# 5. DEBUG COLUMN STATE
# ============================================================

print("\nColumns after cleaning:\n", df.columns.tolist())

# ============================================================
# 6. FIX HRV FEATURES (AUTO-DETECT)
# ============================================================

HRV_FEATURES = []

hrv_keywords = ["sdnn", "rmssd", "lf", "hf", "lf_hf", "sampen"]

for key in hrv_keywords:
    matches = [c for c in df.columns if key in c]
    if matches:
        HRV_FEATURES.append(matches[0])
    else:
        print(f"⚠ Missing HRV feature: {key}")

print("\nFinal HRV features:", HRV_FEATURES)

# ============================================================
# 7. FIX TARGET (AUTO-DETECT)
# ============================================================

targets = [c for c in df.columns if "is_w" in c]

if not targets:
    print("\n❌ Available labels:")
    print([c for c in df.columns if "is_" in c])
    raise ValueError("No valid target found")

TARGET = targets[0]

print("\nUsing TARGET:", TARGET)

# ============================================================
# 8. HEP FEATURES
# ============================================================

channels = ["fc6","poz","c2","f6","po8","fp1","p6","ft10","cz","ft8"]

HEP_FEATURES = [
    c for c in df.columns
    if any(ch in c for ch in channels)
]

HEP_FEATURES = [
    c for c in HEP_FEATURES
    if not c.startswith("is_") and not c.startswith("trans_")
]

print("Total HEP features:", len(HEP_FEATURES))

# ============================================================
# 9. FINAL CLEAN
# ============================================================

required = HRV_FEATURES + HEP_FEATURES + [TARGET]
required = [c for c in required if c in df.columns]

df = df.dropna(subset=required)

print("Final dataset shape:", df.shape)

# ============================================================
# 10. SUBJECT SPLIT
# ============================================================

subjects = df["subject_id"].unique()

np.random.seed(42)
np.random.shuffle(subjects)

n_train = int(0.7 * len(subjects))

train_subjects = subjects[:n_train]
test_subjects  = subjects[n_train:]

train_df = df[df["subject_id"].isin(train_subjects)].copy()
test_df  = df[df["subject_id"].isin(test_subjects)].copy()

print("\nTrain subjects:", train_subjects)
print("Test subjects :", test_subjects)

# ============================================================
# 11. SCALING
# ============================================================

def scale(train, test, features):
    scaler = StandardScaler()
    train[features] = scaler.fit_transform(train[features])
    test[features] = scaler.transform(test[features])
    return train, test

# ============================================================
# 12. GEE MODEL
# ============================================================

def fit_gee(data, features, target):
    X = sm.add_constant(data[features])
    y = data[target]

    model = GEE(
        endog=y,
        exog=X,
        groups=data["subject_id"],
        family=Binomial()
    )
    return model.fit()

# ============================================================
# 13. BACKWARD AIC
# ============================================================

def backward_aic(train_data, features, target):

    current = features.copy()
    best_model = fit_gee(train_data, current, target)
    best_aic = best_model.aic

    print(f"\nInitial AIC ({len(current)} features): {best_aic:.2f}")

    while len(current) > 1:

        candidates = []

        for f in current:
            reduced = [x for x in current if x != f]
            try:
                model = fit_gee(train_data, reduced, target)
                candidates.append((model.aic, f, model))
            except:
                continue

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0])
        new_aic, removed, new_model = candidates[0]

        if new_aic < best_aic:
            print(f"Removed {removed}")
            best_aic = new_aic
            current.remove(removed)
            best_model = new_model
        else:
            break

    return best_model, current

# ============================================================
# 14. EVALUATION
# ============================================================

def evaluate(model, test_df, features, target):

    X = sm.add_constant(test_df[features])
    y = test_df[target]

    y_prob = model.predict(X)

    fpr, tpr, thresholds = roc_curve(y, y_prob)
    j = tpr - fpr
    thresh = thresholds[np.argmax(j)]

    y_pred = (y_prob >= thresh).astype(int)

    return {
        "AUC": roc_auc_score(y, y_prob),
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0)
    }

# ============================================================
# 15. RUN MODELS
# ============================================================

results = {}

# ---- HRV ----
train_h, test_h = scale(train_df.copy(), test_df.copy(), HRV_FEATURES)
model_h, feat_h = backward_aic(train_h, HRV_FEATURES, TARGET)

print("\nHRV Selected Features:", feat_h)
results["HRV"] = evaluate(model_h, test_h, feat_h, TARGET)

# ---- HEP ----
train_e, test_e = scale(train_df.copy(), test_df.copy(), HEP_FEATURES)
model_e, feat_e = backward_aic(train_e, HEP_FEATURES, TARGET)

print("\nHEP Selected Features:", feat_e)
results["HEP"] = evaluate(model_e, test_e, feat_e, TARGET)

# ---- COMBINED ----
ALL = HRV_FEATURES + HEP_FEATURES

train_c, test_c = scale(train_df.copy(), test_df.copy(), ALL)
model_c, feat_c = backward_aic(train_c, ALL, TARGET)

print("\nCOMBINED Selected Features:", feat_c)
results["COMBINED"] = evaluate(model_c, test_c, feat_c, TARGET)

# ============================================================
# 16. RESULTS
# ============================================================

print("\n===== FINAL RESULTS =====")

for k, v in results.items():
    print(f"\n{k}")
    for m, val in v.items():
        print(f"{m}: {val:.3f}")

print("\n===== MODEL SUMMARY (COMBINED) =====\n")
print(model_c.summary())