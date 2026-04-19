# ============================================================
# MULTI-SUBJECT GEE SLEEP STAGE MODEL
# ROC-BASED THRESHOLDING (PAPER-FAITHFUL)
# ============================================================

import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve
)

# ============================================================
# 1. LOAD CLEAN DATASET
# ============================================================

df = pd.read_csv("sleep_hrv_dataset_paper_faithful_old.csv")

print("Dataset shape:", df.shape)
print("Subjects:", sorted(df["subject_id"].unique()))
print("No. of subjects:", df["subject_id"].nunique())

# ============================================================
# 2. FEATURE SET (HRV BASELINE)
# ============================================================

FEATURES = ["SDNN", "RMSSD", "LF", "HF", "LF/HF","SampEn"]
TARGET = "is_W"   # change to is_N2, is_R, etc.

# Drop rows with missing values
df = df.dropna(subset=FEATURES + [TARGET])

# ============================================================
# 3. SUBJECT-WISE TRAIN / TEST SPLIT (AS IN PAPER)
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
print("Train rows:", len(train_df))
print("Test rows :", len(test_df))

# ============================================================
# 4. FEATURE STANDARDIZATION (MANDATORY)
# ============================================================

scaler = StandardScaler()
train_df[FEATURES] = scaler.fit_transform(train_df[FEATURES])
test_df[FEATURES] = scaler.transform(test_df[FEATURES])

# ============================================================
# 5. GEE FIT FUNCTION
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
# 6. BACKWARD AIC FEATURE SELECTION (TRAIN ONLY)
# ============================================================

def backward_aic_selection(train_data, features, target):
    current_features = features.copy()

    best_model = fit_gee(train_data, current_features, target)
    best_aic = best_model.aic

    print(f"Initial AIC ({len(current_features)} features): {best_aic:.2f}")

    improved = True
    while improved and len(current_features) > 1:
        improved = False
        candidates = []

        for f in current_features:
            reduced = [x for x in current_features if x != f]
            try:
                model = fit_gee(train_data, reduced, target)
                candidates.append((model.aic, f, model))
            except Exception:
                continue

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0])
        cand_aic, removed_feature, cand_model = candidates[0]

        if cand_aic < best_aic:
            print(f"Removed {removed_feature} | AIC: {best_aic:.2f} → {cand_aic:.2f}")
            best_aic = cand_aic
            current_features.remove(removed_feature)
            best_model = cand_model
            improved = True

    print("Final selected features:", current_features)
    return best_model, current_features

# ============================================================
# 7. TRAIN FINAL MODEL
# ============================================================

model, selected_features = backward_aic_selection(
    train_df, FEATURES, TARGET
)

print("\n===== FINAL GEE MODEL SUMMARY =====\n")
print(model.summary())

# ============================================================
# 8. TESTING WITH ROC-BASED THRESHOLD (YOU DEN J)
# ============================================================

X_test = sm.add_constant(test_df[selected_features])
y_test = test_df[TARGET]

# Probabilities
y_prob = model.predict(X_test)

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Youden's J statistic
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]

print("\nSelected threshold (Youden's J):", round(best_threshold, 3))

# Final predictions
y_pred = (y_prob >= best_threshold).astype(int)

# ============================================================
# 9. FINAL METRICS (PAPER-CONSISTENT)
# ============================================================

auc = roc_auc_score(y_test, y_prob)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("\n===== TEST SET PERFORMANCE (ROC-BASED) =====")
print("AUC       :", round(auc, 3))
print("Accuracy  :", round(acc, 3))
print("Precision :", round(prec, 3))
print("Recall    :", round(rec, 3))
print("F1-score  :", round(f1, 3))
print("Confusion matrix:\n", cm)

print("\nPipeline completed successfully.")
