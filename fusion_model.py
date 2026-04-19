# ============================================================
# MULTIMODAL SLEEP STAGE DETECTION
# HRV + HEP LATE FUSION (STABLE VERSION)
# Leave-One-Subject-Out Cross Validation
# ============================================================

import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings

from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve
)

warnings.filterwarnings("ignore")

# ============================================================
# 1 LOAD DATASETS
# ============================================================

hep = pd.read_csv("HEP_5min_stagewise.csv")
hrv = pd.read_csv("sleep_hrv_dataset_paper_faithful_old.csv")

print("HEP dataset:", hep.shape)
print("HRV dataset:", hrv.shape)

# ============================================================
# 2 FIX SUBJECT IDS
# ============================================================

hep["subject_id"] = hep["subject"].str.replace("EPCTL","").astype(int)
hrv["subject_id"] = hrv["subject_id"].astype(int)

# ============================================================
# 3 FIX WINDOW COLUMN
# ============================================================

hep = hep.rename(columns={"window_index":"epoch"})
hrv = hrv.rename(columns={"window_id":"epoch"})

# ============================================================
# 4 KEEP COMMON SUBJECTS
# ============================================================

common_subjects = sorted(
    set(hep["subject_id"]).intersection(set(hrv["subject_id"]))
)

print("Common subjects:", common_subjects)

hep = hep[hep["subject_id"].isin(common_subjects)]
hrv = hrv[hrv["subject_id"].isin(common_subjects)]

# ============================================================
# 5 CREATE TARGET
# ============================================================

TARGET_STAGE = "N2"

hep["target"] = (hep["stage"] == TARGET_STAGE).astype(int)
hrv["target"] = hrv["is_N2"]

# ============================================================
# 6 MERGE DATASETS
# ============================================================

merged = pd.merge(
    hrv,
    hep,
    on=["subject_id","epoch"],
    how="inner",
    suffixes=("_hrv","_hep")
)

merged["target"] = merged["target_hep"]

print("Merged dataset:", merged.shape)

# ============================================================
# 7 FEATURE SETS
# ============================================================

HRV_FEATURES = ["SDNN","RMSSD","LF","HF","LF/HF","SampEn"]

HEP_FEATURES = [
    c for c in hep.columns
    if c not in ["subject","subject_id","epoch","stage","target"]
]

# ============================================================
# 8 SUBJECT-WISE NORMALIZATION (EEG)
# ============================================================

merged[HEP_FEATURES] = merged.groupby("subject_id")[HEP_FEATURES].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-6)
)

# ============================================================
# 9 GEE MODEL FUNCTION
# ============================================================

def fit_gee(data, features):

    X = sm.add_constant(data[features])
    y = data["target"]

    model = GEE(
        endog=y,
        exog=X,
        groups=data["subject_id"],
        family=Binomial()
    )

    return model.fit()

# ============================================================
# 10 BACKWARD AIC FEATURE SELECTION
# ============================================================

def backward_aic_selection(train_df, features):

    current = features.copy()

    best_model = fit_gee(train_df, current)
    best_aic = best_model.aic

    improved = True

    while improved and len(current) > 5:

        improved = False
        candidates = []

        for f in current:

            reduced = [x for x in current if x != f]

            try:
                model = fit_gee(train_df, reduced)
                candidates.append((model.aic,f,model))
            except:
                continue

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0])

        cand_aic, removed, cand_model = candidates[0]

        if cand_aic < best_aic:

            best_aic = cand_aic
            current.remove(removed)
            best_model = cand_model
            improved = True

    return best_model, current

# ============================================================
# 11 EVALUATION FUNCTION
# ============================================================

def evaluate(y_true, y_prob):

    y_prob = np.clip(y_prob,1e-6,1-1e-6)

    fpr,tpr,thr = roc_curve(y_true,y_prob)

    j = tpr-fpr
    best = thr[np.argmax(j)]

    y_pred = (y_prob>=best).astype(int)

    return {
        "AUC":roc_auc_score(y_true,y_prob),
        "Accuracy":accuracy_score(y_true,y_pred),
        "Precision":precision_score(y_true,y_pred,zero_division=0),
        "Recall":recall_score(y_true,y_pred,zero_division=0),
        "F1":f1_score(y_true,y_pred,zero_division=0)
    }

# ============================================================
# 12 LOSO CROSS VALIDATION
# ============================================================

subjects = merged["subject_id"].unique()

results = []

for test_subject in subjects:

    print("\nTesting subject:",test_subject)

    train_df = merged[merged["subject_id"]!=test_subject].copy()
    test_df  = merged[merged["subject_id"]==test_subject].copy()

    # ------------------------
    # STANDARDIZATION
    # ------------------------

    scaler_hrv = StandardScaler()
    scaler_hep = StandardScaler()

    train_df[HRV_FEATURES] = scaler_hrv.fit_transform(train_df[HRV_FEATURES])
    test_df[HRV_FEATURES] = scaler_hrv.transform(test_df[HRV_FEATURES])

    train_df[HEP_FEATURES] = scaler_hep.fit_transform(train_df[HEP_FEATURES])
    test_df[HEP_FEATURES] = scaler_hep.transform(test_df[HEP_FEATURES])

    # ------------------------
    # HRV MODEL
    # ------------------------

    hrv_model = fit_gee(train_df,HRV_FEATURES)

    train_df["p_hrv"] = hrv_model.predict(sm.add_constant(train_df[HRV_FEATURES]))
    test_df["p_hrv"] = hrv_model.predict(sm.add_constant(test_df[HRV_FEATURES]))

    train_df["p_hrv"] = np.clip(train_df["p_hrv"],1e-6,1-1e-6)
    test_df["p_hrv"] = np.clip(test_df["p_hrv"],1e-6,1-1e-6)

    # ------------------------
    # HEP MODEL (FEATURE SELECTION)
    # ------------------------

    hep_model,selected_hep = backward_aic_selection(train_df,HEP_FEATURES)

    train_df["p_hep"] = hep_model.predict(sm.add_constant(train_df[selected_hep]))
    test_df["p_hep"] = hep_model.predict(sm.add_constant(test_df[selected_hep]))

    train_df["p_hep"] = np.clip(train_df["p_hep"],1e-6,1-1e-6)
    test_df["p_hep"] = np.clip(test_df["p_hep"],1e-6,1-1e-6)

    # ------------------------
    # CLEAN NAN/INF
    # ------------------------

    train_df = train_df.replace([np.inf,-np.inf],np.nan)
    test_df = test_df.replace([np.inf,-np.inf],np.nan)

    train_df = train_df.dropna(subset=["p_hrv","p_hep"])
    test_df = test_df.dropna(subset=["p_hrv","p_hep"])

    # ------------------------
    # FUSION MODEL
    # ------------------------

    fusion_model = fit_gee(train_df,["p_hrv","p_hep"])

    y_prob = fusion_model.predict(
        sm.add_constant(test_df[["p_hrv","p_hep"]])
    )

    metrics = evaluate(test_df["target"],y_prob)

    results.append(metrics)

# ============================================================
# 13 FINAL RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n================================")
print("FINAL LOSO RESULTS (AVERAGE)")
print("================================")

print(results_df.mean())