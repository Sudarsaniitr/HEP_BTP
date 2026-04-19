import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import mne

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
# 2. CLEAN COLUMN NAMES
# ============================================================

def clean_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("/", "_")
        .str.replace("-", "_")
        .str.replace(r"[^\w_]", "", regex=True)
        .str.lower()
    )
    return df

hrv_df = clean_columns(hrv_df)
hep_df = clean_columns(hep_df)

# ============================================================
# 3. MERGE
# ============================================================

df = pd.merge(hrv_df, hep_df, on=["subject_id","window_id"], how="inner")
df = df.loc[:, ~df.columns.duplicated()]

print("Merged shape:", df.shape)

# ============================================================
# 4. FEATURES
# ============================================================

HRV_FEATURES = []
for key in ["sdnn","rmssd","lf","hf","lf_hf","sampen"]:
    matches = [c for c in df.columns if key in c]
    if matches:
        HRV_FEATURES.append(matches[0])

channels = ["fc6","poz","c2","f6","po8","fp1","p6","ft10","cz","ft8"]

HEP_FEATURES = [
    c for c in df.columns
    if any(ch in c for ch in channels)
]

HEP_FEATURES = [
    c for c in HEP_FEATURES
    if not c.startswith("is_") and not c.startswith("trans_")
]

print("HRV:", HRV_FEATURES)
print("HEP count:", len(HEP_FEATURES))

# ============================================================
# 5. FUNCTIONS
# ============================================================

def scale(train, test, features):
    scaler = StandardScaler()
    train[features] = scaler.fit_transform(train[features])
    test[features] = scaler.transform(test[features])
    return train, test

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

def backward_aic(train_data, features, target):

    current = features.copy()
    best_model = fit_gee(train_data, current, target)
    best_aic = best_model.aic

    print(f"Initial AIC ({len(current)}): {best_aic:.2f}")

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
            print("Removed:", removed)
            best_aic = new_aic
            current.remove(removed)
            best_model = new_model
        else:
            break

    return best_model, current

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
        "F1": f1_score(y, y_pred, zero_division=0),
        "y_true": y.values,
        "y_prob": y_prob
    }

# ============================================================
# 6. MULTI-STAGE LOOP
# ============================================================

targets = [c for c in df.columns if c.startswith("is_")]

ALL_RESULTS = {}
ALL_MODELS = {}

for TARGET in targets:

    print(f"\n===== {TARGET.upper()} =====")

    required = HRV_FEATURES + HEP_FEATURES + [TARGET]
    required = [c for c in required if c in df.columns]

    df_stage = df.dropna(subset=required)

    subjects = df_stage["subject_id"].unique()
    np.random.seed(42)
    np.random.shuffle(subjects)

    n_train = int(0.7 * len(subjects))

    train_df = df_stage[df_stage["subject_id"].isin(subjects[:n_train])]
    test_df  = df_stage[df_stage["subject_id"].isin(subjects[n_train:])]

    results = {}

    # HRV
    train_h, test_h = scale(train_df.copy(), test_df.copy(), HRV_FEATURES)
    model_h, feat_h = backward_aic(train_h, HRV_FEATURES, TARGET)
    results["HRV"] = evaluate(model_h, test_h, feat_h, TARGET)

    # HEP
    train_e, test_e = scale(train_df.copy(), test_df.copy(), HEP_FEATURES)
    model_e, feat_e = backward_aic(train_e, HEP_FEATURES, TARGET)
    results["HEP"] = evaluate(model_e, test_e, feat_e, TARGET)

    # COMBINED
    ALL = HRV_FEATURES + HEP_FEATURES
    train_c, test_c = scale(train_df.copy(), test_df.copy(), ALL)
    model_c, feat_c = backward_aic(train_c, ALL, TARGET)
    results["COMBINED"] = evaluate(model_c, test_c, feat_c, TARGET)

    ALL_RESULTS[TARGET] = results
    ALL_MODELS[TARGET] = model_c

    print("AUC:",
          results["HRV"]["AUC"],
          results["HEP"]["AUC"],
          results["COMBINED"]["AUC"])

    # ROC PER STAGE
    plt.figure()

    for name in ["HRV","HEP","COMBINED"]:
        y_true = results[name]["y_true"]
        y_prob = results[name]["y_prob"]

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr, label=name)

    plt.plot([0,1],[0,1],'--')
    plt.title(TARGET)
    plt.legend()
    plt.show()

# ============================================================
# 7. FINAL PLOTS
# ============================================================

stages = list(ALL_RESULTS.keys())

hrv_auc = [ALL_RESULTS[s]["HRV"]["AUC"] for s in stages]
hep_auc = [ALL_RESULTS[s]["HEP"]["AUC"] for s in stages]
comb_auc = [ALL_RESULTS[s]["COMBINED"]["AUC"] for s in stages]

# LINE PLOT
plt.figure()
plt.plot(stages, hrv_auc, marker='o', label="HRV")
plt.plot(stages, hep_auc, marker='o', label="HEP")
plt.plot(stages, comb_auc, marker='o', label="Combined")
plt.legend()
plt.title("Stage-wise Performance")
plt.grid()
plt.show()

# Δ AUC
improvement = [comb_auc[i] - hrv_auc[i] for i in range(len(stages))]
plt.figure()
plt.plot(stages, improvement, marker='o')
plt.axhline(0, linestyle='--')
plt.title("Improvement (Combined - HRV)")
plt.grid()
plt.show()

# HEATMAP
data = pd.DataFrame({
    "HRV": hrv_auc,
    "HEP": hep_auc,
    "Combined": comb_auc
}, index=[s.upper() for s in stages])

sns.heatmap(data, annot=True, cmap="coolwarm")
plt.title("AUC Heatmap")
plt.show()

# FEATURE + CHANNEL + TOPO MAP
for stage in ALL_MODELS:

    model = ALL_MODELS[stage]
    coefs = model.params.drop("const")

    # Feature importance
    plt.figure()
    coefs.sort_values().tail(15).plot(kind='barh')
    plt.title(f"Top Features - {stage}")
    plt.show()

    # Channel importance
    ch_vals = {}
    for ch in channels:
        vals = [abs(v) for k,v in coefs.items() if ch in k]
        ch_vals[ch.upper()] = np.mean(vals) if vals else 0

    plt.figure()
    pd.Series(ch_vals).sort_values().plot(kind='barh')
    plt.title(f"Channel Importance - {stage}")
    plt.show()

    # Topomap
    ch_names = list(ch_vals.keys())
    values = np.array(list(ch_vals.values()))

    info = mne.create_info(ch_names, sfreq=100, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage)

    plt.figure()
    mne.viz.plot_topomap(values, info, cmap="RdBu_r", contours=0)
    plt.title(f"Topomap - {stage}")