# ============================================================
# HRV DATASET PREPROCESSING (FINAL, PAPER-CONSISTENT)
# ============================================================

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 1. LOAD MERGED DATASET
# ------------------------------------------------------------

INPUT_CSV = "sleep_hrv_dataset_all_subjects_merged.csv"
OUTPUT_CSV = "sleep_hrv_dataset_all_subjects_clean.csv"

df = pd.read_csv(INPUT_CSV)

print("Loaded dataset shape:", df.shape)
print("Subjects:", sorted(df["subject_id"].unique()))

# ------------------------------------------------------------
# 2. BASIC SANITY CHECK
# ------------------------------------------------------------

required_cols = ["SDNN", "RMSSD", "LF", "HF", "LF/HF"]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required HRV columns: {missing}")

# ------------------------------------------------------------
# 3. CLIP PHYSIOLOGICALLY IMPOSSIBLE HRV VALUES
# ------------------------------------------------------------
# 5-min HRV: anything above 500 ms is unrealistic

SDNN_CLIP = 500.0
RMSSD_CLIP = 500.0

df["SDNN"] = df["SDNN"].clip(lower=0, upper=SDNN_CLIP)
df["RMSSD"] = df["RMSSD"].clip(lower=0, upper=RMSSD_CLIP)

# ------------------------------------------------------------
# 4. LOG-TRANSFORM LF AND HF (STANDARD PRACTICE)
# ------------------------------------------------------------
# Avoid log(0) by adding epsilon

EPS = 1e-6

df["LF"] = np.log(df["LF"] + EPS)
df["HF"] = np.log(df["HF"] + EPS)

# ------------------------------------------------------------
# 5. HANDLE LF/HF SAFELY
# ------------------------------------------------------------
# Replace infinities or NaNs (rare edge cases)

df["LF/HF"] = df["LF/HF"].replace([np.inf, -np.inf], np.nan)

# ------------------------------------------------------------
# 6. DROP ROWS WITH BROKEN HRV (VERY FEW)
# ------------------------------------------------------------

before = len(df)
df = df.dropna(subset=["SDNN", "RMSSD", "LF", "HF", "LF/HF"])
after = len(df)

print(f"Dropped {before - after} rows due to invalid HRV values")

# ------------------------------------------------------------
# 7. FINAL SUMMARY (IMPORTANT FOR REPORT)
# ------------------------------------------------------------

print("\nFinal dataset shape:", df.shape)

print("\nHRV summary (after cleaning):")
print(df[["SDNN", "RMSSD", "LF", "HF", "LF/HF"]].describe())

print("\nRows per subject:")
print(df["subject_id"].value_counts().sort_index())

# ------------------------------------------------------------
# 8. SAVE CLEAN DATASET
# ------------------------------------------------------------

df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Clean dataset saved as: {OUTPUT_CSV}")
