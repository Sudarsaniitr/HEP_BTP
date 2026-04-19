import pandas as pd
import os
import re

# ================= CONFIG =================
CSV_FILES = [
    
    "hep_dataset_final_16.csv",
    "hep_dataset_final_11.csv",
    "hep_dataset_final_18.csv",
    "hep_dataset_final_10.csv",
    "hep_dataset_final_20.csv",
    "hep_dataset_final_15.csv",
    "hep_dataset_final_03.csv",
    "hep_dataset_final_04.csv",
    "hep_dataset_final_05.csv",
    "hep_dataset_final_06.csv",
    "hep_dataset_final_08.csv",
    "hep_dataset_final_12.csv",
    "hep_dataset_final_13.csv",
    "hep_dataset_final_14.csv",
    "hep_dataset_final_19.csv",
    "hep_dataset_final_17.csv",
    "hep_dataset_final_28.csv",
    "hep_dataset_final_27.csv",
    "hep_dataset_final_23.csv",
    "hep_dataset_final_22.csv",
   
]

OUTPUT_FILE = "hep_dataset_FINAL_CLEAN_1.csv"
# ==========================================

# -------------------------------
# 1. CLEAN COLUMN NAMES (LIGHT)
# -------------------------------
def clean_column(col):
    col = col.strip()
    col = col.replace(" ", "_")
    col = col.replace("-", "_")
    return col

def normalize_columns(df):
    df.columns = [clean_column(c) for c in df.columns]
    return df

# -------------------------------
# 2. MERGE DUPLICATE COLUMNS
# -------------------------------
def merge_duplicate_columns(df):
    cols = list(df.columns)
    new_df = pd.DataFrame()

    used = set()

    for col in cols:
        if col in used:
            continue

        base = re.sub(r"_\d+$", "", col)

        duplicates = [c for c in cols if re.sub(r"_\d+$", "", c) == base]

        used.update(duplicates)

        if len(duplicates) == 1:
            new_df[base] = df[duplicates[0]]
        else:
            new_df[base] = df[duplicates].bfill(axis=1).iloc[:, 0]

    return new_df

# -------------------------------
# 3. LOAD FILES
# -------------------------------
dfs = []

for file in CSV_FILES:
    if not os.path.exists(file):
        print(f"Skipping missing file: {file}")
        continue

    print(f"Reading: {file}")

    df = pd.read_csv(file)

    df = normalize_columns(df)
    df = merge_duplicate_columns(df)

    dfs.append(df)

# -------------------------------
# 4. USE COLUMN ORDER FROM FIRST FILE
# -------------------------------
base_columns = list(dfs[0].columns)

# Ensure all dfs have same columns (fill missing)
aligned_dfs = []

for df in dfs:
    for col in base_columns:
        if col not in df.columns:
            df[col] = pd.NA  # fill missing

    # IMPORTANT: keep same order
    df = df[base_columns]

    aligned_dfs.append(df)

# -------------------------------
# 5. CONCAT WITHOUT SORTING
# -------------------------------
merged_df = pd.concat(aligned_dfs, ignore_index=True, sort=False)

# -------------------------------
# 6. OPTIONAL CLEAN
# -------------------------------
merged_df = merged_df.drop_duplicates()

# -------------------------------
# 7. SAVE
# -------------------------------
merged_df.to_csv(OUTPUT_FILE, index=False)

print("\n✅ ORDER-PRESERVED MERGE DONE")
print("Shape:", merged_df.shape)
print("Saved:", OUTPUT_FILE)