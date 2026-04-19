import pandas as pd

CSV_OLD = "sleep_hrv_dataset_paper_faithful_old.csv"
CSV_NEW = "sleep_hrv_dataset_paper_faithful.csv"

OUTPUT_CSV = "sleep_hrv_dataset_paper_faithful_merged.csv"

# ------------------------------------------------------------
# 1. LOAD WITH PYTHON ENGINE (TOLERANT PARSER)
# ------------------------------------------------------------

df_old = pd.read_csv(
    CSV_OLD,
    engine="python",
    on_bad_lines="skip"
)

df_new = pd.read_csv(
    CSV_NEW,
    engine="python",
    on_bad_lines="skip"
)

print("Old shape:", df_old.shape)
print("New shape:", df_new.shape)

# ------------------------------------------------------------
# 2. ALIGN COLUMNS (CRITICAL)
# ------------------------------------------------------------

all_columns = sorted(set(df_old.columns) | set(df_new.columns))

df_old = df_old.reindex(columns=all_columns)
df_new = df_new.reindex(columns=all_columns)

# ------------------------------------------------------------
# 3. CONCATENATE
# ------------------------------------------------------------

df_all = pd.concat([df_old, df_new], ignore_index=True)

print("Combined shape (before dedup):", df_all.shape)

# ------------------------------------------------------------
# 4. REMOVE DUPLICATES SAFELY
# ------------------------------------------------------------

if {"subject_id", "window_id"}.issubset(df_all.columns):
    df_all = df_all.drop_duplicates(
        subset=["subject_id", "window_id"],
        keep="first"
    )
else:
    print("⚠ subject_id/window_id not found — skipping dedup")

print("Final shape:", df_all.shape)

# ------------------------------------------------------------
# 5. SORT (OPTIONAL)
# ------------------------------------------------------------

if {"subject_id", "window_id"}.issubset(df_all.columns):
    df_all = df_all.sort_values(
        ["subject_id", "window_id"]
    ).reset_index(drop=True)

# ------------------------------------------------------------
# 6. SAVE
# ------------------------------------------------------------

df_all.to_csv(OUTPUT_CSV, index=False)

print(f"\n✓ Merged dataset saved as {OUTPUT_CSV}")
print("Subjects:", sorted(df_all["subject_id"].dropna().unique()))
