import pandas as pd

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

CSV_PATH = "sleep_hrv_dataset_all_subjects_clean.csv"
df = pd.read_csv(CSV_PATH)

print("Dataset shape:", df.shape)

# ------------------------------------------------------------
# 2. DEFINE STAGES AND TRANSITIONS (AS IN PAPER)
# ------------------------------------------------------------

stage_cols = {
    "N1 stage": "is_N1",
    "N2 stage": "is_N2",
    "N3 stage": "is_N3",
    "R stage":  "is_R",
    "W stage":  "is_W",
}

transition_cols = {
    "transition N1–N2": "trans_N1_N2",
    "transition N2–N3": "trans_N2_N3",
    "transition N2–R":  "trans_N2_R",
    "transition N3–N2": "trans_N3_N2",
    "transition N3–R":  "trans_N3_R",
    "transition R–N2":  "trans_R_N2",
    "transition W–N1":  "trans_W_N1",
}

# ------------------------------------------------------------
# 3. COUNT TOTAL WINDOWS
# ------------------------------------------------------------

total_windows = len(df)
print("Total windows:", total_windows)

# ------------------------------------------------------------
# 4. COMPUTE COUNTS & PERCENTAGES
# ------------------------------------------------------------

rows = []

def add_rows(label_map):
    for label, col in label_map.items():
        if col not in df.columns:
            print(f"Warning: {col} not found, skipping")
            continue

        count = int(df[col].sum())
        percentage = 100 * count / total_windows

        rows.append({
            "Category": label,
            "Count": count,
            "Percentage (%)": round(percentage, 2)
        })

add_rows(stage_cols)
add_rows(transition_cols)

# ------------------------------------------------------------
# 5. CREATE FINAL TABLE
# ------------------------------------------------------------

table_df = pd.DataFrame(rows)

print("\n===== STAGE & TRANSITION DISTRIBUTION =====\n")
print(table_df)

# ------------------------------------------------------------
# 6. OPTIONAL: SAVE TABLE
# ------------------------------------------------------------

table_df.to_csv("stage_transition_distribution.csv", index=False)
print("\nSaved as stage_transition_distribution.csv")
