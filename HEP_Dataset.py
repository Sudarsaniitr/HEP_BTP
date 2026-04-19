import numpy as np
import pandas as pd
import mne
import neurokit2 as nk
import pyedflib
import os

# ================= CONFIG =================
subject_id = 1

edf_path = r"EPCTL01-2025\EPCTL01\EPCTL01 - fixed.edf"
ann_path = r"EPCTL01-2025\EPCTL01\test1.txt"

SELECTED_CHANNELS = [
    "FC6","POZ","C2","F6","PO8",
    "Fp1","P6","FT10","CZ","FT8"
]

ECG_CH = "ECG1"

TMIN, TMAX = -0.2, 0.6
FEATURE_LO, FEATURE_HI = 0.25, 0.6

WINDOW_SIZE = 300  # 5 min 
MIN_EPOCHS = 10

OUTPUT_CSV = "hep_dataset_final_28.csv"

# ============================================================
# 1. READ ANNOTATIONS (HRV-COMPATIBLE)
# ============================================================
ann = pd.read_csv(
    ann_path,
    sep=r"\s+",
    header=None,
    names=["stage", "start", "duration"],
    engine="python"
)

ann["stage"] = ann["stage"].str.upper()
ann["epoch"] = ann["start"] // 30
ann["window_id"] = ann["epoch"] // 10

# ============================================================
# 2. BUILD LABEL STRUCTURE (SAME AS HRV)
# ============================================================
rows = []

for win, g in ann.groupby("window_id"):

    stages = g["stage"].tolist()

    row = {
        "subject_id": subject_id,
        "window_id": win
    }

    # Presence labels
    for s in ["W", "N1", "N2", "N3", "R"]:
        row[f"is_{s}"] = int(s in stages)

    # Transition labels
    for a in ["W", "N1", "N2", "N3", "R"]:
        for b in ["W", "N1", "N2", "N3", "R"]:
            if a != b:
                row[f"trans_{a}_{b}"] = 0

    for i in range(len(stages)-1):
        a, b = stages[i], stages[i+1]
        if a != b:
            row[f"trans_{a}_{b}"] = 1

    rows.append(row)

label_df = pd.DataFrame(rows)

# ============================================================
# 3. ECG → R PEAKS
# ============================================================
f = pyedflib.EdfReader(edf_path)
labels = f.getSignalLabels()

ecg_idx = labels.index(ECG_CH)
fs_ecg = f.getSampleFrequency(ecg_idx)

ecg = f.readSignal(ecg_idx)
f.close()

_, info = nk.ecg_process(ecg, sampling_rate=fs_ecg)
r_peaks = np.array(info["ECG_R_Peaks"])
r_times = r_peaks / fs_ecg

# ============================================================
# 4. LOAD EEG (MEMORY SAFE)
# ============================================================
raw = mne.io.read_raw_edf(
    edf_path,
    preload=False,
    include=SELECTED_CHANNELS,
    verbose=False
)

fs_eeg = raw.info["sfreq"]

# ============================================================
# 5. FEATURE FUNCTION (IMPROVED)
# ============================================================
def extract_features(signal, times):
    idx = np.where((times >= FEATURE_LO) & (times <= FEATURE_HI))[0]
    w = signal[idx]

    return {
        "mean": np.mean(w),
        "mean_abs": np.mean(np.abs(w)),
        "max": np.max(w),
        "min": np.min(w),
        "ptp": np.ptp(w),
        "std": np.std(w),
        "auc": np.trapz(w, times[idx]),
        "latency_pos": times[idx][np.argmax(w)]
    }

# ============================================================
# 6. PROCESS WINDOWS (FIXED ALIGNMENT)
# ============================================================
hep_rows = []

for win in label_df["window_id"]:

    # FIXED: proper alignment using annotations
    win_data = ann[ann["window_id"] == win]

    if win_data.empty:
        continue

    start = win_data["start"].min()
    end = start + WINDOW_SIZE

    print(f"Window {win}: {start}-{end}")

    # R peaks in window
    mask = (r_times >= start) & (r_times < end)
    r_win = r_times[mask]

    if len(r_win) < MIN_EPOCHS:
        continue

    # Load EEG chunk
    s = int(start * fs_eeg)
    e = int(end * fs_eeg)

    data, _ = raw[:, s:e]

    # Filter
    data = mne.filter.filter_data(
        data, fs_eeg, 0.1, 30, verbose=False
    )

    r_local = ((r_win - start) * fs_eeg).astype(int)

    before = int(abs(TMIN) * fs_eeg)
    after = int(TMAX * fs_eeg)

    epochs = []

    for r in r_local:
        ss = r - before
        ee = r + after

        if ss < 0 or ee >= data.shape[1]:
            continue

        ep = data[:, ss:ee]

        # Artifact rejection (NEW)
        if np.max(np.abs(ep)) > 150e-6:
            continue

        # Baseline
        baseline = ep[:, :before].mean(axis=1, keepdims=True)
        ep = ep - baseline

        epochs.append(ep)

    if len(epochs) < MIN_EPOCHS:
        continue

    epochs = np.stack(epochs)
    evoked = epochs.mean(axis=0)

    times = np.linspace(TMIN, TMAX, evoked.shape[1])

    feat_row = {
        "subject_id": subject_id,
        "window_id": win
    }

    for i, ch in enumerate(SELECTED_CHANNELS):

        signal = evoked[i]

        # Normalization (NEW)
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

        feats = extract_features(signal, times)

        for k, v in feats.items():
            feat_row[f"{ch}_{k}"] = v

    # NaN safety (NEW)
    if np.isnan(list(feat_row.values())).any():
        continue

    hep_rows.append(feat_row)

hep_df = pd.DataFrame(hep_rows)

# ============================================================
# 7. FINAL MERGE
# ============================================================
final_df = label_df.merge(
    hep_df,
    on=["subject_id", "window_id"],
    how="inner"
)

# ============================================================
# 8. SAVE
# ============================================================
final_df.to_csv(
    OUTPUT_CSV,
    mode="a",
    header=not os.path.exists(OUTPUT_CSV),
    index=False
)

print("\n✅ FINAL HEP dataset ready")
print("Shape:", final_df.shape)