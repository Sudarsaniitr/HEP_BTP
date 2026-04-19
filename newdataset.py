# ============================================================
# MULTI-SUBJECT SLEEP HRV DATASET BUILDER
# PAPER-FAITHFUL VERSION (Entropy 2024)
# ============================================================

import numpy as np
import pandas as pd
import pyedflib
import neurokit2 as nk
from scipy.signal import welch
from scipy.interpolate import interp1d
import os

# ============================================================
# 1. RR CLEANING (PAPER-CONSISTENT)
# ============================================================

def clean_rr(rr):
    rr = np.asarray(rr, dtype=float)

    # Remove non-physiological RR (ms → sec later)
    rr[(rr < 0.3) | (rr > 2.0)] = np.nan

    if np.sum(~np.isnan(rr)) < 0.5 * len(rr):
        return None

    idx = np.arange(len(rr))
    f = interp1d(idx[~np.isnan(rr)], rr[~np.isnan(rr)],
                 kind="linear", fill_value="extrapolate")
    return f(idx)

# ============================================================
# 2. HRV + SAMPEN (PAPER FEATURES)
# ============================================================

def sample_entropy(x, m=2, r_ratio=0.2):
    r = r_ratio * np.std(x)
    N = len(x)

    def _phi(m):
        X = np.array([x[i:i+m] for i in range(N-m+1)])
        C = np.sum(
            np.max(np.abs(X[:, None] - X[None, :]), axis=2) <= r,
            axis=0
        ) - 1
        return np.sum(C) / (N-m+1)

    try:
        return -np.log(_phi(m+1) / _phi(m))
    except:
        return np.nan


def compute_hrv(rr):
    rr_clean = clean_rr(rr)
    if rr_clean is None or len(rr_clean) < 240:
        return None

    rr_ms = rr_clean * 1000
    sdnn = np.std(rr_ms, ddof=1)
    rmssd = np.sqrt(np.mean(np.diff(rr_ms)**2))

    # Interpolate tachogram @ 4 Hz
    t = np.cumsum(rr_clean)
    fs = 4.0
    t_i = np.arange(0, t[-1], 1/fs)
    rr_i = np.interp(t_i, t[:-1], rr_clean[:-1])

    freqs, psd = welch(rr_i, fs=fs, nperseg=256)

    lf_band = (freqs >= 0.04) & (freqs < 0.15)
    hf_band = (freqs >= 0.15) & (freqs < 0.40)

    lf = np.trapz(psd[lf_band], freqs[lf_band])
    hf = np.trapz(psd[hf_band], freqs[hf_band])

    sampen = sample_entropy(rr_clean)

    return {
        "SDNN": sdnn,
        "RMSSD": rmssd,
        "LF": np.log(lf + 1e-6),
        "HF": np.log(hf + 1e-6),
        "LF/HF": lf / hf if hf > 0 else np.nan,
        "SampEn": sampen
    }

# ============================================================
# 3. PROCESS ONE SUBJECT (SAME STRUCTURE AS YOUR CODE)
# ============================================================

def process_subject(subject_id, edf_path, ann_path, ecg_channel="ECG1"):

    # ---------- Load annotations ----------
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

    # ---------- Window-level labels (PRESENCE-BASED) ----------
    rows = []
    for win, g in ann.groupby("window_id"):
        stages = g["stage"].tolist()

        row = {
            "subject_id": subject_id,
            "window_id": win
        }

        for s in ["W", "N1", "N2", "N3", "R"]:
            row[f"is_{s}"] = int(s in stages)

        # Epoch-level transitions inside window
        for a in ["W", "N1", "N2", "N3", "R"]:
            for b in ["W", "N1", "N2", "N3", "R"]:
                if a != b:
                    row[f"trans_{a}_{b}"] = 0

        for i in range(len(stages)-1):
            a, b = stages[i], stages[i+1]
            if a != b:
                row[f"trans_{a}_{b}"] = 1

        rows.append(row)

    df = pd.DataFrame(rows)

    # ---------- Load ECG ----------
    f = pyedflib.EdfReader(edf_path)
    labels = [l.lower() for l in f.getSignalLabels()]
    ecg_idx = labels.index(ecg_channel.lower())
    fs = f.getSampleFrequency(ecg_idx)
    ecg = f.readSignal(ecg_idx)
    f.close()

    _, info = nk.ecg_process(ecg, sampling_rate=fs)
    rpeaks = info["ECG_R_Peaks"]
    r_times = np.array(rpeaks) / fs
    rr = np.diff(r_times)

    # ---------- HRV per window ----------
    hrv_rows = []
    for win in df["window_id"]:
        start = win * 300
        mask = (r_times[1:] >= start) & (r_times[1:] < start + 300)
        rr_win = rr[mask]

        feats = compute_hrv(rr_win)
        if feats is None:
            continue

        feats["subject_id"] = subject_id
        feats["window_id"] = win
        hrv_rows.append(feats)

    hrv_df = pd.DataFrame(hrv_rows)

    return df.merge(hrv_df, on=["subject_id", "window_id"], how="inner")

# ============================================================
# 4. CSV APPEND LOGIC (UNCHANGED)
# ============================================================

def get_processed_subjects(csv_path):
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path, usecols=["subject_id"])
    return set(df["subject_id"].unique())

# ============================================================
# 5. MAIN DRIVER (SAME AS YOUR CODE)
# ============================================================

if __name__ == "__main__":

    OUTPUT_CSV = "sleep_hrv_dataset_paper_faithful_comnew_newone_1.csv"

    subjects = [
    #    {"id": 21, "edf": "EPCTL21/EPCTL21/EPCTL21.edf", "ann": "EPCTL21/EPCTL21/EPCTL21.txt"},
    #     {"id": 29, "edf": "EPCTL29/EPCTL29/EPCTL29.edf", "ann": "EPCTL29/EPCTL29/EPCTL29.txt"},
    #     {"id": 2, "edf": "EPCTL02/EPCTL02/EPCTL02.edf", "ann": "EPCTL02/EPCTL02/EPCTL02.txt"},
        # {"id": 7, "edf": "EPCTL07/EPCTL07/EPCTL07.edf", "ann": "EPCTL07/EPCTL07/EPCTL07.txt"},
        # {"id": 9, "edf": "EPCTL09/EPCTL09/EPCTL09.edf", "ann": "EPCTL09/EPCTL09/EPCTL09.txt"},
        # {"id": 8, "edf": "EPCTL08/EPCTL08/EPCTL08.edf", "ann": "EPCTL08/EPCTL08/EPCTL08.txt"},
        # {"id":17, "edf": "EPCTL17/EPCTL17/EPCTL17.edf", "ann": "EPCTL17/EPCTL17/EPCTL17.txt"},
        # {"id":6, "edf": "EPCTL06/EPCTL06/EPCTL06.edf", "ann": "EPCTL06/EPCTL06/EPCTL06.txt"},
        # {"id":13, "edf": "EPCTL13/EPCTL13/EPCTL13.edf", "ann": "EPCTL13/EPCTL13/EPCTL13.txt"},
        # {"id":12, "edf": "EPCTL12/EPCTL12/EPCTL12.edf", "ann": "EPCTL12/EPCTL12/EPCTL12.txt"},
        {"id":11, "edf": "EPCTL11/EPCTL11/EPCTL11.edf", "ann": "EPCTL11/EPCTL11/EPCTL11.txt"},
        {"id":19, "edf": "EPCTL19/EPCTL19/EPCTL19.edf", "ann": "EPCTL19/EPCTL19/EPCTL19.txt"},
        {"id":14, "edf": "EPCTL14/EPCTL14/EPCTL14.edf", "ann": "EPCTL14/EPCTL14/EPCTL14.txt"},
        {"id":15, "edf": "EPCTL15/EPCTL15/EPCTL15.edf", "ann": "EPCTL15/EPCTL15/EPCTL15.txt"}
    ]

    processed = get_processed_subjects(OUTPUT_CSV)
    print("Already processed subjects:", processed)

    for subj in subjects:
        if subj["id"] in processed:
            print(f"Skipping subject {subj['id']}")
            continue

        print(f"\nProcessing subject {subj['id']}")

        df_subj = process_subject(
            subj["id"], subj["edf"], subj["ann"]
        )

        write_header = not os.path.exists(OUTPUT_CSV)
        df_subj.to_csv(OUTPUT_CSV, mode="a",
                       header=write_header, index=False)

        print(f"✓ Subject {subj['id']} added ({len(df_subj)} rows)")

    print("\nDATASET BUILD COMPLETE")
