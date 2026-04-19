# ============================================================
# MULTI-SUBJECT SLEEP HRV DATASET BUILDER (FINAL VERSION)
# ============================================================

import numpy as np
import pandas as pd
import pyedflib
import neurokit2 as nk
from scipy.signal import welch
import os

# ============================================================
# SAFE HRV COMPUTATION (LF/HF FIXED)
# ============================================================

def compute_hrv(rr):
    rr = np.asarray(rr, dtype=float)
    if len(rr) < 3:
        return None

    rr_ms = rr * 1000
    sdnn = np.std(rr_ms, ddof=1)
    rmssd = np.sqrt(np.mean(np.diff(rr_ms) ** 2))

    try:
        freqs, psd = welch(rr, fs=4.0, nperseg=min(256, len(rr)))
    except Exception:
        return None

    lf_band = (freqs >= 0.04) & (freqs <= 0.15)
    hf_band = (freqs >= 0.15) & (freqs <= 0.40)

    lf = np.trapz(psd[lf_band], freqs[lf_band]) if lf_band.any() else None
    hf = np.trapz(psd[hf_band], freqs[hf_band]) if hf_band.any() else None

    if lf is None or hf is None or hf == 0:
        lf_hf = None
    else:
        lf_hf = lf / hf

    return {
        "SDNN": sdnn,
        "RMSSD": rmssd,
        "LF": lf,
        "HF": hf,
        "LF/HF": lf_hf
    }

# ============================================================
# PROCESS ONE SUBJECT
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
    ann["window_id"] = (ann["start"] // 300).astype(int)

    # ---------- Build 5-min windows ----------
    rows = []
    for win_id, g in ann.groupby("window_id"):
        total = g["duration"].sum()
        frac = g.groupby("stage")["duration"].sum() / total

        rows.append({
            "subject_id": subject_id,
            "window_id": win_id,
            "start_time": win_id * 300,
            "frac_W": frac.get("W", 0),
            "frac_N1": frac.get("N1", 0),
            "frac_N2": frac.get("N2", 0),
            "frac_N3": frac.get("N3", 0),
            "frac_R": frac.get("R", 0),
        })

    df = pd.DataFrame(rows)

    # ---------- Certain stage labels ----------
    for s in ["W", "N1", "N2", "N3", "R"]:
        df[f"is_{s}"] = (df[f"frac_{s}"] >= 0.7).astype(int)

    # ---------- Transition labels ----------
    stages = ["W", "N1", "N2", "N3", "R"]
    for a in stages:
        for b in stages:
            if a != b:
                df[f"trans_{a}_{b}"] = 0

    for i in range(len(df) - 1):
        for a in stages:
            for b in stages:
                if a != b and df.loc[i, f"is_{a}"] == 1 and df.loc[i+1, f"is_{b}"] == 1:
                    df.loc[i, f"trans_{a}_{b}"] = 1

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
    for _, row in df.iterrows():
        mask = (r_times[1:] >= row["start_time"]) & \
               (r_times[1:] < row["start_time"] + 300)

        rr_win = rr[mask]
        hrv = compute_hrv(rr_win)

        if hrv:
            hrv["subject_id"] = subject_id
            hrv["window_id"] = row["window_id"]
            hrv_rows.append(hrv)

    hrv_df = pd.DataFrame(hrv_rows)

    # ---------- Merge ----------
    final_df = df.merge(
        hrv_df,
        on=["subject_id", "window_id"],
        how="inner"
    )

    # Drop windows with broken HRV
    final_df = final_df.dropna(subset=["SDNN", "RMSSD"])

    return final_df

# ============================================================
# GET ALREADY PROCESSED SUBJECTS
# ============================================================

def get_processed_subjects(csv_path):
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path, usecols=["subject_id"])
    return set(df["subject_id"].unique())

# ============================================================
# MAIN DRIVER
# ============================================================

if __name__ == "__main__":

    OUTPUT_CSV = "sleep_hrv_dataset_all_subjects.csv"

    subjects = [
        # {"id": 28, "edf": "EPCTL28/EPCTL28/EPCTL28.edf", "ann": "EPCTL28/EPCTL28/EPCTL28.txt"},
        # {"id": 27, "edf": "EPCTL27/EPCTL27/EPCTL27.edf", "ann": "EPCTL27/EPCTL27/EPCTL27.txt"},
        # {"id": 23, "edf": "EPCTL23/EPCTL23/EPCTL23.edf", "ann": "EPCTL23/EPCTL23/EPCTL23.txt"},
        # {"id": 22, "edf": "EPCTL22/EPCTL22/EPCTL22.edf", "ann": "EPCTL22/EPCTL22/EPCTL22.txt"},
        # {"id": 20, "edf": "EPCTL20/EPCTL20/EPCTL20.edf", "ann": "EPCTL20/EPCTL20/EPCTL20.txt"},
        # {"id": 10, "edf": "EPCTL10/EPCTL10/EPCTL10.edf", "ann": "EPCTL10/EPCTL10/EPCTL10.txt"},
        # {"id": 5, "edf": "EPCTL05/EPCTL05/EPCTL05.edf", "ann": "EPCTL05/EPCTL05/EPCTL05.txt"},
        # {"id": 4, "edf": "EPCTL04/EPCTL04/EPCTL04.edf", "ann": "EPCTL04/EPCTL04/EPCTL04.txt"},
        # {"id": 3, "edf": "EPCTL03/EPCTL03/EPCTL03.edf", "ann": "EPCTL03/EPCTL03/EPCTL03.txt"},
        # {"id": 26, "edf": "EPCTL26/EPCTL26/EPCTL26.edf", "ann": "EPCTL26/EPCTL26/EPCTL26.txt"},
        {"id": 16, "edf": "EPCTL16/EPCTL16/EPCTL16.edf", "ann": "EPCTL16/EPCTL16/EPCTL16.txt"},
        {"id": 25, "edf": "EPCTL25/EPCTL25/EPCTL25.edf", "ann": "EPCTL25/EPCTL25/EPCTL25.txt"},
        {"id": 18, "edf": "EPCTL18/EPCTL18/EPCTL18.edf", "ann": "EPCTL18/EPCTL18/EPCTL18.txt"},
        {"id": 24, "edf": "EPCTL24/EPCTL24/EPCTL24.edf", "ann": "EPCTL24/EPCTL24.txt"}
    ]

    processed_subjects = get_processed_subjects(OUTPUT_CSV)
    print("Already processed subjects:", processed_subjects)

    for subj in subjects:
        if subj["id"] in processed_subjects:
            print(f"Skipping subject {subj['id']} (already processed)")
            continue

        print(f"\nProcessing subject {subj['id']}")

        df_subj = process_subject(
            subject_id=subj["id"],
            edf_path=subj["edf"],
            ann_path=subj["ann"]
        )

        write_header = not os.path.exists(OUTPUT_CSV)

        df_subj.to_csv(
            OUTPUT_CSV,
            mode="a",
            header=write_header,
            index=False
        )

        print(f"✓ Subject {subj['id']} added ({len(df_subj)} rows)")

    print("\nDATASET BUILD COMPLETE")
