import numpy as np
import pandas as pd
import pyedflib
import neurokit2 as nk
from scipy.signal import welch

# =====================================================
# HRV COMPUTATION (your logic, unchanged)
# =====================================================
def compute_hrv(rr):
    rr = np.asarray(rr, dtype=float)

    if len(rr) < 3:
        return None

    rr_ms = rr * 1000.0

    sdnn = np.std(rr_ms, ddof=1)
    rmssd = np.sqrt(np.mean(np.diff(rr_ms) ** 2))

    try:
        freqs, psd = welch(rr, fs=4.0, nperseg=min(256, len(rr)))
    except Exception:
        return {
            "SDNN": sdnn,
            "RMSSD": rmssd,
            "LF": None,
            "HF": None,
            "LF/HF": None
        }

    lf_band = (freqs >= 0.04) & (freqs <= 0.15)
    hf_band = (freqs >= 0.15) & (freqs <= 0.40)

    lf = np.trapz(psd[lf_band], freqs[lf_band]) if lf_band.any() else None
    hf = np.trapz(psd[hf_band], freqs[hf_band]) if hf_band.any() else None
    lf_hf = lf / hf if (hf not in [None, 0]) else None

    return {
        "SDNN": sdnn,
        "RMSSD": rmssd,
        "LF": lf,
        "HF": hf,
        "LF/HF": lf_hf
    }

# =====================================================
# LOAD ANNOTATION FILE (YOUR FORMAT)
# =====================================================
def load_annotations(txt_path):
    df = pd.read_csv(
        txt_path,
        sep=r"\s+",
        header=None,
        names=["stage", "start", "duration"],
        engine="python"
    )
    df["stage"] = df["stage"].str.upper()
    df["end"] = df["start"] + df["duration"]
    return df

# =====================================================
# BUILD 5-MINUTE WINDOWS (300 seconds)
# =====================================================
def build_5min_windows(ann_df):
    ann_df["window_id"] = (ann_df["start"] // 300).astype(int)

    rows = []
    for win_id, g in ann_df.groupby("window_id"):
        total = g["duration"].sum()

        fractions = (
            g.groupby("stage")["duration"].sum() / total
        ).to_dict()

        row = {
            "window_id": win_id,
            "start_time": win_id * 300,
            "end_time": win_id * 300 + 300,
            "frac_W": fractions.get("W", 0.0),
            "frac_N1": fractions.get("N1", 0.0),
            "frac_N2": fractions.get("N2", 0.0),
            "frac_N3": fractions.get("N3", 0.0),
            "frac_R": fractions.get("R", 0.0),
        }
        rows.append(row)

    return pd.DataFrame(rows)

# =====================================================
# CERTAIN STAGE LABELS (≥ 0.7)
# =====================================================
def add_certain_stage_labels(df, threshold=0.7):
    for stg in ["W", "N1", "N2", "N3", "R"]:
        df[f"is_{stg}"] = (df[f"frac_{stg}"] >= threshold).astype(int)
    return df

# =====================================================
# TRANSITION LABELS
# =====================================================
def add_transition_labels(df):
    stages = ["W", "N1", "N2", "N3", "R"]

    for a in stages:
        for b in stages:
            if a != b:
                df[f"trans_{a}_{b}"] = 0

    for i in range(len(df) - 1):
        for a in stages:
            for b in stages:
                if a == b:
                    continue
                if df.loc[i, f"is_{a}"] == 1 and df.loc[i+1, f"is_{b}"] == 1:
                    df.loc[i, f"trans_{a}_{b}"] = 1

    return df

# =====================================================
# HRV PER 5-MINUTE WINDOW
# =====================================================
def compute_hrv_by_window(edf_path, windows_df, ecg_channel="ECG1"):
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

    rows = []
    for _, row in windows_df.iterrows():
        start, end = row["start_time"], row["end_time"]
        mask = (r_times[1:] >= start) & (r_times[1:] < end)
        rr_win = rr[mask]

        hrv = compute_hrv(rr_win)
        if hrv is None:
            continue

        hrv["window_id"] = row["window_id"]
        rows.append(hrv)

    return pd.DataFrame(rows)

# =====================================================
# MAIN PIPELINE
# =====================================================
if __name__ == "__main__":

    # CHANGE THESE PATHS
    txt_path = "EPCTL04/EPCTL04/EPCTL04.txt"
    edf_path = "EPCTL04/EPCTL04/EPCTL04.edf"

    print("Loading annotations...")
    ann = load_annotations(txt_path)

    print("Building 5-minute windows...")
    windows = build_5min_windows(ann)
    windows = add_certain_stage_labels(windows)
    windows = add_transition_labels(windows)

    print("Computing HRV per window...")
    hrv_df = compute_hrv_by_window(edf_path, windows, ecg_channel="ECG1")

    print("Merging features + labels...")
    final_df = windows.merge(hrv_df, on="window_id", how="inner")

    print("Saving CSV...")
    final_df.to_csv("sleep_hrv_dataset.csv", index=False)

    print("DONE.")
    print("Rows:", len(final_df))
    print("Columns:", list(final_df.columns))
