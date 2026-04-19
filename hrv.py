import numpy as np
import pandas as pd
import pyedflib
import neurokit2 as nk
from scipy.signal import welch

# ---------------------------------------------------
# HRV METRICS
# ---------------------------------------------------
def compute_hrv(rr):
    """Compute SDNN, RMSSD, LF, HF, LF/HF safely."""
    rr = np.asarray(rr, dtype=float)  # convert list → numpy array

    if len(rr) < 3:
        return None

    # -----------------------
    # TIME‑DOMAIN HRV
    # -----------------------
    rr_ms = rr * 1000  # sec → ms

    sdnn = np.std(rr_ms, ddof=1)
    rmssd = np.sqrt(np.mean(np.diff(rr_ms)**2))

    # -----------------------
    # FREQUENCY‑DOMAIN HRV
    # -----------------------
    try:
        # 4 Hz is standard interpolation frequency
        freqs, psd = welch(rr, fs=4.0, nperseg=min(256, len(rr)))
    except Exception:
        # If Welch fails, return time‑domain metrics only
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
    lf_hf = lf / hf if (hf not in [0, None]) else None

    return {
        "SDNN": sdnn,
        "RMSSD": rmssd,
        "LF": lf,
        "HF": hf,
        "LF/HF": lf_hf
    }


# ---------------------------------------------------
# READ SLEEP ANNOTATIONS
# ---------------------------------------------------
def read_stage_annotations(txt_path):
    """
    Reads the annotation file (Stage, Start, Duration).
    Returns a list of tuples: (stage_label, start_time, end_time).
    """
    df = pd.read_csv(txt_path, sep=r"\s+", header=None,
                     names=["Stage", "Start", "Duration"], engine="python")

    stages = []
    for _, row in df.iterrows():
        stage = str(row["Stage"]).strip().upper()
        start = float(row["Start"])
        end = start + float(row["Duration"])
        stages.append((stage, start, end))

    return stages

# ---------------------------------------------------
# GET STAGE FOR A GIVEN TIME
# ---------------------------------------------------
def stage_for_time(t, stage_intervals):
    for stage, start, end in stage_intervals:
        if start <= t < end:
            return stage
    return None

# ---------------------------------------------------
# MAIN FUNCTION: HRV FOR EACH STAGE
# ---------------------------------------------------
def compute_hrv_by_stage(edf_path, txt_path, ecg_channel="ECG1"):
    """
    Returns HRV metrics separately for W, N1, N2, N3, REM.
    """
    # --- Load EDF ---
    f = pyedflib.EdfReader(edf_path)
    labels = f.getSignalLabels()

    # Find ECG channel index
    ecg_norm = ecg_channel.lower()
    labels_norm = [lbl.lower() for lbl in labels]
    if ecg_norm not in labels_norm:
        raise ValueError(f"ECG channel {ecg_channel} not found!")

    ecg_idx = labels_norm.index(ecg_norm)

    fs = f.getSampleFrequency(ecg_idx)
    ecg = f.readSignal(ecg_idx)
    f.close()

    # --- Detect R-peaks ---
    _, info = nk.ecg_process(ecg, sampling_rate=fs)
    rpeaks = info["ECG_R_Peaks"]
    r_times = np.array(rpeaks) / fs  # seconds

    rr = np.diff(r_times)

    # --- Read sleep annotation file ---
    stage_intervals = read_stage_annotations(txt_path)

    # Stage categories
    stage_map = {"W": [], "N1": [], "N2": [], "N3": [], "R": []}

    # Assign each RR interval to stage
    for i in range(1, len(r_times)):
        beat_time = r_times[i]
        stage = stage_for_time(beat_time, stage_intervals)

        if stage in stage_map:
            stage_map[stage].append(rr[i - 1])

    # Compute HRV per stage
    hrv_results = {}
    for stg, rr_list in stage_map.items():
        if len(rr_list) > 2:
            hrv_results[stg] = compute_hrv(rr_list)
        else:
            hrv_results[stg] = None

    return hrv_results

# ---------------------------------------------------
# Example usage
# ---------------------------------------------------
if __name__ == "__main__":
    txt_path = r"EPCTL04\EPCTL04\EPCTL04.txt"
    edf_path = r"EPCTL04\EPCTL04\EPCTL04.edf"       

    results = compute_hrv_by_stage(edf_path, txt_path, ecg_channel="ECG1")

    print("\n===== HRV RESULTS BY SLEEP STAGE =====\n")
    for stg, vals in results.items():
        print(stg, ":", vals)