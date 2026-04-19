import numpy as np
import pandas as pd
import mne
import neurokit2 as nk
import pyedflib
import os

# ================= USER INPUT =================
subject_name = "EPCTL24"

edf_path = r"EPCTL24\EPCTL24.edf"
annotation_path = r"EPCTL24\EPCTL24.txt"

SELECTED_CHANNELS = [
    "FC6","POZ","C2","F6","PO8",
    "Fp1","P6","FT10","CZ","FT8"
]

ECG_CH = "ECG1"

# Epoch parameters
TMIN = -0.2
TMAX = 0.6
BASELINE = (TMIN, 0)

# HEP feature window
FEATURE_LO = 0.25
FEATURE_HI = 0.6

WINDOW_SIZE = 300   # 5 minutes
MIN_EPOCHS = 10

OUTPUT_CSV = f"{subject_name}_HEP_5min_stagewise.csv"
# =================================================


# -------- READ ANNOTATIONS --------
def read_annotations(path):

    ann = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["Stage","Start","Duration"],
        engine="python"
    )

    return ann


# -------- MERGE CONSECUTIVE STAGE SEGMENTS --------
def merge_sleep_stages(annotations):

    merged = []

    current_stage = None
    current_start = None
    current_duration = 0

    for _, row in annotations.iterrows():

        stage = row["Stage"]
        start = float(row["Start"])
        duration = float(row["Duration"])

        if current_stage is None:
            current_stage = stage
            current_start = start
            current_duration = duration
            continue

        if stage == current_stage:
            current_duration += duration

        else:
            merged.append((current_stage,current_start,current_duration))
            current_stage = stage
            current_start = start
            current_duration = duration

    merged.append((current_stage,current_start,current_duration))

    return merged


# -------- HEP FEATURE EXTRACTION --------
def extract_hep_features(signal,times):

    idx = np.where((times>=FEATURE_LO)&(times<=FEATURE_HI))[0]

    window = signal[idx]

    return {
        "mean_amp": np.mean(window),
        "peak_pos": np.max(window),
        "peak_neg": np.min(window),
        "ptp": np.ptp(window),
        "std": np.std(window),
        "auc": np.trapz(window,times[idx])
    }


# ================= START =================
print("Processing subject:",subject_name)

annotations = read_annotations(annotation_path)
merged_stages = merge_sleep_stages(annotations)

# -------- LOAD ECG --------
f = pyedflib.EdfReader(edf_path)

labels = f.getSignalLabels()

ecg_idx = labels.index(ECG_CH)
fs_ecg = f.getSampleFrequency(ecg_idx)

ecg_signal = f.readSignal(ecg_idx)

f.close()

# -------- R PEAK DETECTION --------
_,info = nk.ecg_process(ecg_signal,sampling_rate=fs_ecg)

r_peaks = np.array(info["ECG_R_Peaks"])
r_times = r_peaks/fs_ecg

# -------- OPEN EDF WITHOUT LOADING --------
raw = mne.io.read_raw_edf(
    edf_path,
    preload=False,
    include=SELECTED_CHANNELS,
    verbose=False
)

fs_eeg = raw.info["sfreq"]

rows = []

# ================= STAGE LOOP =================
for stage,start,duration in merged_stages:

    if duration < WINDOW_SIZE:
        continue

    n_windows = int(duration//WINDOW_SIZE)

    for w in range(n_windows):

        win_start = start + w*WINDOW_SIZE
        win_end = win_start + WINDOW_SIZE

        print(f"{stage} | window {w} | {win_start}-{win_end}")

        # --- R peaks in window ---
        mask = (r_times>=win_start)&(r_times<win_end)
        r_win = r_times[mask]

        if len(r_win) < MIN_EPOCHS:
            continue

        # --- read ONLY 5 min EEG ---
        start_sample = int(win_start*fs_eeg)
        stop_sample = int(win_end*fs_eeg)

        data,_ = raw[:,start_sample:stop_sample]

        data = mne.filter.filter_data(
            data,
            sfreq=fs_eeg,
            l_freq=0.1,
            h_freq=30,
            verbose=False
        )

        # --- convert R times to local samples ---
        r_local = ((r_win-win_start)*fs_eeg).astype(int)

        epochs = []

        before = int(abs(TMIN)*fs_eeg)
        after = int(TMAX*fs_eeg)

        for r in r_local:

            s = r-before
            e = r+after

            if s<0 or e>=data.shape[1]:
                continue

            ep = data[:,s:e]

            baseline_samples = before
            baseline_mean = ep[:,:baseline_samples].mean(axis=1,keepdims=True)

            ep = ep-baseline_mean

            epochs.append(ep)

        if len(epochs) < MIN_EPOCHS:
            continue

        epochs = np.stack(epochs)

        evoked = epochs.mean(axis=0)

        times = np.linspace(TMIN,TMAX,evoked.shape[1])

        row = {
            "subject":subject_name,
            "stage":stage,
            "window_index":w
        }

        for ch_i,ch in enumerate(SELECTED_CHANNELS):

            signal = evoked[ch_i]

            feats = extract_hep_features(signal,times)

            for k,v in feats.items():
                row[f"{ch}_{k}"] = v

        rows.append(row)

        del data
        del epochs
        del evoked


# -------- SAVE DATASET --------
df = pd.DataFrame(rows)

df.to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV), index=False)
print("\nDataset created successfully")
print("Shape:",df.shape)
print("Saved:",OUTPUT_CSV)