import pyedflib
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import mne

# -----------------------------
# Parameters
# -----------------------------
edf_path = r"EPCTL18\EPCTL18\EPCTL18.edf"
ecg_ch = "ECG1"
eeg_ch = "Fp2"

# --- NEW: Define the analysis window with a start and end time ---
start_sec = 11000  # Start time in seconds
end_sec = 11100   # End time in seconds
duration_sec = end_sec - start_sec # The total duration is now calculated

hp, lp = 0.1, 30  # preprocessing filter

# -----------------------------
# Read EDF
# -----------------------------
f = pyedflib.EdfReader(edf_path)
ch_labels = f.getSignalLabels()

ecg_idx = ch_labels.index(ecg_ch)
eeg_idx = ch_labels.index(eeg_ch)

fs_ecg = f.getSampleFrequency(ecg_idx)
fs_eeg = f.getSampleFrequency(eeg_idx)

# Calculate start sample and number of samples to read
start_sample_ecg = int(start_sec * fs_ecg)
n_ecg = int(duration_sec * fs_ecg)

start_sample_eeg = int(start_sec * fs_eeg)
n_eeg = int(duration_sec * fs_eeg)

# Read the specified segment of data
ecg_signal = f.readSignal(ecg_idx, start=start_sample_ecg, n=n_ecg)
eeg_signal = f.readSignal(eeg_idx, start=start_sample_eeg, n=n_eeg)
f.close()

# -----------------------------
# R- and T-peak detection
# -----------------------------
signals, info = nk.ecg_process(ecg_signal, sampling_rate=fs_ecg)
r_peaks = info["ECG_R_Peaks"]

delineate_signals, delineate_info = nk.ecg_delineate(
    ecg_signal, r_peaks, sampling_rate=fs_ecg, method="dwt"
)
# --- FIX: Ensure t_peaks is a NumPy array to support boolean indexing ---
t_peaks = np.array(delineate_info["ECG_T_Peaks"])

# -----------------------------
# Build MNE RawArray for EEG
# -----------------------------
info = mne.create_info(ch_names=[eeg_ch], sfreq=fs_eeg, ch_types=["eeg"])
raw = mne.io.RawArray(eeg_signal[np.newaxis, :], info)

# Preprocess EEG
raw.filter(l_freq=hp, h_freq=lp, fir_design="firwin")

# -----------------------------
# Custom Epoching (T → next R - 0.4s)
# -----------------------------
custom_events = []
tmin_list, tmax_list = [], []

for i in range(len(r_peaks) - 1):  # skip last (no next R)
    r = r_peaks[i]
    r_next = r_peaks[i + 1]
    t = t_peaks[i]

    # skip if no valid T in between
    if np.isnan(t) or t <= r or t >= r_next:
        continue

    # relative times (in sec, relative to R-peak at 0)
    tmin = (t - r) / fs_ecg
    tmax = ((r_next - int(0.4 * fs_ecg)) - r) / fs_ecg

    tmin_list.append(tmin)
    tmax_list.append(tmax)

    # create event aligned to R
    eeg_sample = int(r / fs_ecg * fs_eeg)
    custom_events.append([eeg_sample, 0, 1])

# Make events array
events = np.array(custom_events)
event_id = {"Rpeak": 1}

# Use global min/max window (MNE needs fixed size)
global_tmin = min(tmin_list)
global_tmax = max(tmax_list)

epochs = mne.Epochs(
    raw, events=events, event_id=event_id,
    tmin=global_tmin, tmax=global_tmax,
    baseline=None, preload=True, detrend=0
)

# Average HEP
evoked = epochs.average()

# -----------------------------
# Plotting
# -----------------------------
# Adjust the time axis to reflect the actual start time of the data
time_ecg = np.arange(len(ecg_signal)) / fs_ecg + start_sec
time_eeg = np.arange(len(eeg_signal)) / fs_eeg + start_sec

fig, axs = plt.subplots(3, 1, figsize=(14, 9), sharex=False)

# ECG
axs[0].plot(time_ecg, ecg_signal, label="ECG", color="blue")
axs[0].scatter(time_ecg[r_peaks], ecg_signal[r_peaks], color="red", label="R-peaks")

# Before plotting, remove NaN values from t_peaks to prevent IndexError
t_peaks_for_plotting = t_peaks[~np.isnan(t_peaks)].astype(int)
axs[0].scatter(time_ecg[t_peaks_for_plotting], ecg_signal[t_peaks_for_plotting], color="green", label="T-peaks")

axs[0].set_title(f"ECG with R/T Peaks ({start_sec}s - {end_sec}s)")
axs[0].set_xlabel("Time (s)")
axs[0].legend()

# Raw EEG
axs[1].plot(time_eeg, eeg_signal, color="purple")
axs[1].set_title(f"Raw EEG ({eeg_ch})")
axs[1].set_xlabel("Time (s)")

# HEP
axs[2].plot(evoked.times, evoked.data[0]*1e6, label="HEP (averaged)", color="black")
axs[2].axvspan(min(tmin_list), max(tmax_list), color="gray", alpha=0.4, label="HEP window")
axs[2].set_title(f"HEP at {eeg_ch}")
axs[2].set_xlabel("Time from R-Peak (s)")
axs[2].set_ylabel("Amplitude (µV)")
axs[2].legend()

plt.tight_layout()
plt.show()