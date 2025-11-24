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

# Define analysis window (in seconds)
start_sec = 4        # start time
end_sec = 21       # end time
duration_sec = end_sec - start_sec

# Band-pass filter range
hp, lp = 0.1, 30

# -----------------------------
# Read EDF file
# -----------------------------
f = pyedflib.EdfReader(edf_path)
ch_labels = f.getSignalLabels()

ecg_idx = ch_labels.index(ecg_ch)
eeg_idx = ch_labels.index(eeg_ch)

fs_ecg = f.getSampleFrequency(ecg_idx)
fs_eeg = f.getSampleFrequency(eeg_idx)

# Compute start sample and number of samples
start_sample_ecg = int(start_sec * fs_ecg)
n_ecg = int(duration_sec * fs_ecg)

start_sample_eeg = int(start_sec * fs_eeg)
n_eeg = int(duration_sec * fs_eeg)

# Read the specified segments
ecg_signal = f.readSignal(ecg_idx, start=start_sample_ecg, n=n_ecg)
eeg_signal = f.readSignal(eeg_idx, start=start_sample_eeg, n=n_eeg)
f.close()

# -----------------------------
# ECG R- and T-peak detection
# -----------------------------
signals, info = nk.ecg_process(ecg_signal, sampling_rate=fs_ecg)
r_peaks = info["ECG_R_Peaks"]

# Delineate ECG to get T-peaks
delineate_signals, delineate_info = nk.ecg_delineate(
    ecg_signal, r_peaks, sampling_rate=fs_ecg, method="dwt"
)
t_peaks = delineate_info["ECG_T_Peaks"]

# --- FIX: handle both list and array formats safely ---
if isinstance(t_peaks, list):
    # Flatten the list and remove NaNs or empty entries
    flat_t_peaks = []
    for tp in t_peaks:
        if tp is not None and not np.isnan(tp).any():
            flat_t_peaks.append(tp)
    t_peaks = np.array(flat_t_peaks, dtype=float)
else:
    t_peaks = np.array(t_peaks, dtype=float)

# Remove NaN values and convert to integers
t_peaks = t_peaks[~np.isnan(t_peaks)].astype(int)

# -----------------------------
# Build MNE RawArray for EEG
# -----------------------------
info = mne.create_info(ch_names=[eeg_ch], sfreq=fs_eeg, ch_types=["eeg"])
raw = mne.io.RawArray(eeg_signal[np.newaxis, :], info)

# Band-pass filter EEG
raw.filter(l_freq=hp, h_freq=lp, fir_design="firwin")

# -----------------------------
# Custom Epoching (T → next R - 0.4s)
# -----------------------------
custom_events = []
tmin_list, tmax_list = [], []

for i in range(len(r_peaks) - 1):  # skip last (no next R)
    r = r_peaks[i]
    r_next = r_peaks[i + 1]
    t = t_peaks[i] if i < len(t_peaks) else np.nan

    # Skip invalid or missing T-peaks
    if np.isnan(t) or t <= r or t >= r_next:
        continue

    # Relative times in seconds (relative to R-peak at 0)
    tmin = (t - r) / fs_ecg
    tmax = ((r_next - int(0.4 * fs_ecg)) - r) / fs_ecg

    tmin_list.append(tmin)
    tmax_list.append(tmax)

    # Create event aligned to R (in EEG sample space)
    eeg_sample = int(r / fs_ecg * fs_eeg)
    custom_events.append([eeg_sample, 0, 1])

# Make events array
events = np.array(custom_events)
event_id = {"Rpeak": 1}

# Ensure fixed-size epoch window (MNE requirement)
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
time_ecg = np.arange(len(ecg_signal)) / fs_ecg + start_sec
time_eeg = np.arange(len(eeg_signal)) / fs_eeg + start_sec

fig, axs = plt.subplots(3, 1, figsize=(14, 9), sharex=False)

# ECG with peaks
axs[0].plot(time_ecg, ecg_signal, label="ECG", color="blue")
axs[0].scatter(time_ecg[r_peaks], ecg_signal[r_peaks], color="red", label="R-peaks")
axs[0].scatter(time_ecg[t_peaks], ecg_signal[t_peaks], color="green", label="T-peaks")
axs[0].set_title(f"ECG with R/T Peaks ({start_sec}s - {end_sec}s)")
axs[0].set_xlabel("Time (s)")
axs[0].legend()

# Raw EEG
axs[1].plot(time_eeg, eeg_signal, color="purple")
axs[1].set_title(f"Raw EEG ({eeg_ch})")
axs[1].set_xlabel("Time (s)")

# HEP (averaged EEG epochs)
axs[2].plot(evoked.times, evoked.data[0] * 1e6, label="HEP (averaged)", color="black")
axs[2].axvspan(min(tmin_list), max(tmax_list), color="gray", alpha=0.4, label="HEP window")
axs[2].set_title(f"HEP at {eeg_ch}")
axs[2].set_xlabel("Time from R-Peak (s)")
axs[2].set_ylabel("Amplitude (µV)")
axs[2].legend()

plt.tight_layout()
plt.show()
