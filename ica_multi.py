import pyedflib
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import mne
from mne.preprocessing import ICA

# -----------------------------
# Parameters
# -----------------------------
edf_path = r"EPCTL25\EPCTL25\EPCTL25.edf"
ecg_ch_name = "ECG1"
raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

# Print all available annotations
print(raw.annotations)

# --- MODIFIED TIME WINDOW ---
start_sec = 11000  # Start time for analysis
end_sec = 11100  # End time for analysis
duration_sec = end_sec - start_sec # The total duration is now calculated

hp, lp = 0.1, 30  # bandpass filter

# Channels to exclude from EEG
excluded_channels = [
    "ECG1", "ECG2", "leg", "RLEG-", "RLEG+", "LLEG-", "LLEG+",
    "EOG1", "EOG2", "ChEMG1", "ChEMG2",
    "SO1", "SO2", "ZY1", "ZY2", "F11", "F12",
    "FT11", "FT12", "TP11", "TP12", "P11", "P12"
]

# -----------------------------
# Read EDF
# -----------------------------
f = pyedflib.EdfReader(edf_path)
all_labels = f.getSignalLabels()

eeg_ch_labels = [label for label in all_labels if label not in excluded_channels]

# --- MODIFIED SIGNAL READING ---
# Read ECG signal from the specified start time
ecg_idx = all_labels.index(ecg_ch_name)
fs_ecg = f.getSampleFrequency(ecg_idx)
start_sample_ecg = int(start_sec * fs_ecg)
n_ecg = int(duration_sec * fs_ecg)
ecg_signal = f.readSignal(ecg_idx, start=start_sample_ecg, n=n_ecg)

# Read EEG signals from the specified start time
eeg_signals = []
fs_eeg = f.getSampleFrequency(all_labels.index(eeg_ch_labels[0]))
start_sample_eeg = int(start_sec * fs_eeg)
n_eeg = int(duration_sec * fs_eeg)
for ch in eeg_ch_labels:
    idx = all_labels.index(ch)
    eeg_signals.append(f.readSignal(idx, start=start_sample_eeg, n=n_eeg))
f.close()
eeg_signal_stacked = np.array(eeg_signals)

# -----------------------------
# R- and T-peak detection
# -----------------------------
signals, info = nk.ecg_process(ecg_signal, sampling_rate=fs_ecg)
r_peaks = info["ECG_R_Peaks"]
delineate_signals, delineate_info = nk.ecg_delineate(
    ecg_signal, r_peaks, sampling_rate=fs_ecg, method="dwt"
)
t_peaks = delineate_info["ECG_T_Peaks"]

t_peaks_np = np.array(t_peaks)
valid_t_peaks_mask = ~np.isnan(t_peaks_np)
r_peaks = r_peaks[valid_t_peaks_mask]
t_peaks = t_peaks_np[valid_t_peaks_mask].astype(int)

# -----------------------------
# Build MNE RawArray for EEG
# -----------------------------
info = mne.create_info(ch_names=eeg_ch_labels, sfreq=fs_eeg, ch_types="eeg")
raw = mne.io.RawArray(eeg_signal_stacked, info)
raw.set_meas_date(None)
raw.filter(l_freq=hp, h_freq=lp, fir_design="firwin")

# -----------------------------
# ICA for Artifact Removal
# -----------------------------
ica = ICA(n_components=15, random_state=97, max_iter=800)
ica.fit(raw)
ica.exclude = [0, 1]
ica.apply(raw)

# -----------------------------
# Custom Epoching (T → next R - 0.4s)
# -----------------------------
custom_events = []
tmin_list, tmax_list = [], []
for i in range(len(r_peaks) - 1):
    r, r_next, t = r_peaks[i], r_peaks[i + 1], t_peaks[i]
    if np.isnan(t) or t <= r or t >= r_next: continue
    tmin, tmax = (t - r) / fs_ecg, ((r_next - int(0.4 * fs_ecg)) - r) / fs_ecg
    tmin_list.append(tmin)
    tmax_list.append(tmax)
    eeg_sample = int(r / fs_ecg * fs_eeg)
    custom_events.append([eeg_sample, 0, 1])
events = np.array(custom_events)
event_id = {"Rpeak": 1}

epochs = mne.Epochs(
    raw, events=events, event_id=event_id,
    tmin=min(tmin_list), tmax=max(tmax_list),
    baseline=None, preload=True, detrend=0
)
evoked = epochs.average()

# -----------------------------
# Find and Rank HEP Channels
# -----------------------------
max_hep_values = np.max(np.abs(evoked.data), axis=1) * 1e6
sorted_indices = np.argsort(max_hep_values)[::-1]
hep_ranking = [(eeg_ch_labels[idx], max_hep_values[idx]) for idx in sorted_indices]

print("\nEEG channels ranked by HEP amplitude (µV):")
for ch, val in hep_ranking:
    print(f"- {ch}: {val:.2f} µV")

# -----------------------------
# Plot ECG and top HEP channels
# -----------------------------
num_to_plot = min(5, len(eeg_ch_labels))
fig, axs = plt.subplots(2, 1, figsize=(14, 12), sharex=False)

# --- MODIFIED PLOTTING ---
# ECG time axis now reflects the actual time from the file (40-120s)
time_axis = np.arange(len(ecg_signal)) / fs_ecg + start_sec
axs[0].plot(time_axis, ecg_signal, label="ECG", color="blue")
axs[0].scatter(r_peaks / fs_ecg + start_sec, ecg_signal[r_peaks], color="red", label="R-peaks")
axs[0].scatter(t_peaks / fs_ecg + start_sec, ecg_signal[t_peaks], color="green", label="T-peaks")
axs[0].set_title("ECG with R/T Peaks (40-120s)")
axs[0].set_xlabel("Time (s)")
axs[0].legend()

# HEP Plot for Top N Channels
for i in range(num_to_plot):
    ch_index = sorted_indices[i]
    ch_name = eeg_ch_labels[ch_index]
    axs[1].plot(evoked.times, evoked.data[ch_index] * 1e6,
                label=f"{ch_name} ({max_hep_values[ch_index]:.2f} µV)")

axs[1].axvspan(min(tmin_list), max(tmax_list), color="gray", alpha=0.4, label="HEP window")
axs[1].set_title(f"Top {num_to_plot} Channels by HEP Amplitude")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Amplitude (µV)")
axs[1].legend()

plt.tight_layout()
plt.show()