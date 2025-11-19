import pyedflib
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import mne
from mne.preprocessing import ICA

# -----------------------------
# Parameters
# -----------------------------
edf_path = r"EPCTL24\EPCTL24.edf"
ecg_ch_name = "ECG1"

# --- Time Window for Data Loading ---
start_sec = 40
end_sec = 80
duration_sec = end_sec - start_sec

# --- Standard HEP Epoching Window ---
tmin, tmax = -0.2, 0.6  # -200ms to +600ms around the R-peak

hp, lp = 0.1, 30

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
print(f"Using {len(eeg_ch_labels)} standard EEG channels.")

ecg_idx = all_labels.index(ecg_ch_name)
fs_ecg = f.getSampleFrequency(ecg_idx)
start_sample_ecg = int(start_sec * fs_ecg)
n_ecg = int(duration_sec * fs_ecg)
ecg_signal = f.readSignal(ecg_idx, start=start_sample_ecg, n=n_ecg)

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
# R-peak detection
# -----------------------------
signals, info = nk.ecg_process(ecg_signal, sampling_rate=fs_ecg)
r_peaks = info["ECG_R_Peaks"]
print(f"\nFound {len(r_peaks)} R-peaks to use as events.")

# -----------------------------
# Build MNE RawArray for EEG
# -----------------------------
info = mne.create_info(ch_names=eeg_ch_labels, sfreq=fs_eeg, ch_types="eeg")
raw = mne.io.RawArray(eeg_signal_stacked, info)

montage = mne.channels.make_standard_montage('standard_1005')
ch_name_mapping = {ch.upper(): ch for ch in montage.ch_names}
rename_dict = {ch: ch_name_mapping[ch] for ch in raw.ch_names if ch in ch_name_mapping}
raw.rename_channels(rename_dict)
raw.set_montage(montage)

raw.filter(l_freq=hp, h_freq=lp, fir_design="firwin")

# -----------------------------
# ICA for Artifact Removal
# -----------------------------
ica = ICA(n_components=15, random_state=97, max_iter=800)
ica.fit(raw)
ica.exclude = [0, 1]
ica.apply(raw)

# -----------------------------
# Standard R-Peak Locked Epoching
# -----------------------------
event_id = {"R-peak": 1}
events = np.array([[int(r / fs_ecg * fs_eeg), 0, event_id["R-peak"]] for r in r_peaks])

epochs = mne.Epochs(
    raw,
    events=events,
    event_id=event_id,
    tmin=tmin,
    tmax=tmax,
    baseline=(-0.2, 0),
    preload=True,
    detrend=0
)
evoked = epochs.average()

# -----------------------------
# Find Peak HEP Amplitudes for Plotting
# -----------------------------
peak_hep_amplitudes = np.max(np.abs(evoked.data), axis=1)

# -----------------------------
# Plotting
# -----------------------------

# Plot 1: Topographical Map of HEP Amplitudes
fig_topo, ax_topo = plt.subplots(figsize=(7, 7))
mne.viz.plot_topomap(
    data=peak_hep_amplitudes * 1e6, # Convert to µV for the color bar
    pos=evoked.info,
    axes=ax_topo,
    show=False,
    cmap='viridis',
    sensors=True
)
ax_topo.set_title("Topography of Peak HEP Amplitude (µV)", fontsize=16)

# Plot 2: ECG and Averaged HEP Waveforms
fig_waves, axs = plt.subplots(2, 1, figsize=(14, 10))

# ECG Plot
time_axis = np.arange(len(ecg_signal)) / fs_ecg + start_sec
axs[0].plot(time_axis, ecg_signal, label="ECG", color="blue")
axs[0].scatter(r_peaks / fs_ecg + start_sec, ecg_signal[r_peaks], color="red", label="R-peaks")
axs[0].set(title=f"ECG with R-Peaks ({start_sec}-{end_sec}s)", xlabel="Time (s)", ylabel="Amplitude")
axs[0].legend()

# HEP "Butterfly" Waveform Plot
evoked.plot(axes=axs[1], show=False, spatial_colors=True)
axs[1].axvline(0, color='red', linestyle='--', linewidth=1, label='R-Peak at t=0')
axs[1].set_title("Heartbeat Evoked Potentials (All Channels)")
axs[1].legend()

plt.tight_layout()
plt.show()