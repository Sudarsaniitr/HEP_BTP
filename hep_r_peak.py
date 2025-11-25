import pyedflib
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import mne
import warnings

# ----------------------------- USER PARAMETERS -----------------------------
edf_path = r"EPCTL16\EPCTL16\EPCTL16.edf"
ecg_ch = "ECG1"
eeg_ch = "Fp2"

# Segment to read in absolute recording seconds
start_sec = 10030.0
end_sec = 10035.0
duration_sec = end_sec - start_sec

# Preprocessing
hp, lp = 0.1, 30.0  # bandpass for EEG (Hz)

# Epoch window relative to R-peak (R = 0)
TMIN = -0.2   # -200 ms
TMAX = 0.6    # +600 ms
BASELINE = (TMIN, 0.0)  # baseline correction window

# Analysis highlight window on HEP plot
ANALYSIS_LO = 0.25
ANALYSIS_HI = 0.6

# Output plot file (optional)
OUT_PNG = "hep_rref_fixed.png"
# ----------------------------------------------------------------------------

# --- Read EDF snippet ---
f = pyedflib.EdfReader(edf_path)
ch_labels = f.getSignalLabels()

if ecg_ch not in ch_labels:
    raise ValueError(f"ECG channel '{ecg_ch}' not in EDF. Available (first 40): {ch_labels[:40]}")
if eeg_ch not in ch_labels:
    raise ValueError(f"EEG channel '{eeg_ch}' not in EDF. Available (first 40): {ch_labels[:40]}")

ecg_idx = ch_labels.index(ecg_ch)
eeg_idx = ch_labels.index(eeg_ch)

fs_ecg = f.getSampleFrequency(ecg_idx)
fs_eeg = f.getSampleFrequency(eeg_idx)

start_sample_ecg = int(start_sec * fs_ecg)
n_ecg = int(duration_sec * fs_ecg)

start_sample_eeg = int(start_sec * fs_eeg)
n_eeg = int(duration_sec * fs_eeg)

ecg_signal = f.readSignal(ecg_idx, start=start_sample_ecg, n=n_ecg)
eeg_signal = f.readSignal(eeg_idx, start=start_sample_eeg, n=n_eeg)
f.close()

print(f"Read snippet: {duration_sec}s  ECG fs={fs_ecg}Hz  EEG fs={fs_eeg}Hz  samples ECG={len(ecg_signal)} EEG={len(eeg_signal)}")

# --- Detect R-peaks (NeuroKit2) ---
signals, info = nk.ecg_process(ecg_signal, sampling_rate=fs_ecg)
r_peaks = np.array(info.get("ECG_R_Peaks", []), dtype=int)
print("Detected R-peaks (ECG-sample indices, relative to snippet):", len(r_peaks))

if len(r_peaks) == 0:
    raise RuntimeError("No R-peaks detected in snippet — check ECG channel or snippet bounds.")

# optional: get T-peaks for plotting (may be empty)
deline_signals, deline_info = nk.ecg_delineate(ecg_signal, r_peaks, sampling_rate=fs_ecg, method="dwt")
t_peaks_raw = deline_info.get("ECG_T_Peaks", [])
# flatten to 1D ints where possible for plotting (may be list/arrays)
t_peaks_flat = []
if isinstance(t_peaks_raw, (list, tuple)):
    for item in t_peaks_raw:
        try:
            arr = np.asarray(item)
            arr = arr[~np.isnan(arr)]
            if arr.size > 0:
                # choose first if multiple (delineation outputs can vary)
                t_peaks_flat.append(int(arr[0]))
        except Exception:
            pass
else:
    try:
        arr = np.asarray(t_peaks_raw)
        arr = arr[~np.isnan(arr)]
        if arr.size > 0:
            t_peaks_flat = list(arr.astype(int))
    except Exception:
        t_peaks_flat = []

t_peaks = np.array(t_peaks_flat, dtype=int)

# --- Build MNE RawArray for EEG and filter ---
info_mne = mne.create_info(ch_names=[eeg_ch], sfreq=fs_eeg, ch_types=["eeg"])
raw = mne.io.RawArray(eeg_signal[np.newaxis, :], info_mne, verbose=False)

# filtering (safely)
try:
    raw.filter(l_freq=hp, h_freq=lp, fir_design="firwin", verbose=False)
except Exception as e:
    warnings.warn(f"Filtering failed or skipped: {e}")

# --- convert R-peaks (ECG sample indices) -> EEG sample indices ---
r_times_sec = r_peaks.astype(float) / fs_ecg                      # seconds relative to snippet start
r_peaks_eeg = np.round(r_times_sec * fs_eeg).astype(int)          # eeg-sample indices (relative to snippet start)

# --- remove events too close to edges so full epoch fits ---
valid_mask = []
n_samples_eeg = raw.n_times
samples_before = int(np.floor(abs(TMIN) * fs_eeg))
samples_after = int(np.ceil(TMAX * fs_eeg))
for samp in r_peaks_eeg:
    start_idx = samp - samples_before
    stop_idx = samp + samples_after
    valid_mask.append((start_idx >= 0) and (stop_idx < n_samples_eeg))
valid_mask = np.array(valid_mask, dtype=bool)

r_peaks_eeg = r_peaks_eeg[valid_mask]
r_times_sec = r_times_sec[valid_mask]
print(f"R-peaks after boundary check: {len(r_peaks_eeg)} (removed {np.sum(~valid_mask)})")

if len(r_peaks_eeg) == 0:
    raise RuntimeError("No R-peaks remain after boundary check (epoch would exceed snippet bounds).")

# --- build events array for MNE ---
events = np.zeros((len(r_peaks_eeg), 3), dtype=int)
events[:, 0] = r_peaks_eeg
events[:, 2] = 1
event_id = {"R": 1}

# --- create epochs with fixed tmin/tmax and baseline correction ---
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=TMIN, tmax=TMAX,
                    baseline=BASELINE, preload=True, verbose=False, reject_by_annotation=False)
print("Created epochs:", len(epochs), " Epoch shape (ch, time):", epochs.get_data().shape[1:])

if len(epochs) == 0:
    raise RuntimeError("No epochs created after epoching. Check TMIN/TMAX and data length.")

# --- average (evoked) ---
evoked = epochs.average()
hep = evoked.data[0]        # single channel
times = evoked.times        # expected from TMIN to TMAX

# --- plotting ---
time_ecg = np.arange(len(ecg_signal)) / fs_ecg + start_sec
time_eeg = np.arange(len(eeg_signal)) / fs_eeg + start_sec

fig, axs = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)

# ECG with R/T peaks (absolute times)
axs[0].plot(time_ecg, ecg_signal, color="C0", label="ECG")
if len(r_peaks) > 0:
    # use only the valid (kept) R-peaks for markers
    kept_ecg_indices = (r_peaks[valid_mask] if 'valid_mask' in locals() else r_peaks)
    if len(kept_ecg_indices) > 0:
        axs[0].scatter(kept_ecg_indices / fs_ecg + start_sec, ecg_signal[kept_ecg_indices],
                       color="red", s=10, label="R-peaks")
if len(t_peaks) > 0:
    # plot T-peaks that fall into snippet and were found
    t_in_range = t_peaks[(t_peaks >= 0) & (t_peaks < len(ecg_signal))]
    if t_in_range.size > 0:
        axs[0].scatter(t_in_range / fs_ecg + start_sec, ecg_signal[t_in_range],
                       color="black", s=20, label="T-peaks")

axs[0].set_title(f"ECG snippet ({start_sec:.1f}s - {end_sec:.1f}s)")
axs[0].set_xlabel("Time (s)")
axs[0].legend(loc="upper right")

# Raw EEG snippet with vertical lines marking R-peaks
axs[1].plot(time_eeg, eeg_signal, color="C1")
if len(r_peaks_eeg) > 0:
    axs[1].vlines(r_peaks_eeg / fs_eeg + start_sec,
                  ymin=np.min(eeg_signal), ymax=np.max(eeg_signal),
                  color='red', linewidth=0.6, alpha=0.6, label='R-peaks')
axs[1].set_title(f"Raw EEG ({eeg_ch}) snippet")
axs[1].set_xlabel("Time (s)")
axs[1].legend(loc="upper right")

# HEP plot (relative to R-peak)
axs[2].plot(times, hep * 1e6, color="black", label="HEP (avg, µV)")  # convert to µV
# highlight analysis window
axs[2].axvspan(ANALYSIS_LO, ANALYSIS_HI, color="yellow", alpha=0.35, label=f"Analysis window {ANALYSIS_LO}-{ANALYSIS_HI}s")
axs[2].axvline(0.0, color="red", linestyle="--", label="R-peak (0s)")
axs[2].set_xlim(TMIN, TMAX)
axs[2].set_xlabel("Time from R-peak (s)")
axs[2].set_ylabel("Amplitude (µV)")
axs[2].set_title(f"HEP at {eeg_ch}  (n_epochs = {len(epochs)})")
axs[2].legend(loc="upper right")

plt.savefig(OUT_PNG, dpi=150)
print("Saved HEP plot to:", OUT_PNG)
plt.show()
