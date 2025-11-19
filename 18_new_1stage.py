print("!st stage begins letssss goooo")

"""
pipeline_stage1.py

Stage 1: Memory-safe preprocessing and epoch extraction (per-subject).
Key points:
 - Read raw header (preload=False) to avoid loading entire EDFs into RAM.
 - Read only ECG channel to detect R-peaks (NeuroKit2).
 - Create Epochs with preload=True (only epoch segments loaded).
 - Filter epochs (IIR) — memory-friendly.
 - Subsample R-peaks if too many.
 - Robust fallbacks: relax reject if all epochs dropped; drop channels with >40% bad epochs and retry.
 - Save per-subject evoked (.fif) and epoch metadata; write manifest.json.
"""

import os
import json
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import mne
import neurokit2 as nk
import pickle

# ----------------- CONFIG -----------------
SUBJECTS = [
#    {"name": "EPCTL01", "edf_path": r"EPCTL01-2025\EPCTL01\EPCTL01 - fixed.edf", "annotation_path": r"EPCTL01-2025\EPCTL01\test1.txt"}
#    {"name": "EPCTL16", "edf_path": r"EPCTL16\EPCTL16\EPCTL16.edf", "annotation_path": r"EPCTL16\EPCTL16\EPCTL16.txt"}
#    {"name": "EPCTL18", "edf_path": r"EPCTL18\EPCTL18\EPCTL18.edf", "annotation_path": r"EPCTL18\EPCTL18\EPCTL18.txt"}
#    {"name": "EPCTL25", "edf_path": r"EPCTL25\EPCTL25\EPCTL25.edf", "annotation_path": r"EPCTL25\EPCTL25\EPCTL25.txt"}
#    {"name": "EPCTL28", "edf_path": r"EPCTL28\EPCTL28\EPCTL28.edf", "annotation_path": r"EPCTL28\EPCTL28\EPCTL28.txt"},
#    {"name": "EPCTL27", "edf_path": r"EPCTL27\EPCTL27\EPCTL27.edf", "annotation_path": r"EPCTL27\EPCTL27\EPCTL27.txt"},
#    {"name": "EPCTL23", "edf_path": r"EPCTL23\EPCTL23\EPCTL23.edf", "annotation_path": r"EPCTL23\EPCTL23\EPCTL23.txt"},
#    {"name": "EPCTL22", "edf_path": r"EPCTL22\EPCTL22\EPCTL22.edf", "annotation_path": r"EPCTL22\EPCTL22\EPCTL22.txt"},
#    {"name": "EPCTL20", "edf_path": r"EPCTL20\EPCTL20\EPCTL20.edf", "annotation_path": r"EPCTL20\EPCTL20\EPCTL20.txt"},
#    {"name": "EPCTL10", "edf_path": r"EPCTL10\EPCTL10\EPCTL10.edf", "annotation_path": r"EPCTL10\EPCTL10\EPCTL10.txt"},
#    {"name": "EPCTL05", "edf_path": r"EPCTL05\EPCTL05\EPCTL05.edf", "annotation_path": r"EPCTL05\EPCTL05\EPCTL05.txt"},
#    {"name": "EPCTL04", "edf_path": r"EPCTL04\EPCTL04\EPCTL04.edf", "annotation_path": r"EPCTL04\EPCTL04\EPCTL04.txt"},
#    {"name": "EPCTL03", "edf_path": r"EPCTL03\EPCTL03\EPCTL03.edf", "annotation_path": r"EPCTL03\EPCTL03\EPCTL03.txt"},
   {"name": "EPCTL26", "edf_path": r"EPCTL26\EPCTL26\EPCTL26.edf", "annotation_path": r"EPCTL26\EPCTL26\EPCTL26.txt"}
]

OUT_DIR = Path("out_stage1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ECG_CHANNEL_NAME = "ECG1"          # nominal ECG channel to look for
MONTAGE_NAME = "standard_1020"     # or None
TMIN, TMAX = -0.200, 0.600
BASELINE = (TMIN, 0.0)
HP, LP = 0.1, 30.0                 # bandpass for epochs
NOTCH = None
REJECT = dict(eeg=150)         # rejection threshold (can relax in fallback)
MAX_EPOCHS_PER_SUBJECT = 1e9     # to bound memory/time (subsamples R-peaks)
DROP_CHANNEL_THRESHOLD = 0.40     # drop channels if >40% epochs rejected due to them
VERBOSE = True
# ---------------------------------------

# channels to exclude (user-provided)
excluded_channels = [
    "ECG1","ECG2","leg","RLEG-","RLEG+","LLEG-","LLEG+",
    "EOG1","EOG2","ChEMG1","ChEMG2","SO1","SO2",
    "ZY1","ZY2","F11","F12","FT11","FT12","TP11","TP12","P11","P12"
]

def normalize_channel_name(name: str) -> str:
    if name is None:
        return ""
    s = name.strip().lower()
    for suf in ["-ref", "_ref", " ref"]:
        if s.endswith(suf):
            s = s[:-len(suf)].strip()
    if s.endswith("+") or s.endswith("-"):
        s = s[:-1]
    s = s.replace(" ", "").replace(".", "").replace("/", "")
    return s

excluded_set = {normalize_channel_name(c) for c in excluded_channels}

# ----------------- utilities -----------------
def find_ecg_channel_name_from_raw(raw, nominal_ecg_name: str) -> str:
    target = normalize_channel_name(nominal_ecg_name)
    # first exact normalized match
    for ch in raw.ch_names:
        if normalize_channel_name(ch) == target:
            return ch
    # fallback: pick channel whose type is ecg
    try:
        ecg_cands = raw.copy().pick_types(ecg=True).ch_names
        if ecg_cands:
            return ecg_cands[0]
    except Exception:
        pass
    # fallback: search name contains 'ecg'
    for ch in raw.ch_names:
        if 'ecg' in ch.lower():
            return ch
    # final: raise
    raise ValueError("ECG channel not found in header")

def select_eeg_channel_names_filtered(raw, excluded_set):
    try:
        eeg_names = raw.copy().pick_types(eeg=True).ch_names
    except Exception:
        eeg_names = [ch for ch in raw.ch_names if normalize_channel_name(ch) not in excluded_set]
    filtered = [ch for ch in eeg_names if normalize_channel_name(ch) not in excluded_set]
    return filtered

# drop channels that cause >threshold fraction of rejections across epochs
def detect_bad_channels_from_drop_log(epochs: mne.Epochs, threshold: float) -> list:
    # note: epochs.drop_log is a list per-epoch containing list of drop reasons (strings like 'eeg')
    # after calling epochs.drop_bad(...), this may not be present. So better to compute by trying reject per-epoch.
    drop_counts = {ch: 0 for ch in epochs.ch_names}
    n_epochs = len(epochs.events)
    if n_epochs == 0:
        return []
    # we'll iterate epoch by epoch and see which channels exceed reject threshold if present
    # Use epochs.get_data() to inspect amplitude per channel
    data = epochs.get_data()  # shape (n_epochs, n_channels, n_times)
    peak_to_peak = data.ptp(axis=2)  # (n_epochs, n_channels)
    rej_thresh = REJECT.get('eeg', None)
    if rej_thresh is None:
        return []
    for ei in range(peak_to_peak.shape[0]):
        bad_chs = np.where(peak_to_peak[ei, :] > rej_thresh)[0]
        for idx in bad_chs:
            drop_counts[epochs.ch_names[idx]] += 1
    # compute fraction
    bad_channels = [ch for ch, c in drop_counts.items() if (c / max(1, n_epochs)) > threshold]
    return bad_channels

# ----------------- subject processing -----------------
def process_subject(edf_path: str, subj_name: str, annotation_path: Optional[str] = None) -> Tuple[Optional[mne.Evoked], dict]:
    """Process single subject: detect R peaks, create epochs (preload=True), filter epochs, average to evoked.
       Returns (evoked, metadata). If processing fails returns (None, metadata_with_error).
    """
    meta = {"name": subj_name, "edf_path": edf_path, "status": "ok", "n_epochs": 0, "notes": []}
    if not os.path.exists(edf_path):
        meta["status"] = "error"
        meta["error"] = f"EDF not found: {edf_path}"
        return None, meta

    try:
        # 1) read header-only
        raw_hdr = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    except Exception as e:
        meta["status"] = "error"
        meta["error"] = f"Failed to read header: {repr(e)}"
        return None, meta

    # optional annotation presence (we don't apply them by default)
    if annotation_path and os.path.exists(annotation_path):
        meta["notes"].append(f"Annotation file: {annotation_path} (found, not applied)")

    # 2) find ECG channel robustly
    try:
        ecg_channel = find_ecg_channel_name_from_raw(raw_hdr, ECG_CHANNEL_NAME)
        meta["ecg_channel"] = ecg_channel
    except Exception as e:
        meta["status"] = "error"
        meta["error"] = f"ECG channel not found: {repr(e)}"
        return None, meta

    # 3) select EEG channel names after exclusion
    eeg_ch_names = select_eeg_channel_names_filtered(raw_hdr, excluded_set)
    if len(eeg_ch_names) == 0:
        meta["status"] = "error"
        meta["error"] = "No EEG channels available after exclusion"
        return None, meta

    # 4) load ECG channel data only (small)
    try:
        ecg_data = raw_hdr.get_data(picks=[ecg_channel])[0]
        sfreq = raw_hdr.info['sfreq']
        meta["sfreq"] = float(sfreq)
        meta["ecg_samples"] = int(len(ecg_data))
    except Exception as e:
        meta["status"] = "error"
        meta["error"] = f"Failed to load ECG channel data: {repr(e)}"
        return None, meta

    if VERBOSE:
        print(f"Processing {subj_name}: {edf_path}")
        print(f"  Found ECG: {ecg_channel}; EEG channels to use: {len(eeg_ch_names)}")

    # 5) detect R-peaks with NeuroKit2 (works on 1D ECG array).
    try:
        # nk.ecg_process expects time series; it returns dictionary info with ECG_R_Peaks indices
        signals, info_nk = nk.ecg_process(ecg_data, sampling_rate=sfreq)
        r_peaks = np.array(info_nk.get("ECG_R_Peaks", []), dtype=int)
    except Exception as e:
        meta["status"] = "error"
        meta["error"] = f"R-peak detection failed: {repr(e)}"
        return None, meta

    if len(r_peaks) < 5:
        meta["status"] = "error"
        meta["error"] = f"Too few R-peaks detected: {len(r_peaks)}"
        return None, meta

    # 6) optionally subsample R-peaks to MAX_EPOCHS_PER_SUBJECT
    if (MAX_EPOCHS_PER_SUBJECT is not None) and (len(r_peaks) > MAX_EPOCHS_PER_SUBJECT):
        rng = np.random.default_rng(0)
        chosen = np.sort(rng.choice(len(r_peaks), size=MAX_EPOCHS_PER_SUBJECT, replace=False))
        r_peaks = r_peaks[chosen]
        meta["notes"].append(f"Subsampled R-peaks to: {len(r_peaks)}")

    # 7) Build events array (samples, 0, event_id)
    events = np.column_stack([r_peaks, np.zeros_like(r_peaks, dtype=int), np.ones_like(r_peaks, dtype=int)])

    # 8) Now re-open raw with preload=False (header) but we'll pick channels and then create epochs with preload=True
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    except Exception as e:
        meta["status"] = "error"
        meta["error"] = f"Failed to re-open EDF header: {repr(e)}"
        return None, meta

    # Build picks: EEG channels only (exclude ECG from picks_eeg)
    picks_eeg = [ch for ch in eeg_ch_names if ch in raw.ch_names]
    if len(picks_eeg) == 0:
        meta["status"] = "error"
        meta["error"] = "No EEG picks found in raw"
        return None, meta

    # For safety, ensure ECG channel present in raw (we won't include it in epoch picks)
    if ecg_channel not in raw.ch_names:
        # nothing critical; proceed
        if VERBOSE:
            print("  Warning: ECG channel disappeared in re-opened raw (continuing)")

    # pick channels to reduce memory when creating epochs (we still keep only EEG picks in epochs)
    # but keep ECG optionally present in raw when picking (not required)
    try:
        raw.pick_channels(picks_eeg + ([] if ecg_channel not in raw.ch_names else [ecg_channel]))
    except Exception:
        # fallback: pick what exists
        common = [ch for ch in picks_eeg if ch in raw.ch_names]
        raw.pick_channels(common)

    # apply montage if requested (safe to fail)
    if MONTAGE_NAME:
        try:
            mont = mne.channels.make_standard_montage(MONTAGE_NAME)
            raw.set_montage(mont)
        except Exception:
            if VERBOSE:
                print("  Montage apply failed (continuing without montage).")

    # 9) Create epochs with preload=True (this loads only epoch segments into memory)
    try:
        if VERBOSE:
            print("  Creating epochs (preload=True) — epoch segments will be loaded into RAM.")
        epochs = mne.Epochs(raw, events=events, event_id={'R':1},
                            tmin=TMIN, tmax=TMAX, baseline=BASELINE,
                            picks=picks_eeg, preload=True, reject=REJECT, verbose=False)
    except Exception as e:
        meta["status"] = "error"
        meta["error"] = f"Epoch creation failed: {repr(e)}"
        return None, meta

    # if all epochs dropped, try relaxed fallback: remove reject criteria once and retry
    if len(epochs) == 0:
        meta["notes"].append("All epochs dropped with initial reject. Retrying with relaxed reject=None.")
        try:
            epochs = mne.Epochs(raw, events=events, event_id={'R':1},
                                tmin=TMIN, tmax=TMAX, baseline=BASELINE,
                                picks=picks_eeg, preload=True, reject=None, verbose=False)
        except Exception as e:
            meta["status"] = "error"
            meta["error"] = f"Epoch creation retry failed: {repr(e)}"
            return None, meta

    # 10) If many epochs were dropped but not all, detect bad channels and drop them if they cause > DROP_CHANNEL_THRESHOLD of rejections
    # We only do this if some epochs got dropped (i.e., original events count > epochs count)
    original_n = len(r_peaks)
    n_epochs_after = len(epochs)
    if VERBOSE:
        print(f"  Created {n_epochs_after} epochs (original beats: {original_n})")

    if n_epochs_after == 0:
        meta["status"] = "error"
        meta["error"] = "No epochs after attempts"
        return None, meta

    # If a lot of epochs were dropped (e.g. >10%), detect channels that cause many rejections and drop them from raw & retry
    if original_n - n_epochs_after > max(1, 0.1 * original_n):
        # try to detect channels causing rejections by creating a temporary epochs object with reject and checking PTPL
        try:
            # build epochs copy with reject applied to inspect drops
            temp_epochs = mne.Epochs(raw, events=events, event_id={'R':1},
                                     tmin=TMIN, tmax=TMAX, baseline=BASELINE,
                                     picks=picks_eeg, preload=True, reject=REJECT, verbose=False)
            # detect channels to drop
            bad_chs = detect_bad_channels_from_drop_log(temp_epochs, DROP_CHANNEL_THRESHOLD)
            if bad_chs:
                meta["notes"].append(f"Dropping channels due to many rejections (> {DROP_CHANNEL_THRESHOLD*100:.0f}%): {bad_chs}")
                if VERBOSE:
                    print("  Dropping channels from raw and retrying epochs:", bad_chs)
                # drop them from raw and re-create epochs with the original reject
                remaining = [ch for ch in picks_eeg if ch not in bad_chs]
                raw.pick_channels(remaining + ([] if ecg_channel not in raw.ch_names else [ecg_channel]))
                if VERBOSE:
                    print("  Recreating epochs after channel drop...")
                epochs = mne.Epochs(raw, events=events, event_id={'R':1},
                                    tmin=TMIN, tmax=TMAX, baseline=BASELINE,
                                    picks=remaining, preload=True, reject=REJECT, verbose=False)
                n_epochs_after = len(epochs)
                meta["notes"].append(f"Epochs retained after dropping channels: {n_epochs_after}")
            # else nothing to drop
        except Exception:
            # ignore errors here; keep current epochs
            pass

    # 11) Filter epoch data (IIR) — this filters only the loaded epoch windows and is memory-friendly
    try:
        if VERBOSE:
            print("  Filtering epoch data (IIR)...")
        epochs.filter(l_freq=HP, h_freq=LP, method='iir', verbose=False)
    except Exception as e:
        # as fallback try FIR on epochs (safer for short signals), or continue unfiltered
        warnings.warn(f"Epoch filtering failed (IIR): {e}; trying FIR filter on epochs.")
        try:
            epochs.filter(l_freq=HP, h_freq=LP, fir_design='firwin', verbose=False)
        except Exception as e2:
            warnings.warn(f"Epoch filtering failed (FIR) as well: {e2}; continuing without filtering.")

    # 12) Average to subject-level evoked
    try:
        evoked = epochs.average()
    except Exception as e:
        meta["status"] = "error"
        meta["error"] = f"Failed to compute evoked: {repr(e)}"
        return None, meta

    # 13) Save subject outputs: evoked (.fif) and epochs metadata (pickle)
    subj_dir = OUT_DIR / subj_name
    subj_dir.mkdir(parents=True, exist_ok=True)
    evoked_fname = subj_dir / f"{subj_name}_evoked-ave.fif"
    epochs_fname = subj_dir / f"{subj_name}_epochs-epo.fif"
    meta_fname = subj_dir / f"{subj_name}_stage1_meta.pkl"
    try:
        evoked.save(str(evoked_fname), overwrite=True)
    except Exception:
        # fallback: pickle the evoked.data and info
        with open(subj_dir / f"{subj_name}_evoked.pkl", "wb") as f:
            pickle.dump({"data": evoked.data, "ch_names": evoked.ch_names, "times": evoked.times, "info": evoked.info}, f)
        meta["notes"].append("Evoked saved with pickle fallback.")

    try:
        # save epochs to disk as .fif (may be big but per-subject)
        epochs.save(str(epochs_fname), overwrite=True)
    except Exception:
        # fallback: pickle minimal epoch info (data + ch_names + times)
        with open(subj_dir / f"{subj_name}_epochs.pkl", "wb") as f:
            pickle.dump({"data": epochs.get_data(), "ch_names": epochs.ch_names, "times": epochs.times}, f)
        meta["notes"].append("Epochs saved with pickle fallback.")

    # finalize metadata
    meta["n_epochs"] = len(epochs)
    meta["status"] = "ok"
    meta["evoked_file"] = str(evoked_fname) if evoked_fname.exists() else str(subj_dir / f"{subj_name}_evoked.pkl")
    meta["epochs_file"] = str(epochs_fname) if epochs_fname.exists() else str(subj_dir / f"{subj_name}_epochs.pkl")
    return evoked, meta

# ----------------- main stage 1 run -----------------
def run_stage1():
    manifest = {"subjects": [], "config": {
        "HP": HP, "LP": LP, "TMIN": TMIN, "TMAX": TMAX, "REJECT": REJECT,
        "MAX_EPOCHS_PER_SUBJECT": MAX_EPOCHS_PER_SUBJECT
    }}
    for subj in SUBJECTS:
        name = subj.get("name")
        edf_path = subj.get("edf_path")
        ann = subj.get("annotation_path", None)
        try:
            evoked, meta = process_subject(edf_path, name, annotation_path=ann)
        except Exception as e:
            evoked, meta = None, {"name": name, "status": "error", "error": repr(e)}
        manifest["subjects"].append(meta)
        # save manifest after each subject to allow incremental progress
        with open(OUT_DIR / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
    print("Stage1 done. Manifest saved to", (OUT_DIR / "manifest.json").as_posix())

if __name__ == "__main__":
    run_stage1()
