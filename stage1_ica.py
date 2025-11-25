"""
Stage-1 HEP processing with optional ICA + ASR-lite

Saves per-subject NPZ with keys:
  hep       -> ndarray (n_channels, n_times)
  channels  -> list of channel names (matching hep rows)
  times     -> 1D times vector (tmin..tmax)
  subject   -> subject name
  group     -> group label (if provided)
  meta      -> dict with processing flags

Run: python endgame_stage1_with_ica_asr.py
"""

import os
import argparse
import json
import numpy as np
import mne
import neurokit2 as nk
from tqdm import tqdm

# ----------------------
# CONFIG / defaults
# ----------------------
OUT_DIR = "out_stage1_hep"
os.makedirs(OUT_DIR, exist_ok=True)

HP = 0.1
LP = 30.0
TMIN = -0.2
TMAX = 0.6

EXCLUDED_CHANNELS = [
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
            s = s[:-len(suf)]
    if s.endswith("+") or s.endswith("-"):
        s = s[:-1]
    return s.replace(" ", "").replace(".", "").replace("/", "")

EXCLUDED_SET = {normalize_channel_name(c) for c in EXCLUDED_CHANNELS}

def reject_mad(x, factor=5.0):
    """MAD-based artifact detection per-epoch"""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return np.abs(x - med) > (factor * mad)

# ----------------------
# ASR-lite implementation (fast, simple)
# ----------------------
def asr_like_interpolate(raw, picks, sfreq, window_s=0.5, step_s=0.25, thresh=5.0, verbose=False):
    """
    Very simple ASR-like repair:
      - compute channel-wise RMS in sliding windows
      - mark windows where RMS > thresh * median(RMS across windows)
      - for marked windows, replace samples by linear interpolation between window edges (per channel)
    Notes:
      - raw must be loaded for picks (use raw.copy().pick(picks).load_data()) before calling
      - This is conservative and fast; not a full ASR implementation.
    """
    data = raw.get_data(picks=picks)  # shape: (n_ch, n_samples)
    n_ch, n_samps = data.shape
    win = int(round(window_s * sfreq))
    step = int(round(step_s * sfreq))
    if win < 2:
        return 0
    starts = list(range(0, max(1, n_samps - win + 1), max(1, step)))
    rms = np.zeros((n_ch, len(starts)))
    for wi, s in enumerate(starts):
        seg = data[:, s:s+win]
        rms[:, wi] = np.sqrt(np.nanmean(seg**2, axis=1))

    med_rms = np.median(rms, axis=1)  # per channel
    flagged_windows = np.zeros((n_ch, len(starts)), dtype=bool)
    for ch in range(n_ch):
        thr = med_rms[ch] * thresh
        flagged_windows[ch, :] = rms[ch, :] > thr

    # create boolean mask per sample per channel
    sample_flag = np.zeros_like(data, dtype=bool)
    for wi, s in enumerate(starts):
        e = min(s + win, n_samps)
        sample_flag[:, s:e] |= flagged_windows[:, wi][:, None]

    # For each channel, find contiguous flagged segments and interpolate
    repaired_count = 0
    for ch in range(n_ch):
        mask = sample_flag[ch, :]
        if not mask.any():
            continue
        # find contiguous runs
        idx = np.where(mask)[0]
        runs = np.split(idx, np.where(np.diff(idx) != 1)[0]+1)
        for run in runs:
            a = run[0] - 1
            b = run[-1] + 1
            # clip edges
            a_val = data[ch, a] if a >= 0 else None
            b_val = data[ch, b] if b < n_samps else None
            if a >= 0 and b < n_samps:
                # linear interpolation between a and b
                npts = b - a - 1
                interp = np.linspace(a_val, b_val, npts+2)[1:-1]
                data[ch, a+1:b] = interp
            elif a < 0 and b < n_samps:
                # forward-fill
                data[ch, :b] = data[ch, b]
            elif a >= 0 and b >= n_samps:
                data[ch, a+1:] = data[ch, a]
            else:
                # whole signal flagged; leave as is (rare)
                pass
            repaired_count += 1

    # put repaired data back into raw
    raw._data[:, :] = raw._data.copy()  # ensure writable backing
    # careful: raw.get_data(picks=picks) returned new array; we need to write back per pick index
    pick_idxs = mne.pick_channels(raw.ch_names, include=picks, exclude=[])
    # But because we called raw.get_data(picks=picks) on raw with those picks, raw._data must have those channels in same order.
    # Simpler: assume raw has only the picked channels (we will call this function on raw that was reduced to picks only)
    raw._data[:n_ch, :n_samps] = data
    if verbose:
        print(f"ASR-lite: repaired {repaired_count} segments across {n_ch} channels.")
    return repaired_count

# ----------------------
# ICA helpers
# ----------------------
def apply_ica_to_raw(raw, picks_eeg, n_components=0.99, method="fastica", random_state=42, verbose=False):
    """
    Fit ICA on (optionally decimated) data and apply to raw.
      - raw should be pre-filtered (HP/LP) before ICA
      - returns fitted ICA instance (applied in-place to raw)
    """
    ica = mne.preprocessing.ICA(n_components=n_components, method=method, random_state=random_state, verbose=False)

    # fit on copies to avoid modifying original
    raw_for_ica = raw.copy().pick(picks_eeg)
    # optionally downsample to speed up ICA fitting
    try:
        raw_for_ica.resample(sfreq=min(1000.0, raw.info['sfreq']))  # fit ICA at up to 200 Hz
    except Exception:
        pass

    # fit
    ica.fit(raw_for_ica)
    # auto-detect ECG/EOG-related components
    ecg_inds, ecg_scores = ica.find_bads_ecg(raw, method="correlation", ch_name=None)
    eog_inds, eog_scores = [], []
    try:
        eog_inds, eog_scores = ica.find_bads_eog(raw, ch_name=None)
    except Exception:
        pass

    bads = list(set(ecg_inds + eog_inds))
    if verbose:
        print(f"ICA: found ECG inds {ecg_inds}, EOG inds {eog_inds} -> excluding {bads}")

    if len(bads) > 0:
        ica.exclude = bads
        ica.apply(raw)  # apply to original raw (in-place)
    else:
        # still apply even if no exclusions: project back (no-op)
        ica.apply(raw)
    return ica

# ----------------------
# Subject processing
# ----------------------
def process_subject(subj, out_dir, do_ica=True, do_asr=True, asr_thresh=5.0, asr_win=0.5, asr_step=0.25, verbose=True):
    name = subj.get("name")
    edf_path = subj.get("edf_path")
    group = subj.get("group", subj.get("status", "unknown"))

    print(f"\n==== {name} ====")
    if not os.path.exists(edf_path):
        print(f"EDF NOT FOUND: {edf_path}")
        return None

    # 1) open raw (no preload), pick channels of interest
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    sfreq = raw.info['sfreq']

    # Normalize channel names to consistent format & uppercase (helps montage mapping later)
    rename_map = {}
    for ch in raw.ch_names:
        new = ch.upper().replace("-REF", "").replace("_REF", "").strip()
        rename_map[ch] = new
    raw.rename_channels(rename_map)

    # detect channel types
    eeg_chs = []
    ecg_ch = None
    for ch in raw.ch_names:
        ch_low = normalize_channel_name(ch)
        ch_type = raw.get_channel_types(picks=[ch])[0]
        if ch_type == 'ecg' or 'ECG' in ch.upper():
            ecg_ch = ch
            continue
        if ch_type == 'eeg' and ch_low not in EXCLUDED_SET:
            eeg_chs.append(ch)

    if len(eeg_chs) == 0:
        print("No EEG channels found after exclusion — skipping.")
        return None
    if ecg_ch is None:
        # try heuristics to find ECG channel name (contains 'ECG' substring)
        for ch in raw.ch_names:
            if 'ECG' in ch.upper():
                ecg_ch = ch
                break

    print(f"EEG channels kept: {len(eeg_chs)}; ECG: {ecg_ch}")

    # pick only EEG + ECG channels into memory (avoid loading whole file)
    picks = eeg_chs.copy()
    if ecg_ch:
        picks = [ecg_ch] + picks
    raw.pick_channels(picks)
    raw.load_data()   # safe: we've limited to needed channels

    # re-reference to average of EEG channels
    raw.set_eeg_reference('average', projection=False)

    # Optional ASR-lite BEFORE filtering (it reduces transients that spoil filters/ICA)
    if do_asr:
        try:
            repaired = asr_like_interpolate(raw, picks=raw.ch_names, sfreq=sfreq,
                                            window_s=asr_win, step_s=asr_step, thresh=asr_thresh, verbose=verbose)
            if verbose:
                print(f"ASR-like repaired segments: {repaired}")
        except Exception as e:
            print("ASR-lite failed:", e)

    # filter EEG channels only (we picked EEG+ECG; filter picks EEG channels)
    raw.filter(HP, LP, picks=eeg_chs, method='iir', verbose=False)

    # ICA (optional)
    ica = None
    if do_ica:
        try:
            ica = apply_ica_to_raw(raw, picks_eeg=eeg_chs, n_components=0.99, verbose=verbose)
        except Exception as e:
            print("ICA failed:", e)
            ica = None

    # Now prepare for R-peak detection: pick ECG channel signal from raw (if present)
    if ecg_ch is None:
        print("No ECG channel found; cannot detect R-peaks reliably. Skipping subject.")
        return None

    ecg_sig = raw.get_data(picks=[ecg_ch])[0]
    # R-peak detection via neurokit2
    try:
        _, info_nk = nk.ecg_process(ecg_sig, sampling_rate=sfreq)
        r_peaks = np.array(info_nk.get("ECG_R_Peaks", []), dtype=int)
    except Exception as e:
        print("ECG processing failed:", e)
        r_peaks = np.array([], dtype=int)

    print(f"Detected R-peaks: {len(r_peaks)}")
    if len(r_peaks) < 5:
        print("Too few R-peaks — skipping.")
        return None

    # Epoch window in samples
    smin = int(round(TMIN * sfreq))
    smax = int(round(TMAX * sfreq))
    wlen = smax - smin + 1
    times = np.linspace(TMIN, TMAX, wlen)

    # Prepare output HEP matrix (n_channels x wlen)
    hep_mat = np.zeros((len(eeg_chs), wlen), dtype=float)

    # For each channel compute HEP independently
    for i_ch, ch in enumerate(eeg_chs):
        sig = raw.get_data(picks=[ch])[0]
        epochs = []
        # iterate r-peaks and extract epoch
        for r in r_peaks:
            start = r + smin
            end = r + smax
            if start < 0 or end >= len(sig):
                continue
            ep = sig[start:end+1].copy()
            # artifact rejection using MAD (per-epoch)
            if reject_mad(ep).any():
                continue
            # baseline correction full -200..0 ms => baseline_samples = |TMIN|*sfreq
            baseline_samples = int(round(abs(TMIN) * sfreq))
            if baseline_samples > 0:
                baseline_val = np.mean(ep[:baseline_samples])
                ep = ep - baseline_val
            epochs.append(ep)

        if len(epochs) > 0:
            hep_mat[i_ch, :] = np.mean(np.vstack(epochs), axis=0)
        else:
            hep_mat[i_ch, :] = np.zeros(wlen)  # no epochs -> zeros

    # Save per-subject NPZ
    out_file = os.path.join(out_dir, f"{name}_HEP.npz")
    meta = {
        "HP": HP, "LP": LP, "TMIN": TMIN, "TMAX": TMAX,
        "do_ica": bool(do_ica), "do_asr": bool(do_asr),
        "asr_win_s": asr_win, "asr_step_s": asr_step, "asr_thresh": asr_thresh
    }
    np.savez(
        out_file,
        hep=hep_mat,
        channels=np.array(eeg_chs, dtype=object),
        times=times,
        subject=name,
        group=group,
        meta=meta
    )
    print(f"Saved: {out_file}  (HEP shape: {hep_mat.shape})")
    return {"name": name, "group": group, "file": out_file, "n_channels": len(eeg_chs)}

# ----------------------
# MAIN
# ----------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subjects-file", default=None, help="Optional JSON list of subject dicts (name, edf_path, group). If omitted uses embedded SUBJECTS var.")
    p.add_argument("--outdir", default=OUT_DIR)
    p.add_argument("--do-ica", action="store_true", help="Enable ICA (default False unless set).")
    p.add_argument("--do-asr", action="store_true", help="Enable ASR-lite (default False unless set).")
    p.add_argument("--asr-thresh", type=float, default=5.0)
    p.add_argument("--asr-win", type=float, default=0.5)
    p.add_argument("--asr-step", type=float, default=0.25)
    args = p.parse_args()

    out_dir = args.outdir
    os.makedirs(out_dir, exist_ok=True)

    # -- you can either pass subjects-file or edit SUBJECTS below --
    if args.subjects_file:
        with open(args.subjects_file, "r") as f:
            SUBJECTS = json.load(f)
    else:
        # EXAMPLE single-subject list; replace with your subjects
        SUBJECTS = [
            # {"name": "EPCTL26", "edf_path": r"EPCTL26\EPCTL26\EPCTL26.edf", "group": "bad"},
            # ...
        ]
        print("No subjects-file provided. Please edit the script or provide --subjects-file pointing to JSON list.")

    manifest_out = {"subjects": []}

    for subj in SUBJECTS:
        res = process_subject(subj, out_dir, do_ica=args.do_ica, do_asr=args.do_asr,
                              asr_thresh=args.asr_thresh, asr_win=args.asr_win, asr_step=args.asr_step,
                              verbose=True)
        if res:
            manifest_out["subjects"].append(res)

    # save manifest
    manifest_path = os.path.join(out_dir, "manifest_hep.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest_out, f, indent=2)
    print("Done. Manifest:", manifest_path)

if __name__ == "__main__":
    main()
