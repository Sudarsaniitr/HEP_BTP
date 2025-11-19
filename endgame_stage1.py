"""
stage1_hep_optimized.py

Optimized Stage-1 HEP extraction (single-subject .npz outputs).
- Batch reads merged blocks (one read per merged interval).
- Vectorized epoch extraction for all channels from each block.
- Running-sum aggregator (no storing of all epochs).
- Optional downsampling (decimation) per-block to reduce time/memory.
- Progress bars via tqdm.
- Saves per-subject `{name}_HEP.npz` containing:
    hep: (n_channels, n_times)
    channels: list of channel names
    times: time vector (n_times,)
    subject, group

Configurable vars near top.
"""

import os
import json
import numpy as np
import mne
import neurokit2 as nk
from scipy.signal import butter, filtfilt, decimate
from tqdm import tqdm

# -------------------- USER CONFIG --------------------
SUBJECTS = [
    # fill in your subjects here. Example:
    # {"name":"EPCTL01","edf_path":r"EPCTL01\EPCTL01.edf","group":"good"},
    {"name":"EPCTL16","edf_path":r"EPCTL16\EPCTL16\EPCTL16.edf","group":"good"},
    {"name":"EPCTL25","edf_path":r"EPCTL25\EPCTL25\EPCTL25.edf","group":"bad"},
    {"name":"EPCTL18","edf_path":r"EPCTL18\EPCTL18\EPCTL18.edf","group":"bad"},
    {"name":"EPCTL01","edf_path":r"EPCTL01-2025\EPCTL01\EPCTL01 - fixed.edf","group":"good"},
]

OUT_DIR = "out_stage1_hep_opt"
os.makedirs(OUT_DIR, exist_ok=True)

HP = 0.1            # highpass (Hz)
LP = 30.0           # lowpass (Hz)
TMIN = -0.2         # epoch start (s) -200 ms
TMAX = 0.6          # epoch end (s) +600 ms
BASELINE = (TMIN, 0.0)

# Downsample target (set to None to ktaskeep native sfreq). Reasonable choices: 250, 200, 300
TARGET_SFREQ = 250.0

# Maximum RAM (GB) allowed for a single read-block. We'll avoid reading blocks larger than this.
# Set to your machine RAM minus OS/buffer. You told me you have 16 GB; keep safe margin -> use 8 GB default.
MAX_RAM_GB = 16.0

# Merge nearby epoch windows if they are within this many milliseconds (to reduce number of reads).
MERGE_MARGIN_MS = 5.0

# Excluded channels and normalization (you provided these)
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

EXCLUDED_SET = {normalize_channel_name(c) for c in excluded_channels}

# Simple MAD-based reject (vectorized use)
def reject_mad_array(arr, factor=5.0):
    """
    arr: ndarray (..., time)
    returns: boolean mask (...,) True = reject that channel/epoch
    """
    med = np.median(arr, axis=-1)
    mad = np.median(np.abs(arr - med[..., None]), axis=-1)
    # avoid zero mad
    mad = np.where(mad == 0, 1e-12, mad)
    return np.any(np.abs(arr - med[..., None]) > factor * mad[..., None], axis=-1)


# -------------------- Helpers --------------------
def bytes_for_block(n_channels, n_samples, dtype=np.float64):
    return n_channels * n_samples * np.dtype(dtype).itemsize

def safe_block_limit_samples(max_ram_gb, n_channels, dtype=np.float64):
    max_bytes = max_ram_gb * (1024 ** 3)
    return int(max_bytes // (n_channels * np.dtype(dtype).itemsize))

# small utility to design IIR filters (we use filtfilt on small blocks)
def make_filter_coeffs(sfreq, hp, lp):
    b_hp, a_hp = butter(2, hp / (sfreq/2), btype='highpass')
    b_lp, a_lp = butter(4, lp / (sfreq/2), btype='lowpass')
    return (b_hp, a_hp), (b_lp, a_lp)


# -------------------- MAIN PROCESSING --------------------
def process_subject_opt(subj):
    name = subj.get("name", "unknown")
    edf_path = subj.get("edf_path")
    group = subj.get("group", "unknown")
    print("\n" + "="*25 + f" {name} " + "="*25)

    if not os.path.exists(edf_path):
        print("EDF missing:", edf_path)
        return None

    # read header-only to inspect channels & sfreq
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    orig_chs = raw.ch_names.copy()

    # normalize channel names: keep mapping to original names
    rename_map = {}
    for ch in raw.ch_names:
        new = ch  # keep original casing for saved names, but we will use normalized keys for matching montage/exclusion
        rename_map[ch] = new
    # (we won't rename to uppercase globally to avoid changing users' expectations)

    # find ECG channel and EEG channels to keep
    eeg_chs = []
    ecg_ch = None
    for ch in raw.ch_names:
        ch_type = raw.get_channel_types(picks=[ch])[0]
        norm = normalize_channel_name(ch)
        if ch_type == "ecg" or 'ecg' in ch.lower():
            ecg_ch = ch
            continue
        if ch_type == "eeg" and norm not in EXCLUDED_SET:
            eeg_chs.append(ch)

    if ecg_ch is None:
        print("No ECG found — skipping")
        return None
    if len(eeg_chs) == 0:
        print("No EEG channels after exclusions — skipping")
        return None

    print(f"Kept EEG channels: {len(eeg_chs)}; ECG: {ecg_ch}")

    sfreq_native = raw.info['sfreq']
    sfreq_target = TARGET_SFREQ if (TARGET_SFREQ is not None and TARGET_SFREQ < sfreq_native) else sfreq_native
    decim_factor = int(round(sfreq_native / sfreq_target)) if sfreq_target != sfreq_native else 1
    if decim_factor < 1: decim_factor = 1
    if sfreq_native % decim_factor != 0:
        # fine — decimate will still work, but warn
        pass

    print(f"Native sfreq={sfreq_native:.1f} Hz; target sfreq={sfreq_target:.1f} Hz; decimation factor={decim_factor}")

    # compute epoch windows in native samples (we will map to decimated indices per block)
    smin_native = int(np.round(TMIN * sfreq_native))
    smax_native = int(np.round(TMAX * sfreq_native))
    wlen_native = smax_native - smin_native + 1

    # read ECG fully to detect R-peaks (ECG is small relative to full avail, but if huge it's ok)
    # We'll read ECG in header mode and then get_data for picks
    raw_ecg = raw.copy().pick_channels([ecg_ch])
    ecg_data = raw_ecg.get_data()[0]
    _, info_nk = nk.ecg_process(ecg_data, sampling_rate=sfreq_native)
    r_peaks = np.array(info_nk.get("ECG_R_Peaks", []), dtype=int)
    print("Detected R-peaks:", len(r_peaks))
    if len(r_peaks) < 5:
        print("Too few R-peaks -> skip")
        return None

    # valid epoch starts/stops (native)
    starts_native = r_peaks + smin_native
    stops_native = r_peaks + smax_native
    # keep only epochs fully inside recording
    valid_mask = (starts_native >= 0) & (stops_native < raw.n_times)
    starts_native = starts_native[valid_mask]
    stops_native = stops_native[valid_mask]
    if len(starts_native) == 0:
        print("No valid epochs within bounds -> skip")
        return None

    # Build merged intervals (native sample coords). Merge if next start <= current_end + margin_samples
    margin_samples = int(round((MERGE_MARGIN_MS / 1000.0) * sfreq_native))
    intervals = []
    cur_s = int(starts_native[0])
    cur_e = int(stops_native[0])
    for s, e in zip(starts_native[1:], stops_native[1:]):
        s = int(s); e = int(e)
        if s <= cur_e + margin_samples:
            cur_e = max(cur_e, e)
        else:
            intervals.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    intervals.append((cur_s, cur_e))

    print(f"Merged intervals to read: {len(intervals)} (margin {MERGE_MARGIN_MS} ms)")

    # precompute where each epoch belongs (index per interval)
    epoch_indices_per_interval = []
    starts_list = starts_native.tolist()
    stops_list = stops_native.tolist()
    for (blk_s, blk_e) in intervals:
        members = []
        # find indices of epochs where start>=blk_s and stop<=blk_e
        # We'll scan through starts_list for speed
        for i, (S, E) in enumerate(zip(starts_list, stops_list)):
            if S >= blk_s and E <= blk_e:
                # relative positions to block start (native samples)
                rel_s = int(S - blk_s)
                rel_e = int(E - blk_s)
                members.append((i, rel_s, rel_e))
        epoch_indices_per_interval.append(members)

    # Prepare filters (we will apply on decimated (target) data)
    (b_hp, a_hp), (b_lp, a_lp) = make_filter_coeffs(sfreq_target, HP, LP) if sfreq_target>0 else (None,None),(None,None)

    # prepare aggregator (running sum & counts) for decimated wlen
    wlen_dec = int(round((wlen_native) / decim_factor)) if decim_factor > 1 else wlen_native
    # If decimation doesn't divide exact, we'll compute times array accordingly
    hep_sum = np.zeros((len(eeg_chs), wlen_dec), dtype=np.float64)
    hep_count = np.zeros((len(eeg_chs),), dtype=int)

    # compute safe block sample limit to avoid huge memory reads
    max_block_samples = safe_block_limit_samples(MAX_RAM_GB, len(eeg_chs)+1)  # +1 for ECG if picked
    # For safety, limit to at least one epoch length
    min_required = wlen_native
    if max_block_samples < min_required:
        print(f"Warning: computed max samples per block ({max_block_samples}) smaller than epoch length ({min_required}). Reducing MAX_RAM_GB or choose lower TARGET_SFREQ.")
        # Force at least one epoch per block
        max_block_samples = min_required

    # iterate intervals with progress bar; for each, read block (native samples), optionally decimate,
    # slice all epochs in that block for all channels in vectorized way, do reject, baseline, and running aggregate
    n_intervals = len(intervals)
    pbar_intervals = tqdm(range(n_intervals), desc=f"{name} intervals", unit="blk", ncols=100)
    # We'll pick all EEG channels once and read them together to reuse block for all channels
    picks = eeg_chs.copy()  # names
    for ib in pbar_intervals:
        blk_s, blk_e = intervals[ib]
        blk_len_native = blk_e - blk_s + 1
        # if block would be too large (native samples) split it into smaller sub-blocks
        if blk_len_native > max_block_samples:
            # split into subblocks
            n_sub = int(np.ceil(blk_len_native / float(max_block_samples)))
            sub_starts = [blk_s + int(i * max_block_samples) for i in range(n_sub)]
            sub_ends = [min(blk_s + int((i+1) * max_block_samples) - 1, blk_e) for i in range(n_sub)]
            sub_intervals = list(zip(sub_starts, sub_ends))
        else:
            sub_intervals = [(blk_s, blk_e)]

        # For each sub-block, read all EEG channels (and optionally ECG if needed)
        for (ss, se) in sub_intervals:
            # read data for EEG channels in native samples
            # NOTE: raw.get_data(start, stop) uses start inclusive, stop exclusive -> use se+1
            block_native = raw.get_data(picks=picks, start=ss, stop=se+1)  # shape (n_ch, n_samples_native)
            # decimate if needed to target sfreq (per-channel)
            if decim_factor > 1:
                # decimate each channel axis (axis=1) using scipy.signal.decimate (IIR filter). Use zero-phase by applying decimate twice? decimate uses filtfilt when ftype='fir' only in recent versions.
                # We'll use decimate with ftype='iir' (default) per channel (it's faster) — acceptable compromise.
                n_ch, n_samp = block_native.shape
                new_len = int(np.ceil(n_samp / decim_factor))
                block = np.zeros((n_ch, new_len), dtype=np.float64)
                for ci in range(n_ch):
                    # use zero-phase-like behavior by decimating via decimate (which includes anti-aliasing)
                    try:
                        block[ci, :] = decimate(block_native[ci, :], decim_factor, zero_phase=True)
                    except TypeError:
                        # fallback if zero_phase arg not available
                        block[ci, :] = decimate(block_native[ci, :], decim_factor)
                sfreq_block = sfreq_native / decim_factor
            else:
                block = block_native.copy()
                sfreq_block = sfreq_native

            # filter block (apply low+high via filtfilt)
            try:
                block_filt = filtfilt(b_lp, a_lp, block, axis=1)
                block_filt = filtfilt(b_hp, a_hp, block_filt, axis=1)
            except Exception:
                # if filtfilt fails (e.g., too short), skip filtering
                block_filt = block

            # Now find the epochs that fall inside this sub-block (we have precomputed epoch_positions per interval)
            # Need the mapping from epoch relative positions in native samples to decimated indices for this sub-block
            # First compute block native start index (ss) and its decimated index base
            decimated_base = int(round((ss - blk_s) / decim_factor)) if decim_factor>1 else (ss - blk_s)

            # For each epoch belonging to the parent interval ib, check if it falls in this sub-block
            members = epoch_indices_per_interval[ib]
            for (epoch_idx, rel_s_native, rel_e_native) in members:
                # epoch's absolute native start = blk_s + rel_s_native
                abs_start_native = blk_s + rel_s_native
                abs_end_native = blk_s + rel_e_native
                # check if this epoch is inside current sub-block [ss, se]
                if not (abs_start_native >= ss and abs_end_native <= se):
                    continue
                # compute decimated indices inside block_filt
                if decim_factor > 1:
                    rel_start_dec = int(round((abs_start_native - ss) / decim_factor))
                    rel_end_dec = rel_start_dec + wlen_dec - 1
                else:
                    rel_start_dec = abs_start_native - ss
                    rel_end_dec = rel_start_dec + wlen_dec - 1

                # safety bounds
                if rel_start_dec < 0 or rel_end_dec >= block_filt.shape[1]:
                    continue

                # extract epoch for all channels at once (n_ch, wlen_dec)
                epoch_all = block_filt[:, rel_start_dec:rel_end_dec+1]  # shape (n_channels, wlen_dec)

                # artifact rejection per-channel (vectorized)
                rej_mask = reject_mad_array(epoch_all, factor=5.0)  # shape (n_channels,) True if reject

                # baseline correction: baseline_samples computed at decimated rate
                baseline_samples_dec = int(round(abs(TMIN) * sfreq_block))
                if baseline_samples_dec <= 0:
                    baseline_vals = np.zeros(epoch_all.shape[0], dtype=np.float64)
                else:
                    baseline_vals = epoch_all[:, :baseline_samples_dec].mean(axis=1)

                epoch_all = (epoch_all.T - baseline_vals).T  # subtract baseline per channel

                # add to running sums where not rejected
                keep_idx = np.where(~rej_mask)[0]
                if keep_idx.size > 0:
                    hep_sum[keep_idx, :] += epoch_all[keep_idx, :]
                    hep_count[keep_idx] += 1

    # finalize — compute mean for channels with at least 1 epoch
    nonzero = hep_count > 0
    hep_mat = np.zeros_like(hep_sum)
    hep_mat[nonzero, :] = (hep_sum[nonzero, :] / hep_count[nonzero, None])

    # times vector (decimated)
    # note: compute times for decimated vector centered on TMIN..TMAX
    n_times = hep_mat.shape[1]
    times = np.linspace(TMIN, TMAX, n_times)

    # Save
    out_file = os.path.join(OUT_DIR, f"{name}_HEP.npz")
    np.savez(
        out_file,
        hep=hep_mat,
        channels=np.array(eeg_chs, dtype=object),
        times=times,
        subject=name,
        group=group
    )
    print("Saved:", out_file, "HEP shape:", hep_mat.shape, "counts (example first 10):", hep_count[:10].tolist())
    return {"name": name, "group": group, "file": out_file, "n_channels": len(eeg_chs)}

def make_filter_coeffs(sfreq, hp, lp):
    from scipy.signal import butter
    b_hp, a_hp = butter(2, hp / (sfreq/2), btype='highpass')
    b_lp, a_lp = butter(4, lp / (sfreq/2), btype='lowpass')
    return (b_hp, a_hp), (b_lp, a_lp)

# -------------------- RUN --------------------
def main():
    manifest = {"subjects": []}
    for subj in SUBJECTS:
        res = process_subject_opt(subj)
        if res:
            manifest["subjects"].append(res)
    with open(os.path.join(OUT_DIR, "manifest_hep_opt.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("DONE. Manifest saved to", os.path.join(OUT_DIR, "manifest_hep_opt.json"))

if __name__ == "__main__":
    main()
