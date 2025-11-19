"""
stage2_stack_hep.py

Load per-subject HEP NPZ outputs (manifest_hep.json), harmonize channels/times,
attempt to map channels to standard montages for ch_pos, and save stacked arrays:
  out_stage2/group_stacked_data.npz

Outputs saved keys:
 - all_data: shape (n_subjects, n_channels, n_times)
 - ch_names: (n_channels,) canonical channel names chosen (strings)
 - channels: same as ch_names (for backward compat)
 - times: (n_times,) template times vector
 - subjects: (n_subjects,) subject names
 - groups: (n_subjects,) group labels
 - ch_pos: saved dict mapping ch_name -> (x,y,z) when available (may be empty)
"""

import os
import json
import numpy as np
import mne
from collections import OrderedDict

# --------- CONFIG ----------
MANIFEST_STAGE1 = "out_stage1_hep_opt/manifest_hep_opt.json"   # produced by your Stage-1 code
OUT_DIR = "out_stage2_final"
OUT_PATH = os.path.join(OUT_DIR, "group_stacked_data.npz")
os.makedirs(OUT_DIR, exist_ok=True)

# Montages to try (order matters: first try standard_1020 then fallback)
MONTAGES_TO_TRY = ("standard_1020", "standard_1005")

# Normalization function (reuse same logic you used earlier)
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

# --------------------------

def load_manifest(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Stage1 manifest not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

def read_subject_npz(npz_path):
    """Read a subject-level HEP npz. Return dict with keys: hep, channels, times, subject, group"""
    d = np.load(npz_path, allow_pickle=True)
    # expected keys: hep, channels, times, subject, group
    required = ["hep", "channels", "times"]
    for k in required:
        if k not in d:
            raise RuntimeError(f"NPZ {npz_path} missing required key: {k}. Keys: {list(d.keys())}")
    out = {
        "hep": d["hep"],
        "channels": [str(x) for x in np.atleast_1d(d["channels"]).tolist()],
        "times": np.array(d["times"], dtype=float),
        "subject": str(d["subject"]) if "subject" in d else None,
        "group": str(d["group"]) if "group" in d else None,
        "path": npz_path
    }
    d.close()
    return out

def try_get_ch_pos_from_montages(ch_names, montages_to_try=MONTAGES_TO_TRY):
    """Try to build ch_pos mapping (channel->pos) using standard montages.
       Returns ch_pos dict and tried list [(montage_name, matched_count), ...]
    """
    ch_pos = {}
    tried = []
    norm_map_cache = {}
    # We'll attempt exact matches then normalized matches.
    for mname in montages_to_try:
        try:
            mont = mne.channels.make_standard_montage(mname)
            mpos = mont.get_positions().get("ch_pos", {})
            # exact matches
            matched = 0
            for ch in ch_names:
                if ch in mpos:
                    ch_pos[ch] = np.asarray(mpos[ch], dtype=float)
                    matched += 1
            # normalized matches for remaining
            if matched < len(ch_names):
                # build normalized map of montage keys once
                mm = {normalize_channel_name(k): k for k in mpos.keys()}
                for ch in ch_names:
                    if ch in ch_pos:
                        continue
                    key = normalize_channel_name(ch)
                    if key in mm:
                        mch = mm[key]
                        ch_pos[ch] = np.asarray(mpos[mch], dtype=float)
                        matched += 1
            tried.append((mname, matched))
        except Exception:
            tried.append((mname, 0))
    return ch_pos, tried

def harmonize_and_stack(subject_entries, resample_to_times=None):
    """
    subject_entries: list of dicts returned by read_subject_npz
    resample_to_times: None or target times array (1D). If None, uses times from first subject as template.
    Returns stacked array all_data (S, C, T), ch_names (C,), times (T,), subjects, groups
    """
    n_subj = len(subject_entries)
    if n_subj == 0:
        raise RuntimeError("No subjects provided")

    # collect channel sets and times
    channel_sets = [set(e["channels"]) for e in subject_entries]
    # compute intersection of channel names using normalized names, but we want canonical names from first subject
    # We'll pick canonical ordering from the subject with the largest channel set (usually first)
    # Approach: determine common normalized names, then pick canonical ch name for each normalized entry from the majority subject
    # Build normalized to canonical mapping by preferring exact appearances
    norm_to_candidates = {}
    for e in subject_entries:
        for ch in e["channels"]:
            key = normalize_channel_name(ch)
            norm_to_candidates.setdefault(key, []).append(ch)

    # Identify normalized names present in all subjects
    norm_sets_per_subject = [set(normalize_channel_name(ch) for ch in e["channels"]) for e in subject_entries]
    common_norms = set.intersection(*norm_sets_per_subject)

    if len(common_norms) == 0:
        raise RuntimeError("No common channels across subjects after normalization. Check channel naming / exclusions.")

    # Choose canonical name for each normalized name:
    # strategy: pick the most common original candidate (mode) across subjects
    canonical_map = {}
    for norm in sorted(common_norms):
        cands = norm_to_candidates.get(norm, [])
        # pick the most frequent candidate string
        if len(cands) == 0:
            continue
        uniq, counts = np.unique(cands, return_counts=True)
        canonical = uniq[np.argmax(counts)]
        canonical_map[norm] = canonical

    # Now canonical ch list (sorted by canonical_map keys to be deterministic)
    canonical_norms_sorted = sorted(canonical_map.keys())
    ch_names = [canonical_map[n] for n in canonical_norms_sorted]

    # Build per-subject reordering to canonical ch_names
    reorder_indices = []
    for e in subject_entries:
        # map normalized->index for that subject
        subj_norm_to_idx = {normalize_channel_name(ch): i for i, ch in enumerate(e["channels"])}
        idxs = [subj_norm_to_idx[n] for n in canonical_norms_sorted]
        reorder_indices.append(idxs)

    # harmonize times: choose template
    if resample_to_times is None:
        template_times = subject_entries[0]["times"]
    else:
        template_times = np.array(resample_to_times, dtype=float)
    T = template_times.size

    # For each subject, either assert times equal or resample (linear interp) to template
    aligned_data = []
    for i, e in enumerate(subject_entries):
        subj_hep = np.array(e["hep"])
        subj_times = np.array(e["times"], dtype=float)
        # reorder channels
        subj_hep_reordered = subj_hep[reorder_indices[i], :]
        if subj_times.shape[0] == T and np.allclose(subj_times, template_times):
            aligned = subj_hep_reordered
        else:
            # interpolate per-channel to template_times
            aligned = np.zeros((subj_hep_reordered.shape[0], T), dtype=float)
            for ci in range(subj_hep_reordered.shape[0]):
                aligned[ci, :] = np.interp(template_times, subj_times, subj_hep_reordered[ci, :])
        aligned_data.append(aligned)

    all_data = np.stack(aligned_data, axis=0)  # (subjects, channels, times)
    subjects = [e["subject"] if e["subject"] is not None else os.path.splitext(os.path.basename(e["path"]))[0] for e in subject_entries]
    groups = [e["group"] if e["group"] is not None else "unknown" for e in subject_entries]

    return all_data, ch_names, template_times, subjects, groups

def build_ch_pos_saveable(ch_names):
    """Try to obtain 3D positions for ch_names using montages; return dict ch->pos and diagnostics"""
    ch_pos, tried = try_get_ch_pos_from_montages(ch_names, montages_to_try=MONTAGES_TO_TRY)
    return ch_pos, tried

# ------------------ main ------------------
def main():
    print("Loading Stage-1 manifest:", MANIFEST_STAGE1)
    manifest = load_manifest(MANIFEST_STAGE1)
    subj_records = manifest.get("subjects", [])
    if len(subj_records) == 0:
        raise RuntimeError("No subjects in Stage-1 manifest.")

    # find each subject NPZ path (manifest entries are expected to include 'file' with path in Stage-1)
    subject_npzs = []
    for rec in subj_records:
        # Stage-1 manifest entries typically had 'file' or 'file' key; try several fallbacks
        fpath = rec.get("file") or rec.get("file_path") or rec.get("evoked_file") or rec.get("epochs_file") or rec.get("out_file")
        # if manifest entries used a different key, also try a constructed filename based on subject name
        if not fpath:
            # try default expected out file naming
            candidate = os.path.join("out_stage1_hep", f"{rec.get('name')}_HEP.npz")
            if os.path.exists(candidate):
                fpath = candidate
        if not fpath:
            raise RuntimeError(f"Could not find NPZ file location for subject record: {rec}")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Subject NPZ not found: {fpath}")
        subject_npzs.append(fpath)

    # read all subject NPZs
    subject_entries = []
    print("Reading subject NPZs (count={}):".format(len(subject_npzs)))
    for p in subject_npzs:
        print(" -", p)
        subject_entries.append(read_subject_npz(p))

    # Harmonize and stack
    all_data, ch_names, times, subjects, groups = harmonize_and_stack(subject_entries, resample_to_times=None)

    n_subj, n_ch, n_t = all_data.shape
    print(f"Stacked data shape: {all_data.shape} (subjects, channels, times)")

    # try to find channel positions via montages
    ch_pos, tried = build_ch_pos_saveable(ch_names)
    print("Montage matching results:", tried)
    print(f"Total channel positions obtained: {len(ch_pos)} / {len(ch_names)}")
    if len(ch_pos) < len(ch_names):
        print("Warning: some channels missing positions for montage. Stage-3 will need to handle missing positions (fall back to spatial identity).")

    # Save NPZ with keys expected by Stage-3
    np.savez(
        OUT_PATH,
        all_data=all_data,
        ch_names=np.array(ch_names, dtype=object),
        channels=np.array(ch_names, dtype=object),
        times=times,
        subjects=np.array(subjects, dtype=object),
        groups=np.array(groups, dtype=object),
        ch_pos=ch_pos  # dict saved into NPZ (pickle-like)
    )
    print("Saved stacked arrays and metadata to:", OUT_PATH)

if __name__ == "__main__":
    main()
