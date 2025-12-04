
"""
Stage-2: Group-level HEP data stacking and channel position mapping

Load per-subject HEP NPZ outputs (manifest_hep.json), harmonize channels/times,
attempt to map channels to standard montages for ch_pos, and save stacked arrays:
  out_stage2_final/group_stacked_data.npz

Outputs saved keys:
 - all_data: shape (n_subjects, n_channels, n_times)
 - ch_names: (n_channels,) canonical channel names chosen (strings)
 - channels: same as ch_names (for backward compat)
 - times: (n_times,) template times vector
 - subjects: (n_subjects,) subject names
 - groups: (n_subjects,) group labels
 - ch_pos: saved dict mapping ch_name -> (x,y,z) when available 
"""
import os
import json
import numpy as np
import mne
import pprint
import ppprint

# --------- CONFIG ----------
MANIFEST_STAGE1 = "out_stage1_hep_opt/manifest_hep_opt.json"   # produced by your Stage-1 code (includes paths and grouup information)
OUT_DIR = "out_stage2_final"
OUT_PATH = os.path.join(OUT_DIR, "group_stacked_data.npz")
os.makedirs(OUT_DIR, exist_ok=True)

# Montages to try (order matters: first try standard_1020 then fallback)
MONTAGES_TO_TRY = ("standard_1020", "standard_1005")

# Normalization function 
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
    # normalize channels to strings list
    channels_arr = d["channels"]
    # handle weird object arrays
    if isinstance(channels_arr, np.ndarray) and channels_arr.dtype == object and channels_arr.size == 1:
        # maybe inner is a list-like
        inner = channels_arr[0]
        channels = [str(x) for x in np.atleast_1d(inner)]
    else:
        channels = [str(x) for x in np.atleast_1d(channels_arr).tolist()]
    out = {
        "hep": np.array(d["hep"], dtype=float),
        "channels": channels,
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
    for mname in montages_to_try:
        try:
            mont = mne.channels.make_standard_montage(mname)
            mpos = mont.get_positions().get("ch_pos", {})
            matched = 0
            # exact matches
            for ch in ch_names:
                if ch in mpos:
                    ch_pos[ch] = np.asarray(mpos[ch], dtype=float)
                    matched += 1
            # normalized matches for remaining
            if matched < len(ch_names):
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
    # pprint.pprint(ch_pos)
    return ch_pos, tried

def build_ch_pos_fill(ch_names, ch_pos_partial, k_neighbors=4):
    """
    Fill missing channel positions by assigning nearest valid channel coordinates.
    Returns a dict ch->(x,y,z) for all ch_names.
    Also returns adjacency matrix (n_ch,n_ch) built via kNN on filled coords.
    """
    n = len(ch_names)
    coords = np.zeros((n, 3), dtype=float)
    valid = np.zeros(n, dtype=bool)
    for i, ch in enumerate(ch_names):
        if ch in ch_pos_partial:
            coords[i] = np.asarray(ch_pos_partial[ch], dtype=float)
            valid[i] = True
    if valid.sum() == 0:
        # If nothing present, return empty dict and identity adjacency
        coords_dict = {ch: [0.0, 0.0, 0.0] for ch in ch_names}
        adj = np.eye(n, dtype=int)
        return coords_dict, adj
    # fill missing coords by nearest valid coordinate (Euclidean)
    valid_idx = np.where(valid)[0]
    for i in range(n):
        if not valid[i]:
            # compute distances to valid coords; if coords[i] are zero, compute using zeros but compare to valid coords
            d = np.linalg.norm(coords[valid_idx] - coords[i], axis=1)
            best = valid_idx[np.argmin(d)]
            coords[i] = coords[best]
            valid[i] = True
    # build pairwise distances
    D = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    adj = np.zeros((n, n), dtype=int)
    k = min(k_neighbors, n - 1)
    for i in range(n):
        idx = np.argsort(D[i])[:k]
        adj[i, idx] = 1
    adj = ((adj + adj.T) > 0).astype(int)
    np.fill_diagonal(adj, 0)
    coords_dict = {ch_names[i]: coords[i].tolist() for i in range(n)}
    return coords_dict, adj

def harmonize_and_stack(subject_entries, resample_to_times=None):
    """
    subject_entries: list of dicts returned by read_subject_npz
    resample_to_times: None or target times array (1D). If None, uses times from first subject as template.
    Returns stacked array all_data (S, C, T), ch_names (C,), times (T,), subjects, groups
    """
    n_subj = len(subject_entries)
    if n_subj == 0:
        raise RuntimeError("No subjects provided")

    # Collect normalized channel sets for intersection
    norm_sets = [set(normalize_channel_name(ch) for ch in e["channels"]) for e in subject_entries]
    common_norms = set.intersection(*norm_sets)
    if len(common_norms) == 0:
        raise RuntimeError("No common channels across subjects after normalization. Check channel naming / exclusions.")

    # Build mapping of normalized -> candidate names, choose most frequent canonical
    norm_to_candidates = {}
    for e in subject_entries:
        for ch in e["channels"]:
            key = normalize_channel_name(ch)
            norm_to_candidates.setdefault(key, []).append(ch)
    canonical_map = {}
    for norm in sorted(common_norms):
        cands = norm_to_candidates.get(norm, [])
        if len(cands) == 0:
            continue
        uniq, counts = np.unique(cands, return_counts=True)
        canonical = uniq[np.argmax(counts)]
        canonical_map[norm] = canonical

    canonical_norms_sorted = sorted(canonical_map.keys())
    ch_names = [canonical_map[n] for n in canonical_norms_sorted]

    # Build per-subject reorder indices
    reorder_indices = []
    for e in subject_entries:
        subj_map = {normalize_channel_name(ch): idx for idx, ch in enumerate(e["channels"])}
        idxs = [subj_map[n] for n in canonical_norms_sorted]
        reorder_indices.append(idxs)

    # times template
    if resample_to_times is None:
        template_times = subject_entries[0]["times"]
    else:
        template_times = np.array(resample_to_times, dtype=float)
    T = template_times.size

    aligned_data = []
    for i, e in enumerate(subject_entries):
        subj_hep = np.array(e["hep"], dtype=float)
        subj_times = np.array(e["times"], dtype=float)
        subj_hep_reordered = subj_hep[reorder_indices[i], :]
        if subj_times.shape[0] == T and np.allclose(subj_times, template_times):
            aligned = subj_hep_reordered
        else:
            aligned = np.zeros((subj_hep_reordered.shape[0], T), dtype=float)
            for ci in range(subj_hep_reordered.shape[0]):
                aligned[ci, :] = np.interp(template_times, subj_times, subj_hep_reordered[ci, :])
        aligned_data.append(aligned)

    all_data = np.stack(aligned_data, axis=0)  # (subjects, channels, times)
    subjects = [e["subject"] if e["subject"] is not None else os.path.splitext(os.path.basename(e["path"]))[0] for e in subject_entries]
    groups = [e["group"] if e["group"] is not None else "unknown" for e in subject_entries]

    return all_data, ch_names, template_times, subjects, groups

# ------------------ main ------------------
def main():
    print("Loading Stage-1 manifest:", MANIFEST_STAGE1)
    manifest = load_manifest(MANIFEST_STAGE1)
    subj_records = manifest.get("subjects", [])
    if len(subj_records) == 0:
        raise RuntimeError("No subjects in Stage-1 manifest.")

    # find each subject NPZ path (manifest entries expected to include 'file' or 'out_file' or similar)
    subject_npzs = []
    for rec in subj_records:
        fpath = rec.get("file") or rec.get("file_path") or rec.get("out_file") or rec.get("path") or rec.get("file_path")
        if not fpath:
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
    ch_pos_partial, tried = try_get_ch_pos_from_montages(ch_names, montages_to_try=MONTAGES_TO_TRY)
    print("Montage matching results:", tried)
    print(f"Total channel positions obtained from montages: {len(ch_pos_partial)} / {len(ch_names)}")

    # Fill missing positions using nearest-neighbor fill
    ch_pos_filled, ch_adj = build_ch_pos_fill(ch_names, ch_pos_partial, k_neighbors=4)
    print(f"Filled channel positions: {len(ch_pos_filled)} / {len(ch_names)}")

    # Save NPZ with keys expected by Stage-3
    print("Saving Stage-2 output to:", OUT_PATH)
    np.savez(
        OUT_PATH,
        all_data=all_data,
        ch_names=np.array(ch_names, dtype=object),
        channels=np.array(ch_names, dtype=object),
        times=times,
        subjects=np.array(subjects, dtype=object),
        groups=np.array(groups, dtype=object),
        ch_pos=ch_pos_filled
    )
    pprint.pprint(ch_pos_filled)
    print("Saved stacked NPZ with full 3D channel positions.")
    print("Saved stacked arrays and metadata to:", OUT_PATH)

if __name__ == "__main__":
    main()
