
"""
Diagnose a Stage-2 stacked NPZ created for Stage-3.

Usage:
    python diagnose_stacked.py /path/to/group_stacked_data.npz

Prints diagnostics and writes a small JSON report next to the NPZ.
"""
import os
import sys
import json
import math
import numpy as np
from collections import Counter

def unpack_array_like(x):
    """Return a list of strings from ndarray/list/object arrays."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, np.ndarray):
        if x.dtype == object and x.size == 1:
            inner = x[0]
            if isinstance(inner, (list, tuple)):
                return list(inner)
            try:
                return [str(u) for u in inner.tolist()]
            except Exception:
                return [str(inner)]
        try:
            return [str(u) for u in x.tolist()]
        except Exception:
            return [str(u) for u in x.flatten().tolist()]
    return [str(x)]

def normalize_name(s):
    return ''.join(ch for ch in str(s).lower() if ch.isalnum())

def try_montage_match(ch_names, montage_name="standard_1020"):
    try:
        import mne
    except Exception:
        return None, "mne_not_available"
    mont = mne.channels.make_standard_montage(montage_name)
    mpos = mont.get_positions().get("ch_pos", {})
    matched = 0
    norm_map = {normalize_name(k): k for k in mpos.keys()}
    matched_list = []
    for ch in ch_names:
        if ch in mpos:
            matched += 1
            matched_list.append(ch)
        else:
            k = normalize_name(ch)
            if k in norm_map:
                matched += 1
                matched_list.append(ch)
    return matched, len(ch_names)

def build_knn_adj_from_positions(ch_names, ch_pos_map, k=4):
    """Return adjacency matrix and degrees or None if not possible."""
    n = len(ch_names)
    coords = np.zeros((n,3))
    valid = np.zeros(n, bool)
    for i,ch in enumerate(ch_names):
        if ch in ch_pos_map:
            coords[i] = np.asarray(ch_pos_map[ch], dtype=float)
            valid[i] = True
    if valid.sum() == 0:
        return None, None
    # fill missing coords with nearest valid coordinate
    valid_idx = np.where(valid)[0]
    for i in range(n):
        if not valid[i]:
            d = np.linalg.norm(coords[valid_idx] - coords[i], axis=1)
            coords[i] = coords[valid_idx[np.argmin(d)]]
            valid[i] = True
    D = np.sqrt(((coords[:,None,:] - coords[None,:,:])**2).sum(-1))
    np.fill_diagonal(D, np.inf)
    adj = np.zeros((n,n), dtype=int)
    kk = min(k, n-1)
    for i in range(n):
        idx = np.argsort(D[i])[:kk]
        adj[i, idx] = 1
    adj = ((adj + adj.T) > 0).astype(int)
    np.fill_diagonal(adj, 0)
    deg = adj.sum(axis=0)
    return adj, deg

def compute_tmap_summary(all_data, groups):
    """If exactly 2 groups, compute observed t-map (channels x times) summary."""
    uniq = sorted(set(groups.tolist()))
    if len(uniq) != 2:
        return None
    from scipy import stats
    gA, gB = uniq
    idxA = np.where(groups == gA)[0]
    idxB = np.where(groups == gB)[0]
    XA = all_data[idxA]
    XB = all_data[idxB]
    n1, n2 = XA.shape[0], XB.shape[0]
    m1 = XA.mean(axis=0)
    m2 = XB.mean(axis=0)
    v1 = XA.var(axis=0, ddof=1)
    v2 = XB.var(axis=0, ddof=1)
    df = n1 + n2 - 2
    sp2 = ((n1-1)*v1 + (n2-1)*v2) / max(df, 1)
    denom = np.sqrt(sp2 * (1.0/n1 + 1.0/n2))
    denom[denom == 0] = np.nan
    tmap = (m1 - m2) / denom
    tcrit = float(stats.t.ppf(1 - 0.05/2.0, df)) if df > 0 else None
    suprath = np.sum(np.abs(tmap) > (tcrit if tcrit is not None else np.inf))
    return dict(df=int(df), tcrit=tcrit, suprathreshold_nodes=int(suprath), total_nodes=int(tmap.size),
                shape=tmap.shape, tmap_min=float(np.nanmin(tmap)), tmap_max=float(np.nanmax(tmap)))

def main(npz_path):
    if not os.path.exists(npz_path):
        print("NPZ not found:", npz_path); return 1
    print("Loading:", npz_path)
    npz = np.load(npz_path, allow_pickle=True)
    print(" Keys in NPZ:", list(npz.keys()))
    report = {}
    # load arrays
    all_data = npz.get("all_data", None)
    raw_ch = npz.get("ch_names", npz.get("channels", None))
    times = npz.get("times", None)
    subjects = npz.get("subjects", None)
    groups = npz.get("groups", None)
    ch_pos_raw = npz.get("ch_pos", None)

    ch_names = unpack_array_like(raw_ch) if raw_ch is not None else None
    report['ch_names_len'] = len(ch_names) if ch_names is not None else None

    print("\nBasic summary")
    if all_data is not None:
        print(" all_data.shape:", all_data.shape)
        report['all_data_shape'] = list(all_data.shape)
        print(" dtype:", all_data.dtype, " NaNs:", int(np.isnan(all_data).sum()))
        report['nan_count'] = int(np.isnan(all_data).sum())
    else:
        print(" all_data not present!")

    if times is not None:
        times = np.asarray(times, dtype=float)
        print(" times len:", times.shape[0], "first..last:", float(times[0]), float(times[-1]))
        report['n_times'] = int(times.shape[0])
    if subjects is not None:
        subjects = np.asarray(subjects).astype(str)
        print(" subjects count:", subjects.shape[0], "examples:", subjects.tolist()[:6])
        report['n_subjects'] = int(subjects.shape[0])
    if groups is not None:
        groups = np.asarray(groups).astype(str)
        print(" groups:", sorted(set(groups.tolist())), "counts:", dict(Counter(groups.tolist())))
        report['groups'] = dict(Counter(groups.tolist()))

    if ch_names is not None:
        print(" ch_names count:", len(ch_names), "example first 20:", ch_names[:20])

    # mismatch check
    if all_data is not None and ch_names is not None:
        S,C,T = all_data.shape
        if C != len(ch_names):
            print(" WARNING: channel-name count does not match data channels:", len(ch_names), "vs", C)
            report['ch_mismatch'] = dict(ch_names=len(ch_names), data_ch=C)

    # attempt montage match
    try:
        match1020 = try_montage_match(ch_names, "standard_1020")
        match1005 = try_montage_match(ch_names, "standard_1005")
        report['montage_matches'] = dict(standard_1020=match1020, standard_1005=match1005)
        print("\nMontage matching (approx):", report['montage_matches'])
    except Exception as E:
        print(" Montage matching failed:", str(E))

    # ch_pos presence
    ch_pos = {}
    if ch_pos_raw is not None:
        # safe-unpack if saved as object
        try:
            if isinstance(ch_pos_raw, np.ndarray) and ch_pos_raw.dtype == object and ch_pos_raw.size == 1:
                entry = ch_pos_raw[0]
                if isinstance(entry, dict):
                    ch_pos = {str(k): np.asarray(v, dtype=float) for k,v in entry.items()}
                else:
                    ch_pos = {}
            elif isinstance(ch_pos_raw, dict):
                ch_pos = {str(k): np.asarray(v, dtype=float) for k,v in ch_pos_raw.items()}
            else:
                ch_pos = dict(ch_pos_raw)
        except Exception:
            ch_pos = {}
    print(" ch_pos entries:", len(ch_pos))
    report['ch_pos_count'] = len(ch_pos)

    # try build adjacency if positions exist
    if ch_names is not None:
        ch_pos_map = ch_pos.copy()
        # if no positions but mne available, try to fetch montage positions and match normalized names
        if len(ch_pos_map) == 0:
            try:
                import mne
                mont = mne.channels.make_standard_montage("standard_1020")
                mpos = mont.get_positions().get("ch_pos", {})
                norm_map = {normalize_name(k):k for k in mpos.keys()}
                for ch in ch_names:
                    k = normalize_name(ch)
                    if ch in mpos:
                        ch_pos_map[ch] = np.asarray(mpos[ch], dtype=float)
                    elif k in norm_map:
                        ch_pos_map[ch] = np.asarray(mpos[norm_map[k]], dtype=float)
                print(" built ch_pos_map from 1020 montage:", len(ch_pos_map))
            except Exception:
                pass
        if len(ch_pos_map) > 0:
            adj, deg = build_knn_adj_from_positions(ch_names, ch_pos_map, k=4)
            if adj is not None:
                print(" Adjacency degrees (min, mean, max):", int(deg.min()), float(deg.mean()), int(deg.max()))
                report['adj_deg_min'] = int(deg.min()); report['adj_deg_mean']=float(deg.mean()); report['adj_deg_max']=int(deg.max())
            else:
                print(" Could not build adjacency from positions.")
        else:
            print(" No usable channel positions to build spatial adjacency.")

    # compute tmap summary if two groups
    if all_data is not None and groups is not None:
        t_summary = compute_tmap_summary(all_data, groups)
        if t_summary is not None:
            print("\nT-map summary (two-group):", t_summary)
            report['tmap_summary'] = t_summary
        else:
            print("\nNot exactly two groups - skipping tmap summary.")

    # Save report
    out_json = os.path.splitext(npz_path)[0] + "_diagnose_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved diagnose report ->", out_json)
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_stacked.py /path/to/group_stacked_data.npz")
        sys.exit(1)
    main(sys.argv[1])
