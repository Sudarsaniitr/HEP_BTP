#!/usr/bin/env python3
"""
npz_to_readable_chpos.py
Robust conversion of stacked HEP NPZ -> human-readable files, with explicit
handling of ch_pos stored in multiple formats.

Usage:
    python npz_to_readable_chpos.py path/to/group_stacked_data.npz --out readable_out --compress
"""

import os
import argparse
import numpy as np
import json
import csv
import gzip
import pprint

def _unpack(obj):
    """Convert numpy object arrays, scalars -> python lists/values or dicts."""
    if obj is None:
        return None
    # numpy ndarray
    if isinstance(obj, np.ndarray):
        # object dtype: often wraps dict/list/strings
        if obj.dtype == object:
            try:
                # if it's length-1 array containing a dict, return that dict
                if obj.size == 1:
                    inner = obj.reshape(-1)[0]
                    # try recursively unpack
                    return _unpack(inner)
                # otherwise convert each element
                return [_unpack(x) for x in obj.tolist()]
            except Exception:
                return [str(x) for x in obj.flatten().tolist()]
        else:
            # numeric array, return as-is (caller decides)
            return obj
    # numpy scalar
    if isinstance(obj, (np.generic,)):
        return obj.item()
    # python dict/list/tuple/str etc.
    if isinstance(obj, dict):
        # convert numpy arrays inside dict to lists
        out = {}
        for k,v in obj.items():
            out[str(k)] = _unpack(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_unpack(x) for x in obj]
    return obj

def _parse_ch_pos(candidate):
    """
    Robustly parse 'ch_pos' like objects into a dict: {ch_name: [x,y,z]}.
    Candidate can be:
      - dict already -> convert inner arrays to lists
      - ndarray/object array containing a dict -> extract
      - list of tuples (name, [x,y,z])
      - structured array with fields ('name','pos') etc.
    Returns (ch_pos_dict, source_description)
    """
    if candidate is None:
        return {}, "none"

    # direct dict
    if isinstance(candidate, dict):
        d = {}
        for k,v in candidate.items():
            if isinstance(v, np.ndarray):
                d[str(k)] = v.tolist()
            else:
                d[str(k)] = _unpack(v)
        return d, "dict"

    # numpy object array possibly containing a dict or list
    if isinstance(candidate, np.ndarray) and candidate.dtype == object:
        # try if single-element array wrapping a dict
        try:
            if candidate.size == 1:
                inner = candidate.reshape(-1)[0]
                return _parse_ch_pos(inner)
        except Exception:
            pass
        # try iterate to find a dict inside
        for el in candidate.flatten():
            if isinstance(el, dict):
                return _parse_ch_pos(el)
        # check if it's list of pairs
        try:
            seq = candidate.tolist()
            if all(isinstance(x, (list, tuple)) and len(x) >= 2 for x in seq):
                d = {}
                for item in seq:
                    name = str(item[0])
                    pos = _unpack(item[1])
                    d[name] = pos
                return d, "list_of_pairs"
        except Exception:
            pass

    # list/tuple of pairs
    if isinstance(candidate, (list, tuple)):
        # e.g. [('Fz', [x,y,z]), ...]
        if all(isinstance(x, (list, tuple)) and len(x) >= 2 for x in candidate):
            d = {}
            for item in candidate:
                name = str(item[0])
                pos = _unpack(item[1])
                d[name] = pos
            return d, "list_of_pairs"

    # structured array or recarray
    if hasattr(candidate, "dtype") and hasattr(candidate, "tolist"):
        try:
            lst = candidate.tolist()
            # try convert similar to above
            if isinstance(lst, (list, tuple)):
                for el in lst:
                    if isinstance(el, dict):
                        return _parse_ch_pos(el)
                # try elements like (name, pos)
                if all(isinstance(x, (list, tuple)) and len(x) >= 2 for x in lst):
                    d = {}
                    for item in lst:
                        d[str(item[0])] = _unpack(item[1])
                    return d, "structured_pairs"
        except Exception:
            pass

    # fallback: string representation
    try:
        return { "raw": str(candidate) }, "raw_string"
    except Exception:
        return {}, "unknown"

def write_gz_csv(path_gz, row_headers, col_headers, matrix):
    with gzip.open(path_gz, "wt", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["channel"] + [str(t) for t in col_headers])
        for i,row in enumerate(matrix):
            writer.writerow([row_headers[i]] + ["{:.6g}".format(x) for x in row])

def save_ch_pos_json(ch_pos_dict, out_dir):
    path = os.path.join(out_dir, "ch_pos.json")
    with open(path, "w", encoding="utf8") as f:
        json.dump(ch_pos_dict, f, indent=2)
    print("Saved channel positions JSON:", path)

def main(npz_path, out_dir="readable_out", compress=True):
    os.makedirs(out_dir, exist_ok=True)
    npz = np.load(npz_path, allow_pickle=True)
    keys = list(npz.keys())
    print("NPZ keys:", keys)

    # Unpack common fields
    raw_all_data = npz.get("all_data", None)
    all_data = raw_all_data if not isinstance(raw_all_data, np.ndarray) or raw_all_data.dtype != object else _unpack(raw_all_data)
    ch_names_raw = npz.get("ch_names", npz.get("channels", None))
    ch_names = _unpack(ch_names_raw)
    times_raw = npz.get("times", None)
    times = _unpack(times_raw)
    subjects_raw = npz.get("subjects", None)
    subjects = _unpack(subjects_raw)
    groups_raw = npz.get("groups", None)
    groups = _unpack(groups_raw)

    # ch_pos may be stored under several keys: ch_pos, ch_pos_filled, ch_pos_xyz, etc.
    chpos_candidates = []
    for key in ("ch_pos", "ch_pos_filled", "ch_pos_xyz", "ch_pos_coords", "chpos"):
        if key in npz:
            chpos_candidates.append((key, npz[key]))
    # If no direct key, check 'info' or other saved objects
    if not chpos_candidates and "info" in npz:
        info = _unpack(npz["info"])
        if isinstance(info, dict) and "ch_pos" in info:
            chpos_candidates.append(("info.ch_pos", info["ch_pos"]))

    ch_pos = {}
    ch_pos_source = "none"
    tried = []
    if chpos_candidates:
        for (k, cand) in chpos_candidates:
            parsed, src = _parse_ch_pos(_unpack(cand))
            tried.append((k, src, len(parsed)))
            if parsed:
                # merge into ch_pos (do not overwrite existing)
                for kk,vv in parsed.items():
                    if kk not in ch_pos:
                        ch_pos[kk] = vv
                if ch_pos_source == "none":
                    ch_pos_source = k
    else:
        # try keys embedded as object arrays/dicts anywhere in npz
        for k in keys:
            if k in ("all_data","ch_names","channels","times","subjects","groups"):
                continue
            v = npz[k]
            parsed, src = _parse_ch_pos(_unpack(v))
            if parsed:
                tried.append((k, src, len(parsed)))
                for kk,vv in parsed.items():
                    if kk not in ch_pos:
                        ch_pos[kk] = vv
                if ch_pos_source == "none":
                    ch_pos_source = k

    # If still empty, try standard montages mapping (best-effort)
    if not ch_pos and ch_names is not None:
        try:
            import mne
            def normalize_channel_name(s):
                return "".join(ch for ch in str(s).lower() if ch.isalnum())
            montages_to_try = ["standard_1020", "biosemi64", "standard_1005"]
            for mname in montages_to_try:
                try:
                    mont = mne.channels.make_standard_montage(mname)
                    mpos = mont.get_positions().get("ch_pos", {})
                    matched = 0
                    for ch in ch_names:
                        if ch in mpos:
                            ch_pos[ch] = np.asarray(mpos[ch], dtype=float).tolist()
                            matched += 1
                    if matched < len(ch_names):
                        mm = {normalize_channel_name(k): k for k in mpos.keys()}
                        for ch in ch_names:
                            if ch in ch_pos:
                                continue
                            key = normalize_channel_name(ch)
                            if key in mm:
                                mch = mm[key]
                                ch_pos[ch] = np.asarray(mpos[mch], dtype=float).tolist()
                                matched += 1
                    tried.append(("montage_"+mname, matched))
                except Exception:
                    tried.append(("montage_"+mname, 0))
        except Exception:
            tried.append(("mne_not_available", 0))

    # Pretty print ch_pos and tried
    print("ch_pos extraction summary:")
    print("  source chosen:", ch_pos_source)
    print("  ch_pos len:", len(ch_pos))
    pprint.pprint(list(ch_pos.items())[:10])  # show first 10 entries
    print("tried sources summary:", tried)

    # Save ch_pos if available
    if ch_pos:
        save_ch_pos_json(ch_pos, out_dir)

    # Now proceed to write human-readable outputs for all_data etc.
    if isinstance(all_data, np.ndarray):
        S, C, T = all_data.shape
        print(f"all_data shape: subjects={S}, channels={C}, times={T}")
        # write ch_names
        ch_names_list = [str(x) for x in ch_names] if ch_names is not None else [f"ch{i}" for i in range(C)]
        with open(os.path.join(out_dir, "ch_names.txt"), "w", encoding="utf8") as f:
            for ch in ch_names_list:
                f.write(ch+"\n")
        # times
        times_list = list(times) if isinstance(times, (list, np.ndarray)) else [f"t{i}" for i in range(T)]
        with open(os.path.join(out_dir, "times.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_index","time_s"])
            for i,t in enumerate(times_list):
                writer.writerow([i, t])
        # summary
        subjects_list = [str(x) for x in subjects] if subjects is not None else [f"subject_{i:03d}" for i in range(S)]
        groups_list = [str(x) for x in groups] if groups is not None else ["" for _ in range(S)]
        with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["subject","group","n_channels","n_times"])
            for i in range(S):
                writer.writerow([subjects_list[i], groups_list[i] if i < len(groups_list) else "", C, T])

        # per-subject saving
        perdir = os.path.join(out_dir, "per_subject")
        os.makedirs(perdir, exist_ok=True)
        for i in range(S):
            subj = subjects_list[i] if i < len(subjects_list) else f"subject_{i:03d}"
            subj_dir = os.path.join(perdir, subj)
            os.makedirs(subj_dir, exist_ok=True)
            mat = all_data[i]  # C x T
            filename = os.path.join(subj_dir, "HEP_matrix.csv.gz" if compress else "HEP_matrix.csv")
            if compress:
                write_gz_csv(filename, ch_names_list, times_list, mat)
            else:
                with open(filename, "w", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["channel"] + [str(t) for t in times_list])
                    for ri,row in enumerate(mat):
                        writer.writerow([ch_names_list[ri]] + ["{:.6g}".format(x) for x in row])
            meta = {"subject": subj, "group": groups_list[i] if i < len(groups_list) else None,
                    "n_channels": mat.shape[0], "n_times": mat.shape[1]}
            with open(os.path.join(subj_dir, "meta.json"), "w", encoding="utf8") as f:
                json.dump(meta, f, indent=2)
            print("saved subject:", subj, "->", filename)
    else:
        print("No numeric all_data ndarray found. Available keys:", keys)

    # overall metadata
    metadata = {
        "npz_path": npz_path,
        "keys": keys,
        "ch_pos_len": len(ch_pos),
        "ch_pos_source": ch_pos_source,
        "tried": tried
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf8") as f:
        json.dump(metadata, f, indent=2)
    print("Done. Outputs in:", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", help="stacked NPZ path")
    parser.add_argument("--out", default="readable_out", help="output directory")
    parser.add_argument("--compress", action="store_true", help="gzip per-subject CSVs")
    args = parser.parse_args()
    main(args.npz, args.out, args.compress)


