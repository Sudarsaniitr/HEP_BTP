
"""
hep_peak_vs_sleep.py

Compute peak (max absolute) HEP for a channel in a specified time window
(0.20 - 0.50 s by default) from per-subject stage-1 NPZ outputs,
then plot vs sleep-efficiency values provided in the Stage-1 manifest.

Usage:
  python hep_peak_vs_sleep.py \
      --manifest out_stage1_hep_opt/manifest_hep_opt.json \
      --channel FP2 \
      --tmin 0.20 --tmax 0.50 \
      --out hep_peak_vs_sleep_FP2.png
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def normalize_channel_name(name: str) -> str:
    if name is None:
        return ""
    s = str(name).strip().lower()
    for suf in ["-ref", "_ref", " ref"]:
        if s.endswith(suf):
            s = s[:-len(suf)].strip()
    if s.endswith("+") or s.endswith("-"):
        s = s[:-1]
    # remove whitespace and punctuation that commonly differ
    s = s.replace(" ", "").replace(".", "").replace("/", "").replace("-", "")
    return s

def find_channel_index(channels, target):
    """Return index of channel matching target (normalized), or None."""
    tgt = normalize_channel_name(target)
    norm_map = {normalize_channel_name(c): i for i, c in enumerate(channels)}
    return norm_map.get(tgt, None)

def _unpack_array_like(x):
    """Handle arrays/lists/obj arrays from NPZ and return python list of strings."""
    import numpy as _np
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, _np.ndarray):
        if x.dtype == object and x.size == 1:
            inner = x[0]
            if isinstance(inner, (list, tuple, _np.ndarray)):
                return [str(u) for u in inner]
            return [str(inner)]
        try:
            return [str(u) for u in x.tolist()]
        except Exception:
            return [str(u) for u in x.flatten().tolist()]
    return [str(x)]

def read_subject_hep(npz_path):
    """Return dict with keys: hep (2D array ch x t), channels (list), times (1d)"""
    d = np.load(npz_path, allow_pickle=True)
    # expected keys: hep, channels, times OR hep, ch_names, times etc.
    if 'hep' in d:
        hep = np.array(d['hep'])
    elif 'all_data' in d:
        raise RuntimeError(f"File {npz_path} looks like a stacked file (all_data). Expected subject-level HEP npz with 'hep'.")
    else:
        raise RuntimeError(f"NPZ {npz_path} missing key 'hep'")

    # channels
    if 'channels' in d:
        channels = _unpack_array_like(d['channels'])
    elif 'ch_names' in d:
        channels = _unpack_array_like(d['ch_names'])
    else:
        raise RuntimeError(f"NPZ {npz_path} missing 'channels' or 'ch_names'")

    if 'times' not in d:
        raise RuntimeError(f"NPZ {npz_path} missing 'times'")

    times = np.array(d['times'], dtype=float)
    d.close()
    return {'hep': hep, 'channels': channels, 'times': times}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="out_stage1_hep_opt/manifest_hep_opt.json", help="Stage1 manifest JSON (with per-subject 'file' and 'sleep_eff')")
    p.add_argument("--channel", default="FT10", help="channel to use (case-insensitive, tolerant)")
    p.add_argument("--tmin", type=float, default=0.20, help="window start (s)")
    p.add_argument("--tmax", type=float, default=0.50, help="window end (s)")
    p.add_argument("--out", default="hep_peak_vs_sleep_FP2.png", help="output plot file")
    p.add_argument("--sleep-key", default="sleep_eff", help="field name in manifest for sleep efficiency")
    p.add_argument("--sleep-percent", action="store_true", help="if sleep_eff is in percent (0-100) convert to 0-1")
    p.add_argument("--min-subjects", type=int, default=3, help="minimum matched subjects to compute correlation")
    args = p.parse_args()

    if not os.path.exists(args.manifest):
        raise FileNotFoundError("Manifest not found: " + args.manifest)

    with open(args.manifest, "r") as f:
        manifest = json.load(f)

    subj_records = manifest.get("subjects", manifest.get("Subjects", []))
    if not subj_records:
        raise RuntimeError("No subjects found in manifest JSON under key 'subjects' or 'Subjects'.")

    values = []
    sleep_vals = []
    matched_subjects = []

    for rec in subj_records:
        name = rec.get("name") or rec.get("subject") or rec.get("id")
        # find NPZ: prefer explicit file entry, else try standard filename
        npz_path = rec.get("file") or rec.get("out_file") or rec.get("hep_file")
        if not npz_path:
            # try default location
            guess = os.path.join("out_stage1_hep", f"{name}_HEP.npz")
            if os.path.exists(guess):
                npz_path = guess
        if not npz_path or not os.path.exists(npz_path):
            print(f"Skipping {name}: subject-level NPZ not found (tried '{npz_path}')")
            continue

        # read HEP
        try:
            subj = read_subject_hep(npz_path)
        except Exception as e:
            print(f"Skipping {name}: failed reading NPZ {npz_path}: {e}")
            continue

        ch_idx = find_channel_index(subj['channels'], args.channel)
        if ch_idx is None:
            print(f"Skipping {name}: channel {args.channel} not present in {npz_path}")
            continue

        times = subj['times']
        # choose time mask inclusive
        tmask = (times >= args.tmin) & (times <= args.tmax)
        if not tmask.any():
            print(f"Skipping {name}: no timepoints within [{args.tmin},{args.tmax}] in {npz_path}")
            continue

        hep_ts = subj['hep']
        # safety: shape check
        if hep_ts.ndim != 2:
            print(f"Skipping {name}: unexpected hep shape {hep_ts.shape} in {npz_path}")
            continue

        # extract channel row
        chan_ts = hep_ts[ch_idx, :]

        # compute peak absolute (max abs) within window
        segment = chan_ts[tmask]
        if np.all(np.isnan(segment)):
            print(f"Skipping {name}: all NaN in selection window")
            continue
        peak_abs = float(np.nanmax(np.abs(segment)))

        # read sleep eff
        sleep_val = rec.get(args.sleep_key, None)
        if sleep_val is None:
            print(f"Skipping {name}: no '{args.sleep_key}' in manifest entry")
            continue
        try:
            sval = float(sleep_val)
            if args.sleep_percent and sval > 1.0:
                sval = sval / 100.0
        except Exception:
            print(f"Skipping {name}: invalid sleep value '{sleep_val}'")
            continue

        # store
        values.append(peak_abs)
        sleep_vals.append(sval)
        matched_subjects.append(name)

    n_matched = len(values)
    print(f"\nSubjects matched with channel {args.channel} and sleep_eff: {n_matched}")
    if n_matched == 0:
        print("No matches — exiting.")
        return

    values = np.array(values)
    sleep_vals = np.array(sleep_vals)

    # correlation if enough subjects
    if n_matched >= args.min_subjects:
        r, p = stats.pearsonr(values, sleep_vals)
        slope, intercept, r_val, p_val_linreg, stderr = stats.linregress(sleep_vals, values)
        print(f"Pearson r = {r:.4f}, p = {p:.4e}")
    else:
        r = p = None
        slope = intercept = None
        print(f"Too few matched subjects for correlation (need >= {args.min_subjects}).")

    # Plot
    plt.figure(figsize=(6,5))
    plt.scatter(sleep_vals, values, s=60, alpha=0.9, edgecolor='k')
    if slope is not None:
        xs = np.linspace(min(sleep_vals), max(sleep_vals), 100)
        ys = slope * xs + intercept
        plt.plot(xs, ys, label=f"LinReg (slope={slope:.3g})", linewidth=2)
    plt.xlabel("Sleep efficiency (fraction)")
    plt.ylabel(f"Peak |HEP| @ {args.channel} ({args.tmin}-{args.tmax}s)")
    title = f"{args.channel}: peak-absolute HEP ({args.tmin}-{args.tmax}s) vs sleep efficiency\nn={n_matched}"
    if r is not None:
        title += f"  r={r:.3f}, p={p:.3e}"
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved plot: {args.out}")

    # print table
    print("\nMatched subjects (name, sleep_eff, peak_abs_HEP):")
    for nm, se, mv in zip(matched_subjects, sleep_vals, values):
        print(f"  {nm:<10}  {se:0.3f}  {mv:0.6e}")

if __name__ == "__main__":
    main()


