
"""
plot_tstats_allch.py

Compute t-statistic timecourses for every channel from stage-2 stacked data
and save a summary CSV + optional per-channel plots and full t-map.

Usage:
    python plot_tstats_allch.py path/to/group_stacked_data.npz \
        --outdir out_stage2_tstats \
        --groupA good --groupB bad \
        --plot

If group labels are embedded in the npz under "group_names" or "groups",
you may omit --groupA/--groupB (the script will try to auto-detect).

python plot_tstats_allch.py "out_stage2_fixed/group_stacked_data_fixed.npz" --outdir out_stage2_tstats --groupA good --groupB bad --plot
>>

"""

#  python plot_tstats_allch.py "out_stage2_fixed/group_stacked_data_fixed.npz" \
#         --outdir out_stage2_tstats \
#         --groupA good --groupB bad \
#         --plot

import argparse
import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import stats

def compute_t_timecourse(XA, XB):
    """
    XA: array shape (n1, n_times)
    XB: array shape (n2, n_times)
    returns tvec shape (n_times,)
    Uses pooled variance (two-sample t, equal variance assumption).
    """
    n1 = XA.shape[0]
    n2 = XB.shape[0]
    if n1 < 1 or n2 < 1:
        return np.full(XA.shape[1], np.nan)
    m1 = XA.mean(axis=0)
    m2 = XB.mean(axis=0)
    v1 = XA.var(axis=0, ddof=1) if n1 > 1 else np.zeros(XA.shape[1])
    v2 = XB.var(axis=0, ddof=1) if n2 > 1 else np.zeros(XA.shape[1])
    df = n1 + n2 - 2
    sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / df if df > 0 else np.zeros_like(v1)
    # avoid zero division
    denom = np.sqrt(sp2 * (1.0 / n1 + 1.0 / n2))
    denom[denom == 0] = np.nan
    tvec = (m1 - m2) / denom
    return tvec

def safe_load_npz(path):
    npz = np.load(path, allow_pickle=True)
    return npz

def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", help="Stage2 stacked npz file (group_stacked_data.npz)")
    p.add_argument("--outdir", default="out_tstats", help="output directory")
    p.add_argument("--groupA", default=None, help="name of group A (e.g. good)")
    p.add_argument("--groupB", default=None, help="name of group B (e.g. bad)")
    p.add_argument("--alpha", type=float, default=0.05, help="alpha for t_crit (two-tailed)")
    p.add_argument("--plot", action="store_true", help="save per-channel t plots (png)")
    p.add_argument("--channels", default=None, help="comma-separated channel list to restrict plotting (optional)")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    npz = safe_load_npz(args.npz)
    # detect keys
    # expected keys: 'all_data' (subjects, channels, times), 'ch_names', 'times', 'group_names' (or 'group_labels')
    if 'all_data' in npz:
        all_data = npz['all_data']  # (subjects, channels, times)
    elif 'data' in npz:
        all_data = npz['data']
    else:
        # try common variants
        keys = list(npz.keys())
        raise KeyError(f"No 'all_data' found in {args.npz}. Keys present: {keys}")

    # channel names
    if 'ch_names' in npz:
        ch_names = [str(x) for x in npz['ch_names'].tolist()]
    elif 'ch_labels' in npz:
        ch_names = [str(x) for x in npz['ch_labels'].tolist()]
    else:
        # fallback
        n_ch = all_data.shape[1]
        ch_names = [f"Ch{c}" for c in range(n_ch)]

    # times
    if 'times' in npz:
        times = npz['times']
    else:
        # fallback to sample indices
        n_times = all_data.shape[2]
        times = np.arange(n_times)

    # group names/assignment
    # expect a parallel list group_names length subjects, or 'groups' key
    group_names = None
    for key in ('group_names', 'group_labels', 'groups', 'subject_groups'):
        if key in npz:
            group_names = [str(x) for x in npz[key].tolist()]
            break

    n_subj = all_data.shape[0]
    if group_names is None:
        # try to accept groupA/groupB from args and assume groups are first nA then nB? Not safe.
        if args.groupA is None or args.groupB is None:
            raise RuntimeError("Group labels not found in npz. Provide --groupA and --groupB (and ensure npz contains a 'groups' array).")
        # if groups not in file, assume user stacked subjects in metadata order. We'll create single group 'unknown'
        group_names = ['unknown'] * n_subj
    else:
        if len(group_names) != n_subj:
            print("Warning: group_names length doesn't match subjects; overriding to 'unknown' for all.")
            group_names = ['unknown'] * n_subj

    # determine unique groups and indices
    unique_groups = sorted(set(group_names))
    if args.groupA and args.groupB:
        grpA = args.groupA
        grpB = args.groupB
    else:
        if len(unique_groups) >= 2:
            grpA, grpB = unique_groups[0], unique_groups[1]
        else:
            # fallback
            grpA = unique_groups[0]
            grpB = 'unknown'

    idxA = [i for i, g in enumerate(group_names) if g == grpA]
    idxB = [i for i, g in enumerate(group_names) if g == grpB]
    if len(idxA) == 0 or len(idxB) == 0:
        raise RuntimeError(f"Could not find subjects for groups: {grpA} (n={len(idxA)}), {grpB} (n={len(idxB)}). Available groups: {unique_groups}")

    nA, nB = len(idxA), len(idxB)
    print(f"Loaded {n_subj} subjects, channels={len(ch_names)}, times={len(times)}")
    print(f"Groups: {grpA} (n={nA}), {grpB} (n={nB})")

    # Prepare outputs
    ch_count = len(ch_names)
    t_map = np.zeros((ch_count, len(times)), dtype=float)

    # compute t timecourse per channel
    for ci in range(ch_count):
        XA = all_data[idxA, ci, :]  # shape (nA, times)
        XB = all_data[idxB, ci, :]  # shape (nB, times)
        tvec = compute_t_timecourse(XA, XB)
        t_map[ci, :] = tvec

    # degrees of freedom
    df = nA + nB - 2
    if df > 0:
        tcrit = stats.t.ppf(1 - args.alpha / 2.0, df)
    else:
        tcrit = np.nan
    print(f"Degrees of freedom: {df}, two-tailed t_crit (alpha={args.alpha}) = {tcrit}")

    # Prepare summary CSV
    rows = []
    for ci, ch in enumerate(ch_names):
        tvec = t_map[ci]
        # absolute peak
        if np.all(np.isnan(tvec)):
            peak_idx = None
            peak_t = np.nan
            peak_time = np.nan
        else:
            peak_idx = int(np.nanargmax(np.abs(tvec)))
            peak_t = float(tvec[peak_idx])
            peak_time = float(times[peak_idx])
        meanA_at_peak = float(np.nanmean(all_data[idxA, ci, peak_idx])) if peak_idx is not None else np.nan
        meanB_at_peak = float(np.nanmean(all_data[idxB, ci, peak_idx])) if peak_idx is not None else np.nan
        rows.append({
            "channel": ch,
            "ch_index": ci,
            "peak_t": peak_t,
            "peak_time_s": peak_time,
            "peak_time_idx": peak_idx if peak_idx is not None else -1,
            "meanA_at_peak": meanA_at_peak,
            "meanB_at_peak": meanB_at_peak,
            "nA": nA,
            "nB": nB
        })

    df_out = pd.DataFrame(rows)
    csv_out = os.path.join(args.outdir, "tstats_summary_per_channel.csv")
    df_out.to_csv(csv_out, index=False)
    print("Saved summary CSV:", csv_out)

    # save full t-map and metadata
    np.savez_compressed(os.path.join(args.outdir, "tstats_full.npz"),
                        t_map=t_map, ch_names=np.array(ch_names, dtype=object),
                        times=times, groups=group_names,
                        grpA=grpA, grpB=grpB, tcrit=tcrit)
    print("Saved full t-map to tstats_full.npz")

    # optionally create plots per channel (or subset)
    if args.plot:
        plot_dir = os.path.join(args.outdir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        want_channels = None
        if args.channels:
            want_channels = [c.strip() for c in args.channels.split(",")]
        for ci, ch in enumerate(ch_names):
            if want_channels and ch not in want_channels:
                continue
            tvec = t_map[ci]
            fig, ax = plt.subplots(figsize=(8, 2.2))
            ax.plot(times, tvec, label="t")
            if not math.isnan(tcrit):
                ax.axhline(tcrit, linestyle='--', linewidth=1.25)
                ax.axhline(-tcrit, linestyle='--', linewidth=1.25)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("t")
            ax.set_title(f"{ch} (peak t={df_out.loc[ci,'peak_t']:.3f} @ {df_out.loc[ci,'peak_time_s']:.3f}s)")
            ax.legend(loc="upper right")
            plt.tight_layout()
            fname = os.path.join(plot_dir, f"t_timecourse_{ci:03d}_{ch}.png")
            fig.savefig(fname, dpi=150)
            plt.close(fig)
        print("Saved per-channel plots (in {})".format(plot_dir))

    print("Done.")

if __name__ == "__main__":
    main()
