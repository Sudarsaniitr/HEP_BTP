"""
Per-channel HEP plots: one PNG per channel showing mean ± SEM for two groups.

Coommand to run this code:
    avg_hep.py --npz /mnt/data/group_stacked_data.npz --groupA good --groupB bad --outdir out_plots_per_channel
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import re

def sanitize_fname(s: str) -> str:
    s2 = re.sub(r'[^\w\-_\. ]', '_', s)
    return s2.replace(" ", "_")

def unpack_ch_names(x):
    # robust unpacking similar to pipeline code
    if isinstance(x, (list, tuple)):
        return [str(u) for u in x]
    a = np.atleast_1d(x)
    if a.dtype == object and a.size == 1:
        inner = a[0]
        if isinstance(inner, (list, tuple, np.ndarray)):
            return [str(u) for u in inner]
        return [str(inner)]
    try:
        return [str(u) for u in a.tolist()]
    except Exception:
        return [str(u) for u in a.flatten().tolist()]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, help="stacked NPZ path (stage2 output).")
    p.add_argument("--groupA", default="good", help="name of group A (plotted in blue).")
    p.add_argument("--groupB", default="bad", help="name of group B (plotted in orange).")
    p.add_argument("--outdir", default="out_plots_per_channel", help="output directory for per-channel PNGs.")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--show_progress", action="store_true", help="show progress bar (default: True).")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    npz = np.load(args.npz, allow_pickle=True)
    all_data = npz["all_data"]            # (subjects, channels, times)
    groups = npz["groups"].astype(str)
    times = npz["times"].astype(float)

    raw_ch = npz.get("ch_names", npz.get("channels"))
    if raw_ch is None:
        ch_names = [f"CH{i}" for i in range(all_data.shape[1])]
    else:
        ch_names = unpack_ch_names(raw_ch)

    # ensure length consistency
    n_subj, n_ch_data, n_t = all_data.shape
    if len(ch_names) != n_ch_data:
        # pad/truncate as needed
        if len(ch_names) < n_ch_data:
            ch_names = ch_names + [f"CH_PAD_{i}" for i in range(n_ch_data - len(ch_names))]
        else:
            ch_names = ch_names[:n_ch_data]

    print(f"Loaded NPZ: {args.npz}")
    print(f"Data shape (subjects, channels, times): {all_data.shape}")
    print(f"Resolved {len(ch_names)} channel names. Example first 8: {ch_names[:8]}")

    # pick groups
    unique = np.unique(groups)
    if args.groupA not in unique or args.groupB not in unique:
        if len(unique) >= 2:
            grpA, grpB = unique[0], unique[1]
            print(f"Warning: requested group names not found. Falling back to first two groups: {grpA}, {grpB}")
            args.groupA, args.groupB = grpA, grpB
        else:
            raise RuntimeError("Need at least two groups in NPZ.")

    idxA = np.where(groups == args.groupA)[0]
    idxB = np.where(groups == args.groupB)[0]
    XA = all_data[idxA]   # (nA, ch, t)
    XB = all_data[idxB]

    meanA = XA.mean(axis=0) if XA.size else np.zeros((n_ch_data, n_t))
    meanB = XB.mean(axis=0) if XB.size else np.zeros((n_ch_data, n_t))
    semA = XA.std(axis=0, ddof=1) / np.sqrt(max(1, XA.shape[0])) if XA.size else np.zeros((n_ch_data, n_t))
    semB = XB.std(axis=0, ddof=1) / np.sqrt(max(1, XB.shape[0])) if XB.size else np.zeros((n_ch_data, n_t))

    # determine global y-range for nicer consistent scaling (optional)
    all_vals = np.concatenate([meanA, meanB], axis=0)
    vmin = np.nanmin(all_vals)
    vmax = np.nanmax(all_vals)
    pad = 0.1 * (vmax - vmin) if (vmax - vmin) != 0 else 1.0
    global_ylim = (vmin - pad, vmax + pad)

    # iteration with progress bar
    iterator = range(n_ch_data)
    if args.show_progress:
        iterator = tqdm(iterator, desc="Channels")

    for i in iterator:
        ch = ch_names[i]
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        ax.plot(times, meanA[i], label=f"{args.groupA} (n={len(idxA)})", color="tab:blue", lw=1.5)
        ax.fill_between(times, meanA[i] - semA[i], meanA[i] + semA[i], alpha=0.25, color="tab:blue")

        ax.plot(times, meanB[i], label=f"{args.groupB} (n={len(idxB)})", color="tab:orange", lw=1.5)
        ax.fill_between(times, meanB[i] - semB[i], meanB[i] + semB[i], alpha=0.25, color="tab:orange")

        ax.set_title(f"HEP — {ch}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (a.u.)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(linestyle=":", alpha=0.5)
        ax.set_xlim(times[0], times[-1])
        ax.set_ylim(global_ylim)

        # save
        fname = sanitize_fname(f"{i:02d}_{ch}.png")
        outp = os.path.join(args.outdir, fname)
        plt.tight_layout()
        plt.savefig(outp, dpi=args.dpi)
        plt.close(fig)

    print(f"Saved per-channel plots to: {args.outdir}")

if __name__ == "__main__":
    main()
