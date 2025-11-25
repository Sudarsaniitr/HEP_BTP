"""
Plot butterfly (all channels overlayed) for HEP data.

Usage examples:
  # single-subject HEP npz (contains 'hep', 'channels', 'times')
  python plot_butterfly.py out_stage1_hep/EPCTL05_HEP.npz

  # stacked stage2 NPZ (all_data shape = subjects x channels x times):
  python plot_butterfly.py /mnt/data/group_stacked_data.npz --index 0

  # or select by subject name
  python plot_butterfly.py /mnt/data/group_stacked_data.npz --subject EPCTL05
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def robust_unpack_npz(npz_path):
    d = np.load(str(npz_path), allow_pickle=True)
    keys = set(d.keys())

    # Case A: per-subject HEP file
    if "hep" in d and ("channels" in d or "ch_names" in d):
        hep = np.array(d["hep"])
        ch_names = [str(x) for x in (d.get("channels", d.get("ch_names")))]
        times = np.array(d["times"]) if "times" in d else np.linspace(-0.2, 0.6, hep.shape[1])
        meta = {"type": "single", "path": str(npz_path)}
        d.close()
        return hep, ch_names, times, meta

    # Case B: stacked NPZ
    if "all_data" in d:
        all_data = np.array(d["all_data"])   # shape (subjects, channels, times)
        ch_names = None
        for key in ("ch_names", "channels", "channel_names"):
            if key in d:
                ch_names = [str(x) for x in np.atleast_1d(d[key]).tolist()]
                break
        if ch_names is None:
            # fallback generate names
            ch_names = [f"Ch{i}" for i in range(all_data.shape[1])]
        times = np.array(d["times"]) if "times" in d else np.linspace(-0.2, 0.6, all_data.shape[2])
        subjects = [str(x) for x in (d["subjects"].tolist() if "subjects" in d else [f"sub{i}" for i in range(all_data.shape[0])])]
        groups = [str(x) for x in (d["groups"].tolist() if "groups" in d else ["unknown"]*all_data.shape[0])]
        meta = {"type": "stacked", "all_data": all_data, "subjects": subjects, "groups": groups, "path": str(npz_path)}
        d.close()
        return None, ch_names, times, meta
    d.close()
    raise RuntimeError(f"NPZ {npz_path} does not contain 'hep' or 'all_data' keys. Found keys: {list(keys)}")

def plot_butterfly(hep, ch_names, times, subject_name=None, out_png=None, analysis_window=(0.25,0.6), show_mean=True):
    """
    hep: (n_channels, n_times)
    ch_names: list of channel names length n_channels
    times: (n_times,)
    """
    n_ch, n_t = hep.shape
    fig, ax = plt.subplots(figsize=(10,6))
    # plot each channel faintly
    for i in range(n_ch):
        ax.plot(times, hep[i,:], linewidth=0.6, alpha=0.6)
    # plot mean across channels thicker
    # highlight analysis window
    lo, hi = analysis_window
    ax.axvspan(lo, hi, color='orange', alpha=0.25, label=f'Analysis window {lo}-{hi}s')

    ax.axvline(0.0, color='red', linestyle='--', label='R-peak (0s)')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV) (or original units)")
    title_sub = f" — {subject_name}" if subject_name is not None else ""
    ax.set_title(f"Butterfly plot (all channels overlaid){title_sub}")
    ax.legend(loc='upper right')
    ax.grid(alpha=0.2)

    plt.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=180)
        print("Saved:", out_png)
    plt.show()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", help="Path to a subject-level HEP npz (hep,channels,times) OR a stacked NPZ (all_data, ch_names, times, subjects)")
    p.add_argument("--index", type=int, default=None, help="If stacked NPZ, subject index to plot (0-based).")
    p.add_argument("--subject", type=str, default=None, help="If stacked NPZ, subject name to select.")
    p.add_argument("--out", type=str, default=None, help="Optional output PNG path.")
    p.add_argument("--analysis_window", nargs=2, type=float, default=(0.25,0.6), help="Analysis window to highlight (lo hi) in seconds.")
    args = p.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(args.npz)

    maybe_hep, ch_names, times, meta = robust_unpack_npz(npz_path)

    if meta["type"] == "single":
        hep = maybe_hep
        subj_name = Path(meta["path"]).stem
    else:
        # stacked
        all_data = meta["all_data"]  # (S, C, T)
        subjects = meta["subjects"]
        if args.subject is not None:
            if args.subject not in subjects:
                raise RuntimeError(f"Subject '{args.subject}' not found in stacked NPZ. Available: {subjects}")
            idx = subjects.index(args.subject)
        elif args.index is not None:
            idx = args.index
            if idx < 0 or idx >= len(subjects):
                raise IndexError(f"index out of range: {idx}")
        else:
            # default pick 0
            idx = 0
        hep = all_data[idx, :, :]
        subj_name = subjects[idx]

    # sanity checks
    hep = np.array(hep, dtype=float)
    if hep.ndim != 2:
        raise RuntimeError("HEP array must be 2D (channels x times). Got shape: " + str(hep.shape))
    if len(ch_names) != hep.shape[0]:
        # If names mismatch, just create generic names
        ch_names = [f"Ch{i}" for i in range(hep.shape[0])]

    # convert to µV if values look like volts (optional heuristic)
    # (we will not forcibly convert; user can modify if needed)

    out_png = args.out or (Path.cwd() / f"butterfly_{subj_name}.png")
    plot_butterfly(hep, ch_names, times, subject_name=subj_name, out_png=str(out_png), analysis_window=tuple(args.analysis_window))

if __name__ == "__main__":
    main()
