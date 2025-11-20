#!/usr/bin/env python3
"""
analyze_stage3_results.py

Load a Stage-3 results.pkl (cluster-based permutation output) and produce diagnostics:
 - text summary of clusters
 - histograms of permutation null distributions (pos / neg)
 - overlay of observed cluster masses on null histograms
 - t-map image (channels x time) saved as PNG
 - table (CSV) of clusters with p-values
 - attempt topomap snapshots for significant clusters (requires mne)

Usage:
    python analyze_stage3.py /path/to/results.pkl --outdir analysis_stage3_report
"""
import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import csv
from textwrap import shorten

try:
    import mne
except Exception:
    mne = None

def load_results(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def ensure_out(outdir):
    os.makedirs(outdir, exist_ok=True)
    return outdir

def print_summary(res):
    print("------ Stage-3 results summary ------")
    print(f"t_obs shape: {np.array(res['t_obs']).shape}")
    print(f"Degrees of freedom: {res.get('df')}")
    print(f"Cluster-forming thresholds (two-sided/pos/neg): {res.get('t_crit_two')} / {res.get('t_crit_pos')} / {res.get('t_crit_neg')}")
    print()
    print("Groups / subjects:")
    groups = res.get("groups", [])
    subjects = res.get("subjects", [])
    for subj, grp in zip(subjects, groups):
        print(f"  {subj:15s} -> {grp}")
    print()
    print(f"Number of observed clusters: {len(res.get('clusters_info', []))}")
    print()

def save_clusters_csv(res, outdir):
    csv_path = os.path.join(outdir, "clusters_table.csv")
    rows = []
    for c in res.get("clusters_info", []):
        rows.append({
            "index": c.get("index"),
            "sign": c.get("sign", ""),
            "mass": c.get("mass"),
            "pval": c.get("pval"),
            "channels": ";".join(c.get("channels", [])),
            "time_start_s": c.get("time_range", (None, None))[0] if "time_range" in c else (c.get("time_idx")[0] if c.get("time_idx") else None),
            "time_end_s": c.get("time_range", (None, None))[1] if "time_range" in c else (c.get("time_idx")[-1] if c.get("time_idx") else None),
            "n_channels": len(c.get("channels", [])),
            "n_timepoints": len(c.get("time_idx", [])) if c.get("time_idx") else 0
        })
    # write csv
    with open(csv_path, "w", newline='', encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["index","sign","mass","pval","channels"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print("Saved clusters table ->", csv_path)
    return csv_path

def plot_null_histograms(res, outdir):
    pos_null = np.asarray(res.get("pos_null", []))
    neg_null = np.asarray(res.get("neg_null", []))
    cluster_masses_pos = np.asarray(res.get("cluster_masses_pos", []))
    cluster_masses_neg = np.asarray(res.get("cluster_masses_neg", []))

    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    # positive
    ax = axes[0]
    if pos_null.size:
        ax.hist(pos_null, bins=50, alpha=0.8, label="pos null")
        ax.axvline(np.median(pos_null), color="C1", linestyle="--", label="null median")
    if cluster_masses_pos.size:
        ax.scatter(cluster_masses_pos, np.zeros(len(cluster_masses_pos)), color="red", marker="v", label="observed pos masses")
    ax.set_title("Positive cluster null distribution")
    ax.legend()
    # negative
    ax = axes[1]
    if neg_null.size:
        ax.hist(neg_null, bins=50, alpha=0.8, label="neg null")
        ax.axvline(np.median(neg_null), color="C1", linestyle="--", label="null median")
    if cluster_masses_neg.size:
        ax.scatter(cluster_masses_neg, np.zeros(len(cluster_masses_neg)), color="red", marker="v", label="observed neg masses")
    ax.set_title("Negative cluster null distribution (magnitudes)")
    ax.legend()
    plt.tight_layout()
    p = os.path.join(outdir, "null_distributions.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("Saved null histograms ->", p)
    return p

def plot_tmap(res, outdir, max_channels_to_show=71):
    t_obs = np.asarray(res["t_obs"])
    times = np.asarray(res["times"])
    ch_names = list(res.get("ch_names", []))
    nchan = t_obs.shape[0]
    # create an image with channels on y and time on x
    fig, ax = plt.subplots(1,1, figsize=(12, max(3, nchan/6)))
    im = ax.imshow(t_obs, aspect='auto', origin='lower',
                   extent=[times[0], times[-1], 0, nchan])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channels (index)")
    ax.set_title("t-statistic map (channels x time)")
    # optionally label y ticks sparsely
    if nchan <= max_channels_to_show:
        ax.set_yticks(np.arange(nchan)+0.5)
        ax.set_yticklabels(ch_names)
    else:
        # sparse ticks
        step = max(1, nchan // 20)
        yt = np.arange(0, nchan, step) + 0.5
        ax.set_yticks(yt)
        ax.set_yticklabels([ch_names[int(i)] for i in np.arange(0, nchan, step)])
    plt.colorbar(im, ax=ax, label="t")
    plt.tight_layout()
    p = os.path.join(outdir, "tmap_image.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("Saved tmap image ->", p)
    return p

def topomap_for_cluster(res, cluster_info, outdir):
    """
    Attempt topomap snapshot for a cluster. Requires mne and reasonable montage mapping.
    cluster_info expects 'time_idx' or 'time_range' and ch_names in res.
    """
    if mne is None:
        print("mne not available — skipping topomap.")
        return None
    ch_names = list(res.get("ch_names", []))
    times = np.asarray(res.get("times"))
    t_obs = np.asarray(res.get("t_obs"))
    # find midpoint time index
    if "time_idx" in cluster_info and cluster_info["time_idx"]:
        mid_idx = int(np.median(cluster_info["time_idx"]))
    elif "time_range" in cluster_info and cluster_info["time_range"][0] is not None:
        tr = cluster_info["time_range"]
        mid_time = 0.5*(tr[0] + tr[1])
        mid_idx = int(np.argmin(np.abs(times - mid_time)))
    else:
        mid_idx = t_obs.shape[1] // 2
    # create info + montage
    try:
        info = mne.create_info(ch_names, sfreq=1000.0, ch_types='eeg')
        # set montage (try 1020)
        montage = mne.channels.make_standard_montage("standard_1020")
        info.set_montage(montage, on_missing='ignore')
        arr = t_obs[:, mid_idx]
        ev = mne.EvokedArray(arr[:, None], info, tmin=0.0)
        fig = ev.plot_topomap(times=[0.0], ch_type='eeg', show=False)
        # handle plotting object types
        fig_obj = fig[0] if isinstance(fig, list) else fig
        fname = os.path.join(outdir, f"cluster_{cluster_info['index']}_topomap.png")
        try:
            fig_obj.suptitle(f"Cluster {cluster_info['index']} topomap (t={times[mid_idx]:.3f}s, p={cluster_info.get('pval')})")
        except Exception:
            pass
        fig_obj.savefig(fname, dpi=150)
        plt.close(fig_obj)
        return fname
    except Exception as e:
        # fail quietly and return None
        return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_pkl", help="path to results.pkl produced by stage3")
    p.add_argument("--outdir", default="analysis_stage3_report")
    p.add_argument("--top_n_clusters", type=int, default=10, help="number of top clusters to create topomaps for (by mass)")
    args = p.parse_args()

    outdir = ensure_out(args.outdir)
    res = load_results(args.results_pkl)
    print_summary(res)

    # Save cluster table CSV
    if res.get("clusters_info"):
        save_clusters_csv(res, outdir)

    # Plot null histograms + overlay observed masses
    plot_null_histograms(res, outdir)

    # tmap image
    plot_tmap(res, outdir)

    # list top clusters by p-value and mass
    clusters = res.get("clusters_info", [])
    if len(clusters) == 0:
        print("No clusters found in results.")
        return

    # sort clusters: significant first (pval ascending), then by mass magnitude desc
    def sort_key(ci):
        p = ci.get("pval")
        # treat None/NaN as large p
        p = float(p) if p is not None else 1.0
        return (p, -abs(ci.get("mass", 0.0)))
    clusters_sorted = sorted(clusters, key=sort_key)

    print("\nTop clusters (sorted by p then mass):")
    for i, ci in enumerate(clusters_sorted[:args.top_n_clusters]):
        idx = ci.get("index")
        sign = ci.get("sign", "")
        pval = ci.get("pval")
        mass = ci.get("mass")
        chans = ci.get("channels", [])
        tr = ci.get("time_range", None)
        timestr = f"{tr[0]:.3f}..{tr[1]:.3f}" if tr else str(ci.get("time_idx"))
        print(f"Cluster {idx:3d} | sign={sign:8s} p={pval} mass={mass:.3f} | channels={len(chans)} time={timestr}")
        # try topomap
        topo_png = topomap_for_cluster(res, ci, outdir)
        if topo_png:
            print("   Topomap saved:", topo_png)

    print("\nFinished. Report saved in:", outdir)

if __name__ == "__main__":
    main()
