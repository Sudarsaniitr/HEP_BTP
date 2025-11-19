import pickle
import numpy as np
import sys
import matplotlib.pyplot as plt

def load_results(path):
    print(f"Loading Stage-3 results from: {path}\n")
    with open(path, "rb") as f:
        res = pickle.load(f)

    # --- BASIC KEYS ---
    print("Keys in results file:")
    for k in res.keys():
        print(" ", k)
    print("\n")

    # --- BASIC FIELDS ---
    t_obs = res["t_obs"]              # shape (n_channels, n_times)
    df = res["df"]
    t_crit = res["t_crit"]
    clusters_info = res["clusters_info"]
    cluster_masses = res["cluster_masses"]
    cluster_pvals = res["cluster_pvals"]
    ch_names = res["ch_names"]
    times = res["times"]
    groups = res["groups"]
    subjects = res["subjects"]

    print(f"t_obs shape: {t_obs.shape}")
    print(f"Degrees of freedom: {df}")
    print(f"Cluster-forming threshold t_crit: {t_crit}")
    print()

    # --- GROUP SUMMARY ---
    print("Group labels for subjects:")
    for s, g in zip(subjects, groups):
        print(f"  {s:15s} -> {g}")
    print()

    # --- CLUSTER SUMMARY ---
    print("Number of observed clusters:", len(cluster_masses))
    if len(cluster_masses) > 0:
        for i, info in enumerate(clusters_info):
            print(f"Cluster {i}:")
            print(f"   mass = {info['mass']:.3f}")
            print(f"   p-value = {info['pval']}")
            print(f"   channels involved (count {len(info['channels'])}):")
            print("      ", info['channels'])
            print(f"   time range (s): {info['time_range_s']}")
            print()

    sig_clusters = [i for i, p in enumerate(cluster_pvals) if p < 0.05]
    print("Significant clusters (p < 0.05):", sig_clusters)
    print()

    # --- OPTIONAL: PLOT |t| across channels (simple heatmap) ---
    try:
        plt.figure(figsize=(10, 5))
        plt.imshow(np.abs(t_obs), aspect="auto",
                   extent=[times[0], times[-1], 0, len(ch_names)],
                   origin="lower")
        plt.colorbar(label="|t|")
        plt.title("Absolute t-map (channels × time)")
        plt.xlabel("Time (s)")
        plt.ylabel("Channel index")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Plotting failed:", e)

    return res

# Main entry point (run from terminal)
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_stage3_results.py path/to/results.pkl")
        sys.exit(1)

    load_results(sys.argv[1])
