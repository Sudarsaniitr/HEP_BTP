import os
import numpy as np
import matplotlib.pyplot as plt

# ================= CONFIG =================
BASE_DIR = "out_stage1_hep_by_stage"   # parent directory
STAGES = ["W", "N1", "N2", "N3", "R"]
TARGET_CHANNEL = "FC6"
# ==========================================


def load_stage_group_average(base_dir, stage, target_channel, target_group):
    """
    Load all subject HEPs for a given stage and group,
    extract FC6, and compute mean ± std.
    """
    stage_dir = os.path.join(base_dir, stage)

    heps = []
    times_ref = None

    for fname in os.listdir(stage_dir):
        if not fname.endswith(".npz"):
            continue

        data = np.load(os.path.join(stage_dir, fname), allow_pickle=True)

        if data["group"].item() != target_group:
            continue

        channels = list(data["channels"])
        if target_channel not in channels:
            continue

        ch_idx = channels.index(target_channel)
        heps.append(data["hep"][ch_idx])

        if times_ref is None:
            times_ref = data["times"]

    if len(heps) == 0:
        raise RuntimeError(
            f"No data for stage={stage}, group={target_group}, channel={target_channel}"
        )

    heps = np.array(heps)
    return heps.mean(axis=0), heps.std(axis=0), times_ref


# ================== PLOTTING ==================
for stage in STAGES:

    hep_good, std_good, times = load_stage_group_average(
        BASE_DIR, stage, TARGET_CHANNEL, "good"
    )

    hep_bad, std_bad, _ = load_stage_group_average(
        BASE_DIR, stage, TARGET_CHANNEL, "bad"
    )

    plt.figure(figsize=(8, 5))

    plt.plot(times, hep_good, color="green", lw=2, label="Good Sleepers")
    plt.fill_between(
        times,
        hep_good - std_good,
        hep_good + std_good,
        color="green",
        alpha=0.3
    )

    plt.plot(times, hep_bad, color="red", lw=2, label="Bad Sleepers")
    plt.fill_between(
        times,
        hep_bad - std_bad,
        hep_bad + std_bad,
        color="red",
        alpha=0.3
    )

    plt.axvline(0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Time (s) relative to R-peak")
    plt.ylabel("Amplitude (µV)")
    plt.title(f"HEP at FC6 — {stage}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
