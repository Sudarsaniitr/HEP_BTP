import os
import numpy as np
import matplotlib.pyplot as plt

# ================= CONFIG =================
DATA_DIR = "out_stage1_hep_opt"   # Folder containing Stage-1 .npz files
TARGET_CHANNEL = "FC6"                # Change to FC4, Fp2, etc.
# ==========================================


def load_group_average(data_dir, target_channel, target_group):
    """
    Load all subject-level HEPs for a group and average them
    for a specific EEG channel.
    """
    heps = []
    times_ref = None

    for fname in os.listdir(data_dir):
        if not fname.endswith(".npz"):
            continue

        fpath = os.path.join(data_dir, fname)
        data = np.load(fpath, allow_pickle=True)

        group = data["group"].item()
        if group != target_group:
            continue

        channels = list(data["channels"])
        if target_channel not in channels:
            continue

        ch_idx = channels.index(target_channel)
        hep_ch = data["hep"][ch_idx]

        heps.append(hep_ch)

        if times_ref is None:
            times_ref = data["times"]

    if len(heps) == 0:
        raise RuntimeError(
            f"No data found for group='{target_group}' and channel='{target_channel}'"
        )

    heps = np.array(heps)
    mean_hep = heps.mean(axis=0)
    std_hep = heps.std(axis=0)

    return mean_hep, std_hep, times_ref


# ============ LOAD GROUP AVERAGES ============
hep_good, std_good, times = load_group_average(
    DATA_DIR, TARGET_CHANNEL, "good"
)

hep_bad, std_bad, _ = load_group_average(
    DATA_DIR, TARGET_CHANNEL, "bad"
)


# ================== PLOT ====================
plt.figure(figsize=(8, 5))

plt.plot(times, hep_good, color="green", lw=2, label="Good Sleep Efficiency")
plt.fill_between(
    times,
    hep_good - std_good,
    hep_good + std_good,
    color="green",
    alpha=0.3
)

plt.plot(times, hep_bad, color="red", lw=2, label="Bad Sleep Efficiency")
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
plt.title(f"HEP Comparison at {TARGET_CHANNEL}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
