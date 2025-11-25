import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------
# Load Stage-2 Data
# -------------------------
npz_path = "out_stage2_final/group_stacked_data.npz"
data = np.load(npz_path, allow_pickle=True)

all_data = data["all_data"]      # shape (subjects, channels, times)
ch_names = data["ch_names"]
times = data["times"]
groups = data["groups"]          # array of "good" / "bad"

subjects = all_data.shape[0]
n_ch = all_data.shape[1]
n_t = all_data.shape[2]

print("Data loaded:", all_data.shape)
print("Groups:", set(groups))

# -------------------------
# Split GOOD vs BAD
# -------------------------
good_idx = [i for i, g in enumerate(groups) if g == "good"]
bad_idx  = [i for i, g in enumerate(groups) if g == "bad"]

good_data = all_data[good_idx]   # (ng, 71, 200)
bad_data  = all_data[bad_idx]    # (nb, 71, 200)

print("Good subjects:", len(good_idx))
print("Bad subjects:", len(bad_idx))

# -------------------------
# Compute GROUP AVERAGES
# -------------------------
good_avg = np.mean(good_data, axis=0)   # (71, 200)
bad_avg  = np.mean(bad_data,  axis=0)   # (71, 200)

print("Group-average shape:", good_avg.shape)

# -------------------------
# ANOVA on group averages
# For each channel + timepoint: compare good_avg[ch,t] vs bad_avg[ch,t]
# -------------------------

pvals = np.zeros((n_ch, n_t))

for ch in range(n_ch):
    for t in range(n_t):

        x = good_avg[ch, t]
        y = bad_avg[ch, t]

        # ANOVA between 2 conditions → equivalent to t-test
        F, p = stats.f_oneway([x], [y])

        pvals[ch, t] = p

# -------------------------
# FDR correction
# -------------------------
from statsmodels.stats.multitest import fdrcorrection

p_flat = pvals.flatten()
rej, q_flat = fdrcorrection(p_flat, alpha=0.05)
qvals = q_flat.reshape(pvals.shape)
sig_mask = rej.reshape(pvals.shape)

print("Significant points:", np.sum(sig_mask), "/", n_ch*n_t)

# -------------------------
# Save heatmap
# -------------------------
os.makedirs("anova_groupmean_results", exist_ok=True)

plt.figure(figsize=(14, 8))
sns.heatmap(pvals, cmap="viridis", cbar=True)
plt.title("ANOVA p-values (Good Avg vs Bad Avg)")
plt.xlabel("Timepoints")
plt.ylabel("Channels")
plt.tight_layout()
plt.savefig("anova_groupmean_results/pval_heatmap_groupavg.png", dpi=200)
plt.close()

print("Saved heatmap to anova_groupmean_results/")
print("DONE.")
