import numpy as np

# Load your file
npz = np.load("out_stage2_tstats/tstats_full.npz", allow_pickle=True)

print("Available keys:", list(npz.keys()))

# Extract what’s actually available
t_obs = npz["t_map"]             # The full t-statistic map (channels × times)
ch_names = npz["ch_names"]       # Channel names
times = npz["times"]             # Time axis
groups = npz["groups"]           # Group labels per subject (for reference)
grpA = npz["grpA"].item()        # Name of first group
grpB = npz["grpB"].item()        # Name of second group
tcrit = npz["tcrit"].item()      # Critical t-value

print(f"t_obs shape: {t_obs.shape}")
print(f"tcrit: {tcrit}")
print(f"Groups available: {groups}")
print(f"Group A: {grpA}, Group B: {grpB}")
