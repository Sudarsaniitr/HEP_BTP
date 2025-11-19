import mne
import matplotlib.pyplot as plt

# === Load epochs ===
epochs = mne.read_epochs("out_stage1/EPCTL27/EPCTL27_epochs-epo.fif", preload=False)
print(epochs)
print("Channels:", epochs.ch_names[:10], "...")
print("Number of epochs:", len(epochs))
print("Times (s):", epochs.times[:10])

# Plot one random epoch
epochs.plot(n_epochs=5, n_channels=10, scalings='auto')

# === Load evoked (average) ===
evoked = mne.read_evokeds("out_stage1/EPCTL27/EPCTL27_evoked-ave.fif")[0]
print(evoked)
evoked.plot(spatial_colors=True, titles='Evoked response')

# === Optional topography ===
# evoked.plot_topomap(times=[0.1, 0.2, 0.3, 0.4], title='Topomaps at selected latencies')
plt.show()
