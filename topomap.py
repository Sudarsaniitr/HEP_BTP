import mne
import numpy as np
import matplotlib.pyplot as plt

channels = ["fc6","poz","c2","f6","po8","fp1","p6","ft10","cz","ft8"]

def plot_topomap_stage(stage, model):

    coefs = model.params.drop("const")

    ch_values = {}

    for ch in channels:
        vals = [abs(v) for k,v in coefs.items() if ch in k]
        if vals:
            ch_values[ch.upper()] = np.mean(vals)
        else:
            ch_values[ch.upper()] = 0

    ch_names = list(ch_values.keys())
    values = np.array(list(ch_values.values()))

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=100,
        ch_types="eeg"
    )

    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage)

    plt.figure()
    mne.viz.plot_topomap(
        values,
        info,
        cmap="RdBu_r",
        contours=0,
        show=True
    )

    plt.title(f"HEP Topomap - {stage.upper()}")


# ===============================
# RUN FOR ALL STAGES
# ===============================

for stage in ALL_MODELS:

    plot_topomap_stage(
        stage,
        ALL_MODELS[stage]["COMBINED"]
    )