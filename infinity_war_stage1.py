#!/usr/bin/env python3
"""
Plot HEP per-subject: line-plot of all channels + safe topomaps.

Usage:
    python plot_hep_with_topomap.py <subject_hep_npz> [out_dir]

The NPZ is expected to contain keys:
 - 'hep' : array (n_channels, n_times)
 - 'channels' : array of channel names (len n_channels) OR 'ch_names'
 - 'times' : time vector (n_times)
 - optional: 'ch_pos' dict mapping channel->(x,y,z)

This script is robust to MNE/matplotlib layout issues and uses low-level
mne.viz.plot_topomap to create a grid of topomaps and a single shared colorbar.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import mne
except Exception:
    mne = None

# -------------------------
# helper utilities
# -------------------------
def normalize_name(s):
    if s is None:
        return ""
    return "".join(ch for ch in str(s).lower() if ch.isalnum())

def try_get_ch_pos_from_montages(ch_names, montages_to_try=("standard_1020", "standard_1005")):
    """
    Try mapping channel names to positions using standard MNE montages.
    Returns ch_pos dict (channel->(x,y,z)) and 'tried' list [(mname, matched_count), ...]
    """
    ch_pos = {}
    tried = []
    if mne is None:
        return ch_pos, tried
    for mname in montages_to_try:
        try:
            mont = mne.channels.make_standard_montage(mname)
            mpos = mont.get_positions().get("ch_pos", {})
            matched = 0
            # exact matches first
            for ch in ch_names:
                if ch in mpos:
                    ch_pos[ch] = np.asarray(mpos[ch], dtype=float)
                    matched += 1
            # normalized matches
            if matched < len(ch_names):
                norm_map = {normalize_name(k): k for k in mpos.keys()}
                for ch in ch_names:
                    if ch in ch_pos:
                        continue
                    key = normalize_name(ch)
                    if key in norm_map:
                        mch = norm_map[key]
                        ch_pos[ch] = np.asarray(mpos[mch], dtype=float)
                        matched += 1
            tried.append((mname, matched))
        except Exception:
            tried.append((mname, 0))
    return ch_pos, tried

# -------------------------
# robust topomap plotting
# -------------------------
def plot_topomaps(hep, times, ch_names, ch_pos_map, subject, out_png, montage_name_used=None):
    """
    Create a topomap figure with multiple time panels using safe low-level plotting.
    Avoids evoked.plot_topomap() to prevent matplotlib layout/colorbar engine conflicts.
    Returns out_png path on success, else None.
    """
    if mne is None:
        print("mne not available — skipping topomap plotting.")
        return None

    n_ch, n_t = hep.shape
    # estimate sfreq from times if possible
    sfreq = None
    if times is not None and len(times) > 1:
        dt = float(np.mean(np.diff(times)))
        if dt > 0:
            sfreq = 1.0 / dt

    # create info with channel names
    try:
        info = mne.create_info(ch_names, sfreq=sfreq if sfreq is not None else 1000.0, ch_types='eeg')
    except Exception:
        # fallback if mne version picky about channel name types
        info = mne.create_info(list(map(str, ch_names)), sfreq=sfreq if sfreq is not None else 1000.0, ch_types='eeg')

    if len(ch_pos_map) == 0:
        print("No channel positions available — cannot make topomap. Skipping.")
        return None

    # Only include positions for channels present in ch_names
    ch_pos_for_mne = {ch: tuple(ch_pos_map[ch]) for ch in ch_names if ch in ch_pos_map}
    if len(ch_pos_for_mne) == 0:
        print("No matching channel positions for the subject names — skipping topomap.")
        return None

    # set montage
    try:
        montage = mne.channels.make_dig_montage(ch_pos=ch_pos_for_mne, coord_frame='head')
        info.set_montage(montage, match_case=False)
    except Exception as e:
        try:
            dig = mne.channels.DigMontage(ch_pos=ch_pos_for_mne, coord_frame='head')
            info.set_montage(dig, match_case=False)
        except Exception as e2:
            print("Failed to set montage:", e, " / fallback:", e2)
            return None

    evoked = mne.EvokedArray(hep, info, tmin=times[0] if times is not None else 0.0)

    # choose times to plot: include 0s, global peak, and quartiles
    times_to_plot = []
    if times is not None:
        if 0.0 >= times[0] and 0.0 <= times[-1]:
            times_to_plot.append(0.0)
        mean_abs = np.mean(np.abs(hep), axis=0)
        peak_idx = int(np.nanargmax(mean_abs))
        peak_t = float(times[peak_idx])
        if peak_t not in times_to_plot:
            times_to_plot.append(peak_t)
        q_idx = [int(n_t * q) for q in (0.25, 0.5, 0.75)]
        for qi in q_idx:
            t = float(times[qi])
            if t not in times_to_plot:
                times_to_plot.append(t)
    else:
        # fallback: pick central indices
        n = min(4, n_t)
        times_to_plot = [float(i) for i in range(n)]

    times_to_plot = sorted(times_to_plot)
    n = len(times_to_plot)

    # layout grid
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    # set consistent vmin/vmax across panels (robust)
    dat = evoked.data
    vmin = np.percentile(dat, 2)
    vmax = np.percentile(dat, 98)
    if vmin == vmax:
        vmax = vmin + 1e-6

    # Plot each topomap into its axis using mne.viz.plot_topomap (low-level)
    for i, t in enumerate(times_to_plot):
        if i >= len(axes):
            break
        idx = int(np.argmin(np.abs(evoked.times - t)))
        ax = axes[i]
        try:
            # mne.viz.plot_topomap will draw onto provided ax
            mne.viz.plot_topomap(evoked.data[:, idx], evoked.info, axes=ax, show=False, vmin=vmin, vmax=vmax)
        except Exception as e:
            # fallback to older signature / minor differences
            try:
                mne.viz.plot_topomap(evoked.data[:, idx], evoked.info, axes=ax, show=False)
            except Exception as e2:
                ax.text(0.5, 0.5, f"Topomap failed: {e2}", ha='center')
        ax.set_title(f"{t:.3f}s")

    # turn off unused axes
    for j in range(len(times_to_plot), len(axes)):
        axes[j].axis('off')

    # add a single colorbar using the first axis' image if available
    first_img = None
    for ax in axes:
        if hasattr(ax, "images") and len(ax.images) > 0:
            first_img = ax.images[0]
            break
    if first_img is not None:
        fig.colorbar(first_img, ax=axes[:len(times_to_plot)], orientation='vertical', fraction=0.02)

    fig.suptitle(f"{subject} topomaps (montage={montage_name_used})", fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return out_png

# -------------------------
# main script
# -------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_hep_with_topomap.py <subject_npz> [out_dir]")
        sys.exit(1)

    npz_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("out_stage1_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not npz_path.exists():
        print("File not found:", npz_path)
        sys.exit(1)

    print("Loaded:", npz_path)
    data = np.load(str(npz_path), allow_pickle=True)

    # heuristics for channels key names
    if "hep" in data:
        hep = data["hep"]
    elif "all" in data:
        hep = data["all"]
    else:
        # try common key names
        possible = [k for k in data.keys() if data[k].ndim == 2]
        if len(possible) == 0:
            raise RuntimeError("Cannot find 2D HEP array in NPZ keys: " + ", ".join(data.keys()))
        hep = data[possible[0]]

    # channel names
    ch_names = None
    for key in ("ch_names", "channels", "channel_names"):
        if key in data:
            ch_names = list(data[key])
            break
    if ch_names is None:
        # fallback: infer string elements inside 'channels' or from keys
        if "channels" in data:
            ch_names = list(data["channels"])
        else:
            # try to coerce each element as string
            try:
                ch_names = [str(x) for x in data.get("channels", [])]
            except Exception:
                ch_names = [f"Ch{i}" for i in range(hep.shape[0])]

    times = None
    if "times" in data:
        times = np.array(data["times"], dtype=float)
    else:
        # infer from sampling if 'sfreq' & length are present, else use linspace
        if "sfreq" in data:
            sf = float(data["sfreq"])
            times = np.linspace(-0.2, 0.6, hep.shape[1])
        else:
            times = np.linspace(-0.2, 0.6, hep.shape[1])

    # ch_pos if present
    ch_pos = {}
    if "ch_pos" in data:
        try:
            saved = data["ch_pos"].tolist()
            if isinstance(saved, dict):
                for k,v in saved.items():
                    try:
                        ch_pos[str(k)] = np.asarray(v, dtype=float)
                    except Exception:
                        pass
        except Exception:
            pass

    # If ch_pos is empty, try montages
    if len(ch_pos) == 0 and mne is not None:
        try:
            mont_map, tried = try_get_ch_pos_from_montages(ch_names, montages_to_try=("standard_1020","standard_1005"))
            print("Montage matching results:", tried)
            for k,v in mont_map.items():
                if k not in ch_pos:
                    ch_pos[k] = v
        except Exception:
            pass

    # Logging
    print("HEP shape:", getattr(hep, "shape", None))
    print("Time vector method: times_from_file | n_timepoints:", len(times))
    print("Sampling ~ {:.1f} Hz (dt={:.6f}s)".format(1.0/np.mean(np.diff(times)), np.mean(np.diff(times))))

    subj = npz_path.stem.replace("_HEP","")

    # 1) Plot all channels stacked lines
    fig1, ax1 = plt.subplots(figsize=(10, 12))
    n_ch = hep.shape[0]
    tvec = times
    # offset each channel vertically for readability
    offsets = np.arange(n_ch)[::-1] * (np.nanstd(hep) * 4.0 + 1e-12)
    for i in range(n_ch):
        ax1.plot(tvec, hep[i, :] + offsets[i], linewidth=0.7)
    # annotate channel names on y ticks
    ax1.set_yticks(offsets)
    ax1.set_yticklabels(ch_names[::-1])
    ax1.set_xlabel("Time (s)")
    ax1.set_title(f"{subj} — averaged HEP for each channel (stacked)")

    fig1.tight_layout()
    out1 = out_dir / f"{subj}_HEP_all_channels.png"
    fig1.savefig(out1, dpi=200)
    plt.close(fig1)
    print("Plotting all channel lines ->", out1)

    # 2) Topomap(s)
    out_tmap = out_dir / f"{subj}_HEP_topomaps.png"
    try:
        tresult = plot_topomaps(hep, times, ch_names, ch_pos, subj, str(out_tmap), montage_name_used="auto")
        if tresult is None:
            print("Topomap generation failed.")
        else:
            print("Topomap saved ->", tresult)
    except Exception as e:
        print("Topomap plotting via helper failed with exception:", e)

    print("Done. Outputs in:", out_dir)

if __name__ == "__main__":
    main()
