#!/usr/bin/env python3
"""
hep_topo_absmean.py

Compute the mean absolute HEP per channel in a given analysis window and plot a
smoothed filled topomap with contour lines and channel labels.

Usage:
    python hep_topo_absmean.py <subject_hep.npz> [--tmin 0.25 --tmax 0.6] [--outdir out_topo]

Inputs (NPZ expected):
    'hep' or 'all'           -> ndarray (n_channels, n_times)
    'channels' or 'ch_names' -> channel name list
    'times'                  -> time vector (n_times,)
    optional 'ch_pos'        -> dict channel -> (x,y[,z])

This version improves on interpolation robustness and removes white NaN holes by:
 - trying cubic -> linear -> nearest interpolation
 - filling interior NaNs with nearest values
 - applying gaussian smoothing with renormalization to avoid edge dilution
 - exposing CLI flags to control `grid_res` and `smooth_sigma`
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# optional dependency; if not installed, montage lookup will be skipped
try:
    import mne
except Exception:
    mne = None

def normalize_name(s):
    if s is None:
        return ""
    return "".join(ch for ch in str(s).lower() if ch.isalnum())

def try_get_montage_positions(ch_names, montages=("standard_1020","standard_1005")):
    ch_pos = {}
    tried = []
    if mne is None:
        return ch_pos, tried
    for mname in montages:
        try:
            mont = mne.channels.make_standard_montage(mname)
            mpos = mont.get_positions().get("ch_pos", {})
            matched = 0
            for ch in ch_names:
                if ch in mpos:
                    ch_pos[ch] = np.asarray(mpos[ch], dtype=float)
                    matched += 1
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

def compute_mean_abs(hep, times, tmin=0.25, tmax=0.6):
    if times is None:
        times = np.linspace(-0.2, 0.6, hep.shape[1])
    times = np.array(times, dtype=float)
    idx0 = int(np.searchsorted(times, tmin, side="left"))
    idx1 = int(np.searchsorted(times, tmax, side="right")) - 1
    idx0 = max(0, idx0)
    idx1 = min(hep.shape[1]-1, idx1)
    if idx1 < idx0:
        raise ValueError("Invalid tmin/tmax vs times vector.")
    seg = hep[:, idx0:idx1+1]
    mean_abs = np.mean(np.abs(seg), axis=1)
    return mean_abs, (idx0, idx1)

def make_head_circle(ax, radius=0.5, center=(0.0, 0.0), **kwargs):
    circ = plt.Circle(center, radius, transform=ax.transData, fill=False, lw=2.0, **kwargs)
    ax.add_patch(circ)
    ax.plot([0.0, 0.0], [radius, radius+0.06], color='k', lw=2)

def _label_contrast_color(cmap, norm, value):
    try:
        rgba = cmap(norm(value))
        r, g, b, _ = rgba
        lum = 0.2126*r + 0.7152*g + 0.0722*b
        return 'white' if lum < 0.55 else 'black'
    except Exception:
        return 'white'

def _extract_ch_pos_from_npz(data):
    ch_pos = {}
    if "ch_pos" not in data:
        return ch_pos
    raw = data["ch_pos"]
    # Many possible storage layouts: dict, object-array with .item(), structured, tolist()
    try:
        if isinstance(raw, np.ndarray) and raw.dtype == object:
            item = raw.item()
            if isinstance(item, dict):
                raw = item
    except Exception:
        pass
    try:
        if isinstance(raw, dict):
            for k, v in raw.items():
                try:
                    ch_pos[str(k)] = np.asarray(v, dtype=float)
                except Exception:
                    pass
            return ch_pos
    except Exception:
        pass
    try:
        possible = raw.tolist()
        if isinstance(possible, dict):
            for k, v in possible.items():
                try:
                    ch_pos[str(k)] = np.asarray(v, dtype=float)
                except Exception:
                    pass
    except Exception:
        pass
    return ch_pos

from scipy.interpolate import Rbf

def plot_filled_topomap(vals, ch_names, ch_pos3d, out_png,
                        cmap="viridis", grid_res=400, n_contours=20, title=None,
                        show_labels=True, vmin=None, vmax=None,
                        interpolation_methods=("cubic","linear","nearest"),
                        smooth_sigma=2.0, fill_nearest=True, use_rbf_fallback=True):
    """
    Improved topomap:
      - tries cubic->linear->nearest griddata
      - fills interior NaNs with nearest
      - if NaNs still present and use_rbf_fallback True -> fit RBF to sensors for a smooth global map
      - gaussian smoothing with renormalization to avoid edge dilution
      - contourf with extend='both' to avoid white holes for out-of-range values
    """
    # Collect sensor positions & values (2D)
    xs, ys, vals_plot, names = [], [], [], []
    for i, ch in enumerate(ch_names):
        if ch in ch_pos3d:
            pos = ch_pos3d[ch]
            try:
                pos = np.asarray(pos, dtype=float)
                if pos.size >= 2:
                    x, y = float(pos[0]), float(pos[1])
                else:
                    continue
            except Exception:
                continue
            xs.append(x); ys.append(y)
            vals_plot.append(float(vals[i])); names.append(ch)
    xs = np.array(xs); ys = np.array(ys); vals_plot = np.array(vals_plot)

    if xs.size == 0:
        raise RuntimeError("No channel positions available for plotting topomap.")

    # center & scale coords to head radius ~0.5
    x0, y0 = xs.mean(), ys.mean()
    xs_c = xs - x0; ys_c = ys - y0
    r = np.sqrt(xs_c**2 + ys_c**2)
    max_r = r.max() if r.size>0 else 1.0
    scale = 0.45 / max_r if max_r>0 else 1.0
    xs2 = xs_c * scale; ys2 = ys_c * scale

    # grid
    padding = 0.02
    gx_lin = np.linspace(xs2.min()-padding, xs2.max()+padding, grid_res)
    gy_lin = np.linspace(ys2.min()-padding, ys2.max()+padding, grid_res)
    gx, gy = np.meshgrid(gx_lin, gy_lin)
    grid_r = np.sqrt(gx**2 + gy**2)
    mask_circle = grid_r <= 0.5 + 1e-12

    # primary interpolation: try multiple methods
    grid_z = None
    for method in interpolation_methods:
        try:
            gz = griddata((xs2, ys2), vals_plot, (gx, gy), method=method)
            # accept if any finite value inside head
            if np.isfinite(gz[mask_circle]).any():
                grid_z = gz
                break
        except Exception:
            continue

    # final fallback to nearest
    if grid_z is None:
        grid_z = griddata((xs2, ys2), vals_plot, (gx, gy), method='nearest')

    # fill interior NaNs with nearest (if requested)
    if fill_nearest:
        nan_mask = np.isnan(grid_z)
        nan_inside = np.logical_and(nan_mask, mask_circle)
        if nan_inside.any():
            try:
                grid_nearest = griddata((xs2, ys2), vals_plot, (gx, gy), method='nearest')
                grid_z[nan_inside] = grid_nearest[nan_inside]
            except Exception:
                grid_z[nan_inside] = np.nanmean(vals_plot)

    # if there are still NaNs inside the head AND RBF fallback is enabled,
    # compute a smooth RBF fit (multiquadric) and replace the entire interior
    remaining_nan_inside = np.logical_and(np.isnan(grid_z), mask_circle)
    if remaining_nan_inside.any() and use_rbf_fallback:
        try:
            # RBF often smoother if sensors are not too many (works well for EEG layouts)
            # choose epsilon relative to median sensor spacing
            dists = np.sqrt((xs2[:, None]-xs2[None, :])**2 + (ys2[:, None]-ys2[None, :])**2)
            median_spacing = np.median(dists[dists>0]) if dists.size>xs2.size else np.mean(dists)
            eps = median_spacing if (median_spacing>0 and np.isfinite(median_spacing)) else 0.1
            rbf = Rbf(xs2, ys2, vals_plot, function='multiquadric', epsilon=eps)
            grid_z_rbf = rbf(gx, gy)
            # replace only inside head (this produces a full smooth field)
            grid_z[mask_circle] = grid_z_rbf[mask_circle]
        except Exception:
            # if RBF fails, leave grid_z as-is (nearest-filled)
            pass

    # prepare for smoothing: replace remaining NaNs (outside head) with zeros
    grid_z_no_nan = np.copy(grid_z)
    grid_z_no_nan[np.isnan(grid_z_no_nan)] = 0.0

    # gaussian smoothing + renormalize
    if smooth_sigma is not None and smooth_sigma > 0:
        grid_z_smooth = gaussian_filter(grid_z_no_nan, sigma=smooth_sigma, mode='nearest')
        mask_float = np.zeros_like(grid_z_no_nan, dtype=float)
        mask_float[mask_circle] = 1.0
        mask_smooth = gaussian_filter(mask_float, sigma=smooth_sigma, mode='nearest')
        valid = mask_smooth > 1e-8
        grid_z_clean = np.full_like(grid_z_smooth, np.nan)
        grid_z_clean[valid] = grid_z_smooth[valid] / mask_smooth[valid]
    else:
        grid_z_clean = grid_z.copy()

    # mask outside head
    grid_z_masked = np.ma.array(grid_z_clean, mask=~mask_circle)

    # vmin/vmax defaults
    data_inside = grid_z_masked[~grid_z_masked.mask]
    if vmin is None:
        vmin = 0.0
    if vmax is None:
        # prefer 99th percentile to avoid extreme outliers dominating colors
        vmax = np.nanpercentile(data_inside, 99) if data_inside.size>0 else np.max(vals_plot)
    if vmax <= vmin:
        vmax = vmin + 1e-12

    # plot: use extend='both' so extreme values don't leave white gaps
    fig, ax = plt.subplots(figsize=(6,6))
    levels = np.linspace(vmin, vmax, n_contours)
    cmap_obj = plt.get_cmap(cmap)
    from matplotlib import colors as mcolors
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    cf = ax.contourf(gx, gy, grid_z_masked, levels=levels, cmap=cmap_obj, vmin=vmin, vmax=vmax, antialiased=True, extend='both')
    cs = ax.contour(gx, gy, grid_z_masked, levels=levels, colors='k', linewidths=0.5, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.2e")

    ax.scatter(xs2, ys2, c='k', s=18, zorder=10, edgecolors='w', linewidths=0.6)

    # labels: sample cleaned grid and pick contrast-aware color
    if show_labels:
        sample_vals = []
        for xi, yi in zip(xs2, ys2):
            ix = int(np.argmin(np.abs(gx_lin - xi)))
            iy = int(np.argmin(np.abs(gy_lin - yi)))
            sval = grid_z_clean[iy, ix] if (0 <= iy < grid_res and 0 <= ix < grid_res) else vmin
            sample_vals.append(sval)
        for xi, yi, nm, sval in zip(xs2, ys2, names, sample_vals):
            sval_use = float(sval) if np.isfinite(sval) else vmin
            color = _label_contrast_color(cmap_obj, norm, sval_use)
            ax.text(xi, yi, nm, fontsize=7, ha='center', va='center', color=color,
                    bbox=dict(facecolor='black' if color=='white' else 'white', alpha=0.45, boxstyle='round,pad=0.12'),
                    zorder=11)

    make_head_circle(ax, radius=0.5, color='k')
    ax.set_aspect('equal')
    ax.set_xlim(gx.min(), gx.max())
    ax.set_ylim(gy.min(), gy.max())
    ax.axis('off')

    # colorbar
    cax = fig.add_axes([0.88, 0.22, 0.03, 0.56])
    cb = fig.colorbar(cf, cax=cax)
    cb.set_label("Mean |HEP| (same units as input)")

    if title:
        ax.set_title(title, pad=20, fontsize=14)

    fig.tight_layout(rect=[0,0,0.86,1.0])
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_png


def main():
    p = argparse.ArgumentParser(description="Compute mean absolute HEP and plot a smoothed topomap.")
    p.add_argument("npz", help="subject HEP npz file (from stage1)")
    p.add_argument("--tmin", type=float, default=0.25)
    p.add_argument("--tmax", type=float, default=0.6)
    p.add_argument("--outdir", default="out_topo")
    p.add_argument("--montages", nargs="+", default=["standard_1020", "standard_1005"])
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--grid_res", type=int, default=300, help="interpolation grid resolution (higher -> smoother)")
    p.add_argument("--n_contours", type=int, default=18)
    p.add_argument("--no-labels", action="store_true", help="turn off channel labels on topomap")
    p.add_argument("--vmin", type=float, default=None, help="colorbar min (default 0)")
    p.add_argument("--vmax", type=float, default=None, help="colorbar max (default 99th percentile)")
    p.add_argument("--interp", nargs="+", default=["cubic","linear","nearest"], help="interpolation methods fallback order")
    p.add_argument("--smooth-sigma", type=float, default=1.5, help="Gaussian smoothing sigma (pixels) applied to interpolated grid")
    p.add_argument("--no-fill-nearest", action="store_true", help="Don't fill NaNs with nearest before smoothing")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    data = np.load(args.npz, allow_pickle=True)

    # find hep array
    if "hep" in data:
        hep = np.array(data["hep"], dtype=float)
    elif "all" in data:
        hep = np.array(data["all"], dtype=float)
    else:
        candidate = None
        for k in data.keys():
            v = data[k]
            if isinstance(v, np.ndarray) and v.ndim == 2:
                candidate = v; break
        if candidate is None:
            raise RuntimeError("Could not find 2D HEP array in NPZ.")
        hep = np.array(candidate, dtype=float)

    # channel names
    ch_names = None
    for key in ("ch_names", "channels", "channel_names"):
        if key in data:
            raw_ch = data[key]
            try:
                ch_names = [str(x) for x in np.atleast_1d(raw_ch).tolist()]
            except Exception:
                ch_names = [str(x) for x in raw_ch]
            break
    if ch_names is None:
        ch_names = [f"Ch{i+1}" for i in range(hep.shape[0])]

    # times
    if "times" in data:
        times = np.array(data["times"], dtype=float)
    else:
        times = np.linspace(-0.2, 0.6, hep.shape[1])

    # compute mean absolute in window
    mean_abs, (i0, i1) = compute_mean_abs(hep, times, tmin=args.tmin, tmax=args.tmax)
    print(f"Computed mean-|HEP| for window [{args.tmin},{args.tmax}] -> indices {i0}:{i1} (n={i1-i0+1})")

    # save CSV
    subj = os.path.splitext(os.path.basename(args.npz))[0].replace("_HEP","")
    csv_path = os.path.join(args.outdir, f"{subj}_hep_absmean.csv")
    with open(csv_path, "w") as f:
        f.write("# channel,mean_abs\n")
        f.write(f"# window={args.tmin}-{args.tmax}s, indices={i0}-{i1}, samples={i1-i0+1}\n")
        f.write("channel,mean_abs\n")
        for ch,val in zip(ch_names, mean_abs):
            f.write(f"{ch},{float(val)}\n")
    print("Saved CSV ->", csv_path)

    # channel positions
    ch_pos = _extract_ch_pos_from_npz(data)

    if len(ch_pos) == 0 and mne is not None:
        mont_map, tried = try_get_montage_positions(ch_names, montages=args.montages)
        print("Montage matching results:", tried)
        for k,v in mont_map.items():
            if k not in ch_pos:
                ch_pos[k] = v

    # fill missing channel positions with median of available ones
    if len(ch_pos) > 0:
        coords = np.stack(list(ch_pos.values()), axis=0)
        med = np.median(coords, axis=0)
        for ch in ch_names:
            if ch not in ch_pos:
                ch_pos[ch] = med.copy()
    else:
        raise RuntimeError("No channel positions available; cannot plot topomap.")

    out_png = os.path.join(args.outdir, f"{subj}_hep_absmean_topomap.png")
    title = f"{subj} — Mean |HEP| ({args.tmin}s–{args.tmax}s)"
    saved = plot_filled_topomap(mean_abs, ch_names, ch_pos, out_png,
                                cmap=args.cmap, grid_res=args.grid_res,
                                n_contours=args.n_contours, title=title,
                                show_labels=not args.no_labels,
                                vmin=args.vmin, vmax=args.vmax,
                                interpolation_methods=tuple(args.interp),
                                smooth_sigma=args.smooth_sigma,
                                fill_nearest=not args.no_fill_nearest)
    print("Saved topomap ->", saved)

if __name__ == "__main__":
    main()
