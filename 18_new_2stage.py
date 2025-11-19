
"""
Stage 2 (robust): load per-subject evoked files (from Stage1 manifest), harmonize channels/times,
stack group arrays and save out_stage2/group_stacked_data.npz

Improvements:
 - Normalize channel names across subjects (strip '-Ref', case, punctuation)
 - Use normalized intersection (or normalized union fallback if intersection small)
 - Choose canonical display names (prefer versions without '-Ref')
 - Order channels by standard_1020 montage when possible
 - Save ch_names (unicode array) and ch_pos (positions) for Stage-3
"""
import os, sys, json, numpy as np
try:
    import mne
except Exception:
    raise SystemExit("mne required. pip install mne")
import matplotlib.pyplot as plt
from typing import List, Dict

# ---------- args ----------
manifest_path = sys.argv[1] if len(sys.argv) > 1 else "out_stage1_sub1/manifest.json"
out_npz = sys.argv[2] if len(sys.argv) > 2 else "out_stage2/group_stacked_data.npz"
OUT_DIR = os.path.dirname(out_npz) or "."
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- helpers ----------
def normalize_channel_name(name: str) -> str:
    """Lowercase, strip common suffixes like '-ref','_ref', trailing +/- and non-alnum."""
    if name is None:
        return ""
    s = str(name).strip()
    # remove common suffixes (case-insensitive)
    for suf in ["-ref", "_ref", " ref"]:
        if s.lower().endswith(suf):
            s = s[:-len(suf)]
            s = s.strip()
    # drop trailing '+' or '-' if present (e.g. 'F3+' )
    if s.endswith("+") or s.endswith("-"):
        s = s[:-1].strip()
    # remove whitespace and punctuation, keep alnum only
    s2 = "".join(ch for ch in s.lower() if ch.isalnum())
    return s2

def try_get_montage_order(mnames=("standard_1020","standard_1005")):
    for m in mnames:
        try:
            mont = mne.channels.make_standard_montage(m)
            return mont, mont.ch_names
        except Exception:
            continue
    return None, []

def build_ch_pos_map(ch_names: List[str], montage_names=("standard_1020","standard_1005")) -> Dict[str, List[float]]:
    """Return mapping canonical_name -> [x,y,z] using montage attempts with normalized matching."""
    ch_pos = {}
    for m in montage_names:
        try:
            mont = mne.channels.make_standard_montage(m)
            mpos = mont.get_positions().get("ch_pos", {})
            # normalized map of montage keys
            norm_map = { normalize_channel_name(k): k for k in mpos.keys() }
            for ch in ch_names:
                n = normalize_channel_name(ch)
                if n in norm_map and ch not in ch_pos:
                    mch = norm_map[n]
                    ch_pos[ch] = np.asarray(mpos[mch], dtype=float).tolist()
        except Exception:
            continue
    return ch_pos

# ---------- load manifest & evokeds ----------
with open(manifest_path, "r") as f:
    manifest = json.load(f)

subjects_meta = manifest.get("subjects", [])
if len(subjects_meta) == 0:
    raise SystemExit("No subjects in manifest.")

evokeds = []
names = []
groups = []
per_subject_orig_chs = []   # list of lists (original names)
per_subject_norm_chs = []   # list of sets (normalized)
times_list = []

for subj in subjects_meta:
    name = subj.get("name", "unknown")
    evoked_file = subj.get("evoked_file") or subj.get("evoked") or subj.get("evoked_path")
    group = subj.get("group") or subj.get("group_label") or subj.get("status") or "ok"
    print(f"Loading {name} -> {evoked_file}")
    if not evoked_file or not os.path.exists(evoked_file):
        print("  MISSING evoked file; skipping")
        continue
    try:
        ev_list = mne.read_evokeds(evoked_file, verbose=False)
        ev = ev_list[0] if isinstance(ev_list, list) and len(ev_list) > 0 else ev_list
    except Exception as e:
        print("  Failed to read evoked:", e)
        continue
    evokeds.append(ev)
    names.append(name)
    groups.append(group)
    orig_chs = list(ev.ch_names)
    per_subject_orig_chs.append(orig_chs)
    per_subject_norm_chs.append(set(normalize_channel_name(ch) for ch in orig_chs))
    times_list.append(ev.times)

if len(evokeds) == 0:
    raise SystemExit("No evokeds loaded successfully.")

n_subj = len(evokeds)
print(f"Loaded {n_subj} evokeds.")

# ---------- normalized intersection / union ----------
# compute normalized intersection
norm_intersection = set(per_subject_norm_chs[0])
for s in per_subject_norm_chs[1:]:
    norm_intersection &= s

print("Normalized intersection size:", len(norm_intersection))

# If intersection too small (<3) fall back to normalized union
if len(norm_intersection) < 3:
    print("Intersection too small — falling back to normalized union of channel names.")
    norm_union = set()
    for s in per_subject_norm_chs:
        norm_union |= s
    norm_candidates = norm_union
else:
    norm_candidates = norm_intersection

# Build canonical display name map: normalized -> preferred original (prefer no '-Ref')
norm_to_canonical = {}
# collect all original names across subjects per normalized token
norm_to_originals = {}
for orig_list in per_subject_orig_chs:
    for orig in orig_list:
        n = normalize_channel_name(orig)
        norm_to_originals.setdefault(n, []).append(orig)

# choose canonical: prefer a name without '-ref' (i.e. one whose normalize didn't strip suffix)
for n, originals in norm_to_originals.items():
    # prefer original that does not contain 'ref' (case-insensitive)
    chosen = None
    for o in originals:
        if 'ref' not in o.lower():
            chosen = o
            break
    if chosen is None:
        chosen = originals[0]
    norm_to_canonical[n] = chosen

# Create ordered channel list using montage order if possible
mont, mont_order = try_get_montage_order()
ordered_norms = []
if mont_order:
    # include only normalized names present in montage order
    for mch in mont_order:
        nn = normalize_channel_name(mch)
        if nn in norm_candidates and nn not in ordered_norms:
            ordered_norms.append(nn)
# append remaining norm_candidates
for nn in sorted(norm_candidates):
    if nn not in ordered_norms:
        ordered_norms.append(nn)

# final canonical channel names in order
common_ch = [ norm_to_canonical.get(nn, nn) for nn in ordered_norms ]

print(f"Final channel count chosen: {len(common_ch)}. Example first 20: {common_ch[:20]}")

# ---------- align times (resample/interpolate) ----------
# choose template times from first evoked
template_times = times_list[0]
need_resample = any(not np.allclose(t, template_times) for t in times_list)
if need_resample:
    print("Time bases differ across subjects; will resample/interpolate to template times.")
    # compute target sfreq
    if len(template_times) > 1:
        dt = template_times[1] - template_times[0]
        target_sfreq = 1.0 / dt
    else:
        target_sfreq = None
else:
    target_sfreq = None

aligned_data = []
for ev in evokeds:
    # build data array in order of common_ch using available channels; if missing -> zeros
    nC = len(common_ch)
    nT = len(template_times)
    data = np.zeros((nC, nT))
    for i, ch in enumerate(common_ch):
        # try find exact ch in ev.ch_names first, else normalized match
        if ch in ev.ch_names:
            idx = ev.ch_names.index(ch)
            if need_resample and target_sfreq is not None:
                ev2 = ev.copy()
                try:
                    ev2.resample(sfreq=target_sfreq, npad='auto')
                    data[i, :] = np.interp(template_times, ev2.times, ev2.data[idx, :])
                except Exception:
                    data[i, :] = np.interp(template_times, ev.times, ev.data[idx, :])
            else:
                if len(ev.times) != nT:
                    data[i, :] = np.interp(template_times, ev.times, ev.data[idx, :])
                else:
                    data[i, :] = ev.data[idx, :]
        else:
            # try normalized match
            norm_map = { normalize_channel_name(o): o for o in ev.ch_names }
            nn = normalize_channel_name(ch)
            if nn in norm_map:
                orig = norm_map[nn]
                idx = ev.ch_names.index(orig)
                if need_resample and target_sfreq is not None:
                    ev2 = ev.copy()
                    try:
                        ev2.resample(sfreq=target_sfreq, npad='auto')
                        data[i,:] = np.interp(template_times, ev2.times, ev2.data[idx, :])
                    except Exception:
                        data[i,:] = np.interp(template_times, ev.times, ev.data[idx, :])
                else:
                    if len(ev.times) != nT:
                        data[i,:] = np.interp(template_times, ev.times, ev.data[idx, :])
                    else:
                        data[i,:] = ev.data[idx, :]
            else:
                # channel missing in this subject -> leave zeros
                data[i,:] = 0.0
    aligned_data.append(data)

all_data = np.stack(aligned_data, axis=0)  # (subjects, channels, times)
print("Stacked data shape:", all_data.shape)

# ---------- montage/ch_pos mapping ----------
ch_pos = {}
try:
    mont_names = ("standard_1020", "standard_1005")
    for m in mont_names:
        try:
            mont = mne.channels.make_standard_montage(m)
            mpos = mont.get_positions().get("ch_pos", {})
            # normalized map of montage keys
            norm_map = { normalize_channel_name(k): k for k in mpos.keys() }
            for ch in common_ch:
                nn = normalize_channel_name(ch)
                if nn in norm_map and ch not in ch_pos:
                    mch = norm_map[nn]
                    ch_pos[ch] = np.asarray(mpos[mch], dtype=float).tolist()
        except Exception:
            continue
    print(f"Montage mapping produced {len(ch_pos)} positions of {len(common_ch)} channels.")
except Exception as e:
    print("Montage mapping failed:", e)

# If ch_pos is empty but montage exists, attempt normalized fallback assigning nearest available pos
if len(ch_pos) == 0:
    try:
        mont, mont_order = try_get_montage_order()
        if mont is not None:
            mpos = mont.get_positions().get("ch_pos", {})
            # build norm->pos
            normpos = { normalize_channel_name(k): np.asarray(v, dtype=float) for k,v in mpos.items() }
            # for each common_ch, if normalized key exists, set it
            for ch in common_ch:
                nn = normalize_channel_name(ch)
                if nn in normpos:
                    ch_pos[ch] = normpos[nn].tolist()
            print("Fallback montage normalized matched positions:", len(ch_pos))
    except Exception:
        pass

# ---------- save NPZ (safe ch_names conversions) ----------
# ensure common_ch is list of str
if isinstance(common_ch, str):
    common_ch = [common_ch]
common_ch = [str(c) for c in common_ch]

# unicode array
maxlen = max((len(s) for s in common_ch), default=1)
ch_names_arr = np.array(common_ch, dtype=f'<U{maxlen}')

# serialize ch_pos to plain dict of lists
ch_pos_serializable = { str(k): (np.array(v).tolist() if not isinstance(v, list) else v) for k,v in ch_pos.items() }

np.savez(
    out_npz,
    all_data=all_data,
    ch_names=ch_names_arr,
    channels=ch_names_arr,
    times=template_times,
    subjects=np.array(names, dtype=object),
    groups=np.array(groups, dtype=object),
    ch_pos=ch_pos_serializable
)
print("Saved stacked NPZ to:", out_npz)
print("ch_names len:", len(ch_names_arr))
if len(ch_pos_serializable) > 0:
    print("ch_pos entries saved:", len(ch_pos_serializable))

# ---------- optional quick visualization ----------
unique_groups = sorted(list(set(groups)))
if len(unique_groups) >= 2:
    g1,g2 = unique_groups[0], unique_groups[1]
    arr = all_data
    mean_g1 = arr[np.array(groups)==g1].mean(axis=0)
    mean_g2 = arr[np.array(groups)==g2].mean(axis=0)
    diff = mean_g1 - mean_g2
    # try topomap if we have info/montage positions for at least some channels
    try:
        ev_template = evokeds[0].copy()
        # ensure ev_template has channel names in same order as ch_names_arr
        ev_template = mne.EvokedArray(diff, ev_template.info, tmin=template_times[0])
        times_to_plot = [template_times[len(template_times)//4], 0.0, template_times[len(template_times)//2]]
        fig = ev_template.plot_topomap(times=times_to_plot, ch_type="eeg", show=False)
        if isinstance(fig, list) and len(fig)>0:
            fig0 = fig[0].get_fig() if hasattr(fig[0], "get_fig") else fig[0]
        else:
            fig0 = fig
        fig0.suptitle(f"{g1} – {g2} difference")
        fig0.savefig(os.path.join(OUT_DIR, "group_difference_topomap.png"), dpi=150)
        plt.close(fig0)
    except Exception as e:
        print("Topomap plotting failed (fallback to channel-mean plot):", e)
        plt.figure()
        plt.plot(template_times, diff.mean(axis=0))
        plt.title(f"{g1} – {g2} difference (mean across channels)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "group_difference_wave.png"), dpi=150)
        plt.close()

print("Stage 2 (normalized) finished.")
