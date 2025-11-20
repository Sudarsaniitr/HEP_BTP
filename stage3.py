
"""
Stage-3:
cluster permutation test for stacked HEP data.

Command to run this code : 

    python stage3.py out_stage2_final/group_stacked_data.npz  --groupA good --groupB bad --n_permutations 3500 --n_jobs 4 --outdir out_stage3_final
"""
import os
import argparse
import numpy as np
import pickle
from scipy import stats
from scipy.sparse import csr_matrix, kron, eye as sp_eye
from collections import deque
from joblib import Parallel, delayed
from tqdm import tqdm
import itertools
import math
import traceback

# optional plotting
try:
    import mne
    import matplotlib.pyplot as plt
except Exception:
    mne = None
    plt = None

# ---------------- utility ----------------
def normalize_name(s):
    if s is None: return ""
    s = str(s)
    s = s.strip()
    for suf in ["-ref", "_ref", " ref"]:
        if s.lower().endswith(suf):
            s = s[:-len(suf)]
    s = ''.join(ch for ch in s if ch.isalnum()).lower()
    return s

def _unpack_array_like(x):
    """Unpack NPZ-stored channel arrays robustly to list[str]."""
    import numpy as _np
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, _np.ndarray):
        if x.dtype == object and x.size == 1:
            inner = x[0]
            if isinstance(inner, (list, tuple, _np.ndarray)):
                return [str(u) for u in inner]
            return [str(inner)]
        try:
            return [str(u) for u in x.tolist()]
        except Exception:
            return [str(u) for u in x.flatten().tolist()]
    return [str(x)]

# -------------- t-map --------------------
def compute_t_map(XA, XB):
    # XA/XB: (nsubj, nchan, nt)
    n1 = XA.shape[0]; n2 = XB.shape[0]
    m1 = XA.mean(axis=0); m2 = XB.mean(axis=0)
    v1 = XA.var(axis=0, ddof=1); v2 = XB.var(axis=0, ddof=1)
    df = n1 + n2 - 2
    sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / np.maximum(df, 1)
    denom = np.sqrt(sp2 * (1.0/n1 + 1.0/n2))
    denom[denom == 0] = np.nan
    tmap = (m1 - m2) / denom
    return tmap, int(df)

# -------------- adjacency ----------------
def build_ch_adj_from_positions(ch_pos, ch_names, n_neighbors=4):
    n = len(ch_names)
    coords = np.zeros((n,3), dtype=float)
    valid = np.zeros(n, dtype=bool)
    # try direct keys then normalized match
    pos_keys = {k:np.asarray(v, dtype=float) for k,v in (ch_pos.items() if isinstance(ch_pos, dict) else [])}
    norm_keys = {normalize_name(k): k for k in pos_keys.keys()}
    for i,ch in enumerate(ch_names):
        if ch in pos_keys:
            coords[i] = pos_keys[ch]; valid[i] = True
        else:
            key = normalize_name(ch)
            if key in norm_keys:
                coords[i] = pos_keys[norm_keys[key]]; valid[i] = True
    if valid.sum() == 0:
        return np.eye(n, dtype=int), coords
    valid_idx = np.where(valid)[0]
    for i in range(n):
        if not valid[i]:
            d = np.linalg.norm(coords[valid_idx] - coords[i], axis=1)
            coords[i] = coords[valid_idx[np.argmin(d)]]
            valid[i] = True
    D = np.sqrt(((coords[:,None,:] - coords[None,:,:])**2).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    adj = np.zeros((n,n), dtype=int)
    k = min(n_neighbors, n-1)
    for i in range(n):
        idx = np.argsort(D[i])[:k]
        adj[i, idx] = 1
    adj = ((adj + adj.T) > 0).astype(int)
    np.fill_diagonal(adj, 0)
    return adj, coords

def build_spatio_temporal_adjacency(n_ch, n_t, ch_adj=None, temporal_connectivity=1):
    if ch_adj is None:
        ch_adj = np.eye(n_ch, dtype=int)
    ch_adj = (ch_adj != 0).astype(int)
    # temporal adjacency (simple neighbor connectivity)
    rows=[]; cols=[]
    for i in range(n_t):
        for d in range(1, temporal_connectivity+1):
            j = i + d
            if j < n_t:
                rows.append(i); cols.append(j)
                rows.append(j); cols.append(i)
    data = np.ones(len(rows), dtype=int)
    temporal_adj = csr_matrix((data, (rows, cols)), shape=(n_t, n_t))
    A_spatio = kron(csr_matrix(ch_adj), sp_eye(n_t, format='csr'), format='csr')
    A_temp = kron(sp_eye(n_ch, format='csr'), temporal_adj, format='csr')
    A = (A_spatio + A_temp).astype(bool).astype(int).tocsr()
    return A

# -------------- clusters ----------------
def find_clusters_from_mask(mask_bool, adj_csr):
    supra = np.where(mask_bool)[0]
    if supra.size == 0:
        return []
    indptr = adj_csr.indptr; indices = adj_csr.indices
    supra_set = set(supra.tolist())
    visited = set(); clusters=[]
    from collections import deque
    for s in supra:
        if s in visited: continue
        q = deque([s]); visited.add(s); comp=[]
        while q:
            u = q.popleft(); comp.append(u)
            for ii in range(indptr[u], indptr[u+1]):
                v = indices[ii]
                if v in supra_set and v not in visited:
                    visited.add(v); q.append(v)
        clusters.append(np.array(comp, dtype=int))
    return clusters

# permutation worker (single combo)
def permutation_mass_single(combo, all_data, n_total, n_groupA, t_crit_pos, t_crit_neg, adj_csr):
    labels_bool = np.zeros(n_total, dtype=bool)
    labels_bool[list(combo)] = True
    XA = all_data[labels_bool]; XB = all_data[~labels_bool]
    tmap, _ = compute_t_map(XA, XB)
    flat = tmap.reshape(-1)
    # positive clusters
    mask_pos = flat > t_crit_pos
    mass_pos = 0.0
    if mask_pos.any():
        clusters_pos = find_clusters_from_mask(mask_pos, adj_csr)
        if clusters_pos:
            mass_pos = max(flat[c].sum() for c in clusters_pos)
    # negative clusters (we record absolute of negative cluster mass)
    mask_neg = flat < t_crit_neg
    mass_neg = 0.0
    if mask_neg.any():
        clusters_neg = find_clusters_from_mask(mask_neg, adj_csr)
        if clusters_neg:
            # negative masses are negative sums -> take minimum (most negative) and return abs
            negs = [flat[c].sum() for c in clusters_neg]
            mass_neg = -min(negs)  # positive number representing magnitude
    return mass_pos, mass_neg

# ------------------- main -------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", help="stacked npz produced by stage 2 (npz with keys all_data, groups, subjects, ch_names, times[, ch_pos])")
    p.add_argument("--groupA", default=None)
    p.add_argument("--groupB", default=None)
    p.add_argument("--n_permutations", type=int, default=1000)
    p.add_argument("--n_jobs", type=int, default=1)
    p.add_argument("--outdir", default="out_stage3_fixed")
    p.add_argument("--temporal_only", action="store_true", help="do temporal-only adjacency")
    p.add_argument("--n_neighbors", type=int, default=4)
    p.add_argument("--temporal_connectivity", type=int, default=1)
    p.add_argument("--verbose", type=int, default=1)
    args = p.parse_args()
#   Given in the terminal, refer to the top of the file for command to run this code
    os.makedirs(args.outdir, exist_ok=True)
    if args.verbose: print("Loading stacked data from", args.npz)
    npz = np.load(args.npz, allow_pickle=True)

    if "ch_names" in npz:
        ch_names = _unpack_array_like(npz["ch_names"])
    elif "channels" in npz:
        ch_names = _unpack_array_like(npz["channels"])
    else:
        raise RuntimeError("NPZ lacks 'ch_names' or 'channels'.")

 
    seen=set(); cn=[]
    for c in ch_names:
        if c not in seen:
            cn.append(c); seen.add(c)
    ch_names = cn

    all_data = np.array(npz["all_data"])
    groups = np.array(npz["groups"]).astype(str)
    subjects = np.array(npz["subjects"]).astype(str)
    times = np.array(npz["times"]).astype(float)
    ch_pos = {}
    if "ch_pos" in npz:
        try:
            stored = npz["ch_pos"].tolist()
            if isinstance(stored, dict):
                ch_pos = {str(k): np.asarray(v, dtype=float) for k,v in stored.items()}
        except Exception:
            ch_pos = {}

    S, C, T = all_data.shape
    if args.verbose:
        print(f"Loaded stacked data: {all_data.shape}; Groups present: {sorted(list(set(groups)))}")

    # choose groups
    unique_groups = sorted(list(set(groups.tolist())))
    if args.groupA is None or args.groupB is None:
        if len(unique_groups) == 2:
            grpA, grpB = unique_groups[0], unique_groups[1]
            if args.verbose: print("Auto-selecting groups:", grpA, grpB)
        else:
            raise RuntimeError("Please provide --groupA and --groupB when >2 groups present.")
    else:
        grpA, grpB = args.groupA, args.groupB

    idxA = np.where(groups == grpA)[0].tolist()
    idxB = np.where(groups == grpB)[0].tolist()
    if len(idxA)==0 or len(idxB)==0:
        raise RuntimeError(f"Groups not found. Available groups: {unique_groups}")

    XA = all_data[idxA]; XB = all_data[idxB]
    if args.verbose: print(f"Group {grpA}: {len(idxA)} subjects; {grpB}: {len(idxB)} subjects")

    # observed t-map
    t_obs, df = compute_t_map(XA, XB)
    t_crit_two = stats.t.ppf(1 - 0.05/2, df) if df>0 else np.nan
    # we will use one-sided thresholds for pos/neg:
    t_crit_pos = stats.t.ppf(1 - 0.025, df) if df>0 else np.nan  # ~two-sided split -> similar to two-sided but independent
    t_crit_neg = stats.t.ppf(0.025, df) if df>0 else np.nan

    if args.verbose:
        print(f"Computed t-map; df = {df}; two-sided t_crit={t_crit_two:.3f}")

    # adjacency (spatio-temporal)
    if args.temporal_only:
        ch_adj = np.eye(C, dtype=int)
        coords = None
    else:
        ch_adj = None; coords = None
        # try MNE adjacency from an info object if user stored `info` (not required)
        if mne is not None and "info" in npz:
            try:
                info = npz["info"].tolist()
                if isinstance(info, dict):
                    info_obj = mne.create_info(ch_names, sfreq=1000.0, ch_types='eeg')
                    # try find adjacency (may fail if montage mismatch)
                    ch_adj_mne, ch_names_adj = mne.channels.find_ch_adjacency(info_obj, ch_type='eeg')
                    if ch_adj_mne is not None:
                        ch_adj = (ch_adj_mne != 0).astype(int)
                else:
                    pass
            except Exception:
                ch_adj = None

        if ch_adj is None:
            # try using ch_pos
            if len(ch_pos) > 0:
                try:
                    ch_adj, coords = build_ch_adj_from_positions(ch_pos, ch_names, n_neighbors=args.n_neighbors)
                except Exception:
                    ch_adj = np.eye(C, dtype=int)
            else:
                # attempt to use montages by matching names to standard 1020
                if mne is not None:
                    try:
                        mont = mne.channels.make_standard_montage("standard_1020")
                        mpos = mont.get_positions().get("ch_pos", {})
                        # try normalized mapping
                        mapping = {}
                        for ch in ch_names:
                            key = normalize_name(ch)
                            for mk in mpos:
                                if normalize_name(mk) == key:
                                    mapping[ch] = np.asarray(mpos[mk], dtype=float); break
                        if len(mapping) >= max(3, C//2):
                            ch_adj, coords = build_ch_adj_from_positions(mapping, ch_names, n_neighbors=args.n_neighbors)
                            # ch_adj returned as tuple if mapping used (ensure shape)
                            if isinstance(ch_adj, tuple):
                                ch_adj, coords = ch_adj
                        else:
                            ch_adj = np.eye(C, dtype=int)
                    except Exception:
                        ch_adj = np.eye(C, dtype=int)
                else:
                    ch_adj = np.eye(C, dtype=int)

    adj = build_spatio_temporal_adjacency(C, T, ch_adj=ch_adj, temporal_connectivity=args.temporal_connectivity)

    # observed clusters for pos and neg
    flat_obs = t_obs.reshape(-1)
    mask_pos_obs = flat_obs > t_crit_pos
    mask_neg_obs = flat_obs < t_crit_neg

    clusters_pos_obs = find_clusters_from_mask(mask_pos_obs, adj)
    clusters_neg_obs = find_clusters_from_mask(mask_neg_obs, adj)

    cluster_masses_pos = np.array([flat_obs[c].sum() for c in clusters_pos_obs]) if clusters_pos_obs else np.array([])
    cluster_masses_neg = np.array([-flat_obs[c].sum() for c in clusters_neg_obs]) if clusters_neg_obs else np.array([])

    if args.verbose:
        print("Observed positive clusters:", len(cluster_masses_pos), "; negative clusters:", len(cluster_masses_neg))

    # ----- permutations -----
    n_total = S; nA = len(idxA)
    max_possible = math.comb(n_total, nA) if n_total >= nA else 0
    if args.verbose:
        print("Total possible label combinations:", max_possible)

    if max_possible <= args.n_permutations:
        combos = list(itertools.combinations(range(n_total), nA))
        if args.verbose: print("Enumerating all combinations:", len(combos))
    else:
        rng = np.random.default_rng(42)
        combos = []
        seen = set()
        target = args.n_permutations
        max_attempts = int(max(1e6, target * 20))
        attempts = 0
        while len(combos) < target and attempts < max_attempts:
            c = tuple(sorted(rng.choice(n_total, size=nA, replace=False).tolist()))
            attempts += 1
            if c in seen: continue
            seen.add(c); combos.append(c)
        if args.verbose:
            print(f"Sampled {len(combos)} unique permutations (attempts={attempts}).")

    # checkpointing
    checkpoint_file = os.path.join(args.outdir, "perm_checkpoint.npy")
    pos_maxs = []; neg_maxs = []
    processed = 0
    if os.path.exists(checkpoint_file):
        try:
            saved = np.load(checkpoint_file, allow_pickle=True)
            pos_maxs = list(saved.tolist()[0])
            neg_maxs = list(saved.tolist()[1])
            processed = len(pos_maxs)
            if args.verbose: print(f"Resuming from checkpoint ({processed} permutations done).")
        except Exception:
            pos_maxs=[]; neg_maxs=[]; processed=0

    # function to process a chunk (joblib)
    def process_chunk(chunk):
        out = Parallel(n_jobs=args.n_jobs)(
            delayed(permutation_mass_single)(c, all_data, n_total, nA, t_crit_pos, t_crit_neg, adj)
            for c in chunk
        )
        return out

    # iterate
    total = len(combos)
    if args.verbose:
        pbar = tqdm(total=total, desc="Permutations", unit="perm")
    for start in range(processed, total, 200):
        end = min(total, start+200)
        chunk = combos[start:end]
        try:
            results = process_chunk(chunk)
        except Exception as e:
            print("Permutation chunk error:", e); traceback.print_exc(); results=[]
        for (mp, mn) in results:
            pos_maxs.append(float(mp)); neg_maxs.append(float(mn))
        # checkpoint
        np.save(checkpoint_file, np.array([pos_maxs, neg_maxs], dtype=object))
        if args.verbose:
            pbar.update(len(results))
    if args.verbose:
        pbar.close()

    pos_maxs = np.array(pos_maxs); neg_maxs = np.array(neg_maxs)
    if args.verbose:
        print("Finished permutations; null lengths:", len(pos_maxs), len(neg_maxs))

    # p-values: for each observed cluster mass, p = (count(null >= obs) + 1)/(N+1)
    def pv(m, null_arr):
        if null_arr.size == 0:
            return 1.0
        return (np.sum(null_arr >= m) + 1) / (len(null_arr) + 1)

    pvals_pos = np.array([pv(m, pos_maxs) for m in cluster_masses_pos]) if cluster_masses_pos.size else np.array([])
    pvals_neg = np.array([pv(m, neg_maxs) for m in cluster_masses_neg]) if cluster_masses_neg.size else np.array([])

    # build clusters_info list (unify pos + neg)
    clusters_info = []
    def node_to_ch_time(node):
        ch = node // T; tt = node % T; return ch, tt

    idx = 0
    for i, comp in enumerate(clusters_pos_obs):
        ch_idxs = sorted({node_to_ch_time(n)[0] for n in comp})
        t_idxs = sorted({node_to_ch_time(n)[1] for n in comp})
        info = {
            "index": idx, "sign": "positive",
            "mass": float(cluster_masses_pos[i]),
            "pval": float(pvals_pos[i]) if pvals_pos.size else None,
            "channels_idx": ch_idxs, "time_idx": t_idxs,
            "channels": [ch_names[c] for c in ch_idxs],
            "time_range": (float(times[t_idxs[0]]), float(times[t_idxs[-1]]))
        }
        clusters_info.append(info); idx += 1
    for i, comp in enumerate(clusters_neg_obs):
        ch_idxs = sorted({node_to_ch_time(n)[0] for n in comp})
        t_idxs = sorted({node_to_ch_time(n)[1] for n in comp})
        info = {
            "index": idx, "sign": "negative",
            "mass": float(cluster_masses_neg[i]),
            "pval": float(pvals_neg[i]) if pvals_neg.size else None,
            "channels_idx": ch_idxs, "time_idx": t_idxs,
            "channels": [ch_names[c] for c in ch_idxs],
            "time_range": (float(times[t_idxs[0]]), float(times[t_idxs[-1]]))
        }
        clusters_info.append(info); idx += 1

    out = dict(
        t_obs = t_obs,
        df = int(df),
        t_crit_two = float(t_crit_two),
        t_crit_pos = float(t_crit_pos),
        t_crit_neg = float(t_crit_neg),
        clusters_info = clusters_info,
        cluster_masses_pos = cluster_masses_pos,
        cluster_masses_neg = cluster_masses_neg,
        pos_null = pos_maxs, neg_null = neg_maxs,
        groups = groups.tolist(), subjects = subjects.tolist(),
        ch_names = ch_names, times = times
    )

    # Results are being saved here

    out_file = os.path.join(args.outdir, "results.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(out, f)
    print("Saved:", out_file)

    #  plotting of t-map and clusters
    try:
        if plt is not None and mne is not None:
            fig, ax = plt.subplots(1,1, figsize=(10,4))
            im = ax.imshow(t_obs, aspect='auto', origin='lower', extent=[times[0], times[-1], 0, len(ch_names)])
            ax.set_yticks(np.arange(len(ch_names))+0.5); ax.set_yticklabels(ch_names)
            ax.set_xlabel("Time (s)"); ax.set_title("t_obs (channels x time)")
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, "tmap_image.png"))
            plt.close(fig)
            # try topomap snapshots at cluster midpoints for significant clusters
            for cinfo in clusters_info:
                if cinfo["pval"] is None: continue
                if cinfo["pval"] < 0.05:
                    t0 = 0.5*(cinfo["time_range"][0] + cinfo["time_range"][1])
                    # find nearest time index
                    ti = int(np.argmin(np.abs(times - t0)))
                    # attempt to build info with montage
                    try:
                        info = mne.create_info(ch_names, sfreq=1000., ch_types='eeg')
                        montage = mne.channels.make_standard_montage("standard_1020")
                        info.set_montage(montage, on_missing='ignore')
                        ev = mne.EvokedArray(t_obs[:,ti][:,None], info, tmin=0.0)
                        fig = ev.plot_topomap(times=[0.0], show=False)
                        # save
                        fig0 = fig
                        if isinstance(fig, list) and len(fig)>0:
                            fig0 = fig[0]
                        fig0.suptitle(f"cluster {cinfo['index']} topo")
                        fig0.savefig(os.path.join(args.outdir, f"cluster_{cinfo['index']}_topo.png"))
                        plt.close(fig0)
                    except Exception as e:
                        # ignore topomap failure
                        pass
    except Exception:
        pass

if __name__ == "__main__":
    main()
