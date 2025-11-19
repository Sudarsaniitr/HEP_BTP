    
"""
Stage 3: cluster-based permutation analysis on stacked evoked data
(robust adjacency built from montages/ch_names in the NPZ; no Stage2 changes required)

Usage example:
python 18nov_3stage.py out_stage2/group_stacked_data.npz \
    --groupA good --groupB bad --n_permutations 1000 --outdir out_stage3 --temporal_only 0 --n_jobs -1
"""
import os
import argparse
import numpy as np
import pickle
from scipy import stats
from scipy.sparse import csr_matrix, kron, eye as sp_eye
from collections import deque
import itertools
from joblib import Parallel, delayed
import traceback

try:
    import mne
except Exception:
    mne = None

# ---------------- helper functions ----------------
def compute_t_map(XA, XB):
    """
    XA, XB shapes: (nsubj, nchan, nt)
    returns t_map shape (nchan, nt) and df
    uses pooled variance (two-sample t)
    """
    n1 = XA.shape[0]
    n2 = XB.shape[0]
    m1 = XA.mean(axis=0)
    m2 = XB.mean(axis=0)
    v1 = XA.var(axis=0, ddof=1)
    v2 = XB.var(axis=0, ddof=1)
    df = n1 + n2 - 2
    sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / np.maximum(df, 1)
    # avoid division by zero by small epsilon where sp2 is zero
    denom = np.sqrt(sp2 * (1.0 / n1 + 1.0 / n2))
    denom[denom == 0] = np.nan
    tmap = (m1 - m2) / denom
    return tmap, df

def build_spatio_temporal_adjacency(n_ch, n_t, ch_adj=None, temporal_connectivity=1):
    """
    Build spatio-temporal adjacency matrix as sparse csr matrix:
      A = kron(ch_adj, I_time) + kron(I_ch, temporal_adj)
    ch_adj: (n_ch, n_ch) boolean / int adjacency matrix. If None, create identity (no spatial adjacency).
    temporal_connectivity: number of neighbors on each side -> banded adjacency on time axis
    """
    if ch_adj is None:
        ch_adj = np.eye(n_ch, dtype=int)
    ch_adj = (ch_adj != 0).astype(int)
    rows = []
    cols = []
    for i in range(n_t):
        for d in range(1, temporal_connectivity + 1):
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

def find_clusters_from_mask(mask_bool, adj_csr):
    """
    mask_bool: flattened boolean mask (n_nodes,)
    adj_csr: csr adjacency matrix (n_nodes x n_nodes)
    returns list of arrays of node indices (each cluster)
    """
    n_nodes = mask_bool.size
    supra_nodes = np.where(mask_bool)[0]
    if supra_nodes.size == 0:
        return []
    indptr = adj_csr.indptr
    indices = adj_csr.indices
    supra_set = set(supra_nodes.tolist())
    visited = set()
    clusters = []
    for start_node in supra_nodes:
        if start_node in visited:
            continue
        q = deque([start_node])
        visited.add(start_node)
        comp = []
        while q:
            u = q.popleft()
            comp.append(u)
            for idx in range(indptr[u], indptr[u+1]):
                v = indices[idx]
                if v in supra_set and v not in visited:
                    visited.add(v)
                    q.append(v)
        clusters.append(np.array(comp, dtype=int))
    return clusters

def permutation_mass_single(combo, all_data, n_total, n_groupA, t_crit, adj_csr):
    labels_bool = np.zeros(n_total, dtype=bool)
    labels_bool[list(combo)] = True
    XA = all_data[labels_bool]    # shape (nA, ch, t)
    XB = all_data[~labels_bool]
    tmap, _ = compute_t_map(XA, XB)
    flat = tmap.reshape(-1)
    mask_perm = np.abs(flat) > t_crit
    if not mask_perm.any():
        return 0.0
    clusters = find_clusters_from_mask(mask_perm, adj_csr)
    if len(clusters) == 0:
        return 0.0
    masses = [flat[c].sum() for c in clusters]
    return float(np.max(masses))

# ---------------- adjacency helpers ----------------
def normalize_name(s):
    return ''.join(ch for ch in str(s).lower() if ch.isalnum())

def build_ch_adjacency_from_positions(ch_names, ch_pos_dict, n_neighbors=4):
    """
    Build adjacency (n_ch x n_ch) from ch_pos_dict mapping channel->(x,y,z).
    Missing channels are filled by nearest valid channel coordinates.
    """
    n_ch = len(ch_names)
    coords = np.zeros((n_ch, 3))
    valid = np.zeros(n_ch, dtype=bool)
    for i, ch in enumerate(ch_names):
        if ch in ch_pos_dict:
            coords[i, :] = np.asarray(ch_pos_dict[ch], dtype=float)
            valid[i] = True
    if valid.sum() == 0:
        # nothing we can do
        return np.eye(n_ch, dtype=int)
    # fill missing coords by nearest valid
    valid_idx = np.where(valid)[0]
    for i in range(n_ch):
        if not valid[i]:
            # compute distances to valid coords
            d = np.linalg.norm(coords[valid_idx] - coords[i], axis=1)
            coords[i] = coords[valid_idx[np.argmin(d)]]
            valid[i] = True
    D = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    adj = np.zeros((n_ch, n_ch), dtype=int)
    k = min(n_neighbors, n_ch - 1)
    for i in range(n_ch):
        idx = np.argsort(D[i])[:k]
        adj[i, idx] = 1
    adj = ((adj + adj.T) > 0).astype(int)
    np.fill_diagonal(adj, 0)
    return adj

def try_get_ch_pos_from_montages(ch_names, montages_to_try=("standard_1020", "standard_1005")):
    """
    Try to obtain channel positions mapping for ch_names by checking standard montages.
    Returns (ch_pos_dict, tried_list) where tried_list is [(montage_name, matched_count), ...]
    """
    ch_pos = {}
    tried = []
    for mname in montages_to_try:
        try:
            mont = mne.channels.make_standard_montage(mname)
            mpos = mont.get_positions().get('ch_pos', {})
            matched = 0
            # exact matches first
            for ch in ch_names:
                if ch in mpos:
                    ch_pos[ch] = np.asarray(mpos[ch], dtype=float)
                    matched += 1
            # normalized matches to fill more
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
            continue
    return ch_pos, tried

# ---------------- main ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", help="stacked npz produced by stage 2 (npz with keys all_data, groups, subjects, ch_names, times)")
    p.add_argument("--groupA", default=None, help="name of group A (e.g. good)")
    p.add_argument("--groupB", default=None, help="name of group B (e.g. bad)")
    p.add_argument("--n_permutations", type=int, default=1000)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--chunk_size", type=int, default=200)
    p.add_argument("--outdir", default="out_stage3", help="output directory")
    p.add_argument("--temporal_only", type=int, choices=[0,1], default=0, help="1 = only temporal adjacency (no spatial adjacency)")
    p.add_argument("--n_neighbors", type=int, default=4, help="k for k-NN spatial adjacency")
    p.add_argument("--verbose", type=int, choices=[0,1], default=1)
    args = p.parse_args()

    npz_path = args.npz
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    if args.verbose:
        print("Loading stacked data from", npz_path)

    # ---------------- Robust NPZ loader for channel names ----------------
    npz = np.load(npz_path, allow_pickle=True)
    required_min = ["all_data", "groups", "subjects", "times"]
    for k in required_min:
        if k not in npz:
            raise RuntimeError(f"NPZ missing required key: {k}. Found keys: {list(npz.keys())}")

    def _unpack_array_like(x):
        # Handles: list, tuple, ndarray, object-dtype array containing a list/ndarray
        if isinstance(x, (list, tuple)):
            return list(x)
        import numpy as _np
        if isinstance(x, _np.ndarray):
            # if it's object-dtype with single element that is the real array -> unpack
            if x.dtype == object and x.size == 1:
                inner = x[0]
                # try to convert inner to list
                if isinstance(inner, (list, tuple)):
                    return list(inner)
                try:
                    return [str(u) for u in inner.tolist()]
                except Exception:
                    return [str(inner)]
            # normal ndarray of strings
            try:
                return [str(u) for u in x.tolist()]
            except Exception:
                return [str(u) for u in x.flatten().tolist()]
        # fallback
        return [str(x)]

    if "ch_names" in npz:
        raw_ch = npz["ch_names"]
        ch_names = _unpack_array_like(raw_ch)
        if len(ch_names) == 0:
            raise RuntimeError("Loaded 'ch_names' is empty.")
    elif "channels" in npz:
        raw_ch = npz["channels"]
        ch_names = _unpack_array_like(raw_ch)
        if len(ch_names) == 0:
            raise RuntimeError("Loaded 'channels' is empty.")
    else:
        raise RuntimeError("NPZ contains neither 'ch_names' nor 'channels'.")

    # Deduplicate / sanitize order-preserving
    seen = set()
    ch_names = [c for c in ch_names if not (c in seen or seen.add(c))]

    if args.verbose:
        print(f"Resolved {len(ch_names)} channel names from NPZ (example first 20): {ch_names[:20]}")

    # ---------------- Extract rest of required fields ----------------
    all_data = npz["all_data"]      # (subjects, ch, time)
    groups = npz["groups"].astype(str)
    subjects = npz["subjects"].astype(str)
    times = npz["times"].astype(float)

    # ensure ch_names length matches data second dimension if possible
    try:
        n_subj, n_ch_data, n_t = all_data.shape
        if len(ch_names) != n_ch_data:
            if args.verbose:
                print(f"Warning: number of resolved ch_names ({len(ch_names)}) != data channels ({n_ch_data}).")
            # If lengths mismatch, attempt to continue using data shape for adjacency/indices,
            # but ch_names must align: if ch_names shorter, pad with placeholders; if longer, truncate.
            if len(ch_names) < n_ch_data:
                extra = [f"CH_PAD{i}" for i in range(n_ch_data - len(ch_names))]
                ch_names = ch_names + extra
                if args.verbose:
                    print(f"Padded ch_names to length {len(ch_names)}")
            elif len(ch_names) > n_ch_data:
                ch_names = ch_names[:n_ch_data]
                if args.verbose:
                    print(f"Truncated ch_names to length {len(ch_names)}")
    except Exception:
        # fallback if all_data has unexpected shape
        n_subj = len(subjects)
        n_ch = len(ch_names)
        n_t = len(times)

    n_subj, n_ch, n_t = all_data.shape
    if args.verbose:
        print(f"Loaded stacked data shape: {all_data.shape}")
        print("Groups found:", np.unique(groups).tolist())

    # pick groups
    unique_groups = sorted(list(set(groups.tolist())))
    if args.groupA is None or args.groupB is None:
        if len(unique_groups) == 2:
            grpA, grpB = unique_groups[0], unique_groups[1]
            if args.verbose:
                print("Auto-selecting groups:", grpA, grpB)
        else:
            raise RuntimeError(f"Please supply --groupA and --groupB; found groups: {unique_groups}")
    else:
        grpA, grpB = args.groupA, args.groupB

    idxA = [i for i,g in enumerate(groups.tolist()) if g == grpA]
    idxB = [i for i,g in enumerate(groups.tolist()) if g == grpB]
    if len(idxA) == 0 or len(idxB) == 0:
        raise RuntimeError(f"Could not find subjects for groups: {grpA} (n={len(idxA)}), {grpB} (n={len(idxB)}). Available groups: {unique_groups}")

    XA = all_data[idxA]
    XB = all_data[idxB]
    if args.verbose:
        print(f"Group {grpA}: {len(idxA)} subjects; Group {grpB}: {len(idxB)} subjects")

    # compute observed t-map
    t_obs, df = compute_t_map(XA, XB)
    if args.verbose:
        print("Computed observed t-map; df =", df)
    if df <= 0:
        print("Warning: degrees of freedom <= 0 (too few subjects). t_crit will be NaN.")
    t_crit = stats.t.ppf(1 - 0.05 / 2.0, df) if df > 0 else np.nan
    if args.verbose:
        print("cluster-forming t_crit:", t_crit)

    # ------------ adjacency construction ------------
    if args.temporal_only:
        if args.verbose:
            print("Building temporal-only adjacency.")
        adj = build_spatio_temporal_adjacency(n_ch, n_t, ch_adj=np.eye(n_ch, dtype=int), temporal_connectivity=1)
    else:
        ch_adj = None
        ch_pos = {}

        # 1) if stage2 saved ch_pos in NPZ (maybe not in your case) - prefer it
        if 'ch_pos' in npz:
            try:
                saved = npz['ch_pos'].tolist()
                if isinstance(saved, dict) and len(saved) > 0:
                    ch_pos = {str(k): np.asarray(v, dtype=float) for k,v in saved.items()}
                    missing = [ch for ch in ch_names if ch not in ch_pos]
                    if args.verbose:
                        print(f"Loaded ch_pos from NPZ: {len(ch_pos)} entries, {len(missing)} missing.")
                else:
                    if args.verbose:
                        print("ch_pos present in NPZ but not in expected dict form; ignoring.")
            except Exception:
                if args.verbose:
                    print("Failed reading ch_pos from NPZ; will attempt montages.")

        # 2) try montages (standard_1020 then standard_1005) to map positions for ch_names
        try:
            mont_pos_map, tried = try_get_ch_pos_from_montages(ch_names, montages_to_try=("standard_1020","standard_1005"))
            if args.verbose:
                print("Montage matching results:", tried)
            # merge without overwriting existing keys
            for k,v in mont_pos_map.items():
                if k not in ch_pos:
                    ch_pos[k] = v
            if args.verbose:
                print(f"Total channel positions obtained after montage attempts: {len(ch_pos)} / {len(ch_names)}")
        except Exception as e:
            if args.verbose:
                print("Montage-based mapping failed:", e)

        # 3) if still missing, we'll print example missing channels
        missing = [ch for ch in ch_names if ch not in ch_pos]
        if len(missing) > 0 and args.verbose:
            print(f"{len(missing)} channels missing positions after montage attempts. Example missing: {missing[:10]}")

        # 4) If no positions at all, fallback to spatial identity; else build adjacency via k-NN
        if len(ch_pos) == 0:
            if args.verbose:
                print("No channel positions available; falling back to spatial identity adjacency (no spatial neighbors).")
            ch_adj = np.eye(n_ch, dtype=int)
        else:
            # fill any missing channels by nearest valid channel coords inside build_ch_adjacency_from_positions
            ch_adj = build_ch_adjacency_from_positions(ch_names, ch_pos, n_neighbors=args.n_neighbors)

        adj = build_spatio_temporal_adjacency(n_ch, n_t, ch_adj=ch_adj, temporal_connectivity=1)

    # find observed clusters
    flat_obs = t_obs.reshape(-1)
    mask_obs = np.abs(flat_obs) > t_crit
    clusters_obs = find_clusters_from_mask(mask_obs, adj)
    cluster_masses = np.array([flat_obs[c].sum() for c in clusters_obs]) if clusters_obs else np.array([])
    if args.verbose:
        print("Observed clusters:", len(cluster_masses))

    # ------------ permutations (safe sampling) ------------
    n_total = n_subj
    n_groupA = len(idxA)
    from math import comb
    n_combinations = comb(n_total, n_groupA)
    if args.verbose:
        print("Total possible label combinations:", n_combinations)

    if n_combinations <= 200000:
        combos = list(itertools.combinations(range(n_total), n_groupA))
    else:
        # sample unique combos up to n_permutations without enumerating all combinations
        rng = np.random.default_rng(42)
        combos = []
        seen = set()
        target = args.n_permutations
        # limit attempts to avoid infinite loops in tiny groups
        max_attempts = int(max(1e6, target * 20))
        attempts = 0
        while len(combos) < target and attempts < max_attempts:
            c = tuple(sorted(rng.choice(n_total, size=n_groupA, replace=False).tolist()))
            attempts += 1
            if c in seen:
                continue
            seen.add(c)
            combos.append(c)
        if args.verbose:
            print(f"Sampled {len(combos)} unique permutations (attempts={attempts}).")

    total = len(combos)
    if args.verbose:
        print("Permutations to process:", total)

    # checkpointing
    checkpoint_file = os.path.join(outdir, "perm_checkpoint.npy")
    max_cluster_masses = []
    processed = 0
    if os.path.exists(checkpoint_file):
        try:
            saved = np.load(checkpoint_file)
            max_cluster_masses = list(saved.tolist())
            processed = len(max_cluster_masses)
            if args.verbose:
                print(f"Resuming permutation run from checkpoint ({processed} already done)")
        except Exception:
            if args.verbose:
                print("Warning: failed to load checkpoint; starting fresh.")

    def process_chunk(chunk):
        return Parallel(n_jobs=args.n_jobs)(
            delayed(permutation_mass_single)(c, all_data, n_total, n_groupA, t_crit, adj)
            for c in chunk
        )

    # iterate chunks
    for start in range(processed, total, args.chunk_size):
        end = min(total, start + args.chunk_size)
        chunk = combos[start:end]
        if args.verbose:
            print(f"Processing permutations {start+1}..{end} (chunk size {len(chunk)})")
        try:
            results = process_chunk(chunk)
        except Exception as e:
            print("Error during permutation chunk:", e)
            traceback.print_exc()
            results = []
        max_cluster_masses.extend(results)
        # checkpoint
        np.save(checkpoint_file, np.array(max_cluster_masses))
        if args.verbose:
            print(f"  Checkpoint saved ({len(max_cluster_masses)} permutations done)")

    max_cluster_masses = np.array(max_cluster_masses)
    if args.verbose:
        print("Finished permutations; null distribution length:", len(max_cluster_masses))

    # corrected p-values for observed clusters
    if cluster_masses.size > 0 and len(max_cluster_masses) > 0:
        pvals = np.array([(np.sum(max_cluster_masses >= m) + 1) / (len(max_cluster_masses) + 1) for m in cluster_masses])
    else:
        pvals = np.array([])

    sig_idx = np.where(pvals < 0.05)[0].tolist() if pvals.size else []
    if args.verbose:
        print("Significant cluster indices (p<0.05):", sig_idx)

    # prepare cluster info
    def node_to_ch_time(node):
        ch = node // n_t
        tt = node % n_t
        return ch, tt

    clusters_info = []
    for i, nodes in enumerate(clusters_obs):
        ch_idxs = sorted(list({node_to_ch_time(nd)[0] for nd in nodes}))
        t_idxs = sorted(list({node_to_ch_time(nd)[1] for nd in nodes}))
        info = {
            'index': i,
            'mass': float(cluster_masses[i]),
            'pval': float(pvals[i]) if pvals.size else None,
            'channels_idx': ch_idxs,
            'time_idx': t_idxs,
            'channels': [ch_names[c] for c in ch_idxs],
            'time_range_s': (float(times[t_idxs[0]]), float(times[t_idxs[-1]]))
        }
        clusters_info.append(info)

    out = {
        't_obs': t_obs,
        'df': df,
        't_crit': float(t_crit) if not np.isnan(t_crit) else None,
        'clusters_info': clusters_info,
        'cluster_masses': cluster_masses,
        'cluster_pvals': pvals,
        'max_cluster_masses': max_cluster_masses,
        'groups': groups.tolist(),
        'subjects': subjects.tolist(),
        'ch_names': ch_names,
        'times': times
    }
    out_file = os.path.join(outdir, "results.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(out, f)
    if args.verbose:
        print("Saved results to", out_file)
        if len(sig_idx) == 0:
            print("No significant clusters found.")
        else:
            print("Significant clusters:", sig_idx)

if __name__ == "__main__":
    main()
