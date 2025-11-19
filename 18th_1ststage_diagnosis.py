# inspect_evokeds_from_manifest.py
import sys, json, os
try:
    import mne
except Exception:
    raise SystemExit("Install mne: pip install mne")
manifest_path = sys.argv[1] if len(sys.argv)>1 else "out_stage1/manifest.json"
with open(manifest_path) as f:
    mf = json.load(f)
for s in mf.get("subjects", []):
    name = s.get("name")
    evoked_file = s.get("evoked_file") or s.get("evoked") or s.get("evoked_path")
    print("----", name, "->", evoked_file)
    if not evoked_file or not os.path.exists(evoked_file):
        print("  MISSING")
        continue
    evs = mne.read_evokeds(evoked_file, verbose=False)
    ev = evs[0] if isinstance(evs, list) else evs
    print("  ch count:", len(ev.ch_names))
    print("  first 40 chs:", ev.ch_names[:40])
    print("  data shape:", ev.data.shape)
