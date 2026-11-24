# Sensitivity of the GRooT overlap exclusion to its (arbitrary) thresholds. Requires GRooT_ITV_harmonized.csv (not part of
# the uploaded files for this revision), so it is guarded; run it in your environment.
import os, numpy as np, pandas as pd, itertools
def groot_overlap_grid(groot_path, han, out_csv="groot_overlap_grid.csv", sources=("Chen", "Salahuddin")):
    if not os.path.exists(groot_path): print("GRooT file not found -> skipped:", groot_path); return None
    g = pd.read_csv(groot_path); g["sp"] = g.species.str.replace("_", " ").str.strip()
    g = g[g.reference.str.startswith(sources)].dropna(subset=["MAT", "MAP"])
    h = han.dropna(subset=["MAT", "MAP"]).copy(); h["sp"] = h.species.str.replace("_", " ")
    hs = h.groupby("site").agg(MAT=("MAT", "first"), MAP=("MAP", "first")); hsp = h.groupby("site").sp.apply(set)
    rows = []
    for (ref, site), q in g.groupby(["reference", "site_cluster"]):
        mat, mp = q.MAT.iloc[0], q.MAP.iloc[0]; spset = set(q.sp)
        for hsite, r in hs.iterrows():
            rows.append({"reference": ref, "groot_site": site, "han_site": hsite, "dMAT": abs(mat - r.MAT), "dMAP_pct": 100 * abs(mp - r.MAP) / r.MAP, "shared": len(spset & hsp[hsite])})
    M = pd.DataFrame(rows); out = []
    for dmat, dmap, k in itertools.product([0.25, 0.5, 1, 2], [2.5, 5, 10, 15, 25], [1, 2, 3]):
        hit = M[(M.dMAT <= dmat) & (M.dMAP_pct <= dmap) & (M.shared >= k)]
        out.append({"max_dMAT": dmat, "max_dMAP_pct": dmap, "min_shared_species": k, "n_sites_flagged": hit.groupby(["reference", "groot_site"]).ngroups, "flagged": "; ".join(sorted({f"{a}|{b}" for a, b in zip(hit.reference, hit.groot_site)}))})
    O = pd.DataFrame(out); O.to_csv(out_csv, index=False); return O
