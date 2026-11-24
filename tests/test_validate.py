"""Regression tests: itvcheck must reproduce the published v5 numbers (statsmodels fits + scikit-learn<=1.5 folds).
Run from the package folder:  ITV_DATA=/path/to/data python tests/test_validate.py"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from itvcheck import *
D = os.environ.get("ITV_DATA", "./data").rstrip("/") + "/"
c = pd.read_excel(D + "Combined_SRL_ITV_Han_Weemstra_v2.xlsx")
h = c[(c.data_source == "Han_2025") & c.CV_SRL.notna()].reset_index(drop=True); sp = h.species.str.replace(r"_\d+$", "", regex=True); y = h.CV_SRL.values
r = fit_reml(y, np.ones((len(h), 1)), [h.site_cluster.values, sp.values]); tot = r.vcomp.sum() + r.scale
assert abs(r.vcomp[0] / tot - 0.486) < 0.002 and abs(r.vcomp[1] / tot - 0.197) < 0.002 and abs(r.scale / tot - 0.317) < 0.002, "variance partition differs from v5"
z = (h.MAT - h.MAT.mean()) / h.MAT.std(); X1 = np.c_[np.ones(len(h)), z]; r1 = fit_reml(y, X1, [h.site_cluster.values, sp.values])
assert abs(r1.beta[1] - 0.097) < 0.001 and abs(r1.t[1] - 3.61) < 0.02, "MAT slope / t differ from v5"
m0 = fit_reml(y, np.ones((len(h), 1)), [h.site_cluster.values, sp.values], reml=False); m1 = fit_reml(y, X1, [h.site_cluster.values, sp.values], reml=False)
lr, df, p = lr_test(m0, m1); assert abs(lr - 11.15) < 0.05 and abs(p - 0.0008) < 0.0002, "likelihood-ratio test differs from v5"
X = h.rename(columns={"abl.lat": "abs_lat", "plant_woodiness": "woody", "symbiotic_status": "symb"})[NUMERIC + LOWCARD]
spl = list(LegacyGroupKFold(5).split(X, y, groups=h.site_cluster.values)); o = grouped_oof(X, y, h.site_cluster.values, spl, models={"ElasticNet": make_models()["ElasticNet"]}, n_jobs=1)
r2 = [r2_score(q.y, q.pred) for _, q in o.groupby("fold")]; ref = [-0.537, 0.196, 0.148, -0.498, -0.346]
assert np.allclose(r2, ref, atol=0.005), f"fold R2 {np.round(r2, 3)} differs from v5 {ref}"
print("OK: mixed model, likelihood-ratio test and legacy-fold elastic-net CV reproduce the v5 results.")
