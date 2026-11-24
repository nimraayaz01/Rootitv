"""itvcheck: diagnostics for machine learning studies on compiled trait variability data.
Pure numpy/scipy/scikit learn (no statsmodels). Components:
  fit_reml          crossed-random-effects linear mixed model by REML/ML, optional known measurement variance
  cv_sampling_var   sampling variance of a coefficient of variation; bias correction
  design_sim        calibrated null simulation with site AND species structure, correlated predictors,
                    heteroscedastic sampling noise and a fully nested hyperparameter search
  grouped_oof       nested, site grouped (or leave-one-site-out) out-of-fold predictions
  skill_table       pooled skill against the training mean with a site-cluster bootstrap
"""
import numpy as np, pandas as pd
from scipy import optimize, linalg, stats

# ------------------------------------------------------------------ mixed model
class LMMResult:
    pass

def _design(groups):
    codes, uniq = pd.factorize(pd.Series(groups)); Z = np.zeros((len(codes), len(uniq))); Z[np.arange(len(codes)), codes] = 1.0; return Z

def fit_reml(y, X, group_list, meas_var=None, reml=True, starts=3):
    """y (n,), X (n,p) incl. intercept, group_list: list of label arrays (one random intercept each).
    meas_var: known sampling variance per row (added to the diagonal)."""
    y = np.asarray(y, float); X = np.asarray(X, float); n, p = X.shape
    ZZ = [(lambda Z: Z @ Z.T)(_design(g)) for g in group_list]; m = np.zeros(n) if meas_var is None else np.asarray(meas_var, float)
    def parts(th):
        s = np.exp(th); V = np.diag(m + s[-1]) 
        for k, K in enumerate(ZZ): V = V + s[k] * K
        c = linalg.cho_factor(V, lower=True); return V, c
    def nll(th):
        try: V, c = parts(th)
        except Exception: return 1e10
        VX = linalg.cho_solve(c, X); Vy = linalg.cho_solve(c, y); XtVX = X.T @ VX
        try: cb = linalg.cho_factor(XtVX, lower=True)
        except Exception: return 1e10
        b = linalg.cho_solve(cb, X.T @ Vy); r = y - X @ b; q = r @ linalg.cho_solve(c, r)
        ld = 2 * np.log(np.diag(c[0])).sum(); ldx = 2 * np.log(np.diag(cb[0])).sum()
        val = ld + q + (ldx if reml else 0.0); return 0.5 * val + 0.5 * (n - (p if reml else 0)) * np.log(2 * np.pi)
    v0 = max(np.var(y), 1e-6); best = None; rng = np.random.default_rng(0)
    for s in range(starts):
        th0 = np.log(v0 * (np.array([0.4] * len(ZZ) + [0.3]) * (1 if s == 0 else np.exp(rng.normal(0, .7, len(ZZ) + 1)))))
        o = optimize.minimize(nll, th0, method="L-BFGS-B", bounds=[(-20, 5)] * (len(ZZ) + 1))
        if best is None or o.fun < best.fun: best = o
    th = best.x; V, c = parts(th); VX = linalg.cho_solve(c, X); XtVX = X.T @ VX; cov = linalg.inv(XtVX); b = cov @ (X.T @ linalg.cho_solve(c, y))
    res = LMMResult(); res.beta = b; res.cov = cov; res.se = np.sqrt(np.diag(cov)); res.vcomp = np.exp(th[:-1]); res.scale = float(np.exp(th[-1]))
    res.llf = -best.fun; res.reml = reml; res.n = n; res.p = p; res.t = b / res.se; res.th = th
    res.V_chol = c; res.resid = y - X @ b; return res

def predict_fixed(res, Xnew): return np.asarray(Xnew, float) @ res.beta

def lr_test(res_small, res_big):
    """Likelihood-ratio test; both must be ML fits (reml=False) with the same random structure."""
    lr = 2 * (res_big.llf - res_small.llf); df = res_big.p - res_small.p; return lr, df, float(stats.chi2.sf(max(lr, 0), df))

# ------------------------------------------------------------------ CV sampling noise
def cv_sampling_var(cv, n):
    """Large-sample variance of a coefficient of variation (normal data): CV^2 (1 + 2 CV^2) / (2 n)."""
    cv = np.asarray(cv, float); n = np.asarray(n, float); return cv ** 2 * (1 + 2 * cv ** 2) / (2 * np.maximum(n, 2))
def bias_correct_cv(cv, n):
    """Sokal & Rohlf small-sample correction CV* = CV (1 + 1/(4n))."""
    return np.asarray(cv, float) * (1 + 1 / (4 * np.maximum(np.asarray(n, float), 2)))

# ------------------------------------------------------------------ models / CV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score
from joblib import Parallel, delayed

NUMERIC = ["MAT", "MAP", "abs_lat"]; LOWCARD = ["woody", "symb"]
def make_models(seed=42, n_trees=300):
    return {"ElasticNet": (ElasticNet(max_iter=20000, random_state=seed), {"model__alpha": [0.02, 0.1, 0.5], "model__l1_ratio": [0.2, 0.5, 0.8]}),
            "RandomForest": (RandomForestRegressor(random_state=seed, n_jobs=1, n_estimators=n_trees, min_samples_split=10),
                             {"model__max_depth": [2, 4], "model__min_samples_leaf": [5, 15], "model__max_features": ["sqrt", 0.6]})}
def prep(num=NUMERIC, low=LOWCARD):
    parts = [("num", StandardScaler(), list(num))]
    if low: parts.append(("low", OneHotEncoder(handle_unknown="ignore"), list(low)))
    return ColumnTransformer(parts)

def _one_fold(model_name, est, grid, X, y, w, groups, tr, te, f):
    pipe = Pipeline([("prep", prep()), ("model", est)])
    inner = list(LegacyGroupKFold(min(5, len(np.unique(groups[tr])))).split(X.iloc[tr], y[tr], groups=groups[tr]))
    gs = GridSearchCV(pipe, grid, cv=inner, scoring="r2", n_jobs=1)
    if w is None: gs.fit(X.iloc[tr], y[tr])
    else: gs.fit(X.iloc[tr], y[tr], model__sample_weight=w[tr])
    pred = gs.best_estimator_.predict(X.iloc[te]); trm = np.average(y[tr], weights=None if w is None else w[tr])
    return pd.DataFrame({"model": model_name, "fold": f, "row": te, "y": y[te], "pred": pred, "train_mean": trm})

def grouped_oof(X, y, groups, splits, w=None, models=None, n_jobs=-1):
    models = models or make_models()
    jobs = [(m, e, g, f, tr, te) for m, (e, g) in models.items() for f, (tr, te) in enumerate(splits)]
    out = Parallel(n_jobs=n_jobs)(delayed(_one_fold)(m, e, g, X, y, w, groups, tr, te, f) for m, e, g, f, tr, te in jobs)
    o = pd.concat(out, ignore_index=True); o["site"] = np.asarray(groups)[o.row]; return o

def skill_table(o, w=None, B=2000, seed=42):
    """pooled MSE skill vs training-fold mean (+ precision-weighted version), site-cluster bootstrap CI, pooled R2, fold R2 mean."""
    rows = []; r = np.random.default_rng(seed); ww = np.ones(len(o)) if w is None else np.asarray(w)[o.row.values]
    for m, d in o.groupby("model"):
        ww_m = ww[d.index.values] if len(ww) == len(o) else None
        se_m = (d.y - d.pred) ** 2; se_b = (d.y - d.train_mean) ** 2
        ps = pd.DataFrame({"site": d.site.values, "a": se_m.values, "b": se_b.values, "aw": se_m.values * ww_m, "bw": se_b.values * ww_m}).groupby("site").sum()
        sk = 1 - ps.a.sum() / ps.b.sum(); skw = 1 - ps.aw.sum() / ps.bw.sum(); bs, bsw = [], []
        S = ps.index.values; arr = ps.values
        for _ in range(B):
            ix = r.integers(0, len(S), len(S)); t = arr[ix].sum(0); bs.append(1 - t[0] / t[1]); bsw.append(1 - t[2] / t[3])
        fr2 = [r2_score(g.y, g.pred) for _, g in d.groupby("fold")] if d.fold.nunique() > 1 else [np.nan]
        fsk = [1 - ((g.y - g.pred) ** 2).sum() / ((g.y - g.train_mean) ** 2).sum() for _, g in d.groupby("fold")]
        rows.append({"model": m, "skill": sk, "skill_lo": np.percentile(bs, 2.5), "skill_hi": np.percentile(bs, 97.5), "skill_w": skw, "skill_w_lo": np.percentile(bsw, 2.5), "skill_w_hi": np.percentile(bsw, 97.5),
                     "pooled_R2": r2_score(d.y, d.pred), "mean_fold_R2": np.nanmean(fr2), "folds_beating": int(np.sum(np.array(fsk) > 0)), "n_folds": len(fsk), "n": len(d)})
    return pd.DataFrame(rows)

class LegacyGroupKFold:
    """GroupKFold as in scikit-learn <= 1.5 (greedy, largest group first, ties broken by np.argsort order).
    Newer scikit-learn versions break ties differently, which changes fold membership; this class pins the
    historical assignment so published fold results can be reproduced regardless of installed version."""
    def __init__(self, n_splits=5): self.n_splits = n_splits
    def split(self, X, y=None, groups=None):
        uniq, inv = np.unique(groups, return_inverse=True); sizes = np.bincount(inv); idx = np.argsort(sizes)[::-1]
        load = np.zeros(self.n_splits); g2f = np.zeros(len(uniq), int)
        for gi in idx:
            k = int(np.argmin(load)); load[k] += sizes[gi]; g2f[gi] = k
        fold = g2f[inv]
        for k in range(self.n_splits): yield np.where(fold != k)[0], np.where(fold == k)[0]

# ------------------------------------------------------------------ outcome-independent noise model
def cv_noise_var_n(mean_cv, n):
    """Sampling variance of a CV from n individuals evaluated at a *fixed* expected CV (trait mean), so it depends on n only.
    Using each row's own observed CV (plug-in) ties the variance to the outcome and inflates the noise share."""
    mu = float(mean_cv); return mu ** 2 * (1 + 2 * mu ** 2) / (2 * np.maximum(np.asarray(n, float), 2))

# ------------------------------------------------------------------ fast elastic-net CV for permutation / simulation work
def fold_cache(groups, folds):
    return [list(LegacyGroupKFold(5).split(np.zeros(len(tr)), groups=np.asarray(groups)[tr])) for tr, te in folds]
_GRID = [(a, l) for a in [0.02, 0.1, 0.5] for l in [0.2, 0.5, 0.8]]
def fast_en_cv(NUM, OH, y, folds, inner):
    sse = sseb = 0.0; r2s = []; beats = 0; yt_all = []; pr_all = []
    for (tr, te), inn in zip(folds, inner):
        mu, sd = NUM[tr].mean(0), NUM[tr].std(0); Z = np.c_[(NUM - mu) / sd, OH]; best = None
        for a, l in _GRID:
            e = sum(((y[tr][ite] - ElasticNet(alpha=a, l1_ratio=l, max_iter=5000).fit(Z[tr][itr], y[tr][itr]).predict(Z[tr][ite])) ** 2).sum() for itr, ite in inn)
            if best is None or e < best[0]: best = (e, a, l)
        p = ElasticNet(alpha=best[1], l1_ratio=best[2], max_iter=20000).fit(Z[tr], y[tr]).predict(Z[te]); yt = y[te]
        r2s.append(1 - ((yt - p) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()); a_ = ((yt - p) ** 2).sum(); b_ = ((yt - y[tr].mean()) ** 2).sum()
        sse += a_; sseb += b_; beats += a_ < b_; yt_all.append(yt); pr_all.append(p)
    yt = np.concatenate(yt_all); pr = np.concatenate(pr_all)
    return float(np.mean(r2s)), 1 - sse / sseb, int(beats), 1 - ((yt - pr) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
