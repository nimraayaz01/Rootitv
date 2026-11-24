import os, sys; sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import numpy as np, pandas as pd, time
from itvcheck import *
from scipy import optimize
d=pd.read_pickle('han.pkl'); TR=['SRL','RD','RTD','RN']; t0=time.time()
# ---- F2: noise-aware CV with n-only precision weights (outcome-independent)
res=[]
for t in TR:
    s=d[d[f'cv_{t}'].notna()].reset_index(drop=True); X=s[NUMERIC+LOWCARD].copy(); g=s.site.values; n=s.n.values; y=s[f'cv_{t}'].values
    sp=list(LegacyGroupKFold(5).split(X,y,groups=g)); ystar=bias_correct_cv(y,n); vn=cv_noise_var_n(ystar.mean(),n); tau2=max(np.var(ystar,ddof=1)-vn.mean(),0.15*np.var(ystar,ddof=1)); w=1/(tau2+vn); w=w/w.mean()
    o=grouped_oof(X,ystar,g,sp,w=w,n_jobs=1); sk=skill_table(o,w=w,B=1000); sk.insert(0,'variant','bias-corrected CV + n-only precision weights'); sk.insert(0,'trait',t); res.append(sk); print('F2',t,round(time.time()-t0),flush=True)
pd.concat(res).to_csv('F2_noise_aware_cv_n_only.csv',index=False)
