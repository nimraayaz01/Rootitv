import os, sys; sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import numpy as np, pandas as pd
from itvcheck import *
d=pd.read_pickle('han.pkl'); rows=[]; folds_rows=[]
for t in ['SRL','RD','RTD','RN']:
    s=d[d[f'cv_{t}'].notna()].reset_index(drop=True); X=s[NUMERIC+LOWCARD].copy(); g=s.site.values; n=s.n.values
    sp=list(LegacyGroupKFold(5).split(X,s[f'cv_{t}'],groups=g))
    for variant,y,wfit in [('raw',s[f'cv_{t}'].values,None)]:
        o=grouped_oof(X,y,g,sp,w=wfit); 
        # evaluate with precision weights in BOTH variants (weights are always defined from the sampling-variance model)
        sk=skill_table(o,B=1000); sk.insert(0,'variant',variant); sk.insert(0,'trait',t); rows.append(sk)
        for (m,f),q in o.groupby(['model','fold']): folds_rows.append({'trait':t,'variant':variant,'model':m,'fold':f,'n':len(q),'r2':r2_score(q.y,q.pred),'skill':1-((q.y-q.pred)**2).sum()/((q.y-q.train_mean)**2).sum()})
        print(t,variant,'done',flush=True)
R=pd.concat(rows); R.to_csv('A_cv_by_trait.csv',index=False); pd.DataFrame(folds_rows).to_csv('A_cv_folds_by_trait.csv',index=False)
pd.set_option('display.width',250); print(R.round(3).to_string(index=False))
