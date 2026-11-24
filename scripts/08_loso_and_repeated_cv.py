import os, sys; sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import numpy as np, pandas as pd
from itvcheck import *
d=pd.read_pickle('han.pkl'); rows=[]; rep_rows=[]; R=20
def rand_folds(g,rep,k=5):
    r=np.random.default_rng(1000+rep); ch=np.array_split(r.permutation(np.unique(g)),k); return [(np.where(~np.isin(g,c))[0],np.where(np.isin(g,c))[0]) for c in ch]
for t in ['SRL','RD','RTD','RN']:
    s=d[d[f'cv_{t}'].notna()].reset_index(drop=True); X=s[NUMERIC+LOWCARD].copy(); g=s.site.values; y=s[f'cv_{t}'].values
    en={'ElasticNet':make_models()['ElasticNet']}
    o=grouped_oof(X,y,g,list(LeaveOneGroupOut().split(X,y,groups=g)),models=en,n_jobs=1); sk=skill_table(o,B=1000); sk.insert(0,'design','LOSO'); sk.insert(0,'trait',t); rows.append(sk)
    for rep in range(R):
        o=grouped_oof(X,y,g,rand_folds(g,rep),models=en,n_jobs=1); q=o.groupby('fold').apply(lambda z: r2_score(z.y,z.pred)); 
        rep_rows.append({'trait':t,'rep':rep,'model':'ElasticNet','mean_fold_R2':q.mean(),'pooled_skill':1-((o.y-o.pred)**2).sum()/((o.y-o.train_mean)**2).sum(),'pooled_R2':r2_score(o.y,o.pred)})
    print(t,'EN LOSO + repeated done',flush=True)
# random forest on SRL only (150 trees): LOSO + 8 repeats
s=d[d.cv_SRL.notna()].reset_index(drop=True); X=s[NUMERIC+LOWCARD].copy(); g=s.site.values; y=s.cv_SRL.values
rf={'RandomForest':make_models(n_trees=150)['RandomForest']}
o=grouped_oof(X,y,g,list(LeaveOneGroupOut().split(X,y,groups=g)),models=rf,n_jobs=1); sk=skill_table(o,B=1000); sk.insert(0,'design','LOSO'); sk.insert(0,'trait','SRL'); rows.append(sk); print('RF LOSO done',flush=True)
for rep in range(8):
    o=grouped_oof(X,y,g,rand_folds(g,rep),models=rf,n_jobs=1); q=o.groupby('fold').apply(lambda z: r2_score(z.y,z.pred))
    rep_rows.append({'trait':'SRL','rep':rep,'model':'RandomForest','mean_fold_R2':q.mean(),'pooled_skill':1-((o.y-o.pred)**2).sum()/((o.y-o.train_mean)**2).sum(),'pooled_R2':r2_score(o.y,o.pred)})
pd.concat(rows).to_csv('C_loso_by_trait.csv',index=False); rp=pd.DataFrame(rep_rows); rp.to_csv('C_repeated_cv.csv',index=False)
pd.set_option('display.width',250); print(pd.concat(rows).round(3).to_string(index=False))
print(rp.groupby(['trait','model']).agg(R2_mean=('mean_fold_R2','mean'),R2_lo=('mean_fold_R2',lambda s:s.quantile(.025)),R2_hi=('mean_fold_R2',lambda s:s.quantile(.975)),frac_R2_pos=('mean_fold_R2',lambda s:(s>0).mean()),skill_mean=('pooled_skill','mean'),skill_lo=('pooled_skill',lambda s:s.quantile(.025)),skill_hi=('pooled_skill',lambda s:s.quantile(.975)),frac_skill_pos=('pooled_skill',lambda s:(s>0).mean())).round(3).to_string())
