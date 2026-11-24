import os, sys; sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import numpy as np, pandas as pd
from itvcheck import *
U=os.environ.get('ITV_DATA','./data').rstrip('/')+'/'
d=pd.read_pickle('han.pkl'); B=None
c=pd.read_excel(U+'Combined_SRL_ITV_Han_Weemstra_v2.xlsx'); wm=c[c.data_source=='Weemstra_2021'][['species','altitude_m','MAT','MAP','abl.lat','plant_woodiness','symbiotic_status']].rename(columns={'abl.lat':'abs_lat','plant_woodiness':'woody','symbiotic_status':'symb'})
wi=pd.read_csv(U+'Weemstra_ITV.csv'); W=wi.merge(wm,on=['species','altitude_m'],how='left'); assert W.MAT.notna().all() and len(W)==88, (W.MAT.isna().sum(),len(W))
print('Weemstra MAT range',W.MAT.min(),W.MAT.max(),'altitudes',W.altitude_m.nunique(),'distinct MAT',W.MAT.nunique(),'species',W.species.nunique())
TR={'SRL':[('corrected','CV_SRL')],'RD':[('corrected','CV_RD'),('direct','CV_RD_direct')],'RTD':[('corrected','CV_RTD'),('direct','CV_RTD_direct')]}
rows=[]; slope_rows=[]; r=np.random.default_rng(42)
def fit_models(s,col,noise_aware):
    X=s[NUMERIC+LOWCARD].copy(); y=s[col].values; g=s.site.values; n=s.n.values; w=None
    if noise_aware:
        pass
    out={}
    for m,(est,grid) in make_models().items():
        pipe=Pipeline([('prep',prep()),('model',est)]); inner=list(LegacyGroupKFold(5).split(X,y,groups=g)); gs=GridSearchCV(pipe,grid,cv=inner,scoring='r2',n_jobs=1)
        gs.fit(X,y) if w is None else gs.fit(X,y,model__sample_weight=w); out[m]=gs.best_estimator_
    return out,y,w
for t,cols in TR.items():
    s=d[d[f'cv_{t}'].notna()].reset_index(drop=True); mu,sd=s.MAT.mean(),s.MAT.std(); zs=(s.MAT-mu)/sd; sg=[s.site.values,s.species.values]
    for variant in ['raw']:
        mods,ytr,wtr=fit_models(s,f'cv_{t}',False)
        mv=None
        l0=fit_reml(ytr,np.ones((len(s),1)),sg,meas_var=mv); l1=fit_reml(ytr,np.c_[np.ones(len(s)),zs],sg,meas_var=mv)
        trmean=np.average(ytr,weights=wtr)
        for calc,col in cols:
            y=W[col].values; preds={m:mods[m].predict(W[NUMERIC+LOWCARD]) for m in mods}; preds['TrainMean']=np.full(len(W),trmean); preds['LMM_intercept']=np.full(len(W),l0.beta[0]); preds['LMM_MAT']=l1.beta[0]+l1.beta[1]*((W.MAT-mu)/sd).values
            wW=np.ones(len(y)); 
            mse={k:((y-p)**2) for k,p in preds.items()}; clim=W.altitude_m.values; ualt=np.unique(clim)
            def boot(a,b,wt=None,B=2000):
                rr=np.random.default_rng(42); out=[]
                for _ in range(B):
                    ix=np.concatenate([np.where(clim==u)[0] for u in rr.choice(ualt,len(ualt),replace=True)]); ww=np.ones(len(ix)) if wt is None else wt[ix]; out.append(np.average(a[ix],weights=ww)-np.average(b[ix],weights=ww))
                return np.percentile(out,[2.5,97.5])
            for k in preds:
                row={'trait':t,'calc':calc,'train':variant,'model':k,'rmse':np.sqrt(mse[k].mean()),'rmse_w':np.sqrt(np.average(mse[k],weights=wW)),'bias_obs_minus_pred':(y-preds[k]).mean()}
                for ref in ['TrainMean','LMM_intercept']:
                    if k!=ref: lo,hi=boot(mse[k],mse[ref]); row[f'dMSE_vs_{ref}']=mse[k].mean()-mse[ref].mean(); row[f'dMSE_vs_{ref}_lo']=lo; row[f'dMSE_vs_{ref}_hi']=hi
                rows.append(row)
    print(t,'external models done',flush=True)
pd.DataFrame(rows).to_csv('E_weemstra_external.csv',index=False)
# ---- Han's own in-situ campaigns (Ours_9/10/11) as a held-out origin: train on Global rows only
orows=[]
for t in ['SRL','RD','RTD','RN']:
    s=d[d[f'cv_{t}'].notna()].reset_index(drop=True); tr=s[s.origin=='Global'].reset_index(drop=True); te=s[s.origin=='Ours'].reset_index(drop=True)
    X=tr[NUMERIC+LOWCARD]; y=tr[f'cv_{t}'].values; g=tr.site.values
    for m,(est,grid) in make_models().items():
        pipe=Pipeline([('prep',prep()),('model',est)]); gs=GridSearchCV(pipe,grid,cv=list(LegacyGroupKFold(5).split(X,y,groups=g)),scoring='r2',n_jobs=1).fit(X,y); p=gs.predict(te[NUMERIC+LOWCARD])
        for site,q in te.assign(p=p).groupby('site'):
            orows.append({'trait':t,'model':m,'site':site,'n':len(q),'mean_obs':q[f'cv_{t}'].mean(),'mean_pred':q.p.mean(),'sse':((q[f'cv_{t}']-q.p)**2).sum(),'sse_trainmean':((q[f'cv_{t}']-y.mean())**2).sum()})
O=pd.DataFrame(orows); O.to_csv('E_ours_holdout_by_site.csv',index=False)
T=O.groupby(['trait','model']).agg(n=('n','sum'),sse=('sse','sum'),sse0=('sse_trainmean','sum')); T['skill']=1-T.sse/T.sse0; T['rmse']=np.sqrt(T.sse/T.n); T['rmse_trainmean']=np.sqrt(T.sse0/T.n); T.to_csv('E_ours_holdout_summary.csv')
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
print(T.round(3).to_string())
E=pd.DataFrame(rows); print(E[E.train=='raw'].round(4).to_string(index=False))
