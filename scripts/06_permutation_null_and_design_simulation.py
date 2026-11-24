import os, sys; sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import numpy as np, pandas as pd, time
from itvcheck import *
from scipy import optimize
d=pd.read_pickle('han.pkl'); TR=['SRL','RD','RTD','RN']; t0=time.time()
# ---- F3: site-level permutation null for the elastic-net CV statistics (assumption-light)
rows=[]; NP=int(sys.argv[1]) if len(sys.argv)>1 else 300
for t in TR:
    s=d[d[f'cv_{t}'].notna()].reset_index(drop=True); y=s[f'cv_{t}'].values; g=s.site.values; NUM=s[NUMERIC].values.astype(float); OH=OneHotEncoder(handle_unknown='ignore').fit(s[LOWCARD]).transform(s[LOWCARD]).toarray()
    folds=list(LegacyGroupKFold(5).split(NUM,y,groups=g)); inner=fold_cache(g,folds); obs=fast_en_cv(NUM,OH,y,folds,inner)
    sites=np.unique(g); trip={k:NUM[g==k][0] for k in sites}; r=np.random.default_rng(2024); null=[]
    for _ in range(NP):
        perm=r.permutation(sites); mp=dict(zip(sites,perm)); Np=np.array([trip[mp[k]] for k in g]); null.append(fast_en_cv(Np,OH,y,folds,inner))
    N=np.array(null); rows.append({'trait':t,'n_perm':NP,'obs_meanfoldR2':obs[0],'obs_skill':obs[1],'obs_beats':obs[2],'obs_pooledR2':obs[3],
        'null_meanfoldR2_mean':N[:,0].mean(),'null_meanfoldR2_p2_5':np.percentile(N[:,0],2.5),'null_meanfoldR2_p97_5':np.percentile(N[:,0],97.5),'p_R2_ge_obs':(1+(N[:,0]>=obs[0]).sum())/(NP+1),
        'null_skill_mean':N[:,1].mean(),'null_skill_p2_5':np.percentile(N[:,1],2.5),'null_skill_p97_5':np.percentile(N[:,1],97.5),'p_skill_ge_obs':(1+(N[:,1]>=obs[1]).sum())/(NP+1),'p_beats_ge_obs':(1+(N[:,2]>=obs[2]).sum())/(NP+1)})
    print('F3',t,rows[-1],round(time.time()-t0),flush=True)
pd.DataFrame(rows).to_csv('F3_permutation_null.csv',index=False)
# ---- F4: design simulation, two bracketing decompositions of the non-climate variance
s=d[d.cv_SRL.notna()].reset_index(drop=True); y0=s.cv_SRL.values; n=s.n.values; V=np.var(y0,ddof=1); g=s.site.values; spp=s.species.values
NUM=s[NUMERIC].values.astype(float); OH=OneHotEncoder(handle_unknown='ignore').fit(s[LOWCARD]).transform(s[LOWCARD]).toarray(); folds=list(LegacyGroupKFold(5).split(NUM,y0,groups=g)); inner=fold_cache(g,folds)
plain=fit_reml(y0,np.ones((len(s),1)),[g,spp]); ys=bias_correct_cv(y0,n); vn=cv_noise_var_n(ys.mean(),n); nz=fit_reml(ys,np.ones((len(s),1)),[g,spp],meas_var=vn)
SC={'S1_plain (all non-climate variance is real: site 49%, species 20%, residual 32%)':dict(site=plain.vcomp[0],sp=plain.vcomp[1],res=plain.scale,noise=np.zeros(len(s))),
    'S2_noise (n-only sampling noise 24%, rest real)':dict(site=nz.vcomp[0],sp=nz.vcomp[1],res=nz.scale,noise=vn)}
zrow=((s.MAT-s.MAT.mean())/s.MAT.std()).values; out=[]; NS=int(sys.argv[2]) if len(sys.argv)>2 else 100
for name,P in SC.items():
    for share in [0,0.05,0.10,0.15,0.20,0.30,0.40]:
        r=np.random.default_rng(int(1000*share)+5); res=[]
        for k in range(NS):
            su=dict(zip(np.unique(g),r.normal(0,1,len(np.unique(g))))); spu=dict(zip(np.unique(spp),r.normal(0,1,len(np.unique(spp)))))
            clim=np.sqrt(share*V)*zrow; idio=max(P['site']-share*V,0)
            y=y0.mean()+clim+np.sqrt(idio)*np.array([su[x] for x in g])+np.sqrt(P['sp'])*np.array([spu[x] for x in spp])+r.normal(0,np.sqrt(P['res']),len(y0))+r.normal(0,1,len(y0))*np.sqrt(P['noise'])
            res.append(fast_en_cv(NUM,OH,y,folds,inner)+(float(np.var(clim)/np.var(y,ddof=1)),))
        a=np.array(res); out.append({'scenario':name,'true_share_of_total_var':share,'realized_share':a[:,4].mean(),'exp_mean_fold_R2':a[:,0].mean(),'P_meanR2_gt0':(a[:,0]>0).mean(),'exp_pooled_skill':a[:,1].mean(),'P_skill_gt0':(a[:,1]>0).mean(),'P_beat_ge4of5':(a[:,2]>=4).mean(),'n_sims':NS})
    print('F4',name,round(time.time()-t0),flush=True)
pd.DataFrame(out).to_csv('F4_design_simulation_two_scenarios.csv',index=False); print('ALL DONE')
