import os, sys; sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import numpy as np, pandas as pd, time
from itvcheck import *
from scipy import optimize
d=pd.read_pickle('han.pkl'); TR=['SRL','RD','RTD','RN']; t0=time.time()
# ---- F1: outcome-independent noise model (n only) -> variance partition and MAT tests
rows=[]; tests=[]
for t in TR:
    s=d[d[f'cv_{t}'].notna()].reset_index(drop=True); y=s[f'cv_{t}'].values; n=s.n.values; V=np.var(y,ddof=1)
    ystar=bias_correct_cv(y,n); vn=cv_noise_var_n(ystar.mean(),n); plug=cv_sampling_var(y,n); sg=[s.site.values,s.species.values]; X0=np.ones((len(s),1))
    zM=((s.MAT-s.MAT.mean())/s.MAT.std()).values; zL=((s.abs_lat-s.abs_lat.mean())/s.abs_lat.std()).values
    a=fit_reml(y,X0,sg); b=fit_reml(ystar,X0,sg,meas_var=vn); tot=b.vcomp.sum()+b.scale
    row={'trait':t,'noise_share_n_only':vn.mean()/np.var(ystar,ddof=1),'noise_share_plugin':plug.mean()/V,'ceiling_R2_range_lo':1-plug.mean()/V,'ceiling_R2_range_hi':1-vn.mean()/np.var(ystar,ddof=1),
         'plain_site':a.vcomp[0]/(a.vcomp.sum()+a.scale),'plain_species':a.vcomp[1]/(a.vcomp.sum()+a.scale),'plain_resid':a.scale/(a.vcomp.sum()+a.scale),
         'true_site_of_true':b.vcomp[0]/tot,'true_species_of_true':b.vcomp[1]/tot,'true_resid_of_true':b.scale/tot}
    for tag,yy,mv in [('plain',y,None),('noise_n_only',ystar,vn)]:
        fits={k:fit_reml(yy,(np.c_[np.ones(len(s)),*c] if c else X0),sg,meas_var=mv,reml=False) for k,c in {'null':[],'MAT':[zM],'lat':[zL],'MAT+lat':[zM,zL]}.items()}
        for a_,b_ in [('null','MAT'),('null','lat'),('MAT','MAT+lat'),('lat','MAT+lat')]:
            lr,df,p=lr_test(fits[a_],fits[b_]); tests.append({'trait':t,'model':tag,'test':f'{a_}->{b_}','LR':lr,'p':p})
        for k in fits: tests.append({'trait':t,'model':tag,'test':f'AIC_{k}','LR':-2*fits[k].llf+2*(fits[k].p+3),'p':np.nan})
        rM=fit_reml(yy,np.c_[np.ones(len(s)),zM],sg,meas_var=mv); fx=np.var(np.c_[np.ones(len(s)),zM]@rM.beta); row[f'MAT_beta_per_SD_{tag}']=rM.beta[1]; row[f'MAT_t_{tag}']=rM.t[1]
        row[f'MAT_share_of_true_var_{tag}']=fx/(fx+rM.vcomp.sum()+rM.scale)
    rows.append(row)
pd.DataFrame(rows).to_csv('F1_noise_n_only_partition.csv',index=False); pd.DataFrame(tests).to_csv('F1_lmm_tests.csv',index=False); print('F1 done',round(time.time()-t0),flush=True)
