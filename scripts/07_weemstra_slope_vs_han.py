import os, sys; sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import numpy as np, pandas as pd
from itvcheck import *
U=os.environ.get('ITV_DATA','./data').rstrip('/')+'/'
d=pd.read_pickle('han.pkl'); c=pd.read_excel(U+'Combined_SRL_ITV_Han_Weemstra_v2.xlsx'); wm=c[c.data_source=='Weemstra_2021'][['species','altitude_m','MAT']]
W=pd.read_csv(U+'Weemstra_ITV.csv').merge(wm,on=['species','altitude_m']); rows=[]
def slope(y,x,sp,clim,B=2000):
    df=pd.DataFrame({'y':y,'x':x,'sp':sp}); yc=df.y-df.groupby('sp').y.transform('mean'); xc=df.x-df.groupby('sp').x.transform('mean'); b=(xc*yc).sum()/(xc**2).sum(); res=yc-b*xc
    se=np.sqrt((res**2).sum()/(len(df)-df.sp.nunique()-1)/(xc**2).sum()); ua=np.unique(clim); rr=np.random.default_rng(42); bs=[]
    for _ in range(B):
        ix=np.concatenate([np.where(clim==u)[0] for u in rr.choice(ua,len(ua),replace=True)]); dd=df.iloc[ix]; a=dd.y-dd.groupby('sp').y.transform('mean'); q=dd.x-dd.groupby('sp').x.transform('mean'); bs.append((q*a).sum()/(q**2).sum() if (q**2).sum()>0 else np.nan)
    return b,se,np.nanpercentile(bs,[2.5,97.5]),res.std(ddof=1),(xc**2).sum()
for t,cols in {'SRL':[('corrected','CV_SRL')],'RD':[('corrected','CV_RD'),('direct','CV_RD_direct')],'RTD':[('corrected','CV_RTD'),('direct','CV_RTD_direct')]}.items():
    s=d[d[f'cv_{t}'].notna()].reset_index(drop=True); y=s[f'cv_{t}'].values; sg=[s.site.values,s.species.values]; zs=(s.MAT-s.MAT.mean())/s.MAT.std()
    f=fit_reml(y,np.c_[np.ones(len(s)),zs],sg); hb=f.beta[1]/s.MAT.std(); hse=f.se[1]/s.MAT.std()    # plain mixed model (noise model does not change the slope: see F1)
    for calc,col in cols:
        b,se,ci,rs,sxx=slope(W[col].values,W.MAT.values,W.species.values,W.altitude_m.values); dfree=len(W)-W.species.nunique()-1; tc=stats.t.ppf(.975,dfree)
        pw=lambda beta:float(1-stats.nct.cdf(tc,dfree,beta/np.sqrt(rs**2/sxx))+stats.nct.cdf(-tc,dfree,beta/np.sqrt(rs**2/sxx)))
        rows.append({'trait':t,'calc':calc,'weemstra_slope_per_C':b,'ci_lo':ci[0],'ci_hi':ci[1],'han_slope_per_C':hb,'han_lo':hb-1.96*hse,'han_hi':hb+1.96*hse,'gradient_C':W.MAT.max()-W.MAT.min(),
                     'power_han_slope':pw(hb),'power_han_lower95':pw(max(hb-1.96*hse,0)),'min_detectable_slope_80':float(2.8*np.sqrt(rs**2/sxx)),'z_diff':(b-hb)/np.sqrt(se**2+hse**2)})
pd.DataFrame(rows).to_csv('G_weemstra_slopes_plainLMM.csv',index=False); pd.set_option('display.width',250); print(pd.DataFrame(rows).round(4).T.to_string())
