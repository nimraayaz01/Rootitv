import os, numpy as np, pandas as pd
U=os.environ.get('ITV_DATA','./data').rstrip('/')+'/'
g=pd.read_excel(U+'Source_data.xlsx','root ITV globe'); c=pd.read_excel(U+'Combined_SRL_ITV_Han_Weemstra_v2.xlsx')
ch=c[c.data_source=='Han_2025'].reset_index(drop=True)
assert len(g)==len(ch)==543 and np.allclose(g['cv.SRL'].fillna(-1),ch.CV_SRL.fillna(-1),atol=1e-4) and (g.MAT.round(1)==ch.MAT.round(1)).all()   # row alignment of the two files
d=pd.DataFrame({'species':ch.species.str.replace(r'_\d+$','',regex=True).str.replace(' ','_'),'site':ch.site_cluster,'origin':g.data_origin,'study':g.study_ID,'MAT':g.MAT,'MAP':g.MAP,'abs_lat':g['abl.lat'],'woody':g.plant_woodiness,'symb':g.symbiotic_status,'family':g.family,'n':g.n,
  **{f'cv_{t}':g[f'cv.{t}'] for t in ['SRL','RD','RTD','RN']}, **{f'mean_{t}':g[t] for t in ['SRL','RD','RTD','RN']}})
d.to_pickle('han.pkl'); print(d.shape)
