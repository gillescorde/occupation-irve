import json,sys,csv,math,collections,glob,os
import pandas as pd, numpy as np
S=sys.argv[1]; OUT=sys.argv[2]
allc={x['id']:x for x in json.load(open(f'{S}/dl/countpoints_all.json'))}
sel=json.load(open(f'{S}/dl/sel.json'))
axes=collections.defaultdict(list)
for r in csv.DictReader(open('/mnt/project-files/irve/donnees_reelles/axes_approx.csv')):
    axes[r['axe']].append((float(r['pk']),float(r['lat']),float(r['lon'])))
# compteurs Avatar a <= 5 km des axes, ramenes au format compteurs.csv / comptages.csv de irve/README.md
def proj(ax,lat,lon):
    # projection sur la polyligne de l'axe : (distance km, pk interpolé)
    pts=axes[ax]; best=None
    for (p1,a1,o1),(p2,a2,o2) in zip(pts,pts[1:]):
        k=math.cos(math.radians(lat))*111.32
        ax_,ay=(o1-lon)*k,(a1-lat)*111.32; bx,by=(o2-lon)*k,(a2-lat)*111.32
        dx,dy=bx-ax_,by-ay; L=dx*dx+dy*dy
        t=0 if L==0 else max(0,min(1,-(ax_*dx+ay*dy)/L))
        d=math.hypot(ax_+t*dx,ay+t*dy)
        if best is None or d<best[0]: best=(d,p1+t*(p2-p1))
    return best
def coords(ls):
    return [tuple(map(float,p.split())) for p in ls[ls.index('(')+1:-1].split(',')]
rows=[]
for s in sel:
    c=allc[s['id']]; d,pk=proj(s['axe'],s['lat'],s['lon'])
    sens=0
    ls=c.get('linear_section')
    if ls and ls.startswith('LINESTRING'):
        cc=coords(ls)
        (_,pa),(_,pb)=proj(s['axe'],cc[0][1],cc[0][0]),proj(s['axe'],cc[-1][1],cc[-1][0])
        if abs(pb-pa)>=0.3: sens=1 if pb>pa else 2
    rl=c.get('meas_roadlink') or {}
    route=rl.get('NUMERO') or (c.get('op_road_name') or '')
    sur_axe=(route==s['axe'])
    rows.append(dict(compteur_id=f"AVATAR_{s['id']}",lat=round(s['lat'],6),lon=round(s['lon'],6),axe_rattache=s['axe'],sens=sens,
        pk_projete=round(pk,1),delai_rattachement_min=0.0 if sur_axe else round(d/70*60,1),
        avatar_id=s['id'],nom_avatar=c['count_point_name'],nom_station=c.get('station_name'),route=route,
        nature_route=rl.get('NATURE'),importance=rl.get('IMPORTANCE'),sur_axe=sur_axe,distance_axe_km=round(d,2),
        direction_op=c.get('op_direction'),destination_op=c.get('op_destination'),operateur_avatar=c.get('operator_id')))
cpt=pd.DataFrame(rows)
fs=sorted(glob.glob(f'{S}/dl/raw/*.csv'))
nmois=collections.Counter(os.path.basename(f)[:7] for f in fs)
fs=[f for f in fs if nmois[os.path.basename(f)[:7]]==47]  # mois complets uniquement
m=pd.concat([pd.read_csv(f,sep=';') for f in fs if os.path.getsize(f)>200],ignore_index=True)
m=m.rename(columns={'flow[veh/h]':'debit_vh','speed[km/h]':'vitesse_kmh','perc_flow_predicted':'part_estimee_pct'})
m['horodatage']=pd.to_datetime(m['measure_datetime'].str[:19])
m['compteur_id']='AVATAR_'+m['count_point_id'].astype(str)
m=m.dropna(subset=['debit_vh']).drop_duplicates(['compteur_id','horodatage'])
m['vitesse_kmh']=m['vitesse_kmh'].round(1)
st=m.groupby('compteur_id').agg(nb_heures=('debit_vh','size'),debut=('horodatage','min'),fin=('horodatage','max'),
    debit_moyen_vh=('debit_vh','mean'),part_estimee_moy_pct=('part_estimee_pct','mean'))
jul=m[(m.horodatage>='2026-07-09')&(m.horodatage<'2026-07-16')]
st['heures_9_15_juillet']=jul.groupby('compteur_id').size()
st['part_mesuree_9_15_juillet_pct']=100-jul.groupby('compteur_id')['part_estimee_pct'].mean()
cpt=cpt.merge(st,left_on='compteur_id',right_index=True,how='inner')
cpt['debit_moyen_vh']=cpt['debit_moyen_vh'].round(0); cpt['part_estimee_moy_pct']=cpt['part_estimee_moy_pct'].round(1)
cpt['part_mesuree_9_15_juillet_pct']=cpt['part_mesuree_9_15_juillet_pct'].round(1)
cpt['heures_9_15_juillet']=cpt['heures_9_15_juillet'].fillna(0).astype(int)
Lmax={a:max(p for p,_,_ in v) for a,v in axes.items()}
cpt['hors_agglo_extremites']=[(p>30 and p<Lmax[a]-30) for a,p in zip(cpt.axe_rattache,cpt.pk_projete)]
cpt=cpt.sort_values(['axe_rattache','pk_projete','sens'])
os.makedirs(OUT,exist_ok=True)
cpt.to_csv(f'{OUT}/compteurs.csv',index=False)
m=m[m.compteur_id.isin(cpt.compteur_id)].sort_values(['horodatage','compteur_id'])
m[['horodatage','compteur_id','debit_vh','vitesse_kmh','part_estimee_pct']].to_csv(f'{OUT}/comptages.csv',index=False)
print(len(cpt),'compteurs avec données ;',len(m),'lignes ;',m.horodatage.min(),m.horodatage.max())
ok=cpt[(cpt.heures_9_15_juillet>=120)]
print('couvrant 9-15 juillet :',len(ok)); print(ok.groupby('axe_rattache').agg(n=('compteur_id','size'),sur_axe=('sur_axe','sum'),mesure=('part_mesuree_9_15_juillet_pct','median')))
print(cpt.sens.value_counts())
