import json,sys,csv,math,collections
d=json.load(open(sys.argv[1]))
axes=collections.defaultdict(list)
for r in csv.DictReader(open('/mnt/project-files/irve/donnees_reelles/axes_approx.csv')):
    axes[r['axe']].append((float(r['pk']),float(r['lat']),float(r['lon'])))
def dist(a,b,c,e):
    x=(e-b)*math.cos(math.radians((a+c)/2))*111.32; y=(c-a)*111.32; return math.hypot(x,y)
rows=[]
for x in d:
    p=x['punctual_position']
    if not p: continue
    lon,lat=map(float,p[p.index('(')+1:-1].split())
    rl=x.get('meas_roadlink') or {}
    best=None
    for ax,pts in axes.items():
        for pk,la,lo in pts:
            if abs(la-lat)>0.2: continue
            dd=dist(lat,lon,la,lo)
            if best is None or dd<best[0]: best=(dd,ax,pk)
    if best and best[0]<=10:
        rows.append(dict(id=x['id'],name=x['count_point_name'],st=x['station_name'],lat=lat,lon=lon,op=x['operator_id'],
          num=rl.get('NUMERO'),imp=rl.get('IMPORTANCE'),nature=rl.get('NATURE'),way=x.get('meas_roadlink_way'),
          dir=x.get('op_direction'),dest=x.get('op_destination'),supp=x.get('suppressed'),last=x.get('last_date_quality_data'),
          axe=best[1],pk=best[2],d=round(best[0],2)))
json.dump(rows,open(sys.argv[2],'w'))
print(len(rows))
c=collections.Counter((r['axe']) for r in rows); print(c)
print(collections.Counter(r['supp'] for r in rows))
print(collections.Counter((r['last'] or '')[:7] for r in rows).most_common(15))
for ax in ['A10','A6','A7','A9']:
    rr=[r for r in rows if r['axe']==ax]
    print(ax,len(rr),collections.Counter(r['num'] for r in rr).most_common(8), collections.Counter(r['imp'] for r in rr))
