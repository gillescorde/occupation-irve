import json,sys,time,urllib.request,os
sel=json.load(open(sys.argv[1])); out=sys.argv[2]; os.makedirs(out,exist_ok=True)
ids=[x['id'] for x in sel]
months=[('2026-05-01','2026-05-31'),('2026-06-01','2026-06-30'),('2026-07-01','2026-07-31'),('2026-08-01','2026-08-31'),('2026-09-01','2026-09-30')]
for mi,(a,b) in enumerate(months):
  for i in range(0,len(ids),40):
    fn=f'{out}/{a[:7]}_{i:05d}.csv'
    if os.path.exists(fn): continue
    u=('https://avatar.cerema.fr/api/aggregated_measures/download?with_cp_names=true&with_quality=0&traffic_vars=flow,speed'
       f'&count_point_ids={",".join(map(str,ids[i:i+40]))}&time_zone=Europe%2FParis&aggregation_period=hour&start_time={a}T00:00:00&end_time={b}T23:59:59')
    for k in range(5):
      try:
        req=urllib.request.Request(u,headers={'X-Client-Type':'avatar-dashboard'})
        data=urllib.request.urlopen(req,timeout=600).read(); break
      except Exception as e:
        print('err',fn,e,flush=True); time.sleep(10*(k+1))
    else: continue
    open(fn,'wb').write(data); print(fn,len(data),flush=True); time.sleep(1)
