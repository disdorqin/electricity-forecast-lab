import json,csv
from pathlib import Path
H=Path(__file__).resolve().parent; sel=json.loads(Path('src/TafM_改进源码/outputs/tabm_v21/selector/selector_cutoff_2025-12-31/manifest.json').read_text(encoding='utf-8'))
roles={r['feature_name']:r['role'] for r in sel['feature_roles']}
profiles={'all':{'Strong-BOTH','Strong-DIR','Strong-MAG','Forced-Core'},'drop_mag':{'Strong-BOTH','Strong-DIR','Forced-Core'},'drop_dir':{'Strong-BOTH','Strong-MAG','Forced-Core'}}
out=[]
for p,allowed in profiles.items():
 for role in ['Strong-BOTH','Strong-DIR','Strong-MAG','Forced-Core','Weak']:
  names=sorted([n for n,r in roles.items() if r==role]); out.append({'strong_role_profile':p,'role':role,'frozen_count':len(names),'strong_included':role in allowed,'expected_strong_path_count':sum(len([n for n,r in roles.items() if r==x]) for x in allowed),'feature_names':'|'.join(names)})
(H/'role_inventory.csv').write_text('',encoding='utf-8')
with (H/'role_inventory.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=out[0]);w.writeheader();w.writerows(out)
(H/'role_inventory.json').write_text(json.dumps({'profiles':out},ensure_ascii=False,indent=2),encoding='utf-8')
