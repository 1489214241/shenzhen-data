"""Rebuild Guangming/Pingshan inputs and map-point Gold from frozen local sources."""
import argparse,copy,hashlib,importlib.util,json,os,pickle,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
SZ=ROOT/'数据/深圳';SOURCE=SZ/'光明坪山';STANDARD=SZ/'标准数据';INPUT=SZ/'模型输入'
REPORT=ROOT/'登记/光明坪山生成';REPORT.mkdir(parents=True,exist_ok=True)
DISTRICTS=['光明区','坪山区'];SEEDS=[20261231,20261232,20261233]
sys.path[:0]=[str(ROOT/'应用'),str(Path(__file__).parent/'光明坪山依赖')]
import numpy as np

def read(p):return json.loads(p.read_text('utf-8-sig'))
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('wb') as f:pickle.dump(v,f,protocol=5)
def guard(mode):
 denied=[];opened=set();old=os.path.normcase(os.path.abspath(r'F:\学区划片描述'))
 def check(event,args):
  if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
  path=Path(os.fsdecode(args[0])).absolute();p=os.path.normcase(str(path));parts=path.parts
  forbidden=p==old or p.startswith(old+os.sep)
  if mode=='inputs':
   forbidden|=any(x in ('Gold','评价数据') for x in parts)
   forbidden|=path.name in ('polygons.json','polygon与文字.json','两区_polygon与文字.json','原始地图分组.json')
   forbidden|=path.suffix in ('.geojson','.html','.js','.zip')
   forbidden|='来源存档' in parts or path.name in ('texts.json','guangming_map_browser.json','pingshan_public_map_data.json')
  if forbidden:denied.append(str(path));raise PermissionError('Disallowed source during '+mode+': '+str(path))
  if str(path).startswith(str(ROOT)):opened.add(str(path))
 sys.addaudithook(check)
 return denied,opened

def build_inputs():
 denied,opened=guard('inputs')
 import torch,shapely
 from pyproj import Transformer
 import zoning
 zoning.DEVICE='cpu'
 from zoning import data,rules,evidence as E,model as MD
 from zoning.modernnca_evidence_v2 import expanded_model
 from zoning.conditional_admission import Admission,features,combine
 import nametok as NT
 MD.DEVICE='cpu';torch.set_num_threads(4)
 transform=Transformer.from_crs(4326,32650,always_xy=True)
 def project(x):
  x=np.asarray(x,float);a,b=transform.transform(x[:,0],x[:,1]);return np.stack([a,b],1)
 # Road crossing is unused with no supported reference rows; anchors need only the 2km test.
 # Register empty trees only for the native distance routine. No zero crossing claim is emitted.
 from shapely import STRtree
 E._lines['gp_no_reference_geometry']=SimpleNamespace(by={k:STRtree([]) for k in ['major','secondary','tertiary','local','rr']},tree=STRtree([]),geoms=np.array([],object),L=[],cls=[],key=[])
 audit={}
 models=ROOT/'模型/ModernNCA-LN32/深圳'
 scale=dict(np.load(models/'补校标准化.npz',allow_pickle=False))
 for d in DISTRICTS:
  folder=INPUT/d;folder.mkdir(parents=True,exist_ok=True)
  rawschools=read(SOURCE/d/'schools.json');sites=read(SOURCE/d/'小区数据/坐标组.json')
  annual=read(SOURCE/d/'政策文本/年度原表.json');by_policy={p['record_id']:p for p in annual}
  schools=[]
  for s in rawschools:
   assert len(s['map_markers'])==1 and s['map_markers'][0]['coordinates_wgs84']
   marker=s['map_markers'][0]
   schools.append(dict(school_id=s['school_id'],school_name=s['school_name'],district=d,stage='小学',school_point_bd09=marker['coordinates_bd09'],school_point_wgs84=marker['coordinates_wgs84'],point_role=marker['point_role'],policy_record_ids=s['policy_record_ids']))
  n,S=len(sites),len(schools);coords=project([s['representative_point_wgs84'] for s in sites]);school_xy=project([s['school_point_wgs84'] for s in schools])
  assert (n,S)==({'光明区':(167,48),'坪山区':(107,31)}[d])
  texts=['\n'.join(by_policy[i]['scope_text'] for i in s['policy_record_ids']) for s in schools]
  # Annual source text is a soft name feature only; substring hits never create supervision.
  tokens=[NT.tokens(t) for t in texts];name=np.zeros((n,S,4),np.float32)
  for q,site in enumerate(sites):
   keys=[NT.key(x) for x in site['coordinate_group_building_names']];keys=[k for k in keys if len(k)>=3];best=np.zeros(S)
   for j in range(S):
    lengths=[len(t) for k in keys for t in tokens[j] if t in k or len(k)>=4 and k in t]
    if lengths:best[j]=max(lengths)
   hit=np.flatnonzero(best>0)
   if len(hit)==1:name[q,hit[0],0]=1
   elif len(hit)>1:name[q,hit,1]=1
   name[q,:,2]=np.minimum(best/10,1)
   if len(hit):name[q,best==0,3]=1
  states=np.full((n,S),-1,np.int8);sides=np.full((n,S,4),-1,np.int8)
  negative=np.zeros((n,S),bool);community=np.zeros((n,S,4),np.float32)
  district=data.District(name=d,city='深圳',crs=32650,site_names=[s['building_name'] for s in sites],school_names=[s['school_name'] for s in schools],orig=None,coords=coords,groups=np.array([s['coordinate_group_id'] for s in sites]),school_xy=school_xy,answers={},eval_rows=[],train_answers={},scope=None,not_reference=set(),not_training=set(),community_name=np.full(n,''),community_strict=np.zeros(n,bool),community_school=community,name_match=name,lines_key='gp_no_reference_geometry',site_ids=[s['coordinate_group_id'] for s in sites],school_ids=[s['school_id'] for s in schools],policy_facts={})
  rr=rules.RuleOutput(statuses=sides,negative=negative,known={},kind={})
  cfg=dict(pair_blocks=E.PAIR_BLOCKS,school_blocks=E.SCHOOL_BLOCKS,exclusion_to_model='all')
  native=E.build(district,rr,cfg);assert (native.np_,native.na)==(41,27) and len(native.ref_sites)==0
  control=copy.deepcopy(native);routed=copy.deepcopy(native)
  for x,encoded in [(control,np.zeros((n,S,3),np.float32)),(routed,(states[...,None]==np.array([-1,0,1])).astype(np.float32))]:
   x.pair=np.concatenate([native.pair,encoded[:,native.ref_schools]],-1);x.anchor=np.concatenate([native.anchor,encoded],-1)
   x.pair_blocks=dict(native.pair_blocks,whole_scope=slice(41,44));x.anchor_blocks=dict(native.anchor_blocks,whole_scope=slice(27,30))
   assert (x.np_,x.na)==(44,30)
  label=np.full((n,S),-1,np.float32)
  ys=dict(labels=label,train_ids=np.array([],np.int64),valid_ids=np.array([],np.int32),valid_labels=np.zeros((0,S),bool),fold=np.full(n,-1,np.int32))
  for arm,x in [('native',native),('control',control),('routed',routed)]:
   save(folder/f'{arm}.pkl',x)
   np.savez_compressed(folder/f'{arm}_arrays.npz',pair=x.pair,anchor=x.anchor,ref_sites=x.ref_sites,ref_schools=x.ref_schools,group_bias=x.group_bias,self_mask=x.self_mask,groups=x.groups,y=x.y,supervised=x.supervised,membership=x.membership.numpy())
  save(folder/'district.pkl',district);save(folder/'rules.pkl',rr)
  for arm in ['control','routed']:np.savez_compressed(folder/f'{arm}_labels.npz',**ys)
  np.savez_compressed(folder/'whole_scope.npz',states=states)
  np.savez_compressed(folder/'model_arrays.npz',coords=coords,school_xy=school_xy,statuses=sides,negative=negative,community_name=district.community_name,community_strict=district.community_strict,community_school=community,name_match=name,site_ids=np.array(district.site_ids),school_ids=np.array(district.school_ids))
  metadata=dict(sites=sites,schools=schools,known={},known_kind={},projected_crs=32650,source_package=f'数据/深圳/光明坪山/{d}',policy_reference_sites=0,singleton_training_sites=0,school_order='source schools.json order',site_order='all source coordinate groups; no Gold filtering',supervision='independent policy only; no verified site-community memberships available',gold_in_inputs=False)
  write(folder/'metadata.json',metadata)
  observations=[dict(school_id=s['school_id'],source_text=t,policy_records=[by_policy[i] for i in s['policy_record_ids']],scope_status='site_applicability_unknown',reason='No independently verified site-community or complete geographic condition; retain population/exclusion/transition conditions',whole_scope_state=-1) for s,t in zip(schools,texts)]
  write(folder/'policy_observations.json',observations);write(folder/'evidence_ledger.json',[])
  write(folder/'constraints.json',{s['coordinate_group_id']:dict(positive=[],negative=[]) for s in sites})
  write(folder/'feature_schema.json',dict(native=dict(pair=41,anchor=27),control=dict(pair=44,anchor=30,whole_scope='absent block zero'),routed=dict(pair=44,anchor=30,whole_scope='three-valued unknown [1,0,0]'),pair_blocks={k:[v.start,v.stop] for k,v in routed.pair_blocks.items()},anchor_blocks={k:[v.start,v.stop] for k,v in routed.anchor_blocks.items()},reference_count=0,missing_evidence=['verified site-community relations','verified geographic clause applicability','reference memberships'],crossing='no reference tokens: empty array, no asserted no-road observations',name='exact native tokenizer and four soft name channels; never a fact'))
  probabilities=[];seed_errors={}
  for seed in SEEDS:
   state=torch.load(models/'教师权重'/f'{seed}.pt',map_location='cpu',weights_only=True)
   model=expanded_model(state,seed,'cpu').eval();values=[]
   with torch.no_grad():
    for lo in range(0,n,64):values.append(model(MD.batch(control,np.arange(lo,min(lo+64,n)))).softmax(1).numpy())
   p=np.concatenate(values);assert p.shape==(n,S) and np.isfinite(p).all() and np.allclose(p.sum(1),1)
   # True 41/27 teacher must agree with the zero-expanded 44/30 adapter.
   from zoning.modernnca_multilabel import MultiLabelModernNCA
   original=MultiLabelModernNCA(41,27,seed,32).eval();original.load_state_dict(state)
   with torch.no_grad():check=original(MD.batch(native,np.arange(n))).softmax(1).numpy()
   seed_errors[str(seed)]=float(np.max(np.abs(p-check)));assert np.allclose(p,check,atol=2e-6,rtol=1e-5)
   probabilities.append(p)
  prob=np.stack(probabilities).mean(0)
  feature=np.concatenate([features(routed,prob,lo,min(lo+32,n)) for lo in range(0,n,32)])
  assert feature.shape==(n,S,len(scale['mean'])) and np.isfinite(feature).all()
  admission_shapes=[]
  for seed in SEEDS:
   net=Admission(feature.shape[-1],seed).eval();net.load_state_dict(torch.load(models/'补校权重'/f'{seed}.pt',map_location='cpu',weights_only=True))
   with torch.no_grad():q=net(torch.from_numpy(np.clip((feature-scale['mean'])/scale['std'],-20,20))).sigmoid().numpy()
   assert q.shape==(n,S) and np.isfinite(q).all();admission_shapes.append(list(q.shape))
  np.savez_compressed(SZ/'补校模型输入'/f'{d}.npz',feature=feature,prob=prob,labels=label,original_labels=label,train_ids=ys['train_ids'],valid_ids=ys['valid_ids'],valid_labels=ys['valid_labels'],weight=np.zeros(n,np.float64))
  audit[d]=dict(sites=n,schools=S,references=0,observed_training_rows=0,pair_shape=list(routed.pair.shape),anchor_shape=list(routed.anchor.shape),admission_feature_shape=list(feature.shape),name_feature_rows=int((name>0).any((1,2)).sum()),teacher_weights_loaded=3,admission_weights_loaded=3,teacher_zero_expansion_error=seed_errors,finite_inference=True,gold_read=False,training_run=False,missing_community_relations=True)
  print('INPUT',d,json.dumps(audit[d],ensure_ascii=False),flush=True)
 files=[p for d in DISTRICTS for p in (INPUT/d).rglob('*') if p.is_file()]+[SZ/'补校模型输入'/f'{d}.npz' for d in DISTRICTS]
 write(REPORT/'输入冻结.json',dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in files},districts=audit,denied_read_attempts=denied,opened_paths=sorted(opened),target_gold_or_polygon_read=False))
 return audit

NODE=r"""
const fs=require('fs'),vm=require('vm');const data=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
class Point{constructor(lng,lat){this.lng=lng;this.lat=lat}equals(p){return this.lng===p.lng&&this.lat===p.lat}}
class Bounds{constructor(sw,ne){this.sw=sw;this.ne=ne}getSouthWest(){return this.sw}getNorthEast(){return this.ne}}
class Polygon{constructor(p){this.p=p}getPath(){return this.p}getBounds(){return new Bounds(new Point(Math.min(...this.p.map(p=>p.lng)),Math.min(...this.p.map(p=>p.lat))),new Point(Math.max(...this.p.map(p=>p.lng)),Math.max(...this.p.map(p=>p.lat))))}}
const fn=vm.runInNewContext('('+data.function+')',{BMap:{Point,Bounds,Polygon}},{timeout:1000});
const ctx={isPointInRect(p,b){return p.lng>=b.sw.lng&&p.lng<=b.ne.lng&&p.lat>=b.sw.lat&&p.lat<=b.ne.lat}};
const call=(pt,v)=>+fn.call(ctx,new Point(...pt),new Polygon(v.map(p=>new Point(...p))));
const sq=[[114,22],[114.01,22],[114.01,22.01],[114,22.01]];
const tests=[[[114.005,22.005],1],[[114,22.005],1],[[114,22],1],[[113.99,22],0]];
for(const [p,e] of tests)if(call(p,sq)!==e)throw Error('Boundary predicate failed');
process.stdout.write(JSON.stringify(data.points.map(p=>data.zones.map(z=>call(p,z)))));
"""
def parity(pt,vertices):
 # Independent even-odd ray implementation; unmodified closed path, boundary inclusive.
 x,y=pt;inside=False
 for a,b in zip(vertices,vertices[1:]+vertices[:1]):
  ax,ay=a;bx,by=b;cross=(x-ax)*(by-ay)-(y-ay)*(bx-ax)
  if abs(cross)<1e-14 and min(ax,bx)-1e-12<=x<=max(ax,bx)+1e-12 and min(ay,by)-1e-12<=y<=max(ay,by)+1e-12:return True
  if (ay>y)!=(by>y) and x < ax+(y-ay)*(bx-ax)/(by-ay):inside=not inside
 return inside

def build_gold():
 denied,opened=guard('gold')
 from shapely.geometry import Polygon,Point
 frozen=read(REPORT/'输入冻结.json')
 for rel,h in frozen['files'].items():assert sha(ROOT/rel)==h
 raw=read(SOURCE/'来源存档/深圳市/光明坪山准入核查_20261115/current_sources/guangming_map_browser.json')
 function=raw['point_predicate'];assert not raw['actual_boundary_overrides']
 protocol=read(SOURCE/'构建协议.json');assert protocol['pingshan_gold_basis_user_confirmed'] and protocol['guangming_school_polygon_user_confirmed']
 kernel=REPORT/'点落面核验.cjs';kernel.write_text(NODE,'utf-8')
 summaries={}
 for d in DISTRICTS:
  sites=read(SOURCE/d/'小区数据/坐标组.json');zones=read(SOURCE/d/'polygons.json');schools=read(INPUT/d/'metadata.json')['schools']
  sid=[s['school_id'] for s in schools];assert len(set(sid))==len(sid) and {z['school_id'] for z in zones}==set(sid)
  assert all(z['active_for_computation'] for z in zones)
  if d=='坪山区':assert any(z['source_school_id']==protocol['selected_pingshan_source_school_id'] for z in zones) and not any(z['source_school_id']==protocol['archived_pingshan_source_school_id'] for z in zones)
  request=REPORT/f'{d}_点落面请求.json';write(request,{'function':function,'points':[s['representative_point_bd09'] for s in sites],'zones':[z['vertices'] for z in zones]})
  process=subprocess.run(['node',str(kernel),str(request)],capture_output=True,text=True,encoding='utf-8',check=True)
  path_hits=np.array(json.loads(process.stdout),np.int8)
  matrix=np.zeros((len(sites),len(schools)),np.int8);boundaries={};checks=0;parity_checks=0;invalid=0
  for j,z in enumerate(zones):
   poly=Polygon(z['vertices']);invalid+=int(not poly.is_valid)
   assert poly.is_valid==z['geometry_valid']
   matrix[:,sid.index(z['school_id'])]|=path_hits[:,j]
   for i,site in enumerate(sites):
    p=site['representative_point_bd09'];v=parity(p,z['vertices']);parity_checks+=1
    assert v==bool(path_hits[i,j]),(d,i,j,'parity')
    point=Point(p)
    if poly.is_valid:checks+=1;assert poly.covers(point)==bool(path_hits[i,j]),(d,i,j,'GEOS')
    if poly.boundary.distance(point)==0:boundaries.setdefault(i,[]).append(z['zone_id'])
  records=[];gold=[]
  for i,site in enumerate(sites):
   hits=np.flatnonzero(matrix[i]);zh=np.flatnonzero(path_hits[i]);ids=[sid[j] for j in hits]
   row=dict(site,gold_record_id='point:'+site['coordinate_group_id'],source_map_partition=d,label_kind='official_map_point_membership',label_status='outside_selected_map' if not len(hits) else 'available' if len(hits)==1 else 'multiple_schools',predicate='BMapLib.GeoUtils.isPointInPolygon captured Guangming kernel; BD09 raw path',point_school_ids=ids,point_school_names=[schools[j]['school_name'] for j in hits],point_school_count=len(ids),point_zone_ids=[zones[j]['zone_id'] for j in zh],gold_school_id=ids[0] if len(ids)==1 else None,gold_school_name=schools[hits[0]]['school_name'] if len(ids)==1 else None,on_boundary_zone_ids=boundaries.get(i,[]),invalid_geometry_hit_zone_ids=[zones[j]['zone_id'] for j in zh if not zones[j]['geometry_valid']],unknown_school_ids=[],complete_within_selected_roster=True,complete_within_all_official_schools=False,extra_policy_application_expanded=False,individual_enrollment_eligibility=False,base_scope_policy_alignment_verified=False,live_website_requeried=False,captured_override_inventory_available=(d=='光明区'),map_snapshot='2026-11-15' if d=='光明区' else '2026-10-23',raw_geometry_repaired=False)
   records.append(row)
   if len(ids):gold.append(row)
  base=STANDARD/d
  write(base/'学校数据/schools.json',schools);write(base/'schools.json',schools)
  write(base/'小区数据/坐标组.json',sites);write(base/'小区数据/buildings.json',read(SOURCE/d/'小区数据/buildings.json'))
  write(base/'政策文本/年度原表.json',read(SOURCE/d/'政策文本/年度原表.json'))
  write(base/'评价数据/官方学区范围.json',zones);write(base/'评价数据/全部候选_点落面结果.json',records)
  write(base/'Gold/小区_学校Gold.json',gold)
  np.savez_compressed(base/'Gold/评价标签.npz',membership=matrix,evaluation_mask=matrix.any(1),site_ids=np.array([s['coordinate_group_id'] for s in sites]),school_ids=np.array(sid),path_membership=path_hits,zone_ids=np.array([z['zone_id'] for z in zones]))
  counts=matrix.sum(1);summary=dict(candidate_coordinate_groups=len(sites),schools=len(sid),gold_coordinate_groups=len(gold),single_school=int((counts==1).sum()),multiple_school=int((counts>1).sum()),outside_selected_map=int((counts==0).sum()),max_school_count=int(counts.max()),raw_invalid_paths_preserved=invalid,independent_even_odd_pairs=parity_checks,independent_valid_geos_pairs=checks,boundary_groups=len(boundaries),feature_row_order_aligned=True,geometry_repair=False)
  summaries[d]=summary;write(base/'数据清单/校验结果.json',dict(passed=True,**summary))
  write(base/'实验输入/输入清单.json',dict(model_input=f'../../../模型输入/{d}',admission_input=f'../../../补校模型输入/{d}.npz',evaluation_labels=f'../Gold/评价标签.npz',gold_is_evaluation_only=True,model_arrays_ready=True,gold_generated=True,missing_independent_community_memberships=True))
  print('GOLD',d,json.dumps(summary,ensure_ascii=False),flush=True)
 for rel,h in frozen['files'].items():assert sha(ROOT/rel)==h
 write(REPORT/'Gold验收.json',dict(districts=summaries,feature_hashes_unchanged_after_gold=True,point_kernel_source='数据/深圳/光明坪山/来源存档/深圳市/光明坪山准入核查_20261115/current_sources/guangming_map_browser.json::point_predicate',point_kernel_sha256=hashlib.sha256(function.encode()).hexdigest(),pingshan_kernel_basis='archived page calls BMapLib.GeoUtils1.2; same kernel, raw authorized 2026-10-23 paths, live query overrides not obtained',no_training=True,old_read_attempts=denied))
 return summaries

def verify():
 guard('verify')
 import torch
 from zoning import evidence
 snapshot=read(REPORT/'输入冻结.json')
 for rel,h in snapshot['files'].items():assert sha(ROOT/rel)==h,rel
 for d in DISTRICTS:
  meta=read(INPUT/d/'metadata.json');ids=[s['coordinate_group_id'] for s in meta['sites']];sids=[s['school_id'] for s in meta['schools']]
  with np.load(STANDARD/d/'Gold/评价标签.npz',allow_pickle=False) as z:
   assert z['site_ids'].tolist()==ids and z['school_ids'].tolist()==sids
   rows=read(STANDARD/d/'Gold/小区_学校Gold.json');expected={ids[i]:{sids[j] for j in np.flatnonzero(z['membership'][i])} for i in np.flatnonzero(z['evaluation_mask'])}
   assert {r['coordinate_group_id']:set(r['point_school_ids']) for r in rows}==expected
  for arm in ['control','routed']:
   with (INPUT/d/f'{arm}.pkl').open('rb') as f:x=pickle.load(f)
   assert x.n==len(ids) and x.school_n==len(sids) and (x.np_,x.na)==(44,30)
   assert x.membership.shape==(len(sids),len(sids)) and torch.equal(x.membership,torch.eye(len(sids)))
   assert np.all(x.y==-1) and not len(x.supervised) and not len(x.ref_sites)
   with np.load(INPUT/d/f'{arm}_labels.npz',allow_pickle=False) as z:assert (z['labels']==-1).all() and not len(z['train_ids']) and not len(z['valid_ids'])
  assert np.all(x.anchor[:,:,-3:]==np.array([1,0,0]))
 write(REPORT/'总验收.json',dict(passed=True,districts=DISTRICTS,input_hashes_verified=True,full_population_and_school_alignment=True,target_gold_excluded_from_features=True,model_weight_compatibility=True,training_run=False,independent_community_memberships_ready=False))
 print('VERIFY PASS',flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['inputs','gold','verify','all']);a=p.parse_args()
 if a.mode!='verify' and (REPORT/'总验收.json').exists():raise FileExistsError('Completed snapshot exists; preserve it and create a separately versioned build before regeneration')
 if a.mode=='all':
  for mode in ['inputs','gold','verify']:subprocess.run([sys.executable,'-B','-X','utf8',__file__,mode],check=True)
 elif a.mode=='inputs':build_inputs()
 elif a.mode=='gold':build_gold()
 else:verify()
if __name__=='__main__':main()
