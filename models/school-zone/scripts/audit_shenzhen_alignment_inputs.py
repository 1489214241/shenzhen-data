"""Publish source-only reference changes and every school's remaining limits."""
import pickle
import shenzhen_full_inputs as F

OUT=F.REPO/'outputs/shenzhen_aligned_model_20261111'
INPUT=F.REPO/'data/深圳市/规则对齐输入_20261111'
OLD=F.REPO/'data/深圳市/完整输入_20261111'


def nodes(a):
    yield a
    for c in a.get('children',[]):yield from nodes(c)


def main():
    F.guard()
    changes={};details={}
    lines=['逐校规则对齐与剩余限制台账（冻结输入；未使用目标评分）',
           'unknown 是执行状态，不是官方文字缺失。行政边界来自独立 OSM，不是学区面。',
           '参照是政策规则产生的输入估计；不是额外人工核验标签。学校数按目录校区计。','']
    for d in ['龙华区','龙岗区']:
        before=F.read(OLD/d/'model_metadata.json')['known'];after=F.read(INPUT/d/'model_metadata.json')['known']
        summary=dict(reference_sites_before=len(before),reference_sites_after=len(after),
            school_coverage_before=len({s for ss in before.values() for s in ss}),
            school_coverage_after=len({s for ss in after.values() for s in ss}),
            new_reference_sites=len(after.keys()-before.keys()),withdrawn_reference_sites=len(before.keys()-after.keys()),
            changed_reference_sets=sum(set(before[k])!=set(after[k]) for k in before.keys()&after.keys()))
        changes[d]=summary
        meta=F.read(INPUT/d/'model_metadata.json');sites=meta['sites'];schools=meta['schools']
        details[d]=[dict(site_id=sites[int(k)]['coordinate_group_id'],name=sites[int(k)]['building_name'],
            before=[schools[s]['school_name'] for s in before.get(k,[])],after=[schools[s]['school_name'] for s in after.get(k,[])])
            for k in sorted(before.keys()|after.keys(),key=int) if set(before.get(k,[]))!=set(after.get(k,[]))]
        rows=F.read(INPUT/d/'policy_observations.json')
        admins={r['policy_name']:r for r in F.read(INPUT/d/'admin_geometry_audit.json')}
        rings={tuple(r['names']):r for r in F.read(INPUT/d/'ring_geometry_audit.json')}
        roads=list(F.read(INPUT/d/'road_identity_audit.json').values())
        lines += [d,str(summary),'']
        for j,r in enumerate(rows):
            lines += [f"{j+1}. {r['school_name']}",f"原文：{r['source_text']}",
                f"结构类型：{r['ast']['op']}；目录状态：{r['scope_status']}",
                f"作为参照涉及的位置数：{sum(j in ss for ss in after.values())}；名称命中数：{r['named_sites']}"]
            for e in r['evidence']:lines.append(f"来源：{e['source_url']} {e['source_position']}")
            limits=[]
            for a in nodes(r['ast']):
                if a['op'] in ['unknown','conditional']:limits.append(f"未完整执行：{a.get('text')}；原因：{a.get('reason')}")
                elif a['op'] in ['admin','membership']:
                    rec=admins.get(a['text'],{})
                    if rec.get('status')!='independent_osm_boundary':limits.append(f"无唯一独立行政边界：{a['text']}；社区点仅为代理线索")
                elif a['op']=='ring':
                    rec=rings.get(tuple(a['names']),{})
                    if rec.get('status')!='unique_closed_ring':limits.append(f"围合未唯一执行：{'、'.join(a['names'])}；{rec.get('status','not_evaluated')}")
                elif a['op']=='road':
                    for name,di in a['pairs']:
                        match=[v for v in roads if v['policy_name']==name and v['inward']==di]
                        if match and all(v['status']!='ready' for v in match):limits.append(f"道路几何缺失：{name}以{di}")
            lines+=list(dict.fromkeys(limits))+['']
    source_coverage={}
    for d in ['天心区','芙蓉区','雨花区','开福区']:
        with (OUT/'inputs'/f'{d}.pkl').open('rb') as f:x=pickle.load(f)
        source_coverage[d]=dict(sites=x.n,schools=x.school_n,reference_sites=len(set(x.ref_sites)),reference_schools=len(set(x.ref_schools)))
    F.write(OUT/'参照变化_不读Gold.json',changes)
    F.write(OUT/'参照逐例变化_不读Gold.json',details)
    F.write(OUT/'长沙输入覆盖对照.json',source_coverage)
    (OUT/'逐校对齐与剩余限制.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(changes,flush=True);print(source_coverage,flush=True)


if __name__=='__main__':main()
