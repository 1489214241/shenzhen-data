"""Verify original residential cells, source proof and every scored prediction."""
import argparse
from collections import defaultdict,Counter
import pickle
import re
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
from bs4 import BeautifulSoup
import experiment_shenzhen_residential_supplement as D
from inventory_shenzhen_community_profiles import norm

F=D.F;R=D.R


def source():
    F.guard();D.verify()
    dest=D.OUT/'source_replay_audit.json'
    if dest.exists():raise FileExistsError('Preserve source audit')
    root=D.SOURCE.parent;ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    # Independent OOXML read checks every table cell and declared merge anchor.
    with zipfile.ZipFile(root/'longgang2024_residential.xlsx') as z:
        strings=[''.join(t.text or '' for t in n.findall('.//s:t',ns)) for n in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)]
        tree=ET.fromstring(z.read('xl/worksheets/sheet1.xml'));values={}
        for cell in tree.findall('.//s:sheetData/s:row/s:c',ns):
            assert cell.find('s:f',ns) is None,'Source unexpectedly contains formulas'
            v=cell.find('s:v',ns)
            if v is None:continue
            values[cell.attrib['r']]=strings[int(v.text)] if cell.attrib.get('t')=='s' else v.text
        merges={n.attrib['ref'] for n in tree.findall('.//s:mergeCell',ns)}
    rows=F.read(root/'longgang_review_queue.json')['statements']
    assert len(rows)==48
    for row in rows:
        n=row['row'];assert int(values[f'A{n}'])==n-3
        assert values[f'D{n}']==row['residential_name']
        for col,key in [('B','street'),('C','community')]:
            anchor=row['cells'][key]
            if anchor!=f'{col}{n}':
                containing=[m for m in merges if re.fullmatch(col+r'\d+:'+col+r'\d+',m) and int(re.findall(r'\d+',m)[0])<=n<=int(re.findall(r'\d+',m)[1])]
                assert len(containing)==1 and containing[0].split(':')[0]==anchor
            expected=row['quote']['street_cell_value' if key=='street' else 'community_cell_value']
            assert values[anchor]==expected
    lh=F.read(root/'longhua_review_queue.json')
    soup=BeautifulSoup((root/'longhua2022.html').read_text('utf8'),'html.parser')
    ps=[p.get_text('',strip=True) for p in soup.find_all('p')]
    first=ps.index('“生活垃圾分类绿色小区”');last=ps.index('“生活垃圾分类好家庭”',first)
    section=''.join(ps[first+1:last])
    for row in lh['statements']:assert row['quote'] in section
    all_statements=lh['statements']+rows
    prior=F.read(R.ROOT/'community_membership_sources/review_v1/membership_facts.json')
    accepted=F.read(D.SOURCE/'membership_facts.json')['facts'];ledger=F.read(D.SOURCE/'review_ledger.json')['rows']
    assert len(ledger)==74 and len(accepted)==14
    source_pairs=set()
    for f in accepted:
        meta=F.read(R.ROOT/'round03_partial_labels/inputs'/f"{f['district']}.json")
        site=next(s for s in meta['sites'] if s['coordinate_group_id']==f['site_id'])
        assert site['coordinate_group_building_names']==f['site_names']
        found=[s for s in all_statements if s['district']==f['district'] and (s['street'],s['community'])==(f['street'],f['community']) and all(norm(n)==norm(s['residential_name']) for n in f['site_names'])]
        assert len(found)==1 and found[0]['publication_date']<='2026-06-01'
        assert not any(p['district']==f['district'] and p['site_id']==f['site_id'] for p in prior['facts'])
        assert not f['complete_school_set'] and not f['membership_absence_is_negative']
        source_pairs.add((f['district'],f['site_id'],f['community']))
    count=Counter()
    for d in R.TARGETS:
        raw=F.read(D.SOURCE/'school_facts'/f'{d}.json')['facts'];final=F.read(D.OUT/'facts'/f'{d}.json')
        positives={(r['site_id'],r['school_id']):r for r in final['facts']}
        pending={(r['site_id'],r['school_id']) for r in final['pending_eligibility_only']}
        assert set(positives).isdisjoint(pending)
        assert set(positives)|pending=={(r['site_id'],r['school_id']) for r in raw}
        for r in raw:
            assert (d,r['site_id'],r['community']) in source_pairs
            has_member=any(w['kind']=='official_community_membership' and w['community']==r['community'] and w['street']==r['street'] for w in r['proof'])
            assert has_member==((r['site_id'],r['school_id']) in positives)
            if has_member:assert positives[r['site_id'],r['school_id']]==r
        count['accepted_relations']+=len(positives);count['held_street_only']+=len(pending)
    assert count==dict(accepted_relations=12,held_street_only=4)
    F.write(dest,dict(original_xlsx_rows=48,html_explicit_community_rows=26,html_missing_community_rows=7,
        reviewed_rows=74,new_memberships=14,counts=dict(count),original_merged_cells_verified=True,
        source_hashes_verified=True,target_gold_read=False,target_polygon_read=False,code_sha256=F.sha(__file__)))
    print('SOURCE_AUDIT',dict(count),flush=True)


def score():
    from zoning.evaluate import bootstrap
    D.verify();dest=R.ROOT/'round20_independent_completion_audit.json'
    if dest.exists():raise FileExistsError('Preserve score audit')
    assert F.read(D.OUT/'source_replay_audit.json')['original_merged_cells_verified']
    c=F.read(D.OUT/'protocol.json');claimed=F.read(D.OUT/'results.json')['reports']
    for d,h in F.read(D.OUT/'results.json')['gold_sha256'].items():assert F.sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    gold={(r['district'],r['site_id']):r for r in F.read(R.BASE/'scored_cases.json')};xy={};facts={}
    for d in R.TARGETS:
        with (R.NEW/d/'district.pkl').open('rb') as h:dist=pickle.load(h)
        xy.update({(d,s):p for s,p in zip(dist.site_ids,dist.coords)})
        facts[d]=defaultdict(set)
        for f in F.read(D.OUT/'facts'/f'{d}.json')['facts']:facts[d][f['site_id']].add(f['school_id'])
    checks={};filecount=0
    for arm in c['arms']:
        for obj in c['objectives']:
            for label in c['labels']:
                scored=[];all_keys=[];unscored=[]
                for d in R.TARGETS:
                    before=F.read(D.parent_path(arm,obj,label,d));after=F.read(D.OUT/'predictions'/arm/obj/label/f'{d}.json');filecount+=1
                    assert [r['site_id'] for r in before]==[r['site_id'] for r in after]
                    for a,b in zip(before,after):
                        key=(d,b['site_id']);all_keys.append(key);expected=set(a['after'])|facts[d][b['site_id']]
                        assert b['district']==d and set(b['before'])==set(a['after'])
                        assert set(b['after'])==expected and set(b['added_school_ids'])==expected-set(a['after'])
                        assert len(b['after'])==len(expected)
                        g=gold[key]
                        if not g['has_map_gold']:unscored.append(b);continue
                        scored.append(dict(b,gold=set(g['gold']),d=d,block=str(tuple(int(v//1000) for v in xy[key]))))
                assert len(all_keys)==len(set(all_keys))==1802 and len(scored)==1761 and len(unscored)==41
                subsets=[('all',scored)]+[(d,[r for r in scored if r['d']==d]) for d in R.TARGETS]+[
                    ('single',[r for r in scored if len(r['gold'])==1]),('multi',[r for r in scored if len(r['gold'])>1])]
                groups={}
                for name,rows in subsets:
                    expected=claimed[arm][obj][label][name];summary={};changes=[]
                    for field in ['before','after']:
                        sets=[(set(r[field]),r['gold']) for r in rows]
                        stats=dict(n=len(rows),correct=sum(a==b for a,b in sets),fp=sum(len(a-b) for a,b in sets),fn=sum(len(b-a) for a,b in sets))
                        stats['exact_accuracy']=stats['correct']/stats['n']
                        assert stats==expected[field];summary[field]=stats
                    for r in rows:changes.append(int(set(r['after'])==r['gold'])-int(set(r['before'])==r['gold']))
                    summary.update(corrected=changes.count(1),harmed=changes.count(-1),delta_pp=float(np.mean(changes)*100),
                        delta_95_pp=bootstrap(rows,changes,seed=20261104,draws=3000))
                    assert summary==expected;groups[name]=summary
                checks[f'{arm}/{obj}/{label}']=groups
    assert filecount==88
    F.write(dest,dict(results=checks,prediction_files=filecount,all_scored_groups_recomputed=220,
        source_relations=12,no_street_only_additions=True,no_transitive_additions=True,original_gold_unchanged=True,
        independent_validation=False,adopted=False,code_sha256=F.sha(__file__)))
    print('ALL88PREDICTION_FILES_AND220METRIC_GROUPS_VERIFIED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['source','score']);globals()[p.parse_args().phase]()
