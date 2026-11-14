"""Read the official residential-only table; resolve only declared merged cells."""
from collections import Counter
from pathlib import Path
import re
import openpyxl
from bs4 import BeautifulSoup
import shenzhen_full_inputs as F
from inventory_shenzhen_community_profiles import norm

ROOT=F.REPO/'outputs/shenzhen_resolution_20261112'
OUT=ROOT/'municipal_residential_membership_sources'
PAGE='https://www.lg.gov.cn/zwfw/zdfw/csgl/tzgg/content/post_12425962.html'
URL='https://www.lg.gov.cn/attachment/1/1589/1589277/12425962.xlsx'


def extract():
    path=OUT/'longgang2024_residential.xlsx'
    page=OUT/'longgang2024_notice.html'
    soup=BeautifulSoup(page.read_text('utf8'),'html.parser')
    assert soup.find('meta',attrs={'name':'PubDate'})['content']=='2025-06-03'
    assert any(a['href']==URL and '住宅区' in a.get_text() for a in soup.find_all('a',href=True))
    wb=openpyxl.load_workbook(path,data_only=False)
    assert wb.sheetnames==['Sheet1'];ws=wb['Sheet1']
    assert (ws.max_row,ws.max_column)==(51,6)
    assert [ws.cell(3,c).value for c in range(1,7)]==['序号','街道','社区','场所名称','申报类型','激励对象']
    def resolve(row,col):
        cell=ws.cell(row,col)
        if cell.value is not None:return cell.value,cell.coordinate
        merges=[m for m in ws.merged_cells.ranges if cell.coordinate in m]
        assert len(merges)==1,(row,col)
        m=merges[0];parent=ws.cell(m.min_row,m.min_col)
        return parent.value,parent.coordinate
    rows=[]
    for row in range(4,52):
        assert ws.cell(row,1).value==row-3
        assert ws.cell(row,5).value=='单位' and ws.cell(row,6).value=='住宅区'
        st,st_cell=resolve(row,2);co,co_cell=resolve(row,3)
        street=re.fullmatch(r'(.+街道)\s*（(\d+)个）',st).group(1)
        name=ws.cell(row,4).value
        rows.append(dict(district='龙岗区',street=street,community=co,residential_name=name,
            quote=dict(street_cell_value=st,community_cell_value=co,residential_cell_value=name),
            cells=dict(street=st_cell,community=co_cell,residential=f'D{row}'),sheet='Sheet1',row=row,
            publication_date='2025-06-03',source_url=URL,notice_url=PAGE,
            source_sha256=F.sha(path),notice_sha256=F.sha(page)))
    for st,n in Counter(x['street'] for x in rows).items():
        raw=next(x['quote']['street_cell_value'] for x in rows if x['street']==st)
        assert int(re.search(r'（(\d+)个）',raw)[1])==n
    return rows


def main():
    F.guard();dest=OUT/'longgang_review_queue.json'
    if dest.exists():raise FileExistsError('Preserve source inventory')
    rows=extract()
    sites=F.read(ROOT/'round03_partial_labels/inputs/龙岗区.json')['sites']
    prior=F.read(ROOT/'community_membership_sources/review_v1/membership_facts.json')['facts']
    profiles=F.read(ROOT/'community_membership_sources/profile_inventory.json')['profiles']
    for r in rows:
        exact=[];partial=[]
        for s in sites:
            names=s['coordinate_group_building_names'];sid=s['coordinate_group_id']
            entry=dict(site_id=sid,site_names=names,prior_memberships=[dict(street=f['street'],community=f['community']) for f in prior if f['site_id']==sid and f['district']==r['district']])
            if all(norm(n)==norm(r['residential_name']) for n in names):exact.append(entry)
            elif any(norm(n) in norm(r['residential_name']) or norm(r['residential_name']) in norm(n) for n in names):partial.append(entry)
        r.update(exact_coordinate_group_matches=exact,partial_name_matches=partial,
            profile_identities=sorted({(p['street'],p['community']) for p in profiles if p['district']==r['district'] and p['community']==r['community']}),
            admitted=False,status='pending_identity_conflict_and_temporal_review',complete_school_set=False)
    summary=dict(residential_rows=len(rows),streets=len({r['street'] for r in rows}),
        exact_coordinate_groups=len({h['site_id'] for r in rows for h in r['exact_coordinate_group_matches']}),
        exact_without_prior=len({h['site_id'] for r in rows for h in r['exact_coordinate_group_matches'] if not h['prior_memberships']}),
        school_relations_generated=0,target_gold_read=False,target_polygons_read=False)
    F.write(dest,dict(summary=summary,statements=rows))
    links=[Path(__file__),OUT/'longgang2024_notice.html',OUT/'longgang2024_residential.xlsx',dest,
        ROOT/'round03_partial_labels/inputs/龙岗区.json',ROOT/'community_membership_sources/review_v1/membership_facts.json',ROOT/'community_membership_sources/profile_inventory.json']
    F.write(OUT/'longgang_inventory_manifest.json',dict(files={str(p):F.sha(p) for p in links},target_gold_read=False,admitted=False))
    print(summary,flush=True)
    for r in rows:
        if r['exact_coordinate_group_matches'] or r['partial_name_matches']:
            print(r['row'],r['street'],r['community'],r['residential_name'],'EXACT',r['exact_coordinate_group_matches'],'PARTIAL',[h['site_names'] for h in r['partial_name_matches']],flush=True)


if __name__=='__main__':main()
