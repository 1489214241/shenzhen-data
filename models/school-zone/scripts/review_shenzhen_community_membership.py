"""Freeze an exhaustive source-only review of the 219 literal-hit paragraphs.

This is a reviewed evidence ledger, not an automatic substring classifier.
Selections below were made after reading every paragraph and its proposed site
names, without loading target labels, predictions, or school polygons. Missing
names remain unknown. Publication dates are preserved, never backdated.
"""
from collections import Counter, defaultdict
from pathlib import Path
import re
import shenzhen_full_inputs as F
from inventory_shenzhen_community_profiles import norm

ROOT = F.REPO / 'outputs/shenzhen_resolution_20261112/community_membership_sources'
OUT = ROOT / 'review_v1'

# Exact reviewed dataset names supported by the residential/industrial/building
# membership clause. A decorative dot or whitespace is the only normalization.
# '*' is permissible only for paragraphs whose EVERY proposed name was reviewed.
SELECT = {
    0:'*',1:'*',4:'*',5:'*',10:'湖润名苑|荣辉花园|兆利花园',12:'*',15:'*',18:'*',19:'*',
    29:'*',31:'*',34:'*',47:'*',48:'清湖宝湖居',49:'*',51:'*',
    54:'合能城市阳光花园',55:'*',57:'*',65:'*',71:'*',74:'南源商业大厦',
    76:'*',78:'*',81:'*',83:'*',86:'德兴花园|粤宝花园|泽润华庭|宝丽花园',87:'*',
    89:'*',93:'大芬油画苑|兴业花园',94:'*',96:'*',98:'*',99:'*',101:'*',
    102:'半岛苑|中海怡瑞山居|龙园意境|倚山别苑|来座山',103:'*',105:'*',106:'*',
    109:'*',110:'*',117:'*',118:'*',121:'*',122:'*',123:'*',124:'*',125:'*',126:'*',
    131:'富豪花园|恒大都会广场',132:'富豪花园|恒大都会广场',
    133:'中海月朗苑|万科紫悦山',134:'中海月朗苑|万科紫悦山',139:'*',140:'*',
    147:'*',148:'*',149:'康乐花园|信义锦绣花园|中海怡美山庄|天颂雅苑|泉井湖小区|悦心园|乐清小区',
    151:'*',154:'*',155:'*',157:'长发楼|水晶花园|华龙新村',160:'尚水天成|海心汇福园',
    162:'龙翔花园|又一村花园|秀丽山庄|青青家园|金沙花园|下水径新村',163:'岸上林居',
    166:'*',167:'国香尚居|玉湖山畔',169:'*',170:'*',172:'*',175:'*',181:'合正丹郡',
    183:'*',185:'*',188:'*',191:'*',193:'安鸿峰景苑',194:'*',197:'*',201:'*',204:'*',206:'*',
    209:'名居广场|阳光天健城|奥林华府|星河时代|中粮祥云|缤纷世纪|龙城华府|园景花园|海逸雅居|雅庭名苑|琳珠华庭|顺景花园|星龙园',
    210:'保利上城|美地里园|鸿进花园|东都中央王座|吉祥来花园|龙翔花园|鸿基花园|余岭山庄|风临域|德馨楼|珑城原点广场',
    211:'华策中心城|恒大城市之光',212:'*',213:'*',214:'嘉欣园|黄阁翠苑',
    215:'佳兆业未来花园|嘉亿爵悦|睿智华庭|天昊华庭|天健现代城|水蓝湾',216:'*',217:'*',
    218:'新亚洲花园|新鸿进花园|碧湖玫瑰园|碧湖花园',
}

# A review reason for EVERY paragraph without an accepted membership clause.
# These rejections concern this evidence, not a negative membership label.
REASONS = {}
def reason(ids, text):
    for i in ids:
        assert i not in REASONS
        REASONS[i] = text

reason([2,7,14,24,50,53,58,60,62,63,64,67,80,95,97,135,136,141,142,150,152,161,177,186,190,192,196],
       '办公或服务地址不证明整座小区属于该社区，且可能仅限楼栋、房间或旧址。')
reason([3,6,16,26,36,40,66,68,69,73,82,84,107,108,145,146,173,195,207],
       '只在幼儿园、社康、活动中心或周边服务机构名称中命中，不据此确认同名小区整段归属。')
reason([8,11,13,17,22,25,35,39,52,56,61,72,75,77,79,85,88,90,104,111,112,115,116,119,120,129,130,137,138,143,144,156,159,164,176,184,189,200,202],
       '党组织或服务关系不是独立行政归属依据；其他段落的明确范围列举另行保留。')
reason([9,44,45,70,92,158,171,182,187],
       '边界、相邻或附近对象的提及不证明范围包含，不产生正负归属标签。')
reason([20,21,23,38,41,174,178,179,198],
       '命中社区本身名称、道路、电话或活动叙述；数据点与整个行政单元的身份未确认。')
reason([27,37,100], '荣誉或业委会活动叙述未给出明确管辖范围。')
reason([28,30,32,33,205], '同名街区或片区仍需与数据中的具体住宅实体核实身份，不直接按前缀转换。')
reason([42,43,46], '包含相邻地区、商业设施或历史分设描述，不能据此将数据中的同名地点整体归入。')
reason([59], '只列玉翠新村AB区、C区，不能无证据扩展成整个同名小区。')
reason([91,168], '居民小组与同名住宅数据点不是已确认的同一空间实体，保留待核。')
reason([113,114,153,165], '源范围列举的是更完整名称；简称与完整名称的身份尚需独立证明。')
reason([127,128,180,203], '企业或商业叙述与数据中的住宅/地点实体身份尚未独立确认。')
reason([199], '统计口径为户籍人口且明确包含空挂户，不能当作住宅地理归属。')
reason([208], '城市更新项目与数据中的住宅实体、覆盖阶段和范围尚未确认。')

# More specific explanations for mixed paragraphs, including dangerous substrings.
MIXED = {
    10:'浪漫新城原文仅列AB座，不能扩大到整座小区。',
    48:'清湖湾只命中清湖湾花园前缀，身份未确认。',
    54:'依山小筑、龙苑新村仅见党支部名称；不据此确认地理归属。',
    74:'商业中心名称不等同整个同名楼盘。',
    86:'钱排村仅作为社区位置描述出现，非该段辖区清单。',
    93:'百门前工业区是边界参照，非该段管辖列表。',
    102:'茂业城是东界参照；明确管辖关系见茂业社区其他段落。',
    131:'诺德中央道、华晖瑞禧与完整花园名身份未核。',
    132:'诺德中央道、华晖瑞禧与完整花园名身份未核。',
    133:'璞悦山只命中万科璞悦山后缀，尚未确认身份。',
    134:'璞悦山只命中万科璞悦山后缀，尚未确认身份。',
    149:'颂雅苑只是天颂雅苑的子串；宝岭社区另明确列有颂雅苑，不能合并。',
    157:'花语岸、花语馨与完整花园名身份未核。',
    160:'福园小区是海心汇福园小区的子串；下水径居民小组与地点身份未核。',
    162:'下水径只命中下水径新村，不能从子区扩大到整体。',
    163:'龙西社区名称本身不证明同名数据点身份。',
    167:'龙西社区为邻界；承翰陶源与完整花园名身份未核。',
    181:'南岭村为南界相邻社区。',
    193:'统建楼小区过于通用且多处同名；恒大国香山与完整花园名身份未核。',
    209:'爱联社区名称本身不证明同名数据点身份；颐安仅二期、五期不可扩展。',
    210:'万科天誉同时命中花园与中央广场，整体/子项目身份未核。',
    211:'爱联社区是历史分设来源，非现辖范围。',
    214:'爱联社区为相邻边界。',
    215:'京基御景中央仅一期、二期；吉祥里与吉祥里豪庭身份未核。',
    218:'鸿进花园只命中新鸿进花园子串；园景花园只是南侧边界参照。',
}

def review(proposals, profiles):
    assert len(proposals) == 219
    assert set(SELECT).isdisjoint(REASONS)
    assert set(SELECT) | set(REASONS) == set(range(len(proposals)))
    ledger=[]; candidates=[]
    for i,r in enumerate(proposals):
        # Verify that the reviewed paragraph really belongs to the archived profile.
        assert any(p['district']==r['district'] and p['street']==r['street'] and
                   p['community']==r['community'] and p['source']==r['source'] and
                   p['paragraphs'][r['paragraph_index']]==r['quote'] for p in profiles
                   if len(p['paragraphs'])>r['paragraph_index'])
        all_names={n for h in r['hits'] for n in h['site_names']}
        selected = all_names if SELECT.get(i)=='*' else set(SELECT.get(i,'').split('|'))-{''}
        assert selected <= all_names,(i,selected-all_names)
        assert all(norm(n) in norm(r['quote']) for n in selected)
        hits=[]
        for h in r['hits']:
            supported=set(h['site_names']) <= selected
            hr=dict(h, membership_supported_by_paragraph=supported,
                    reason='官方明确范围列举；坐标组所有名称均有本段支持。' if supported else
                    REASONS.get(i,MIXED.get(i,'坐标组有名称尚未获得完整、无条件的本段支持。')))
            hits.append(hr)
            if supported:
                candidates.append(dict(district=r['district'],street=r['street'],community=r['community'],
                    site_id=h['site_id'],site_names=h['site_names'],paragraph_review_id=i,
                    source=r['source'],quote=r['quote'],complete_school_set=False,
                    enrollment_eligibility=False,kind='official_profile_geographical_membership'))
        ledger.append(dict(review_id=i,**r,reviewed_hits=hits,selected_names=sorted(selected),
                           paragraph_reason=REASONS.get(i,MIXED.get(i,'明确管辖列表；仅采用逐一核验的实体。'))))
    grouped=defaultdict(list)
    for c in candidates:grouped[c['district'],c['site_id']].append(c)
    facts=[]; pending=[]
    for (d,sid),rows in grouped.items():
        identities={(r['street'],r['community']) for r in rows}
        # A shared spelling is not proof of a shared place. Never select by Gold,
        # school polygon, nearest school, or which choice improves performance.
        if len(identities)>1:
            pending.append(dict(district=d,site_id=sid,reason='multiple_community_identity_conflict',evidence=rows))
            continue
        street,community=next(iter(identities))
        dates=[date[:10] for r in rows for date in r['source']['page_dates']
               if re.match(r'^\d{4}-\d{2}-\d{2}',date)]
        # Prespecified conservative pre-enrollment cutoff, not a claim that an
        # old page is an authoritative survey of the full 2026 boundary.
        usable=[r for r in rows if any(re.match(r'^\d{4}-\d{2}-\d{2}',x) and
                    x[:10]<='2026-06-01' for x in r['source']['page_dates'])]
        f=dict(district=d,street=street,community=community,site_id=sid,site_names=rows[0]['site_names'],
               evidence=rows,pre_enrollment_evidence=usable,source_dates=sorted(set(dates)),
               complete_school_set=False,enrollment_eligibility=False,
               membership_absence_is_negative=False,source_age_limitation=True,
               status='source_supported_pre_enrollment' if usable else 'pending_temporal_confirmation')
        (facts if usable else pending).append(f)
    return ledger,facts,pending

def main():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve reviewed source snapshot')
    inv=F.read(ROOT/'profile_inventory.json')
    proposals=F.read(ROOT/'literal_hit_review_queue.json')['proposals']
    manifest=F.read(ROOT/'inventory_manifest.json')
    for rel,h in manifest['files'].items():assert F.sha(ROOT/rel)==h,rel
    ledger,facts,pending=review(proposals,inv['profiles'])
    summary={d:dict(reviewed_paragraphs=sum(r['district']==d for r in ledger),
        supported_site_community_facts=sum(f['district']==d for f in facts),
        pending_sites=sum(f['district']==d for f in pending),
        supported_paragraph_site_mentions=sum(h['membership_supported_by_paragraph'] for r in ledger
            if r['district']==d for h in r['reviewed_hits'])) for d in F.SZ.DISTRICTS}
    F.write(OUT/'review_ledger.json',dict(paragraphs=ledger,target_gold_read=False,target_polygon_read=False))
    F.write(OUT/'membership_facts.json',dict(facts=facts,pending=pending,summary=summary,
        rule='Only explicit reviewed geographical inclusion; all coordinate-group names required; no negative labels.',
        cutoff='2026-06-01',temporal_limitation='Publication date is not proof of unchanged boundaries in 2026. Later-only evidence held out.',
        target_gold_read=False,target_polygon_read=False,adopted=False))
    F.write(OUT/'manifest.json',dict(code_sha256=F.sha(__file__),
        source_inventory_manifest_sha256=F.sha(ROOT/'inventory_manifest.json'),
        files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()}))
    print(summary,flush=True)
    for r in pending:
        if r.get('reason'):print('CONFLICT',r['site_id'],[(e['community'],e['site_names']) for e in r['evidence']],flush=True)

if __name__=='__main__':main()
