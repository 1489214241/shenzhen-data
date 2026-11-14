"""Source-only Shenzhen policy adapter using the actual Changsha primitives.

No school catchment geometry, evaluation labels, or prediction-derived facts.
The parser preserves unknown children; representation coverage is not certainty.
"""
import ast
import collections
import importlib.util
import re
import sys
import types
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, shape
from shapely.ops import unary_union, polygonize

from zoning import rules

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT/'科研推进/四至补全_20261028/执行器副本/rules/src'
SOURCE_FILES = [NATIVE/'longhua/rules.py', NATIVE/'name_matching.py',
                NATIVE/'address_scope.py', NATIVE/'road_solution/geometry.py',
                NATIVE/'v3_names.py',NATIVE/'contextual_names.py']


def load_native():
    # Import only the pure definitions, not the old dataset-owning runtime.
    p = SOURCE_FILES[0]; tree = ast.parse(p.read_text(encoding='utf-8'))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'phase_rule_class': break
        if isinstance(node, ast.ImportFrom) and node.level: continue
        if isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ORIGINAL_RULE_CLASS' for t in node.targets): continue
        nodes.append(node)
    phase = types.ModuleType('_aligned_native_phase'); sys.modules[phase.__name__] = phase
    exec(compile(ast.Module(nodes, type_ignores=[]), str(p), 'exec'), phase.__dict__)
    p = SOURCE_FILES[1]; tree = ast.parse(p.read_text(encoding='utf-8'))
    tree.body = [n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module=='longhua.rules')]
    names = types.ModuleType('_aligned_native_names')
    names.__dict__.update({k:getattr(phase,k) for k in ['identity','normalize','parse_scope','number']})
    exec(compile(tree,str(p),'exec'),names.__dict__)
    modules = []
    for i,p in enumerate(SOURCE_FILES[2:4]):
        spec=importlib.util.spec_from_file_location(f'_aligned_native_{i}',p)
        m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);modules.append(m)
    safety=types.ModuleType('_aligned_native_name_safety')
    tree=ast.parse(SOURCE_FILES[4].read_text(encoding='utf-8'))
    tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module=='contextual_names')]
    recall=next(n for n in ast.parse(SOURCE_FILES[5].read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='recall_key')
    tree.body.append(recall)
    exec(compile(tree,str(SOURCE_FILES[4]),'exec'),safety.__dict__)
    return names, *modules, safety


NM, ADDRESS, GEO, SAFE = load_native()


def split_top(text, separators='、，,；;。\n'):
    depth=0; start=0; result=[]
    for i,c in enumerate(text):
        if c in '([（【': depth+=1
        elif c in ')]）】': depth=max(0,depth-1)
        elif depth==0 and c in separators:
            if text[start:i].strip():result.append(text[start:i].strip())
            start=i+1
    if text[start:].strip():result.append(text[start:].strip())
    return result


def node(op, **kw):return dict(op=op,**kw)
def combine(op, children):return children[0] if len(children)==1 else node(op,children=children)
def walk(a):
    yield a
    for c in a.get('children',[]):yield from walk(c)


def canonical_school(s):
    s=re.sub(r'[（(][^()（）]*[)）]','',s)
    return re.sub(r'深圳市|深圳|龙岗区|龙华区|教育集团|集团|小学部|小学招生|招生|[*\s]','',s)


class Parser:
    def __init__(self, inventory, old_parse):
        self.old_parse=old_parse
        self.index=collections.defaultdict(set)
        for i,s in enumerate(inventory):
            name=s['school_name']
            tail=re.split(r'(?:集团[）)]|教育集团)',name)[-1]
            for v in [name,canonical_school(name),tail,canonical_school(tail)]:self.index[v].add(i)

    def ref(self,text):
        t=re.sub(r'(?:的)?招生(?:范围|片区|地段)$','',text).strip()
        t=re.sub(r'学校小学$','学校',t)
        parts=re.split('和|及|、',t); children=[]
        for p in parts:
            match=self.index.get(p,set()) or self.index.get(canonical_school(p),set())
            if len(match)==1:children.append(node('ref',school=next(iter(match)),text=p))
            elif len(parts)>1 and not re.search('学校|小学|中学|招生',p):children.append(self.parse(p))
            else:children.append(node('unknown',text=p,reason='school_reference_identity_unresolved'))
        return combine('or',children)

    def parse(self,text):
        raw=text
        t=text.replace('居委会居委会','居委会').replace('（','(').replace('）',')').replace('[','(').replace(']',')').strip(' 。;；\n')
        # These sentences express school relationships, retained separately in the
        # existing sourced policy package, never converted into a spatial union.
        t=re.sub(r'[。]该招生范围与[^。]*共享[^。]*[。]?','',t)
        t=re.sub(r'[。]注：从\d+年秋季开始，该校整体搬迁至[^。]*办学[。]?','',t)
        t=re.sub(r'\(此楼盘为[^()]*共享招生地段\)','',t)
        t=re.sub(r'^与[^：:]+共享区域[：:]','',t)
        t=re.sub(r'(?<=以[东西南北])(?:以及|及|和|与)','、',t)
        if t!=raw.strip(' 。;；\n') and not t:return node('unknown',text=raw,reason='no_base_scope')
        # Explicit population conditions stay attached; no whole-list promotion.
        if re.search('非深户|深户生|户籍|村民子女|原住民|原本地|统筹开班',t):
            return node('conditional',text=raw,reason='population_or_integrated_qualification_requires_facts')
        # Circled digits are list markers, not part of residential identities.
        t=re.sub(r'^[①②③④⑤⑥⑦⑧⑨⑩]\s*','',t)
        # Explicit colon enumerations retain the listed houses, not the entire
        # community as an unconditional membership assertion.
        listing=re.fullmatch(r'([^()：:]+(?:社区|居委会))[：:](?:含[：:]?)?(.+)',t)
        if listing:return self.parse(listing[2])
        listing=re.fullmatch(r'(.+(?:片区|区域))[：:]([①②③④⑤⑥⑦⑧⑨⑩].+)',t)
        if listing:return combine('or',[self.parse(listing[1]),self.parse(listing[2])])
        t=re.sub(r'^(?:含|包括|包含)[：:]','',t)
        address=ADDRESS.parse_range(t.replace('从',''))
        if address:return node('address',text=t,address=list(address))
        ranges=re.split(r'(?:范围[一二三四五六七八九十]+[：:]|(?:^|(?<=[。；;\n]))\s*\d+[.、．])',t)
        if len(ranges)>1:
            return combine('or',[self.parse(v) for v in ranges if v.strip(' 。;；\n')])
        # A route with turns is not one named line. Keep it unknown before
        # handing simple cardinal clauses to the legacy parser.
        if re.search(r'(?:往[东西南北].*[—－-]|沿.+往[东西南北])',t) and '以' in t and not re.search(r'[：:]|含|包括|包含',t):
            return node('unknown',text=raw,reason='turning_route_requires_segment_geometry')
        # Four-sided expressions accepted by the original parser, unchanged.
        old=self.old_parse(t.replace('(','（').replace(')','）'))
        if old['op']=='road':return old
        plain=re.sub(r'(?:的)?(?:围成|围合|合围)(?:的)?(?:区域|范围)$','',t)
        plain=re.sub(r'(?:的)?(?:区域|住宅区域|片区|部分|范围|住宅|居民小组)$','',plain)
        old=self.old_parse(plain.replace('(','（').replace(')','）'))
        if old['op']=='road':return old
        single=re.fullmatch(r'([^()、，,。;；]+(?:路|街|道|河|高速|铁路))以([东西南北])',plain)
        if single and not re.search('街道|社区|居委会|居民',single[1]):return node('road',pairs=[[single[1],single[2]]],explicit_current_names={},text=raw)
        simple_sides=split_top(plain)
        if len(simple_sides)>1 and not re.search('街道|社区|居委会|居民',plain) and all(re.fullmatch(r'[^()、，,。;；]+(?:路|街|道|河|高速|铁路)以[东西南北]',v) for v in simple_sides):
            return node('road',pairs=[[v[:-2],v[-1]] for v in simple_sides],explicit_current_names={},text=raw)
        # Trailing parentheses may be an exclusion, explicit include, or a local restriction.
        depth=0; opening=None
        if t.endswith(')'):
            for i in range(len(t)-1,-1,-1):
                if t[i]==')':depth+=1
                elif t[i]=='(':
                    depth-=1
                    if depth==0:opening=i;break
        if opening is not None:
            base,tail=t[:opening],t[opening+1:-1]
            # Only bind parentheses if they belong to the whole current clause.
            # Split a preceding list first, unless it is a road conjunction/ring.
            prefix_parts=split_top(base)
            checkbase=re.sub(r'(?:围合|围成|合围)(?:的)?(?:区域|范围)$','',base)
            roadlike=self.parse(checkbase)['op']=='road' or bool(re.fullmatch(r'[^()]+(?:围合|围成|合围)(?:的)?(?:区域|范围)',base))
            # A four-sided scope remains one conjunction when a terminal
            # parenthesis lists included houses or exclusions. Do not let the
            # generic preceding-list branch turn each side into an alternative.
            roadlike = roadlike or (len(prefix_parts)>1 and all(
                re.match(r'^[东西南北]{1,2}(?:和[东西南北])?至', v) for v in prefix_parts))
            if len(prefix_parts)>1 and not roadlike:
                return combine('or',[self.parse(v) for v in prefix_parts[:-1]]+[self.parse(prefix_parts[-1]+'('+tail+')')])
            # "Community [X excluded; includes A,B]" is a scoped exclusion
            # plus an explicit list, not a conjunction with an opaque tail.
            tail_parts=split_top(tail,separators='；;')
            if len(tail_parts)==2 and tail_parts[0].endswith('除外') and re.match(r'^(?:含|包括|包含)[：:]',tail_parts[1]):
                return combine('or',[node('minus',children=[self.parse(base),self.parse(tail_parts[0][:-2])],text=raw),self.parse(tail_parts[1])])
            if not re.search('社区|街道|居委会|居民小组|以[东西南北]|围合|围成',base) and NM.parse_scope(t,{NM.normalize(base)}) is not None:
                return node('named',text=t)
            ex=re.fullmatch(r'(?:不包含|不包括|不含|除)(.+?)(?:除外|以外|外)?',tail)
            if not ex:ex=re.fullmatch(r'(.+?)(?:除外|以外)',tail)
            if ex:
                ext=ex[1]
                # A named list introduced by "school scope中的" is an explicit list,
                # not exclusion of the whole referenced school.
                ext=re.sub(r'^.+?招生范围中的','',ext)
                return node('minus',children=[self.parse(base),self.parse(ext)],text=raw)
            inc=re.fullmatch(r'(?:包括|包含|含|即)[：:]?(.+)',tail)
            if inc:
                if re.search('不含|不包括|不包含|除外|以外',inc[1]):
                    return node('unknown',text=raw,reason='mixed_inclusion_exclusion_requires_scoped_parse')
                return combine('or',[self.parse(base),self.parse(re.sub(r'等(?:楼盘|住宅区域|区域)?$','',inc[1]))])
            only=re.fullmatch(r'(?:只含|仅含|仅限)(.+)',tail)
            if only:return self.parse(only[1])
            if re.search(r'社区|居委会|居民小组|街道|村$',base):
                local=self.parse(tail)
                # Explicit housing enumeration in a community clause identifies
                # those houses; a road condition must intersect the community.
                if all(x['op'] in ['or','named','address'] for x in walk(local)):return local
                return combine('and',[self.parse(base),local])
            # Explicit residential aliases are parsed by Changsha's matcher.
            if not NM.alias_rejection(tail):return node('named',text=t)
            return node('unknown',text=raw,reason='unresolved_parenthetical_condition')
        # Unparenthesized explicit including-list is independently named evidence.
        inc=re.search(r'[，,：:]\s*(?:含以下花园|包括|包含|含)[：:]?',t)
        if inc:return combine('or',[self.parse(t[:inc.start()]),self.parse(re.sub(r'等(?:住宅区域|区域)?$','',t[inc.end():]))])
        ex=re.search(r'[，,]\s*(?:不包含|不包括|不含)',t)
        if ex:return node('minus',children=[self.parse(t[:ex.start()]),self.parse(t[ex.end():])],text=raw)
        if re.fullmatch(r'.+招生(?:范围|片区|地段)',t):return self.ref(t)
        # Road ring: keep it a ring, never invent independent cardinal sides.
        ring=re.fullmatch(r'(.+?)(?:围合|围成|合围)(?:的)?(?:区域|范围|范围内的住宅区域)',t)
        if ring:
            names=re.split('[、，,—－-]|及',ring[1])
            names=[v.strip() for v in names if v.strip()]
            if len(names)>=3:
                if any(re.search('以[东西南北]|社区|片区|由|方向|期',v) for v in names):
                    return node('unknown',text=raw,reason='mixed_ring_requires_clause_segmentation')
                return node('ring',names=names,text=raw)
        # Do not split an unresolved ring into bare road names; those are
        # boundary operands, including when the ring is an exclusion.
        ring_parts=split_top(t)
        boundary_prefix=all(re.fullmatch(r'(?:由)?[^()]+(?:铁路|路|街|大道|河|高速)',v) for v in ring_parts[:-1])
        if re.search('围合|围成|合围',t) and (t.startswith('由') or len(ring_parts)==1 or boundary_prefix):
            return node('unknown',text=raw,reason='qualified_ring_requires_scoped_parse')
        # 东至/北至 boundary wording; inward direction is the opposite side.
        adminside=re.fullmatch(r'[东西南北和]+至([^()、，,。;；]+街道)(?:行政)?区域界线',t)
        if adminside:return node('admin',text=adminside[1],source_text=raw)
        directional_parts=split_top(plain)
        if len(directional_parts)>1 and all(re.match(r'^[东西南北]{1,2}(?:和[东西南北])?至',v) for v in directional_parts):
            return combine('and',[self.parse(v) for v in directional_parts])
        sides=[]
        for part in split_top(plain):
            m=re.fullmatch(r'([东西南北])至(.+)',part)
            if not m:break
            sides.append([m[2],dict(zip('东西南北','西东北南'))[m[1]]])
        else:
            if sides:return node('road',pairs=sides,explicit_current_names={},text=raw)
        # Street constraint applies to the entire subsequent road conjunction.
        if re.fullmatch(r'(?:面向)?(.+街道)(?:招生)?',t):return node('admin',text=re.sub('^面向|招生$','',t))
        m=re.fullmatch(r'([^()、，,。;；]+?街道)(.+)',t)
        if m:return combine('and',[node('admin',text=m[1]),self.parse(m[2])])
        m=re.fullmatch(r'([^()、，,。;；]+?(?:社区|居委会))(?:范围内|的)?(.+以[东西南北].*)',t)
        if m and not any(v in m[2] for v in ['社区','居委会','(',')']):return combine('and',[node('membership',text=m[1]),self.parse(m[2])])
        # Community prefix for an explicitly enumerated list is provenance, not
        # permission to extend the list to the whole community.
        t=re.sub(r'[^、，,。;；]+?社区的','',t)
        parts=split_top(t)
        if len(parts)>1:return combine('or',[self.parse(p) for p in parts])
        t=t.strip(' 。').removesuffix('等')
        if re.fullmatch(r'.+(?:社区|居委会|居民小组)',t):return node('membership',text=t)
        if re.fullmatch(r'(?:面向)?(.+街道)(?:招生)?',t):return node('admin',text=re.sub('^面向|招生$','',t))
        # Explicit street numbers, including Shenzhen's "从1至100号" layout.
        address=ADDRESS.parse_range(t.replace('从',''))
        if address:return node('address',text=t,address=list(address))
        if re.search(r'以[东西南北]|[东西南北]至|除外|不含|部分|片区|街道|招生|附近|两侧|围合|围成|范围|区域|所辖|居民|住户|往[东西南北]|到|至|以外|沿线',t):
            return node('unknown',text=raw,reason='unsupported_spatial_or_identity_qualifier')
        if t:return node('named',text=t)
        return node('unknown',text=raw,reason='empty_scope')


class Evaluator:
    def __init__(self,sites,schools,inventory,policy,byname,old_parse,side,member,project,strict_line,policy_named=None,admin_features=()):
        self.sites=sites;self.schools=schools;self.inventory=inventory;self.policy=policy
        self.policy_named=policy_named or {}
        self.n=len(sites);self.m=len(schools);self.byname=byname
        self.side=side;self.member=member;self.project=project;self.strict_line=strict_line
        self.coords=project([s['representative_point_wgs84'] for s in sites])
        self.parser=Parser(inventory,old_parse)
        self.asts=[self.parser.parse(r['base_scope_text']) for r in inventory]
        self.rows=[];self.row_site=[]
        for q,s in enumerate(sites):
            for name in sorted(set(s['coordinate_group_building_names'])):
                self.rows.append(dict(building_id=s['coordinate_group_id'],building_name=name,source_district=policy['district']))
                self.row_site.append(q)
        self.cache={};self.geometry_audit=[];self.name_audit=[];self.trace=[]
        self.point_names=[set(NM.normalize(v) for v in s['coordinate_group_building_names']) for s in sites]
        self.admin_index=collections.defaultdict(list);self.admin_audit=[]
        district_shapes=[shape(f['geometry']) for f in admin_features if f['properties'].get('name')==policy['district'] and f['properties'].get('valid')]
        district_shape=unary_union(district_shapes) if district_shapes else None
        for f in admin_features:
            if not f['properties'].get('valid'):continue
            g=shape(f['geometry'])
            if not district_shape or not g.area or g.intersection(district_shape).area/g.area<.95:continue
            for name in {f['properties'].get('name'),f['properties'].get('name:zh')} - {None}:
                self.admin_index[name].append(f)

    def unknown(self):return np.full(self.n,-1,np.int8)

    def admin(self,name):
        key=('admin',name)
        if key in self.cache:return self.cache[key].copy()
        # Only a redundant 居委会 after a complete 社区 name is removed.
        identity=re.sub(r'(?<=社区)居委会$','',name)
        fs=self.admin_index.get(identity,[])
        record=dict(policy_name=name,identity=identity,source_area_ids=[f['properties']['osm_area_id'] for f in fs],status='identity_missing_or_ambiguous')
        v=self.unknown()
        if len(fs)==1:
            g=shape(fs[0]['geometry'])
            from shapely.ops import transform
            from pyproj import Transformer
            g=transform(Transformer.from_crs(4326,32650,always_xy=True).transform,g)
            ps=shapely.points(self.coords)
            v=np.where(shapely.distance(g.boundary,ps)<=50,-1,np.where(shapely.contains(g,ps),1,0)).astype(np.int8)
            record['status']='independent_osm_boundary';record['counts']={str(k):int((v==k).sum()) for k in [-1,0,1]}
        self.admin_audit.append(record);self.cache[key]=v.copy();return v

    def match(self,text,exclude=False):
        key=('named',text,exclude)
        if key in self.cache:return self.cache[key].copy()
        result=NM.match_named_lists(self.rows,[dict(school_name='source clause',named_list_text=text)],district=self.policy['district'])
        v=np.zeros(self.n,np.int8) if exclude else self.unknown()
        review=set(result['review_rows'])
        ambiguous=set()
        for i,q in enumerate(self.row_site):
            if i in review or result['phase_limits'][i,0]:ambiguous.add(q)
            elif result['named'][i,0]==1:v[q]=1
        for q in ambiguous:
            if v[q]!=1:v[q]=-1
        if exclude:
            terms=[r['policy_term'] for r in NM.policy_terms(text)]
            for q,s in enumerate(self.sites):
                if v[q]!=1 and any(SAFE.similar(n,t) for n in s['coordinate_group_building_names'] for t in terms):v[q]=-1;ambiguous.add(q)
        self.name_audit.append(dict(text=text,exclusion=exclude,hit=np.flatnonzero(v==1).tolist(),ambiguous=sorted(ambiguous),events=result['events']))
        self.cache[key]=v.copy();return v

    def ring(self,names):
        key=('ring',tuple(names))
        if key in self.cache:return self.cache[key].copy()
        record=dict(names=names,status='unresolved',source_osm_ids=[])
        geoms=[]
        for name in names:
            fs={f['properties']['osm_id']:f for f in self.byname.get(name,[])}
            if not fs:
                record.update(status='independent_boundary_geometry_missing',missing=name)
                self.geometry_audit.append(record);self.cache[key]=self.unknown();return self.unknown()
            record['source_osm_ids']+=sorted(fs)
            geoms.append(unary_union([LineString(self.project(f['geometry']['coordinates'])) for f in fs.values()]))
        faces=list(polygonize(unary_union(geoms)))
        # No gap bridges, no arbitrary school-centred window, no face selection by Gold.
        valid=[p for p in faces if p.is_valid and all(p.boundary.intersection(g).length>=50 for g in geoms)]
        record.update(faces=len(faces),candidates=len(valid),status='unique_closed_ring' if len(valid)==1 else 'ring_missing_or_ambiguous')
        v=self.unknown()
        if len(valid)==1:
            ps=shapely.points(self.coords);p=valid[0]
            v=np.where(shapely.distance(p.boundary,ps)<=50,-1,np.where(shapely.contains(p,ps),1,0)).astype(np.int8)
            record['geometry']=shapely.geometry.mapping(p);record['geometry_crs']=32650
        self.geometry_audit.append(record);self.cache[key]=v.copy();return v

    def evaluate(self,a,j,stack=(),exclude=False,partial=False):
        op=a['op']
        if op=='named':return self.match(a['text'],exclude)
        if op=='address':
            street,lo,hi,parity=a['address'];v=np.zeros(self.n,np.int8) if exclude else self.unknown()
            for q,s in enumerate(self.sites):
                for name in s['coordinate_group_building_names']:
                    p=ADDRESS.parse_point(name)
                    if p and p[0]==street and lo<=p[1]<=hi and (parity is None or p[1]%2==(1 if parity=='odd' else 0)):v[q]=1
            return v
        if op=='road':return rules.AND([self.side(name,di,a.get('explicit_current_names',{}).get(name)) for name,di in a['pairs']],self.n)
        if op=='ring':return self.ring(a['names'])
        if op=='membership':
            proxy=self.member(a['text'],j,partial=partial)
            boundary=self.admin(a['text'])
            # Boundary-band points remain unknown, not overridden with an office proxy.
            rec=next(r for r in self.admin_audit if r['policy_name']==a['text'])
            return boundary if rec['status']=='independent_osm_boundary' else proxy
        if op=='admin':return self.admin(a['text'])
        if op in ['or','and']:
            vals=[self.evaluate(c,j,stack,exclude,partial or op=='and') for c in a['children']]
            return (rules.OR if op=='or' else rules.AND)(vals,self.n)
        if op=='minus':
            base=self.evaluate(a['children'][0],j,stack,partial=True)
            ex=self.evaluate(a['children'][1],j,stack,True)
            return rules.AND([base,np.where(ex<0,-1,1-ex).astype(np.int8)],self.n)
        if op=='ref':
            k=a['school']
            if k in stack or k==j:return self.unknown()
            if self.inventory[k]['status'].endswith('_pending'):return self.unknown()
            return self.evaluate(self.asts[k],k,stack+(j,))
        return self.unknown()

    def positives(self,a,j,stack=()):
        """Only independent explicit names; never mine names from an unknown clause."""
        if a['op'] in ['named','address']:return self.evaluate(a,j,stack)
        if a['op']=='or':return rules.OR([self.positives(c,j,stack) for c in a['children']],self.n)
        if a['op']=='minus':
            base=self.positives(a['children'][0],j,stack);ex=self.evaluate(a['children'][1],j,stack,True)
            return rules.AND([base,np.where(ex<0,-1,1-ex).astype(np.int8)],self.n)
        return self.unknown()

    def geometry_ready(self,a):
        if a['op']=='road':return all(self.byname.get(a.get('explicit_current_names',{}).get(n,n)) for n,d in a['pairs'])
        if a['op']=='ring':return any(r['names']==a['names'] and r['status']=='unique_closed_ring' for r in self.geometry_audit)
        if a['op'] in ['named','address']:return True
        if a['op'] in ['membership','admin']:return any(r['policy_name']==a['text'] and r['status']=='independent_osm_boundary' for r in self.admin_audit)
        if a['op'] in ['and','or','minus']:return all(self.geometry_ready(c) for c in a['children'])
        return False

    def direction_nodes(self,a):
        if a['op']=='road':yield a
        elif a['op']=='and':
            for c in a['children']:yield from self.direction_nodes(c)
        elif a['op']=='minus':yield from self.direction_nodes(a['children'][0])

    def positive_terms(self,a):
        if a['op']=='named':return [a['text']]
        if a['op']=='or':return [t for c in a['children'] for t in self.positive_terms(c)]
        if a['op']=='minus':return self.positive_terms(a['children'][0])
        return []

    def build(self):
        active=[not r['status'].endswith('_pending') for r in self.inventory]
        named=np.column_stack([self.positives(a,j) if active[j] else self.unknown() for j,a in enumerate(self.asts)])
        # Run the exact multi-school matcher as well: aliases must be unique across
        # schools and bare project entries must not erase another phase restriction.
        grouped=NM.match_named_lists(self.rows,[dict(school_name=s['school_name'],named_list_text='、'.join(self.positive_terms(a)) if active[j] else '')
            for j,(s,a) in enumerate(zip(self.schools,self.asts))],district=self.policy['district'])
        phase_amb=np.zeros((self.n,self.m),bool)
        for i,q in enumerate(self.row_site):phase_amb[q]|=grouped['phase_limits'][i]
        named[phase_amb]=-1
        review={self.row_site[i] for i in grouped['review_rows']}
        known={q:np.flatnonzero(named[q]==1).tolist() for q in range(self.n) if (named[q]==1).any()}
        for q,ss in self.policy_named.items():known[q]=sorted(set(known.get(q,[]))|set(ss))
        # Retain already sourced exact shared-name rules; never expand university members.
        sid=[s['school_id'] for s in self.schools]
        for q in list(known):
            initial={sid[k] for k in known[q]}
            for rel in self.policy['relations']:
                if rel['kind']=='shared' and initial.intersection(rel.get('trigger_school_ids',[])):
                    known[q]=sorted(set(known[q])|{sid.index(s) for s in rel['school_ids']})
        # Native K1--K8 / X1--X4 / P1--P5 execute the compiled observations.
        columns=[];clauses=[];values=[]
        for j,a in enumerate(self.asts):
            roots=a.get('children',[]) if a['op']=='or' else [a]
            for k,root in enumerate(roots):
                val=self.evaluate(root,j) if active[j] else self.unknown()
                values.append(val)
                is_name=root['op'] in ['named','address']
                sides=[dict(column=len(columns),direction='',feature='composed_scope',status='ready' if self.geometry_ready(root) else 'partial')]
                columns.append(val)
                # Native direction channels only for true cardinal conjunctions.
                if active[j]:
                    for road in self.direction_nodes(root):
                        for name,di in road['pairs']:
                            alias=road.get('explicit_current_names',{}).get(name)
                            v=self.side(name,di,alias)
                            sides.append(dict(column=len(columns),direction=dict(zip('东西南北','西东北南'))[di],feature=name,status='ready' if self.byname.get(alias or name) else 'missing'))
                            columns.append(v)
                exclusions=[]
                if root['op']=='minus':
                    ex=self.evaluate(root['children'][1],j,exclude=True)
                    exclusions=[dict(text=str(root['children'][1]),kind='composed',broad=False,hit=np.flatnonzero(ex==1).tolist(),ambiguous=np.flatnonzero(ex<0).tolist(),resembles={})]
                clauses.append(dict(school=j,school_name=self.schools[j]['school_name'],clause=k+1,text=self.inventory[j]['base_scope_text'],kind='qualified_or_named' if is_name else 'area',
                    admitted_by_parse=active[j],parse_complete=True,sides=sides,admin=None,exclusions=exclusions,ast=root))
        O=dict(city='深圳',n=self.n,school_n=self.m,name_known=known,names=[s['building_name'] for s in self.sites],
               generic=[bool(SAFE.generic(*s['coordinate_group_building_names'])) for s in self.sites],review=sorted(review),name_conflict=sorted(review),geo_excluded=[],clauses=clauses)
        A=dict(side_values=np.stack(columns,axis=1),name_negative=np.zeros((self.n,self.m),bool))
        S=dict(rules.PRESETS['unified_20261106']['芙蓉区'])
        hits=np.zeros((self.n,self.m),bool);st=np.full((self.n,self.m,4),-1,np.int8);neg=hits.copy()
        audits=[]
        for j in range(self.m):
            cls=[c for c in clauses if c['school']==j]
            hits[:,j],st[:,j],neg[:,j]=rules.school(j,cls,O,A,S)
            hits[phase_amb[:,j],j]=False
            st[phase_amb[:,j],j]=-1
            for q,ss in known.items():
                if j in ss:neg[q,j]=False
            rootvals=rules.OR([self.evaluate(a,j) for a in ([self.asts[j]] if active[j] else [])],self.n)
            audits.append(dict(school_id=sid[j],school_name=self.schools[j]['school_name'],source_text=self.inventory[j]['base_scope_text'],evidence=self.inventory[j]['evidence'],
                scope_status=self.inventory[j]['status'],ast=self.asts[j],positive_estimates=int(hits[:,j].sum()),named_sites=int((named[:,j]==1).sum()),
                negative=int(neg[:,j].sum()),unknown=int((rootvals<0).sum()),unresolved=[a for a in walk(self.asts[j]) if a['op'] in ['unknown','conditional','admin']]))
        result,kind=rules.sites(O,hits,neg,S)
        pending=[]
        for q in range(self.n):
            if q not in result:
                cand=np.flatnonzero(hits[q]&~neg[q]).tolist();ids={sid[k] for k in cand}
                if len(cand)>1 and any(r['kind']=='shared' and r['rule_id'].startswith('EQ-') and set(r['school_ids'])==ids for r in self.policy['relations']):
                    result[q]=cand;kind[q]='boundary'
                elif len(cand)>1:pending.append(dict(site_id=self.sites[q]['coordinate_group_id'],candidate_school_ids=sorted(ids)))
            if q in result:
                initial={sid[k] for k in result[q]}
                for r in self.policy['relations']:
                    if r['kind']=='shared' and initial.intersection(r.get('trigger_school_ids',[])):
                        result[q]=sorted(set(result[q])|{sid.index(s) for s in r['school_ids']})
        return dict(known=result,kind=kind,statuses=st,negative=neg,audits=audits,pending=pending,
                    observations=O,observation_arrays=A,named_audit=dict(clauses=self.name_audit,grouped=grouped['school_audit'],events=grouped['events'],review_sites=sorted(review)),geometry_audit=self.geometry_audit,admin_audit=self.admin_audit)
