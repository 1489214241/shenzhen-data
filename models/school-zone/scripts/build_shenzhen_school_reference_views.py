"""Freeze school-wide absent-reference views and independently recomputed borders."""
from pathlib import Path
import shutil
import numpy as np
import experiment_shenzhen_source_multireferences as X
from shenzhen_school_reference_views import plans,boundary_support,apply_view
from zoning import evidence as E,rules

F=X.F;R=X.R
OUT=F.REPO/'data/深圳市/来源学校参照遮蔽输入_20261112'


def main():
    F.guard();c=X.verify();R.check(X.B.OUT,'manifest.json')
    if OUT.exists():raise FileExistsError('Preserve source mask snapshot')
    cfg=F.read(X.B.CACHE/'manifest.json')['config'];xs,ys=X.load(c,'all_multi_training');links={};audit={}
    F.write(OUT/'protocol.json',dict(unit='School: remove every reference relation to hidden schools, not random properties.',
        fraction='Two complementary halves per partition, floor(S/2) and ceil(S/2), no rate search.',
        views='Two train partitions giving4fixed views, one separate stress partition giving2views; seed20261118+districtindex.',
        curriculum='Training alternates full and masked views. Four train masks cycle,40totalsteps; no extra loss passes.',
        unchanged='Query school anchors/policy, complete or partial training labels, candidate school roster, all target inputs.',
        borders='Remove original known sets touching hidden schools before recomputing learned boundaries. Never turn a partial multi set into singleton support.',
        boundary_labels='Only original source rule-derived singleton support; added complete multi training references do not become negative border labels.',
        prior_negative='Random-property/spatial-property masks tested20261107 and adaptation A1 tested20261110 were not adopted. This tests school-wide absence under the new multi-reference input.',
        target_gold_read=False,target_polygon_read=False,adopted=False))
    for di,d in enumerate(c['sources']):
        x=xs[d];dist=X.B.source_district(d,links);known=rules.run(dist,**cfg['rules']).known
        assert not dist.answers and not dist.train_answers
        ls=E.lines(dist.lines_key or dist.city)
        original=E.learned_boundary(ls,dist.coords,dist.groups,known,x.ref_sites)
        np.testing.assert_array_equal(original,x.pair[:,:,x.pair_blocks['learned_boundary']])
        audit[d]={}
        for name,hidden in plans(x.school_n,20261118+di).items():
            columns=np.flatnonzero(~hidden[x.ref_schools]);support=boundary_support(known,hidden)
            boundary=E.learned_boundary(ls,dist.coords,dist.groups,support,x.ref_sites[columns])
            view=apply_view(x,columns,boundary)
            assert not hidden[view.ref_schools].any();np.testing.assert_array_equal(view.anchor,x.anchor)
            for block,sl in x.pair_blocks.items():
                if block!='learned_boundary':np.testing.assert_array_equal(view.pair[:,:,sl],x.pair[:,columns,sl])
            dest=OUT/d/f'{name}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest,columns=columns,boundary=boundary,hidden_schools=hidden)
            audit[d][name]=dict(hidden_schools=np.flatnonzero(hidden).tolist(),reference_pairs=len(columns),
                reference_school_coverage=len(set(view.ref_schools)),original_rule_support_sites=len(known),retained_rule_support_sites=len(support),
                changed_boundary_cells=int((boundary!=original[:,columns]).sum()),source_labels_changed=False,anchors_unchanged=True)
            print('VIEW',d,name,audit[d][name]['reference_pairs'],audit[d][name]['reference_school_coverage'],flush=True)
    links.update(F.read(X.OUT/'input_links.json')['files'])
    for bank in [X.B.OUT,X.TARGET]:
        links[str(bank/'manifest.json')]=F.sha(bank/'manifest.json');links.update({str(bank/p):h for p,h in F.read(bank/'manifest.json')['files'].items()})
    for p in [Path(__file__),F.REPO/'scripts/shenzhen_school_reference_views.py',F.REPO/'tests/test_shenzhen_school_reference_views.py',F.REPO/'zoning/evidence.py']:
        dest=OUT/'code'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);links[str(p)]=F.sha(p)
    F.write(OUT/'input_audit.json',audit);F.write(OUT/'input_links.json',dict(files=links))
    F.write(OUT/'manifest.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},target_gold_read=False))


if __name__=='__main__':main()
