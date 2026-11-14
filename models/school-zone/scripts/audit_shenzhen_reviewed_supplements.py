"""Source-only identity completion audit on the entire historical population."""
from pathlib import Path
from shenzhen_reviewed_supplements import ReviewedSupplements,F,R


def main():
    F.guard();out=R.ROOT/'reviewed_supplements_completion_20261113'
    if out.exists():raise FileExistsError('Preserve reviewed supplements audit')
    parent=R.ROOT/'round31_explicit_scope/combined';R.check(parent,'prediction_freeze.json')
    reports=[];links={str(Path(__file__)):F.sha(__file__)};total=0;relations=0
    for d in R.TARGETS:
        s=ReviewedSupplements(d);links.update(s.hashes);relations+=sum(map(len,s.extra.values()))
        for path in sorted((parent/'predictions').rglob(d+'.json')):
            old=F.read(path);new=s.augment_historical(old);changes=[]
            for a,b in zip(old,new):
                expected=set(a['after'])|s.extra.get(a['site_id'],set())
                assert set(b['after'])==expected
                assert all(a[k]==b[k] for k in a if k!='after')
                assert not(set(b['after'])&s.negative.get(a['site_id'],set()))
                if set(a['after'])!=expected:changes.append(dict(site_id=a['site_id'],before=a['after'],after=b['after']))
            total+=len(old);links[str(path)]=F.sha(path)
            reports.append(dict(source=str(path),rows=len(old),changed_rows=changes))
    assert len(reports)==88 and total==79288 and relations==4
    F.write(out/'audit.json',dict(prediction_files=len(reports),checked_rows=total,reviewed_relations=relations,
        changed_rows=sum(len(r['changed_rows']) for r in reports),historical_other_fields_unchanged=True,no_new_graph_expansion=True,
        target_gold_read=False,target_predictions_generated=False,reports=reports))
    p=F.REPO/'scripts/shenzhen_reviewed_supplements.py';links[str(p)]=F.sha(p)
    F.write(out/'manifest.json',dict(sources=links,files={'audit.json':F.sha(out/'audit.json')}))
    print('REVIEWED_SUPPLEMENTS',total,'rows',sum(len(r['changed_rows']) for r in reports),'changed',flush=True)


if __name__=='__main__':main()
