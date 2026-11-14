"""Compare accelerated all-mask geometry with completed native-mask files."""
import time
import numpy as np
import experiment_shenzhen_source_multireferences as X
import build_shenzhen_school_reference_views as V
from shenzhen_school_reference_views import plans
from shenzhen_batched_reference_boundaries import compute
from zoning import rules


def main():
    X.F.guard();c=X.verify();xs,_=X.load(c,'all_multi_training');d=c['sources'][0];x=xs[d]
    cfg=X.F.read(X.B.CACHE/'manifest.json')['config'];dist=X.B.source_district(d,{})
    known=rules.run(dist,**cfg['rules']).known;p=plans(x.school_n,20261118);start=time.monotonic()
    values=compute(dist,known,x.ref_sites,p)
    np.testing.assert_array_equal(values[0],x.pair[:,:,x.pair_blocks['learned_boundary']])
    for i,key in enumerate(p,1):
        saved=np.load(V.OUT/d/f'{key}.npz');np.testing.assert_array_equal(values[i][:,saved['columns']],saved['boundary'])
    dest=X.R.ROOT/'school_mask_geometry_acceleration_audit.json'
    if dest.exists():raise FileExistsError('Preserve verification')
    X.F.write(dest,dict(district=d,full_and_six_native_views_bitwise_identical=True,seconds=time.monotonic()-start,
        historical_helper_called_without_setup_or_labels=True,target_gold_read=False))
    print('ALL7_BOUNDARIES_BITWISE_IDENTICAL',time.monotonic()-start,flush=True)


if __name__=='__main__':main()
