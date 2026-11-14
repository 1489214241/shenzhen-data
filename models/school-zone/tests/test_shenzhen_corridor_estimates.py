import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import shapely
from shapely.geometry import box,LineString
from build_shenzhen_corridor_scope_estimates import consensus_inside,unconditional_rings


def test_boundary_disagreement_and_outside_remain_unknown():
    lines=[LineString([(0,0),(0,1000)]),LineString([(0,0),(1000,0)]),LineString([(1000,0),(1000,1000)]),LineString([(1000,1000),(0,1000)])]
    points=shapely.points([[500,500],[25,500],[75,500],[1500,500]])
    regions=[box(10,10,990,990),box(25,25,975,975),box(100,100,900,900)]
    np.testing.assert_array_equal(consensus_inside(points,lines,regions),[1,-1,-1,-1])


def test_ring_with_other_conditions_is_not_promoted():
    ring=dict(op='ring',names=['a','b','c'])
    assert list(unconditional_rings(dict(op='or',children=[ring])))==[('a','b','c')]
    for op in ['and','minus','conditional']:
        assert not list(unconditional_rings(dict(op=op,children=[ring,dict(op='unknown')])))
