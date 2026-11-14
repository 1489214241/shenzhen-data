import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_shenzhen_community_reference_inputs import union_positive_references


def test_partial_supplement_preserves_old_positives_and_deduplicates():
    known={0:[0],1:[1,2]};kind={0:'named',1:'boundary'}
    facts=[dict(site_id='a',school_id='z'),dict(site_id='c',school_id='y')]*2
    out,k,added,touched=union_positive_references(known,kind,['a','b','c'],['x','y','z'],facts)
    assert out=={0:[0,2],1:[1,2],2:[1]}
    assert known=={0:[0],1:[1,2]}
    assert k=={0:'named',1:'boundary',2:'boundary'}
    assert added=={(0,2),(2,1)} and touched=={0,2}


def test_unmatched_identity_cannot_silently_enter_reference_bank():
    with pytest.raises(KeyError):
        union_positive_references({}, {}, ['a'], ['x'], [dict(site_id='missing',school_id='x')])
    with pytest.raises(KeyError):
        union_positive_references({}, {}, ['a'], ['x'], [dict(site_id='a',school_id='missing')])


def test_duplicate_identity_is_rejected():
    with pytest.raises(ValueError):
        union_positive_references({}, {}, ['a','a'], ['x'], [])
