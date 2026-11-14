import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from shenzhen_resident_feature_patch import dependent_blocks, patch_features


def fixture():
    states = np.ones((2, 2, 4), dtype=np.int8)
    negative = np.zeros((2, 2), dtype=np.float32)
    x = SimpleNamespace(ref_sites=np.array([0, 1]), ref_schools=np.array([0, 1]),
        pair_blocks={'policy': slice(0,13), 'agreement': slice(13,17), 'disagreement': slice(17,21)},
        anchor_blocks={'policy': slice(0,13)}, pair=np.zeros((2,2,22), np.float32),
        anchor=np.zeros((2,2,14), np.float32))
    blocks, anchor = dependent_blocks(x, states, negative)
    for key, value in blocks.items(): x.pair[:,:,x.pair_blocks[key]] = value
    x.anchor[:,:,:13] = anchor
    x.pair[:,:,-1] = 7
    x.anchor[:,:,-1] = 9
    return x, states, negative


def test_unknown_identity_is_not_positive_or_disagreement():
    x, before, negative = fixture()
    after = before.copy(); after[0,0,:] = -1
    y = patch_features(x, before, after, negative)
    np.testing.assert_array_equal(y.pair[0,0,:12], np.tile([1,0,0],4))
    assert not y.pair[0,0,13:21].any()
    # A reference whose identity becomes unknown cannot confirm another query.
    assert not y.pair[1,0,13:21].any()
    np.testing.assert_array_equal(y.anchor[0,0,:12], np.tile([1,0,0],4))
    np.testing.assert_array_equal(y.pair[:,:,-1], x.pair[:,:,-1])
    np.testing.assert_array_equal(y.anchor[:,:,-1], x.anchor[:,:,-1])
    assert x.pair[0,0,2] == 1  # Original frozen inputs are untouched.


def test_rejects_patch_against_different_baseline():
    x, before, negative = fixture()
    x.pair[0,0,0] = 1
    with pytest.raises(AssertionError): patch_features(x, before, before.copy(), negative)
