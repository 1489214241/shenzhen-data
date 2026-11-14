from zoning.school_gap_reference import fill_school_gaps

def test_existing_partial_counts_and_one_payment_for_shared_set():
    r=fill_school_gaps(['a','b','c'],[dict(school='a',group='old')],
        [dict(q=1,group='shared',schools=['b','c'])])
    assert len(r[1]['selected'])==1
    assert r[1]['counts']==dict(a=1,b=1,c=1)
    assert r[1]['selected'][0]['requested_school']=='b'
    assert r[2]['deficits']==dict(a=1,b=1,c=1)

def test_duplicate_points_do_not_fill_multiple_slots_and_gaps_remain():
    r=fill_school_gaps(['a','b'],[],[dict(q=1,group='x',schools=['a']),dict(q=2,group='x',schools=['a'])])
    assert len(r[3]['selected'])==1
    assert r[3]['deficits']==dict(a=2,b=3)

def test_nested_targets_and_no_extra_sample_for_satisfied_school():
    r=fill_school_gaps(['a'],[dict(school='a',group='x')],
        [dict(q=i,group=str(i),schools=['a']) for i in range(5)])
    assert not r[1]['selected']
    assert len(r[2]['selected'])==1 and len(r[3]['selected'])==2
    assert r[2]['selected'][0]==r[3]['selected'][0]
