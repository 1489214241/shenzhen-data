from zoning.full_policy_prior import condition_state,one_pass,candidate_pool,choose,compile_pool,choose_compiled


def test_unknown_and_exclusion_are_not_dropped():
    site={'building_name':'X'}
    assert condition_state(['and',['named','X'],['side','R','东']],site,set())==-1
    assert condition_state(['minus',['named','X'],['named','Y']],site,set())==-1
    assert condition_state(['named','X'],site,set())==1


def test_union_hypothesis_is_not_transitive_expansion():
    graph={'edges':[{'trigger_school_id':'a','implied_school_id':'b'},
                    {'trigger_school_id':'b','implied_school_id':'c'}]}
    assert one_pass(['a'],{'relations':[]},[graph])=={'a','b'}
    records=[{'school_ids':['a','b'],'candidate_id':'ab'},{'school_ids':['b','c'],'candidate_id':'bc'}]
    prior={'verified_members':[]}
    one=candidate_pool(records,prior,[],[],[],[],False)
    two=candidate_pool(records,prior,[],[],[],[],True)
    assert frozenset('abc') not in dict(one) and frozenset('abc') in dict(two)
    assert len(dict(two)[frozenset('abc')])==2


def test_partial_positives_preserved_excluded_never_enters():
    rec=[{'school_ids':['a','b'],'candidate_id':'ab'},{'school_ids':['c'],'candidate_id':'c'}]
    pool=candidate_pool(rec,{'verified_members':['a']},[],[],[],['c'],True)
    assert all('a' in s and 'c' not in s for s,b in pool)


def test_membership_ranking_and_university_not_group_choice():
    pool=[(frozenset(['a','b']),['ab']),(frozenset(['a']),['a'])]
    assert choose(pool,[.9,.8],['a','b'],.5)[0]==['a','b']
    assert choose(pool,[.9,.8],['a','b'],.5,[['a','b']])[0]==['a']
    assert choose(pool,[.9,.8],['a','b'],.5,[['a','b']],[['a','b']])[0]==['a','b']


def test_vectorized_selection_matches_scalar():
    pool=[(frozenset(['a']),['a']),(frozenset(['a','b']),['ab']),
          (frozenset(['b','c']),['bc'])]
    for q in [[.9,.8,.7],[.1,.8,.9],[.5,.5,.5]]:
        compiled=compile_pool(pool,['a','b','c'],[['a','b']])
        assert choose_compiled(compiled,q,.6)[0]==choose(pool,q,['a','b','c'],.6,[['a','b']])[0]
