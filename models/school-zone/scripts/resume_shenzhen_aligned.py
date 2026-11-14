"""Resume complete, hashed seed checkpoints with the frozen training function.

Only skip fully completed seeds. An interrupted seed is trained from its original
initialization, using the unmodified frozen optimizer loop and prediction export.
"""
import ast
import inspect
import shutil
from pathlib import Path
import experiment_shenzhen_aligned as A

B=A.R.B


def main():
    B.guard();B.check_frozen()
    assert not (A.OUT/'prediction_freeze.json').exists(), 'Predictions already frozen'
    c=B.read(A.OUT/'protocol.json')
    history=B.read(A.OUT/'training_history.json')
    valid={f'{d}/{s}' for d in B.TARGETS for s in c['seeds']}
    assert set(history)<=valid
    for key,v in history.items():
        d,s=key.split('/')
        assert len(v['loss'])==c['steps'] and B.np.isfinite(v['loss']).all()
        assert B.sha(A.OUT/'checkpoints'/f'{d}_{s}.pt')==v['checkpoint_sha256']
    frozen=A.OUT/'code/scripts/experiment_shenzhen_current.py'
    assert B.sha(Path(B.__file__))==B.sha(frozen)
    source=inspect.getsource(B.train)
    old='    history = {}'
    assert source.count(old)==1
    source=source.replace(old,"    history = read(OUT/'training_history.json')")
    marker="            optimizer = torch.optim.Adam(m.parameters(), lr=c['lr'])"
    assert source.count(marker)==1
    insert='''            if f'{d}/{seed}' in history:
                saved = history[f'{d}/{seed}']
                assert saved['training'] == traininfo
                assert saved['checkpoint_sha256'] == sha(ck)
                m.load_state_dict(torch.load(ck, map_location=DEVICE, weights_only=True))
                all_probs.append(T.predict_proba(m, x, chunk=64))
                log('RESUMED_COMPLETE_SEED', d, seed)
                continue
'''
    source=source.replace(marker,insert+marker)
    # The complete optimizer loop and export statements are identical to the
    # frozen function. Only history loading and a completed-seed branch differ.
    tree=ast.parse(source);assert tree.body[0].name=='train'
    ns=dict(B.__dict__)
    exec(compile(tree,str(Path(__file__))+'::<frozen_train_resume>','exec'),ns)
    snap=A.OUT/'resume_code';snap.mkdir(exist_ok=True)
    shutil.copyfile(__file__,snap/Path(__file__).name)
    (snap/'effective_train.py').write_text(source,encoding='utf8')
    B.write(A.OUT/'resume_manifest.json',dict(completed_checkpoints_before=history,
        frozen_train_sha256=B.sha(frozen),resume_sha256=B.sha(Path(__file__)),
        effective_train_sha256=B.sha(snap/'effective_train.py'),
        protocol_sha256=B.sha(A.OUT/'protocol.json'),target_gold_read=False,
        policy='reuse complete hashed seeds only; all other seeds restart full frozen training'))
    ns['train']()
    current=B.read(A.OUT/'training_history.json')
    for key,v in history.items():assert current[key]==v
    B.write(A.OUT/'resume_verification.json',dict(passed=True,
        preserved_complete_seeds=len(history),total_complete_seeds=len(current),
        protocol_unchanged=True,completed_checkpoint_hashes_unchanged=True))


if __name__=='__main__':main()
