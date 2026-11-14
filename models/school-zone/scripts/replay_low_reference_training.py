"""Replay all12 saved models against frozen target probabilities, without Gold."""
import numpy as np
import torch
import experiment_low_reference_training as R


def main():
    R.guard();c=R.verify();checked={}
    for name in ['selection_freeze.json','prediction_freeze.json']:
        for p,h in R.read(R.OUT/name).items():assert R.sha(R.OUT/p)==h
    for d in c['targets']:
        data=R.E.pkl(R.OUT/'target_inputs'/f'{d}.pkl');q=data['query_ids'];t=R.M.task(data['x'],q)
        for arm in c['arms']:
            saved=np.load(R.OUT/'target_probabilities'/f'{d}_{arm}.npz')['seeds']
            for j,seed in enumerate(c['seeds']):
                model=R.new_model(arm,seed,c)
                model.load_state_dict(torch.load(R.OUT/'weights'/arm/f'{seed}.pt',weights_only=True,map_location=R.E.DEVICE))
                replay=R.M.predict(model,t,'multilabel');np.testing.assert_array_equal(replay,saved[j])
                checked[f'{d}/{arm}/{seed}']=dict(rows=len(q),max_difference=float(np.abs(replay-saved[j]).max()))
    R.write(R.OUT/'weight_replay_audit.json',dict(passed=True,models=12,district_model_replays=24,
        replayed_query_predictions=12*1095,target_gold_read=False,details=checked))
    print('ALL12 WEIGHT REPLAYS PASSED',flush=True)


if __name__=='__main__':main()
