"""Run the selected, verified Shenzhen community pipeline on its frozen dataset.

The consumed config pins the input snapshot and checkpoint comparison. It is not
a promise of independent validation or an interface for arbitrary new sites.
"""
from pathlib import Path
import argparse
import experiment_shenzhen_community_methods as C


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default=str(C.F.REPO/'configs/shenzhen_community_selected.json'))
    args=ap.parse_args()
    config=C.F.read(Path(args.config))
    folder=(C.F.REPO/config['experiment']).resolve()
    if not folder.is_relative_to(C.F.REPO.resolve()):
        raise ValueError('Experiment must be inside this repository')
    C.OUT=folder
    assert C.F.sha(folder/'selected_method.json')==config['selection_sha256']
    assert C.F.sha(folder/'results.json')==config['results_sha256']
    choice=C.F.read(folder/'selected_method.json')
    assert choice['method']==config['method']
    C.selected_replay()


if __name__=='__main__':
    main()
