"""Run the three-arm gated round17 target comparison using the shared frozen driver."""
import argparse
import experiment_shenzhen_multireference_target as T
import experiment_shenzhen_relative_multireferences as D

T.DECODER_EXPERIMENT=D
T.OUT=D.R.ROOT/'round17_relative_multireference_target'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','replay','score','score_audit']);getattr(T,p.parse_args().phase)()
