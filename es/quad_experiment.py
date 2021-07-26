import ray
from sacred import Experiment
from sacred.observers import FileStorageObserver
from quad_head import run_quad_fdm
from argparse import Namespace

ex = Experiment('QuadFDM')
ex.observers.append(FileStorageObserver('quad_fdm_runs'))

@ex.named_config
def default():
    ex.add_config('configs/quad.yaml')

@ex.automain
def run(_run, _config):
    ray.init()
    conf = Namespace(**_config)
    run_quad_fdm(conf, _run)