import ray
from sacred import Experiment
from sacred.observers import FileStorageObserver
from quad_head import run_quad_fdm
from argparse import Namespace

ex = Experiment('QuadFDM')
ex.observers.append(FileStorageObserver('quad_fdm_runs'))

@ex.named_config
def quad():
    ex.add_config('configs/quad.yaml')

@ex.named_config
def quadspider():
    ex.add_config('configs/quadspider.yaml')

@ex.named_config
def hcopter():
    ex.add_config('configs/hcopter.yaml')

@ex.named_config
def hexring():
    ex.add_config('configs/hexring.yaml')

@ex.named_config
def hplane():
    ex.add_config('configs/hplane.yaml')

@ex.named_config
def hex():
    ex.add_config('configs/hex.yaml')

@ex.named_config
def design1():
    ex.add_config('configs/design1.yaml')

@ex.automain
def run(_run, _config):
    ray.init()
    conf = Namespace(**_config)
    run_quad_fdm(conf, _run)