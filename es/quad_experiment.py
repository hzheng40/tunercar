import ray
from sacred import Experiment
from sacred.observers import FileStorageObserver
from head import run_tunercar
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
    run_tunercar(conf, _run)