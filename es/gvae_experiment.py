import ray
from sacred import Experiment
from sacred.observers import FileStorageObserver
from gvae_head import run_gvae_fdm
from argparse import Namespace

ex = Experiment('GVAEFDM')
ex.observers.append(FileStorageObserver('gvae_fdm_runs'))

@ex.named_config
def default():
    ex.add_config('configs/gvae.yaml')

@ex.automain
def run(_run, _config):
    ray.init()
    conf = Namespace(**_config)
    run_gvae_fdm(conf, _run)