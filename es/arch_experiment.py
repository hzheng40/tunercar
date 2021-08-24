import ray
from sacred import Experiment
from sacred.observers import FileStorageObserver
from arch_head import run_arch_fdm
from argparse import Namespace

ex = Experiment('ArchFDM')
ex.observers.append(FileStorageObserver('arch_fdm_runs'))

@ex.named_config
def default():
    ex.add_config('configs/arch.yaml')

@ex.automain
def run(_run, _config):
    ray.init()
    conf = Namespace(**_config)
    run_arch_fdm(conf, _run)