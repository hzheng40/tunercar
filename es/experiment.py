import ray
from sacred import Experiment
from sacred.observers import FileStorageObserver
from head import run_tunercar
from argparse import Namespace

ex = Experiment('TunerCar')
ex.observers.append(FileStorageObserver('tunercar_runs'))

@ex.named_config
def map0():
    ex.add_config('configs/config_map0.yaml')

@ex.named_config
def map1():
    ex.add_config('configs/config_map1.yaml')

@ex.named_config
def map2():
    ex.add_config('configs/config_map2.yaml')

@ex.named_config
def map3():
    ex.add_config('configs/config_map3.yaml')

@ex.named_config
def map4():
    ex.add_config('configs/config_map4.yaml')

@ex.automain
def run(_run, _config):
    ray.init()
    conf = Namespace(**_config)
    run_tunercar(conf, _run)