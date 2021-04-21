import ray
from sacred import Experiment
from sacred.observers import FileStorageObserver
from head import run_tunercar
from argparse import Namespace

ex = Experiment('TunerCar')
ex.observers.append(FileStorageObserver('tunercar_runs'))

@ex.named_config
def default():
    ex.add_config('configs/default.yaml')

@ex.named_config
def default_random():
    ex.add_config('configs/default_random.yaml')

@ex.named_config
def default_50ctrl():
    ex.add_config('configs/default_50ctrl.yaml')

@ex.named_config
def default_200ctrl():
    ex.add_config('configs/default_200ctrl.yaml')

@ex.named_config
def default_400ctrl():
    ex.add_config('configs/default_400ctrl.yaml')

@ex.named_config
def default_monza():
    ex.add_config('configs/default_monza.yaml')

@ex.named_config
def default_spielberg():
    ex.add_config('configs/default_spielberg.yaml')

@ex.named_config
def default_spielberg_unbounded():
    ex.add_config('configs/default_spielberg_unbounded.yaml')

@ex.named_config
def default_spielberg_stanley():
    ex.add_config('configs/default_spielberg_stanley.yaml')

@ex.named_config
def default_spielberg_lqr():
    ex.add_config('configs/default_spielberg_lqr.yaml')

@ex.named_config
def default_lowest():
    ex.add_config('configs/default_spielberg_lowestpop.yaml')

@ex.named_config
def default_lower():
    ex.add_config('configs/default_spielberg_lowerpop.yaml')

@ex.named_config
def default_highest():
    ex.add_config('configs/default_spielberg_highestpop.yaml')

@ex.named_config
def default_higher():
    ex.add_config('configs/default_spielberg_higherpop.yaml')

@ex.named_config
def default_stanley():
    ex.add_config('configs/default_stanley.yaml')

@ex.named_config
def default_lqr():
    ex.add_config('configs/default_lqr.yaml')

@ex.named_config
def pso():
    ex.add_config('configs/pso.yaml')

@ex.named_config
def pso_lowest():
    ex.add_config('configs/pso_lowest.yaml')

@ex.named_config
def pso_lower():
    ex.add_config('configs/pso_lower.yaml')

@ex.named_config
def pso_highest():
    ex.add_config('configs/pso_highest.yaml')

@ex.named_config
def pso_higher():
    ex.add_config('configs/pso_higher.yaml')

@ex.named_config
def twopointsde():
    ex.add_config('configs/2pointsde.yaml')

@ex.named_config
def twopointsde_dimension():
    ex.add_config('configs/2pointsde_dimension.yaml')

@ex.named_config
def twopointsde_large():
    ex.add_config('configs/2pointsde_large.yaml')

@ex.named_config
def noisyde():
    ex.add_config('configs/noisyde.yaml')

@ex.named_config
def noisyde_dimension():
    ex.add_config('configs/noisyde_dimension.yaml')

@ex.named_config
def noisyde_large():
    ex.add_config('configs/noisyde_large.yaml')

@ex.named_config
def tbpsa():
    ex.add_config('configs/tbpsa.yaml')

@ex.named_config
def oneplusone():
    ex.add_config('configs/oneplusone.yaml')

@ex.named_config
def random():
    ex.add_config('configs/random.yaml')

@ex.automain
def run(_run, _config):
    ray.init()
    conf = Namespace(**_config)
    run_tunercar(conf, _run)