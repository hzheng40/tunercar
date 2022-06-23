# Overview
This is the reference implementation for our paper:

<em><b>TunerCar: A Superoptimization Toolchain for Autonomous Racing</b></em>

## Citing

If you find this code useful in your work, please consider citing:

```
@inproceedings{o2020tunercar,
  title={TunerCar: A superoptimization toolchain for autonomous racing},
  author={O’Kelly, Matthew and Zheng, Hongrui and Jain, Achin and Auckley, Joseph and Luong, Kim and Mangharam, Rahul},
  booktitle={2020 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={5356--5362},
  year={2020},
  organization={IEEE}
}
```
```
@article{zhengcombinatorial,
  title={Combinatorial and Parametric Gradient-Free Optimization for Cyber-Physical System Design},
  author={Zheng, Hongrui and Betz, Johannes and Ramamurthy, Arun and Jin, Hyunjee and Mangharam, Rahul}
}
```
```
@article{zheng2022gradient,
  title={Gradient-free Multi-domain Optimization for Autonomous Systems},
  author={Zheng, Hongrui and Betz, Johannes and Mangharam, Rahul},
  journal={arXiv preprint arXiv:2202.13525},
  year={2022}
}
```
This implementation also uses the F1TENTH Gym environment from: [https://github.com/f1tenth/f1tenth_gym/tree/exp_py](https://github.com/f1tenth/f1tenth_gym/tree/exp_py)

If you find the simulator useful in your work, please consider citing:

```
@inproceedings{o2020textscf1tenth,
  title={textscF1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning},
  author={O’Kelly, Matthew and Zheng, Hongrui and Karthik, Dhruv and Mangharam, Rahul},
  booktitle={NeurIPS 2019 Competition and Demonstration Track},
  pages={77--89},
  year={2020},
  organization={PMLR}
}
```

## Installation
You can install all dependencies through pip with:

```bash
$ pip3 install gym \
               ray \
               sacred \
               tqdm \
               numpy \
               scipy \
               numba \
               pyyaml \
               nevergrad
$ git clone https://github.com/f1tenth/f1tenth_gym.git
$ cd f1tenth_gym
$ git checkout exp_py
$ pip3 install -e gym/
```

## Configuration and Running Experiments

### Setup

Before running all experiments, you'll have to clone the race tracks repo that contains all scaled-down Formula 1 race tracks:

```bash
$ cd tunercar/es/maps
$ git clone https://github.com/f1tenth/f1tenth_racetracks.git
```

### Example experiment

An example experiment can be ran by:
```bash
$ cd tunercar/es
$ python3 experiment.py with default
```

This will create a directory ```tunercar/es/tunercar_runs``` that contains all the logs and files created during the experiment.

To change a configuration of an experiment, please checkout ```experiment.py```. You can add an named configuration by providing a yaml file in the ```tunercar/es/configs``` directory (For an example, see ```config_map0.yaml```), and adding a function in ```experiment.py```:

```python
@ex.named_config
def custom_config():
    ex.add_config('configs/custom_config.yaml')
```

Then an experiment with the custom config can be ran by:
```bash
$ cd tunercar/es
$ python3 experiment.py with custom_config
```

#Acknowledgement
This work is supported in part by the Defense Advanced Research Projects Agency (DARPA) under the program Symbiotic Design for Cyber Physical Systems (SDCPS) Contract FA8750-20-C-0542 (Systemic Generative Engineering). The views, opinions, and/or findings expressed are those of the author(s) and do not necessarily reflect the view of DARPA.
