import numpy as np
import pickle
from os import listdir
from os.path import isfile, join

def parse_file_name(f_name):
    split = f_name.split('.')[0].split('_')
    map = split[0]
    optim_method = split[-3]
    popsize = int(split[-2].split('pop')[-1])
    budget = int(split[-1].split('budget')[-1])
    return map, optim_method, popsize, budget

if __name__ == "__main__":
    npz_path = '../es/tunercar_runs/npzs'
    npz_names = [f for f in listdir(npz_path) if isfile(join(npz_path, f))]
    results = [(np.min(np.load(join(npz_path, f_name))['lap_times']), *parse_file_name(f_name)) for f_name in npz_names]
    results.sort()
    print([(t, o, m) for t, m, o, p, b in results])