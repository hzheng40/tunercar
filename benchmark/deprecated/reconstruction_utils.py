import numpy as np
import nevergrad as ng

def load_saved_data(exp_name, npz_dir='../es/tunercar_runs/npzs/', pkl_dir='../es/tunercar_runs/optims_pkl/'):
    dat = np.load(npz_dir + exp_name + '.npz')
    optim = ng.optimizers.base.Optimizer.load(pkl_dir + exp_name + '_optim.pkl')
    lap_times = dat['lap_times']
    general_params = dat['general_params']
    controller_params = dat['controller_params']
    perturb_vector = dat['perturb_params']
    controller_type = dat['controller']
    return optim, lap_times, general_params, controller_params, perturb_vector, controller_type

def recover_params(raw, conf):
    """
    From normalized vector to actual parameters vector

    Args:
        raw (numpy.ndarray (n, )): normalized vector found by optimizers
        conf (argparse.Namespace): configuration struct

    Return:
        work (dict): converted work dictionary
    """
    work = {}
    # general
    work['perturb'] = conf.left_bound + raw[:conf.num_ctrl]*(conf.right_bound - conf.left_bound)
    work['mass'] = conf.mass_min + raw[conf.num_ctrl]*(conf.mass_max - conf.mass_min)
    work['lf'] = conf.lf_min + raw[conf.num_ctrl + 1]*(conf.lf_max - conf.lf_min)
    work['vel_min'] = conf.v_lower_min + raw[conf.num_ctrl + 2]*(conf.v_lower_max - conf.v_lower_min)
    work['vel_max'] = conf.v_upper_min + raw[conf.num_ctrl + 3]*(conf.v_upper_max - conf.v_upper_min)
    # controller
    if conf.controller == 'stanley':
        work['kpath'] = conf.kpath_min + raw[-1]*(conf.kpath_max - conf.kpath_min)
    elif conf.controller == 'lqr':
        work['q1'] = conf.q1_min + raw[-5]*(conf.q1_max - conf.q1_min)
        work['q2'] = conf.q2_min + raw[-4]*(conf.q2_max - conf.q2_min)
        work['q3'] = conf.q3_min + raw[-3]*(conf.q3_max - conf.q3_min)
        work['q4'] = conf.q4_min + raw[-2]*(conf.q4_max - conf.q4_min)
        work['r'] = conf.r_min + raw[-1]*(conf.r_max - conf.r_min)
    else:
        # default to pp
        work['tlad'] = conf.tlad_min + raw[-1]*(conf.tlad_max - conf.tlad_min)

    return work