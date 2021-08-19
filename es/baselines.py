"""
Baselines found by random searching, passes some of the tests, but have bad lqr params
"""
import numpy as np

# quad_baseline1 is highest scoring with RandomSearch seed 123
quad_baseline1 = np.array([ 2.60000000e+01,  1.70000000e+01,  5.00000000e+00,  4.00000000e+00,
                       1.00000000e+01,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                       1.00000000e+00,  2.06000000e+02,  1.06000000e+02,  1.57000000e+02,
                       2.55000000e+02,  6.00000000e+00,  3.80000000e+01,  3.80000000e+01,
                       3.10000000e+01,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
                       0.00000000e+00,  4.83785956e+02,  3.73685013e+02,  7.56345072e+02,
                       8.72621576e+02,  2.84571354e+01,  2.60426587e+01,  6.23043247e+01,
                       9.97925992e+01,  2.45323565e+00,  1.66716652e+00,  1.95164670e+00,
                       7.94821247e-01,  4.78369637e-01,  1.42196907e+00,  1.61500806e+00,
                       1.13017873e+00,  7.65670138e-01,  1.14171064e+00,  1.76207902e+00,
                       2.00088341e+00,  1.77935201e+00,  1.31697451e+00,  2.16657834e+00,
                       1.42920867e+00,  1.75972459e+00,  1.91122897e+00,  1.87740770e+00,
                       2.38628847e+00,  6.98187913e+00,  1.20576802e+01,  1.43490932e+01,
                       1.60497639e+01,  6.63037687e-01,  1.34347446e+00, -3.19835893e+00,
                      -1.96116314e+00])
quad_scores1 = np.array([359.,  10.,  10., 349.])

# quad_baseline2 is highest scoring with RandomSearch seed 456
quad_baseline2 = np.array([ 2.60000000e+01,  1.90000000e+01,  1.30000000e+01,  0.00000000e+00,
                       5.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                       0.00000000e+00,  1.43000000e+02,  5.20000000e+01,  3.27000000e+02,
                       2.10000000e+02,  7.80000000e+01,  6.00000000e+00,  8.00000000e+00,
                       5.50000000e+01,  0.00000000e+00,  1.00000000e+00,  1.00000000e+00,
                       1.00000000e+00,  4.72834860e+02,  6.59774017e+02,  3.31209957e+02,
                       4.76189639e+02,  5.41916272e+01,  1.07440528e+01,  7.98668232e+01,
                       5.53149960e+01,  2.04607019e+00,  1.70705367e+00,  1.15075842e+00,
                       1.12926230e+00,  9.73267135e-01,  3.79611009e-01,  1.39078354e+00,
                       7.15235178e-01,  2.56703159e+00,  1.02866896e+00,  2.35793270e+00,
                       1.46595126e+00,  1.87212994e+00,  8.54397913e-01,  2.03775782e+00,
                       1.42827260e+00,  1.11270996e+00,  1.73728135e+00,  9.51424845e-01,
                       1.76160748e+00,  1.10250950e+01,  6.32899741e+00,  1.54392024e+01,
                       1.19637782e+01, -1.39091792e-01, -3.08689430e+00, -2.33703901e-01,
                       8.67978904e-01])
quad_scores2 = np.array([364.,  10.,  10., 277.])

# hexring baseline1 is highest scoring with PortfolioDiscreteOnePlusOne seed 456
hexring_baseline1 = np.array([ 2.30000000e+01,  1.40000000e+01,  1.70000000e+01,  1.40000000e+01,
                               6.00000000e+00,  3.00000000e+00,  2.00000000e+00,  1.00000000e+00,
                               1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                               1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
                               1.10000000e+02,  3.58000000e+02,  7.80000000e+01,  1.79000000e+02,
                               5.20000000e+01,  1.49000000e+02,  9.00000000e+00,  2.50000000e+01,
                               3.10000000e+01,  2.40000000e+01,  2.50000000e+01,  1.70000000e+01,
                               1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
                               0.00000000e+00,  1.00000000e+00,  4.07012681e+02,  2.64371822e+02,
                               5.57818219e+02,  4.65365066e+02,  6.02678988e+02,  4.98141552e+02,
                               4.20132709e+02,  4.81602803e+02,  4.56164241e+02,  6.16287892e+01,
                               7.13784741e+01,  3.09162042e+01,  3.81486771e+01,  7.15068517e+01,
                               7.24131005e+01,  9.42443146e-01,  1.20195309e+00,  1.30662602e+00,
                               1.81151800e+00,  1.23729300e+00,  2.01361229e+00,  9.36063313e-01,
                               1.11032191e+00,  1.48535392e+00,  9.04854589e-01,  1.39844693e+00,
                               2.17836894e+00,  1.06961939e+00,  1.88993878e+00,  1.77330217e+00,
                               2.38121353e+00,  1.62370624e+00,  1.49270557e+00,  1.99891192e+00,
                               2.81879700e+00,  1.59848664e+01,  1.00000000e+01,  1.34364582e+01,
                               8.97167202e+00, -1.61946782e-01, -5.23129612e-01,  1.15548249e+00,
                              -2.43736024e+00])
hexring_scores1 = np.array([410., 310.,  10., 210.])