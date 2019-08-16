import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

# %% import public modules
import matplotlib.pyplot as plt
plt.close("all")

from scipy import signal
import autograd.numpy as np


# %% import custom modules
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum

from classes.environment import OpticalField, OpticalField_CW, OpticalField_Pulse
from classes.components import Fiber, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter, PhaseShifter, DelayLine, PhotonicAWG
from classes.experiment import Experiment


#%%
env = OpticalField_CW(n_samples=2**14, window_t=10e-9, peak_power=1)
# env = OpticalField_Pulse(n_samples=2 ** 14, profile='gaussian', pulse_width=100e-12, f_rep=100e6, n_pulses=15, peak_power=1)
# env = OpticalField_Pulse(n_samples=2 ** 15, profile='gaussian', pulse_width=100e-12, f_rep=50e7, n_pulses=30, peak_power=1)

# components = {
#     0: PhotonicAWG(),
# 	1: WaveShaper()
# }
# adj = [(0, 1)]

# ps = PhaseShifter()
# ps.SIGMA = [0.1 * np.pi]
# components = {
# 	1: PowerSplitter(),
# 	2: PhaseShifter(),
# 	4: ps,
# 	3: PowerSplitter(),
# }
# adj = [ (1, 2), (2, 3), (1,4), (4, 3) ]
# at = {2: [2.0*np.pi], 4: [0.0 * np.pi]}

# def grad_descent()

# components = {
# 	1: PowerSplitter(),
# 	2: PhaseShifter(),
# 	3: PowerSplitter(),
# }
# adj = [ (1, 2), (2, 3), (1,3) ]

components = {
               0:PhaseModulator(),
               1:WaveShaper(),
            }
adj = [(0,1)]
at = {   0: [ 0.8234132112176106,
             12000000000.0],
        1: [ 0.6060667463824007,
             0.0,
             0.721833819341576,
             0.9503376429345127,
             0.6234058812515285,
             4.888443104871878,
             0.4101923968206471,
             1.9730094674171068,
             0.5122717909947406,
             2.191008207394641]
         }

# %% initialize the experiment, and perform all the pre-processing steps
exp = Experiment()
exp.buildexperiment(components, adj)
exp.checkexperiment()

exp.make_path()
exp.check_path()
exp.inject_optical_field(env.At)
exp.draw(node_label='both', title='')


exp.setattributes(at)
exp.simulate(env)
At = exp.nodes[exp.measurement_nodes[0]]['output']
fit = env.fitness(At)
print("Fitness: {}".format(fit))

#%%
fig = plt.figure()
plt.plot(env.f, PSD(At, env.dt, env.df), label='output')
plt.plot(env.f, PSD(env.At0, env.dt, env.df), label='input', alpha=0.4, ls='--')
plt.legend()
plt.show()

#%%
fig = plt.figure()
plt.plot(env.t, P(At), label='output')
plt.plot(env.t, P(env.At0), label='input', alpha=0.4, ls='--')
plt.legend()
plt.show()

#%%
fig = plt.figure()
exp.visualize(env, at)
plt.show()

#%%
print('Starting analysis')
exp.init_fitness_analysis(at, env, method='LHA', verbose=True)
lha_stability, (H0, eigvals, eigvecs) = exp.run_analysis(at, verbose=True)

x_opt, node_lst, idx_lst, sigma_lst, mu_lst, at_names = exp.experiment_info_as_list(at)

run_other_analysis = True
if run_other_analysis:
	exp.init_fitness_analysis(at, env, method='UDR', verbose=True)
	udr_stability, *tmp = exp.run_analysis(at, verbose=True)

	exp.init_fitness_analysis(at, env, method='MC', verbose=True)
	mc_stability, *tmp = exp.run_analysis(at, verbose=True)

#%%
xvals = np.arange(len(lha_stability))
plt.figure(figsize=[9,6])
plt.stem(xvals-0.2, udr_stability/np.max(udr_stability), label='UDR', markerfmt='ro', linefmt ='r--')
plt.stem(xvals-0.0, mc_stability/np.max(mc_stability), label='MC', linefmt='g--', markerfmt='go')
plt.stem(xvals+0.2, np.abs(lha_stability)/np.max(lha_stability), label='LHA', linefmt='b--', markerfmt='bo')
# ticknames = ['{} : {}'.format(exp.nodes[node]['info'].disp_name, exp.nodes[node]['info'].AT_NAME[idx]) for node, idx in zip(node_lst, idx_lst)]
ticknames = ['Parameter{}'.format(i) for i in range(len(xvals))]
plt.ylabel('Relative sensitivity of fitness')
plt.xticks(xvals, ticknames, rotation=90)
plt.legend()
plt.show()
