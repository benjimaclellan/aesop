import autograd.numpy as np
from itertools import count
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum
import matplotlib.pyplot as plt
import matplotlib

"""
ASOPE
|- components.py

Each component in the simulations is described by a custom 'Component' class. Within this class various physical parameters are stored (for example dispersion of a fiber) which are neeed for the simulation of an experimental setup containing that component, but also variables for the optimization. 

Each component class a number of important class variables:
    - type = what type of component is this (awg, powersplitter, fiber, etc)
    - N_PARAMETERS = how many parameters on the component are to be optimized (as a list)
    - UPPER = upper bound on your optimization parameters (as a list of length N_PARAMETERS)
    - LOWER = lower bound on your optimization parameters (as a list of length N_PARAMETERS)
    - DTYPE = the datatype of each parameter you are optimizing, either int or float (as a list)
    - DSCRTVAL = the discretization step (resolution) when generating a random attribute for the parameters (None if you want continuous)
    - FINETUNE_SKIP = this is for the fine-tuning using gradient descent. As some parameters are integers (for example, the number of steps on the AWG) and the grad-desc cannot deal with ints, we skip is. This is a list of the indices to skip
    - splitter = Defines whether this component will have more than one output or input (only certain component types support multiple input/outputs)
    
There is also important class functions:
    - datasheet() = contains all the information about the component
    - simulate() = simulates the transformation of the component to the input
    - mutate() = in the GA, used to mutate the attributes on each component
    - newattribute() = will create random attribute for ALL parameters of the component
    
    - randomattribute() = based on the settings for the component (bounds, discretization, etc), will generate ONE random attribute (setting) for the component
"""

class Component(object):
    def __init__(self):
        """
            Initialize each component, and saves the datasheet to as class variables
        """
        self.datasheet()
        self.updateinstances()
        self.name = str(self.type) + str(self.id)
        
    def updateinstances(self):
        """
            Keeps track of how many of each component is in the setup to ensure they are distinguishable (fiber0, fiber1, etc)
        """
        self.id = next(self._num_instances)
    
    def resetinstances(self):
        self._num_instances = count(0)
    
    def datasheet(self):
        """
            Different for each component, but saves all important parameters for the physical device
        """
        return
    
    def simulate(self):
        """
            Simulates the transformation of the component on the pulse
        """
        raise ValueError('Not implemented yet')  
    
    def newattribute(self):
        """
            Creates a list of attributes (parameters) for the device
        """
        at = []
        for i in range(self.N_PARAMETERS):
            at.append(self.randomattribute(self.LOWER[i], self.UPPER[i], self.DTYPE[i], self.DSCRTVAL[i]))
        self.at = at
        return at

    def mutate(self):
        """
            Mutates the list of attributes (parameters) for the device, used in the GA
        """
        mut_loc = np.random.randint(0, self.N_PARAMETERS)
        self.at[mut_loc] = self.randomattribute(self.LOWER[mut_loc], self.UPPER[mut_loc], self.DTYPE[mut_loc],
                                                self.DSCRTVAL[mut_loc])
        return self.at
        
        
    def randomattribute(self, low=0.0, high=1.0, dtypes='float', dscrtval=None):
        """
            Common function for creating a new random attribute, based on the bounds (upper and lower), datatype, and discretization
        """
        
        if dtypes == 'float':
            if dscrtval is not None:  
                at = round(np.random.uniform( low, high )/dscrtval) * dscrtval 
            else:
                at = np.random.uniform( low, high )
                
        elif dtypes == 'int':
            if dscrtval is not None:    
                at = np.round(np.random.randint( low/dscrtval, high/dscrtval))*dscrtval
            else: 
                at = np.random.randint(low, high)
        else:
            raise ValueError('Unknown datatype when making a new attribute')
    
        return at

# ----------------------------------------------------------
class Input(Component):
    """
    """
    _num_instances = count(0)
    def datasheet(self):
        self.type = 'input'
        self.disp_name = 'Input'
        self.N_PARAMETERS = 0
        self.UPPER = []
        self.LOWER = []
        self.SIGMA = []
        self.MU = []
        self.DTYPE = []
        self.DSCRTVAL = []
        self.FINETUNE_SKIP = []
        self.splitter = False
        self.AT_NAME = []

    def simulate(self, env, field,  visualize=False):
        field = env.field
        if visualize:
            self.lines = ()
        return field


class Output(Component):
    """
    """
    _num_instances = count(0)
    def datasheet(self):
        self.type = 'output'
        self.disp_name = 'Output'
        self.N_PARAMETERS = 0
        self.UPPER = []
        self.LOWER = []
        self.SIGMA = []
        self.MU = []
        self.DTYPE = []
        self.DSCRTVAL = []
        self.FINETUNE_SKIP = []
        self.splitter = False
        self.AT_NAME = []

    def simulate(self, env, field,  visualize=False):
        field = env.field
        if visualize:
            self.lines = ()
        return field




# ----------------------------------------------------------
# Here we now implement each component - and more can be added easily or adapted to a different purpose (ie quantum).
# It is also trivial to change how the device is simulated without changing the rest of the code, provided the general format is followed
# ----------------------------------------------------------
#%%
class Dummy(Component):
    """
    """
    _num_instances = count(0)
    def datasheet(self):
        self.type = ''
        self.disp_name = ''
        self.N_PARAMETERS = []
        self.UPPER = []
        self.LOWER = []
        self.SIGMA = []
        self.MU = []
        self.DTYPE = []
        self.DSCRTVAL = []
        self.FINETUNE_SKIP = []
        self.splitter = False
        self.AT_NAME = []
    def simulate(self, env, field, visualize = False):
        if visualize:
            self.lines = (('t', None, 'DUMMY'),)
            # self.lines = (('f',ampmask,'WaveShaper Amplitude Mask'),('f', phasemask,'WaveShaper Phase Mask'),)
        return field
    def newattribute(self):
        return []
    def mutate(self):
        return []


class Fiber(Component):
    """
        Simple dispersive fiber. Only considers second order dispersion for now.
    """
    _num_instances = count(0)
    def datasheet(self):
        self.type = 'fiber'
        self.disp_name = 'Dispersion Compensating Fiber'
        self.beta = [2e-20]     # second order dispersion (SI units)
        self.N_PARAMETERS = 1
        self.UPPER = [5000]
        self.LOWER = [0]
        self.DTYPE = ['float']
        self.DSCRTVAL = [None]
        self.SIGMA = [0.15]
        self.MU = [0.0]
        self.FINETUNE_SKIP = []
        self.splitter = False
        self.AT_NAME = ['fiber length']

    def simulate(self, env, field, visualize=False):

        # attribute list is extracted. For fiber, only one parameter which is length
        fiber_len = self.at[0]

        # calculate the dispersion operator in the spectral domain
        D = np.zeros(env.f.shape).astype('complex')
        factorial = 1
        for n in range(0, len(self.beta)):
#            D += self.beta[n] * np.power(2*np.pi*env.f, n+2) / np.math.factorial(n+2)
            D += self.beta[n] * np.power(2*np.pi*env.f, n+2) / factorial 
            factorial *= n
            
            
        # apply dispersion
        Af = np.exp(fiber_len * -1j * D) * FFT(field, env.dt)
        field = IFFT( Af, env.dt )

        # this visualization functionality was broken, may be fixed later
        if visualize:
            self.lines = (('f',D, 'Dispersion'),)

        return field

    def newattribute(self):
        at = [self.randomattribute(self.LOWER[0], self.UPPER[0], self.DTYPE[0], self.DSCRTVAL[0])]
        self.at = at
        return at

    def mutate(self):
        at = [self.randomattribute(self.LOWER[0], self.UPPER[0], self.DTYPE[0], self.DSCRTVAL[0])]
        self.at = at
        return at

# ----------------------------------------------------------
class PhaseModulator(Component):
    """
        Electro-optic phase modulator, driven by a single sine tone (for now).
    """
    _num_instances = count(0)
    def datasheet(self):
        self.type = 'phasemodulator'
        self.disp_name = 'Electro-Optic Phase Modulator'
        self.vpi = 1
        # self.N_PARAMETERS = 3
        # self.UPPER = [2*np.pi, 1e9, 2*np.pi]   # max shift, frequency
        # self.LOWER = [0, 1e9, 0]
        # self.SIGMA = [0.1, 1e6, 0.05*np.pi]
        # self.MU = [0.0, 0.0, 0.0]
        # self.DTYPE = ['float', 'float', 'float']
        # self.DSCRTVAL = [None, None, None]
        self.N_PARAMETERS = 2
        self.UPPER = [2 * np.pi, 12e9]  # max shift, frequency
        self.LOWER = [0, 12e9]
        self.SIGMA = [0.1, 0.1e7]
        self.MU = [0.0, 0.0]
        self.DTYPE = ['float', 'float']
        self.DSCRTVAL = [None, None]
        self.FINETUNE_SKIP = []
        self.splitter = False
        self.AT_NAME = ['EOM Modulation Depth', 'EOM{} Modulation Frequency']

    def simulate(self, env, field,  visualize=False):

        # plt.figure(figsize=[2, 2.5])
        # maskf = (env.f >= -4 * 12e9) & (env.f <= 4 * 12e9)
        # psd = PSD(field, env.dt, env.df)
        # plt.plot(env.f[maskf] / 1e9, psd[maskf] / np.max(psd), color='black', label='PSD')
        # # plt.legend()
        # plt.xticks([-50, 0, 50], [-50, 0, 50]), plt.yticks([0, 0.5, 1.0],[0, 0.5, 1.0])
        # plt.xlabel('Frequency (GHz)')
        # plt.ylabel('PSD')
        # plt.savefig("/home/benjamin/Documents/Communication - Patents/ASOPE/Figures/awg_allsteps/cw_transfer.eps",
        #             bbox="tight")
        # plt.show()


        # extract attributes (parameters) of driving signal
        M = self.at[0]       # phase amplitude [V/Vpi]
        NU = self.at[1]      # frequency [Hz]
        # OFFSET = self.at[2]
        OFFSET = 0
        BIAS = 0
        phase = (M)*(np.cos(2*np.pi* NU * env.t + OFFSET)+BIAS)

        # apply phase shift temporally
        field = field * np.exp(1j * phase)

        # plt.figure(figsize=[2,2.5])
        # maskf = (env.f >= -4*12e9) & (env.f <= 4*12e9)
        # psd = PSD(field, env.dt, env.df)
        # plt.plot(env.f[maskf]/1e9,  psd[maskf]/np.max(psd), color='black', label='PSD')
        # plt.xticks([-50, 0, 50], [-50, 0, 50]), plt.yticks([0, 0.5, 1.0], [0, 0.5, 1.0])
        # plt.xlabel('Frequency (GHz)')
        # # plt.legend()
        # plt.ylabel(" ")
        # # plt.xticks([-50, 0, 50], 3 * [" "]), plt.yticks([0, 0.5, 1.0], 3 * [" "])
        # plt.savefig("/home/benjamin/Documents/Communication - Patents/ASOPE/Figures/awg_allsteps/eom_transfer.eps", bbox="tight")
        # plt.show()

        if visualize:
            self.lines = (('t',phase, 'Phase Modulation'),)

        return field

    def newattribute(self):

        at = []
        for i in range(self.N_PARAMETERS):
            at.append(self.randomattribute(self.LOWER[i], self.UPPER[i], self.DTYPE[i], self.DSCRTVAL[i]))
        self.at = at
        return at

    def mutate(self):
        mut_loc = np.random.randint(0, self.N_PARAMETERS)
        self.at[mut_loc] = self.randomattribute(self.LOWER[mut_loc], self.UPPER[mut_loc],        self.DTYPE[mut_loc], self.DSCRTVAL[mut_loc])
        return self.at


#%%
class WaveShaper(Component):
    """
        Waveshaper, with amplitude and phase masks. Currently, the mask profile are made with a polynomial (parameters are the polynomial coefficients) and then clipped to valid levels (0-1 for amplitude, 0-2pi for phase). This is admittedly likely not the best solution - but for now it can work.
    """
    _num_instances = count(0)
    def datasheet(self):
        self.type = 'waveshaper'
        self.disp_name = 'Programmable Filter'
        self.bitdepth = 8
        self.res = 12e9     # resolution of the waveshaper
        self.n_windows = 5
        self.N_PARAMETERS = 2*self.n_windows
        self.UPPER = self.n_windows*[1.0] + self.n_windows*[2*np.pi]
        self.LOWER = self.n_windows*[0.0] + self.n_windows*[0.0]
        self.SIGMA = self.n_windows*[0.05] + self.n_windows*[0.05*np.pi]
        self.MU = self.n_windows*[0.0] + self.n_windows*[0.0]
        self.DTYPE = self.n_windows * ['float'] + self.n_windows * ['float']

        self.DSCRTVAL = self.N_PARAMETERS * [None]
        # self.DSCRTVAL = self.n_windows * [1/(2**self.bitdepth-1)] + self.n_windows * [2*np.pi/(2**self.bitdepth-1) ]
        self.FINETUNE_SKIP = []
        self.splitter = False

        self.AT_NAME = ['WS Amplitude Window {}'.format(j - int(np.floor(self.n_windows/2))) for j in range(0, self.n_windows, 1)] + ['WS Phase Window {}'.format(j - int(np.floor(self.n_windows/2))) for j in range(0, self.n_windows, 1)]


    def temp_func(self, left, right, i, vals):
        if i >= left and i < right:
            return vals[i-left][0]
        else:
            return 0

    def simulate(self, env, field, visualize = False):
        # Slice at into the first half (amp) and last half (phase)
        ampvalues = self.at[0:self.n_windows]
        # ampvalues = [self.at[0], 0, self.at[self.n_windows-1]]
        phasevalues = self.at[self.n_windows:]
        # phasevalues = [0, 0, np.pi]

        n = np.floor(env.n_samples/((1/env.dt)/self.res)).astype('int')
        N = np.shape(env.f)[0]
        tmp = np.ones((n,1))

        a = np.array([i*tmp for i in ampvalues])
        p = np.array([i*tmp for i in phasevalues])

        amp1 = np.concatenate(a)
        phase1 = np.concatenate(p)

        left = np.floor((env.n_samples - amp1.shape[0])/2).astype('int')
        right = env.n_samples - np.ceil((env.n_samples - amp1.shape[0])/2).astype('int')

        # we will pad amp1 and phase1 with zeros so they are the correct size
        padleft = np.zeros((left, 1))
        padright = np.zeros((N-right, 1))

#        if right - left > env.n_samples:
#            left = 0
#            right = env.n_samples
##            raise Warning('Frequency window less than the resolution of waveshaper')
#            print('Frequency window less than the resolution of waveshaper')
#            phase1 = np.ones_like(env.f) * phasevalues[0]
#            amp1 = np.ones_like(env.f) * ampvalues[0]

        # Concatenate the arrays together
        # We cannot use array assignment as it is not supported by autograd
        ampmask = np.concatenate((padleft, amp1, padright), axis=0)
        phasemask = np.concatenate((padleft, phase1, padright), axis=0)

        Af = ampmask * np.exp(1j*(phasemask)) * FFT(field, env.dt)
        field = IFFT( Af, env.dt )

        # plt.figure(figsize=[2,2.5])
        # maskf = (env.f >= -4*12e9) & (env.f <= 4*12e9)
        # psd = PSD(field, env.dt, env.df)
        # plt.plot(env.f[maskf]/1e9, psd[maskf]/np.max(psd), color='black')
        # plt.plot(env.f[maskf]/1e9, ampmask[maskf], color='grey', ls='-', label="Amplitude")
        # plt.plot(env.f[maskf]/1e9, phasemask[maskf]/np.pi/2, color='grey', ls=':', label="Phase Mask")
        # # plt.legend()
        # plt.xticks([-50, 0, 50], [-50, 0, 50]), plt.yticks([0, 0.5, 1.0], [0, 0.5, 1.0])
        # plt.xlabel('Frequency (GHz)')
        # plt.ylabel(" ")
        # # plt.xticks([-50,0,50], 3*[" "]), plt.yticks([0,0.5,1.0], 3*[" "])
        # plt.savefig("/home/benjamin/Documents/Communication - Patents/ASOPE/Figures/awg_allsteps/ws_transfer.eps", bbox="tight")
        # plt.show()

        if visualize:
            self.lines = (('f',ampmask,'WaveShaper Amplitude Mask'),('f', phasemask,'WaveShaper Phase Mask'),)
        return field

    def newattribute(self):
        at = []
        for i in range(self.N_PARAMETERS):
            at.append(self.randomattribute(self.LOWER[i], self.UPPER[i], self.DTYPE[i], self.DSCRTVAL[i]))
        self.at = at
        return at


    def mutate(self):
        mut_loc = np.random.randint(0, self.N_PARAMETERS)
        self.at[mut_loc] = self.randomattribute(self.LOWER[mut_loc], self.UPPER[mut_loc],        self.DTYPE[mut_loc], self.DSCRTVAL[mut_loc])
        return self.at


#%%
class PowerSplitter(Component):
    """
        Simple balanced (3dB for two arms) power splitter. No parameters to optimize.
    """
    _num_instances = count(0)
    def datasheet(self):
        self.type = 'powersplitter'
        self.disp_name = 'Power Splitter'
        self.N_PARAMETERS = 0
        self.UPPER = []
        self.LOWER = []
        self.SIGMA = []
        self.MU = []
        self.DTYPE = ['float']
        self.DSCRTVAL = [None]
        self.FINETUNE_SKIP = []
        self.splitter = True
        self.AT_NAME = []

    def simulate(self, env, At_in, num_outputs, visualize=False):
        # ensure there is maximum 2 inputs/outputs (for now)

        num_inputs = At_in.shape[1]
        assert num_inputs <= 2
        assert num_outputs <= 2

        # this is kinda overkill, but can be extended to multi-path powersplitters (ie tritters) if wanted
        XX,YY = np.meshgrid(np.linspace(0,num_outputs-1, num_outputs), np.linspace(0,num_inputs-1, num_inputs))

        # in the case of 2x2 splitter, this works, but should check for more arms
        S = (1/max([num_outputs,1])) * np.exp(np.abs(XX - YY) * 1j * np.pi  )

        # apply scattering matrix to inputs and return the outputs
#        At_out = At_in.dot(S)
        At_out = np.dot(At_in, S)

        if visualize:
            self.lines = None

        return At_out

    def newattribute(self):
        at = [self.randomattribute(self.LOWER[0], self.UPPER[0], self.DTYPE[0], self.DSCRTVAL[0])]
        self.at = at
        return at

    def mutate(self):
        at = [self.randomattribute(self.LOWER[0], self.UPPER[0], self.DTYPE[0], self.DSCRTVAL[0])]
        self.at = at
        return at

#%%
class FrequencySplitter(Component):
    """
        Frequency splitter for splitting spectrum into two spatial paths. Currently using one paramter (attribute), which sets where the (even) split occurs. However, it can trivially be extended to more complex selection of wavelengths
    """
    _num_instances = count(0)
    def datasheet(self):
        self.type = 'frequencysplitter'
        self.disp_name = 'Wavelength Selective Splitter'
        self.N_PARAMETERS = 1
        self.UPPER = [0.1]
        self.LOWER = [-0.1]
        self.SIGMA = [0.01]
        self.MU = [0.0]
        self.DTYPE = ['float']
        self.DSCRTVAL = [0.02]
        self.FINETUNE_SKIP = [0]
        self.splitter = True
        self.AT_NAME = ['frequency split']
    def simulate(self, env, At_in, num_outputs, visualize=False):
        # ensuring that, for now, we only have maximum ONE input. Please use a powersplitter for coupling (easier to deal with)

        # ensuring that, for now, we only have maximum two outputs
        num_inputs = At_in.shape[1]
        assert num_inputs <= 2
        assert num_outputs <= 2

        # collect the input (single input path)
        if num_inputs > 1:
            k = np.array([[np.exp(1j*0)], [np.exp(1j*np.pi)]])
            Af_in = FFT(np.sum(np.dot(At_in,k), axis=1), env.dt).reshape(env.n_samples,1)
        else:
            Af_in = FFT(At_in, env.dt)

        if num_outputs > 1:
            # extract the frequency location to split at (can be extended to have two)
            splits = (env.f[0]-env.df, self.at[0]*env.f[-1])
            split1 = min(splits)
            split2 = max(splits)

            # create masks, which are used to select the frequencies on each outgoing spatial path
            mask = np.ones([env.n_samples,2], dtype='complex')
            mask[env.f[:,0] <= split1,0] = 0
            mask[env.f[:,0] > split2,0] = 0

            # second mask is the NOT of the first
            mask[:,1] = np.logical_not(mask[:,0]).astype('float') * np.exp(1j*np.pi)

            if visualize:
                self.lines = (('f', mask[:,0],'WDM Profile'),)

            # apply masks and send to next components
            At_out = IFFT(mask*Af_in, env.dt)

        else:
            At_out = IFFT(Af_in, env.dt)#.reshape(env.n_samples, 1)
            if visualize:
                self.lines = None

        return At_out

    def newattribute(self):
        at = [self.randomattribute(self.LOWER[0], self.UPPER[0], self.DTYPE[0], self.DSCRTVAL[0])]
        self.at = at
        return at

    def mutate(self):
        at = [self.randomattribute(self.LOWER[0], self.UPPER[0], self.DTYPE[0], self.DSCRTVAL[0])]
        self.at = at
        return at
    
#%%
class PhaseShifter(Component):
    """
    """
    _num_instances = count(0)
    def datasheet(self):
        self.type = 'phase shifter'
        self.disp_name = 'Phase Shifter'
        self.N_PARAMETERS = 1
        self.UPPER = [2*np.pi] #[1e-7]
        self.LOWER = [0]
        self.DTYPE = ['float']
        self.DSCRTVAL = [None]
        self.SIGMA = [0.05*np.pi]
        self.MU = [0.0]
        self.FINETUNE_SKIP = []
        self.splitter = False
        self.AT_NAME = ['phase shift']

    def simulate(self, env, field, visualize=False):
        # attribute list is extracted
        phase_shift = self.at[0]

        # length_shift = self.at[0]
        # phase_shift = 2 * np.pi * env.f * length_shift / 299792458

        # apply phase_shift
        Af = np.exp(phase_shift * -1j) * FFT(field, env.dt)
        field = IFFT( Af, env.dt )
        # this visualization functionality was broken, may be fixed later
        if visualize:
            self.lines = (('f',np.ones_like(env.t) * phase_shift, 'Phase Shift'),)
        return field

    def newattribute(self):
        at = [self.randomattribute(self.LOWER[0], self.UPPER[0], self.DTYPE[0], self.DSCRTVAL[0])]
        self.at = at
        return at

    def mutate(self):
        at = [self.randomattribute(self.LOWER[0], self.UPPER[0], self.DTYPE[0], self.DSCRTVAL[0])]
        self.at = at
        return at



# #%%
# class DelayLine(Component):
#     """
#     """
#     _num_instances = count(0)
#     def datasheet(self):
#         self.type = 'delay line'
#         self.disp_name = 'Integrated Delay Line'
#         self.N_PARAMETERS = 3
#         self.delays = [0] + [2**i * 1e-12 for i in range(0, self.N_PARAMETERS-1)]  # bit steps of 1ps
#         self.UPPER = self.N_PARAMETERS * [1]
#         self.LOWER = self.N_PARAMETERS * [0]
#         self.DTYPE = ['float']
#         self.DSCRTVAL = self.N_PARAMETERS * [0.2]
#         self.SIGMA = self.N_PARAMETERS * [0.05]
#         self.MU = self.N_PARAMETERS * [0.0]
#         self.FINETUNE_SKIP = []
#         self.splitter = False
#         self.AT_NAME = ['delay line {}'.format(j) for j in range(0, self.N_PARAMETERS, 1)]
#
#     def simulate(self, env, field, visualize=False):
#         # attribute list is extracted
#
#         ratios = self.at
#
#         tmp = np.fft.fft(field, axis=0)
#         Af_delay = np.zeros_like(field)
#
#         ratio = 1.0
#         for line_i, ratio_i in enumerate(ratios):
#             ratio = ratio * ratio_i
#             delayed_tmp = (1-ratio) * tmp * np.exp(1j * 2 * np.pi * self.delays[line_i] * env.f)
#             Af_delay += delayed_tmp
#
#         field = np.fft.ifft(Af_delay, axis=0)
#         # this visualization functionality was broken, may be fixed later
#         if visualize:
#             def find_nearest(array, value):
#                 array = np.asarray(array)
#                 return (np.abs(array - value)).argmin()
#
#             shift_vis = np.zeros_like(env.t)
#             ratio = 1.0
#             for line_i, ratio_i in enumerate(ratios):
#                 ratio = ratio * ratio_i
#                 shift_vis[find_nearest(env.t, self.delays[line_i])] = 1 - ratio
#
#
#             self.lines = (('t', shift_vis, 'Delay Shifts'),)
#         return field


# %%
class GasSample(Component):
    """
    """
    _num_instances = count(0)

    def datasheet(self):
        self.type = 'gas sample'
        self.disp_name = 'Gas Cell'
        self.N_PARAMETERS = 1
        self.UPPER = [1e13]  #concentration (ppm),
        self.LOWER = [-1e13]
        self.DTYPE = ['float']
        self.DSCRTVAL = [None]
        self.SIGMA = [100e8] #[0.01 * (self.UPPER[0] - self.LOWER[0])]
        self.MU = [0.0]
        self.FINETUNE_SKIP = []
        self.splitter = False
        self.AT_NAME = ['CW Laser Detuning']

        self.z0 = 1 # length of sample (m)
        self.eps0 =  8.8541878128e-12 # permitivvity of free space
        self.c0 = 299792458 # speed of light m/s
        self.f0 = 2*np.pi*self.c0 / 1549.7302e-9  # resonance frequency
        self.gamma0 = 3.5e9 # line width in Hz
        self.e = 1.60217662e-19
        self.me = 9.10938356e-31

    def simulate(self, env, field, visualize=False):
        # C = self.at[0]
        # n = 1 + C * self.e ** 2.0 / self.me / (4 * self.eps0 * self.f0) * (env.f/(np.power(env.f, 2) + np.power(self.gamma0,2)))
        # alpha = C * self.e ** 2.0 / self.me / (2 * self.eps0 * self.c0) * ((env.f- self.f0)/(self.f0)) * (np.power(self.gamma0, 2)/(np.power(env.f, 2) + np.power(self.gamma0,2)))
        # print(max(n))

        # N = self.at[0]
        # fi = 1.0
        N = 3e15
        ni = 1.3
        detune = self.at[0]

        w0 = 2 * np.pi * self.f0
        dw = 2 * np.pi * (env.f + detune)
        w = dw + w0

        eps_complex = np.sqrt(np.power(ni, 2) + (N * self.e * self.e / self.eps0 / self.me) * 4 * np.pi / (
                        (np.power(w0, 2) - np.power(w, 2) - 1j * self.gamma0 * 2 * np.pi * (w))))


        # eps_complex = (np.power(ni, 2) +
        #                 (N*self.e*self.e/self.me) * 1/((4*np.pi*np.pi)*(-np.power(env.f, 2)
        #                                                                               + 2*env.f*self.f0
        #                                                                               - 1j*self.gamma0*(env.f+self.f0))))


        n = np.real(eps_complex)
        alpha = np.imag(eps_complex)

        fieldFFT = FFT(field, env.dt)

        # plt.figure()
        # plt.plot(env.t, P(field), label='P Before')
        # plt.legend()
        # plt.show()
        #
        # plt.figure()
        # plt.plot(env.f, PSD(field, env.dt, env.df), label='PSD Before')
        # plt.legend()
        # plt.show()

        fieldFFT = fieldFFT * np.exp(-alpha * w / self.c0 * self.z0) * np.exp( -1j * n * w / self.c0 * self.z0)
        field = IFFT(fieldFFT, env.dt)

        # plt.figure()
        # plt.plot(env.f, n, label='n')
        # plt.legend()
        # plt.show()
        #
        # plt.figure()
        # plt.plot(env.f, alpha, label='alpha')
        # plt.legend()
        # plt.show()
        #
        # plt.figure()
        # plt.plot(env.f, PSD(fieldFFT, env.dt, env.df), label='PSD')
        # plt.legend()
        # plt.show()

        return field






##%
# %%
class DelayLine(Component):
    """
    """
    _num_instances = count(0)

    def datasheet(self):
        self.type = 'delay line'
        self.disp_name = 'Split-and-Delay Line'
        self.N_PARAMETERS = 4
        self.UPPER = self.N_PARAMETERS * [1]
        self.LOWER = self.N_PARAMETERS * [0]
        self.DTYPE = self.N_PARAMETERS * ['float']
        self.DSCRTVAL = self.N_PARAMETERS * [0.1]
        self.SIGMA = self.N_PARAMETERS * [0.05]
        self.MU = [0.0]
        self.FINETUNE_SKIP = []
        self.splitter = False
        self.AT_NAME = ['Delay {}'.format(j) for j in range(0, self.N_PARAMETERS, 1)]
        return

    def simulate(self, env, field, visualize=False):
        return field