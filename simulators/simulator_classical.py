import numpy as np
from numpy import pi
from classes import Simulator
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import peakutils
import math 

"""
Simulates an experiment given an input and experimental setup.
Each simulation script should be for a given use case (classical pulse propagation, quantum, etc), so that the GA optimization is entirely modular and can be used with any simulation environment

This is the environment which simulates a classical pulse train through a given experiment

"""
class ClassicalSimulator(Simulator):
    
    def __init__(self, fitness_function, **kwargs):
        self.FitnessFunctions = ClassicalFitnessFunctions()
        try:
            self.FitnessFunctions.func = getattr(self.FitnessFunctions, fitness_function)
        except:
            raise ValueError('This is not a valid fitness function!')
        for key, value in kwargs.items():
            setattr(self.FitnessFunctions, key, value)
        return 

    def fitness(self, env):
        fitness = self.FitnessFunctions.func(env)
        return fitness

    def simulate_experiment(self, individual, experiment, env, verbose=False):
        env.reset()
        
        if env.add_noise:
            noise = env.CreateNoise()
            env.AddNoise(noise)
                
        i = 0
        for component in experiment:
            component_values = individual[i:i+1][0]
#            component_values = individual[i:i+component.N_PARAMETERS]
#            i += component.N_PARAMETERS
            i += 1
                        
            if component.component_type == 'fiber':
                self.fiber_effect_propagation(component_values, env, component)
                if verbose: print('Applying a fiber, {}, with values {}'.format(component.name, component_values))
                continue
            
            if component.component_type == 'phasemodulator':
                self.phasemodulator_effect_propagation(component_values, env, component)
                if verbose: print('Applying a phase modulator, {}, with values {}'.format(component.name, component_values))
                continue
    
            if component.component_type == 'waveshaper':
                self.waveshaper_effect_propagation(component_values, env, component)
                if verbose: print('Applying a waveshaper, {}, with values {}'.format(component.name, component_values))
                continue  
            
            if component.component_type == 'awg':
                self.awg_effect_propagation(component_values, env, component)
                if verbose: print('Applying an AWG, {}, with values {}'.format(component.name, component_values))
                continue  
            
            else:
                raise ValueError('This component type does not exist!')
            return


    def fiber_effect_propagation(self, values, env, component):
        """
        Simulates the effect of a fiber on the pulse
        """        
        fiber_len = values[0]   
        D = np.zeros(env.f.shape).astype('complex')
        for n in range(0, len(component.beta)):    
            D += component.beta[n] * np.power(2*pi*env.f, n+2) / np.math.factorial(n+2)
        D = -1j * D
        
        env.Af = np.exp(fiber_len * D) * env.Af
        env.At = env.IFFT( env.Af, env.dt )
        return
    
    
    def phasemodulator_effect_propagation(self, values, env, component):
        """
        Simulates a phase modulator's effect on a pulse
        """
        M = values[0]       # amplitude []
        NU = values[1]   # frequency [Hz]
        PHI = values[2]     # phase offset [rad]
        dPhase = (M/2)*np.cos(2*pi* NU * env.t + PHI) + M/2
        
        ## Apply phase modulator in time-domain, and update frequency
        env.At = env.At * np.exp(1j * dPhase)                
        env.Af = env.FFT(env.At, env.dt)
        return
    
    
    def waveshaper_effect_propagation(self, values, env, component):
        """
        Simulates a waveshaper's effect on the spectrum
        """
        n_slices = component.N_SLICES
        
        phasevalues = values[0:n_slices]
        ampvalues = np.sqrt(values[n_slices:])
        
        
        mask = np.floor(env.f / component.slicewidth)
        
        phasemask = np.zeros(env.f.shape, dtype='float')
        ampmask = np.zeros(env.f.shape, dtype='float')
        
        for i_slice in range(n_slices):
            tmp = (mask == i_slice-np.floor(n_slices/2) )            
            phasemask[tmp] = phasevalues[i_slice]
            ampmask[tmp] = ampvalues[i_slice]
        
        env.Af = ampmask * env.Af * np.exp(1j * phasemask)
        env.At = env.IFFT(env.Af, env.dt)

        return
        
    
    def awg_effect_propagation(self, values, env, component):       
        phase = self.make_awg_phase(values, env, component)
        
        env.At = env.At * np.exp(1j * phase)
        env.Af = env.FFT(env.At, env.dt)
        return 
    
    def make_awg_phase(self, values, env, component):
        
#        holdval = values[0]
#        phasevalues = values[1:]
        
        nlevels = values[0]
        phasevalues = values[1:]
 
#        nlevels = len(phasevalues)
        timeblock = np.round(1/env.dt/env.f_rep).astype('int')
#        timeblock = np.round(1/env.f_rep/holdval/env.dt).astype('int')
        
        tmp = np.ones(timeblock)
        oneperiod = np.array([]).astype('float')
        
        for i in range(0,nlevels):
            oneperiod = np.concatenate((oneperiod, tmp*phasevalues[i]))            
        phasetmp = np.tile(oneperiod, np.ceil(len(env.t)/len(oneperiod)).astype('int') )

        shift1 = timeblock//2
        
        phasetmp = phasetmp[shift1:]
        phase = phasetmp[0:len(env.t)] 
        return phase
# ********************************************

class ClassicalFitnessFunctions():
    def __init__(self):
        return
    def MatchInputPulse(self, env):
        P = env.P(env.At)
#        P0 = env.P(env.At0)
        fitness1 = np.exp( -np.sum( np.power(P - env.P(env.At0), 2)))
#        fitness1 = 2-np.abs( np.sum(P0 - P)/np.sum(P0) )
#        fitness1 = np.abs(np.sum( np.correlate(P, P, mode='same') ))
        fitness2 = np.max(P)
        return (fitness1, fitness2)
    
    def FrequencyShift(self, env):
        
        peak = env.f[np.argmax(env.PSD(env.Af, env.df))]
        fitness = np.exp( np.power((self.shift-peak)/np.abs(self.shift) ,2) * (-2 + np.sign(peak)*np.sign(self.shift)) )
        
        return fitness
    
    def TemporalTalbot(self, env):
        fitness2 = np.max(env.P(env.At))
#        fitness2 = np.exp(-(1-np.max(env.P(env.At))/env.peakP))
            
        peakinds = peakutils.indexes(env.RFSpectrum(env.At, env.dt))
        peakf_dist = np.mean( env.df * np.diff(peakinds) )
        peakf_distTarget = env.f_rep / (self.p/self.q)
        
        if len(peakinds) <= 1:
            fitness1 = -99999999
        else: 
            fitness1 = - np.abs( peakf_distTarget - peakf_dist)
#            fitness1 = np.exp( - np.abs( peakf_distTarget - peakf_dist)/peakf_distTarget)        
        return (fitness1, fitness2)
    
    
    def SpectralTalbot(self, env):
        PSD = env.PSD(env.Af,env.df)
#            fitness2 = np.exp(-(1-np.max(PSD)/np.max(env.PSD(env.Af0, env.df))))
        peakinds = peakutils.indexes(PSD)
        peakf_dist = np.min( env.df * np.diff(peakinds) )
#            peakf_dist = np.median( env.df * np.diff(peakinds) )
#            peakf_distTarget = env.f_rep / (self.p/self.q)
        peakf_distTarget = env.f_rep * (self.p/self.q)

        if len(peakinds) <= 1:
            fitness1 = 0
            fitness2 = -10
        else: 
            fitness1 = np.exp( - np.abs( peakf_distTarget - peakf_dist)/peakf_distTarget)
            fitness2 = np.exp(-np.abs(np.mean(np.abs(np.diff(PSD[peakinds])))/np.mean(PSD[peakinds])))
        return (fitness1, fitness2)
    
#    def SubIntegerImage(self, env):  
#        
#        if self.domain == 'temporal':
#            fitness2 = np.exp(-(1-np.max(env.P(env.At))/env.peakP))
#            
#            peakinds = peakutils.indexes(env.RFSpectrum(env.At, env.dt))
#            
#            peakf_dist = np.mean( env.df * np.diff(peakinds) )
#            peakf_distTarget = env.f_rep / (self.p/self.q)
#            
#            if len(peakinds) <= 1:
#                fitness1 = 0
#            else: 
#                fitness1 = np.exp( - np.abs( peakf_distTarget - peakf_dist)/peakf_distTarget)
#
#
#        elif self.domain == 'spectral':
#            PSD = env.PSD(env.Af,env.df)
##            fitness2 = np.exp(-(1-np.max(PSD)/np.max(env.PSD(env.Af0, env.df))))
#            
#            peakinds = peakutils.indexes(PSD)
#        
#            peakf_dist = np.min( env.df * np.diff(peakinds) )
##            peakf_dist = np.median( env.df * np.diff(peakinds) )
##            peakf_distTarget = env.f_rep / (self.p/self.q)
#            peakf_distTarget = env.f_rep * (self.p/self.q)
#
#            if len(peakinds) <= 1:
#                fitness1 = 0
#                fitness2 = -10
#            else: 
#                fitness1 = np.exp( - np.abs( peakf_distTarget - peakf_dist)/peakf_distTarget)
#                
#                fitness2 = np.exp(-np.abs(np.mean(np.abs(np.diff(PSD[peakinds])))/np.mean(PSD[peakinds])))
#            
#        else:
#            raise ValueError('Not a valid domain to optimize in')
#
#
#        return (fitness1, fitness2)
    
    def SingleFrequencyMax(self, env):
        fitness = np.zeros(len(self.centralfreq))
        for i in range(len(self.centralfreq)):
            inds = (env.f >= self.centralfreq[i]-self.width[i]) & (env.f <= self.centralfreq[i]+self.width[i])
            fitness_i = np.sum( env.PSD(env.Af[inds], env.df) )
#            fitness_i = np.max( env.PSD(env.Af[inds], env.df) )
            fitness[i] = fitness_i
        return (fitness)