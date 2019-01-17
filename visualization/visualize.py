import matplotlib.pyplot as plt


"""
Plots the temporal and spectral power for a given individual
"""
def plot_individual(ind, fitness, env, sim):    
    fig, ax = plt.subplots(2, 1, figsize=(8, 10), dpi=80)
    
    ax[0].set_xlabel('Time (ps)')
    ax[0].set_ylabel('Power [arb]')
    ax[1].set_xlabel('Frequency (THz)')
    ax[1].set_ylabel('PSD [arb]')
    
    ax[0].plot(env.t/1e-12, env.P(env.At0), label='initial')
    ax[1].plot(env.f/1e12, env.PSD(env.Af0, env.df))

    
    ax[0].plot(env.t/1e-12, env.P(env.At), label='final')
    ax[1].plot(env.f/1e12, env.PSD(env.Af, env.df))

    ax[0].set_title('Fitness {}'.format(fitness))
    ax[0].legend()
    
    return fig, ax