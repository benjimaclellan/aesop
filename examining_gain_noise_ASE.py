import autograd.numpy as np
import matplotlib.pyplot as plt

# trying to replicate those sweet graphs from the paper to UNDERSTAND (replication worked)
# TI being clutch and explaining stuff to me: https://www.ti.com/lit/an/slaa652/slaa652.pdf?ts=1596751106433&ref_url=https%253A%252F%252Fwww.google.com%252F 
# RP Photonics: also ace
# Seems useful: https://www.hft.tu-berlin.de/fileadmin/fg154/ONT/Skript/ENG-Ver/EDFA.pdf
# http://notes-application.abcelectronique.com/018/18-27242.pdf


def ratio_to_dB(ratio):
    return 10 * np.log10(ratio)


def dB_to_ratio(dB):
    return 10**(dB / 10)


def W_to_dBM(W):
    return 10 * np.log10(W / (1e-3))


def dBm_to_W(dBm):
    return 1e-3 * 10**(dBm / 10)


alpha = 1.0129
P_max_dBm = 15.4
P_max = dBm_to_W(P_max_dBm)
k1 = 0.0147
k2 = 0.2105

G_0_dB = 42 # guessed from figure
G_0 = dB_to_ratio(G_0_dB)
NF_0 = 4.4 # dB
F_0 = dB_to_ratio(NF_0)


def G(P_in):
    denom = 1 + (G_0 * P_in / P_max)**alpha
    return G_0 / denom


input_signal_power_dBm = np.arange(-40, 5, 1)
input_signal_power = dBm_to_W(input_signal_power_dBm)

G = G(input_signal_power)
G_dB = ratio_to_dB(G)

_, ax = plt.subplots(3, 1)
# ax[0].plot(input_signal_power_dBm, G)
ax[0].plot(input_signal_power_dBm, G_dB)
ax[0].set_xlabel('Input signal power (dBm)')
# ax[0].set_ylabel('Gain (ratio)')
ax[0].set_ylabel('Gain (dB)')

NF = NF_0 + k1 * np.exp(k2 * (G_0_dB - G_dB))
F = dB_to_ratio(NF)
# ax[1].plot(input_signal_power_dBm, F)
ax[1].plot(input_signal_power_dBm, NF)
ax[1].set_xlabel('Input signal power (dBm)')
ax[1].set_ylabel('Noise figure (dB)')
# ax[1].set_ylabel('Noise factor (ratio)')

h = 6.6261e-34
v = 3e8/(1550e-9)
B_0 = 3e8 / (1550e-9)**2 * 60e-9

F_max_db = 5
F_max = dB_to_ratio(5)

# floor_noise_power = 1.38e-23 * 290 * B_0
# floor_noise_power_dBm = W_to_dBM(floor_noise_power)

def P_ASE(F, G):
    return (F * G - 1) * h * v * B_0 # / 2 --> I THINK the division by 2 is just for polarization, but we're ignoring polarization so...
    # return h * v * B_0 * (G * F - 1)
    # return (F - 1) * G * floor_noise_power

P_ase = P_ASE(F, G)
P_ase_dBm = W_to_dBM(P_ase)

ax[2].plot(input_signal_power_dBm, P_ase_dBm) #  P_ase * 1e3)
ax[2].set_xlabel('Input signal power (dBm)')
ax[2].set_ylabel('ASE (dBm)')

plt.show()