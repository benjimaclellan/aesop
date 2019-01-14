"""
Definitions of various standard quantum gates, generalized to higher dimensions. All are created as QuTiP objects
"""
import numpy as np
import qutip as qt

def genpauli_x(d):
    basis = np.eye(d).astype('complex')
    mat = np.zeros([d,d]).astype('complex')
    for l in range(d):
        mat += np.outer( basis[(l+1)%d,:], basis[l,:])
    return qt.Qobj(mat)


def genpauli_z(d):
    w = np.exp(2*np.pi*1j/d)
    basis = np.eye(d).astype('complex')
    mat = np.zeros([d,d]).astype('complex')
    for l in range(d):
        mat += np.outer( basis[l,:], basis[l,:]) * np.power(w, l)
    return qt.Qobj(mat)

def genpauli_y(d):
    x = genpauli_x(d)
    z = genpauli_z(d)
    mat = x*z
    return mat

def genhadamard(d):
    x = genpauli_x(d)
    z = genpauli_z(d)
    mat = (x + z)/np.sqrt(2)
    return mat
