import autograd.numpy as np
import scipy as sp


def Xgate(dim):
	zeroket = np.zeros([dim, 1]).astype('complex')
	X = np.zeros([dim, dim]).astype('complex')

	for ell in range(0, dim):
		ket, bra = np.copy(zeroket), np.copy(zeroket.T)
		ket[((ell + 1) % dim), 0] = 1
		bra[0, ell] = 1
		X += np.dot(ket, bra)
	return X


def Zgate(dim):
	Z = np.diag(np.exp(2 * np.pi * 1j / dim * np.arange(0, dim)))
	return Z


def Ygate(dim):
	return 1j * np.matmul(Xgate(dim), Zgate(dim))


def Hgate(dim):
	return (Xgate(dim) + Zgate(dim)) / 2


# %%
class QuantumGate():
	"""
	"""

	def __init__(self, dim=2):
		self.dim = dim
		self.mat = np.identity(self.dim, dtype='complex')
		return

	def init_fitness(self, gate):
		if gate == 'X':
			self.target_gate = Xgate(self.dim)
		elif gate == 'Y':
			self.target_gate = Ygate(self.dim)
		elif gate == 'Z':
			self.target_gate = Zgate(self.dim)
		elif gate == 'H':
			self.target_gate = Hgate(self.dim)
		else:
			print('Not a defined gate')

		print(self.target_gate)
		self.target_gate_dagger = np.conjugate(self.target_gate.T)
		# self.target_gate_sqrt = sp.linalg.sqrtm(self.target_gate)
		return

	def fitness(self, generated_gate):
		"""
		"""
		return self.fidelity(generated_gate),

	def probability(self, mat):
		return np.trace(np.matmul(np.conjugate(mat.T), mat)) / self.dim

	def fidelity(self, mat):
		# fid = np.power(np.trace(sp.linalg.sqrtm(np.matmul(np.matmul(self.target_gate_sqrt, generated_gate), self.target_gate_sqrt))), 2)
		return np.power(np.abs(np.trace( np.matmul(self.target_gate_dagger, mat)))/self.dim, 2) / self.probability(mat)