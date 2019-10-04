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
	x, y = np.arange(0,dim), np.arange(0, dim)
	X, Y = np.meshgrid(x,y)
	IJ = np.maximum(X, Y) * np.minimum(X, Y)

	gate = np.exp(1j * 2 * np.pi * IJ / dim) /np.sqrt(dim)

	return gate


# %%
class QuantumGate():
	"""
	"""

	def __init__(self, dim=2):
		self.dim = dim
		# self.field = np.identity(self.dim, dtype='complex')
		self.field = np.identity(dim * 3, dtype='complex')

		return

	def submatrix(self, field):
		field_sub = field[self.dim*1: self.dim*2, self.dim*1: self.dim*2]
		return field_sub


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

		self.target_gate_dagger = np.conjugate(self.target_gate.T)
		return

	def fitness(self, field):
		"""
		"""
		if field.shape[0] != self.dim:
			field = self.submatrix(field)
		fit = self.fidelity(field),
		return fit

	def probability(self, field):
		return np.real(np.trace(np.matmul(np.conjugate(field.T), field)) / self.dim)

	def fidelity(self, field):
		# fid = np.power(np.trace(sp.linalg.sqrtm(np.matmul(np.matmul(self.target_gate_sqrt, generated_gate), self.target_gate_sqrt))), 2)
		# return np.power(np.abs(np.trace( np.matmul(self.target_gate_dagger, field)))/self.dim, 2) / self.probability(field)

		return np.power(np.abs(np.trace(np.matmul(np.conjugate(field.T), self.target_gate))), 2) / self.probability(field) /self.dim /self.dim