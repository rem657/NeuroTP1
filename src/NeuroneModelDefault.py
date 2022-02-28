import numpy as np
from scipy.integrate import solve_ivp, odeint
from typing import List, Union, Tuple


class NeuroneModelDefault:
	def __init__(
			self,
			t_init: float = 0.0,
			t_end: float = 500.0,
			t_inter: float = 0.01,
			I_inj: callable = lambda t: 0.0
	):
		self.I_inj = I_inj
		self.time = np.arange(t_init, t_end, t_inter)  # (t_init, t_end)

	def Jacobian(self, v):
		raise NotImplementedError()

	def dVdt(self, *args):
		raise NotImplemented

	def dXdt(self, *args):
		raise NotImplemented

	def compute_model(self, init_cond: list):
		return odeint(self.dXdt, init_cond, self.time)

	def get_eigenvalues(self, *args) -> Tuple[np.ndarray, np.ndarray]:
		eigenval, eigenvect = np.linalg.eig(self.Jacobian(*args))
		return eigenval, eigenvect

	def get_fixed_point(self, v_min: float, v_max: float, numtick: int):
		raise NotImplementedError()

	def get_eigenvalues_at_fixed(self, params: Union[List, float]) -> tuple:
		list_eigenval = []
		list_eigenvects = []
		for param in params:
			eigenvals, eigenvects = self.get_eigenvalues(*param)
			list_eigenval.append(eigenvals)
			list_eigenvects.append(eigenvects)
		return list(zip(*np.real(list_eigenval))), list(zip(*list_eigenvects))


