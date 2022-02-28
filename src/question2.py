# -C_m dV_m/dt = g_K(V_M - E_K) + g_Na(V_M - E_Na) + g_L(V_M - E_L)
from typing import Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# dn/dt = α_n(V)(1 - n) - β_n(V)n
# dm/dt = α_m(V)(1 - m) - β_m(V)m
# dh/dt = α_h(V)(1 - h) - β_h(V)h
from scipy.integrate import odeint

from src.NeuroneModelDefault import NeuroneModelDefault


class HHModel(NeuroneModelDefault):
	def __init__(
			self,
			I_inj: callable,
			C_m: float = 1.0,
			g_Na: float = 120.0,
			g_K: float = 36.0,
			g_L: float = 0.3,
			E_Na: float = 50.0,
			E_K: float = -77.0,
			E_L: float = -54.387,
			t_init: float = 0.0,
			t_end: float = 500.0,
			t_inter: float = 0.01
	):
		super().__init__(t_init, t_end, t_inter, I_inj)
		self.V_repos = -65.0
		self.time = np.arange(t_init, t_end, t_inter)  # (t_init, t_end)
		self.C_m = C_m  # capacitance, uF/cm^2
		self.g_Na = g_Na  # conductance Sodium, mS/cm^3
		self.g_K = g_K  # conductance Potassium, mS/cm^3
		self.g_L = g_L  # conductance leak, mS/cm^3
		self.E_Na = E_Na  # Potentiel Nernst Sodium, mV
		self.E_K = E_K  # Potentiel Nernst Potassium, mV
		self.E_L = E_L  # Potentiel Nernst leak, mV
		self.I_inj = I_inj  # 10 * (t > 100) - 10 * (t > 200) + 35 * (t > 300) - 35 * (t > 400)

	@staticmethod
	def alpha_n(V: float):
		return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))

	@staticmethod
	def beta_n(V: float):
		return 0.125 * np.exp(-(V + 65) / 80.0)

	@staticmethod
	def alpha_m(V: float):
		return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))

	@staticmethod
	def beta_m(V: float):
		return 4.0 * np.exp(-(V + 65.0) / 18.0)

	@staticmethod
	def alpha_h(V: float):
		return 0.07 * np.exp(-(V + 65.0) / 20.0)

	@staticmethod
	def beta_h(V):
		return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

	def I_Na(self, V: float, m: float, h: float):
		return self.g_Na * m ** 3 * h * (V - self.E_Na)

	def I_K(self, V: float, n: float):
		return self.g_K * n ** 4 * (V - self.E_K)

	def I_L(self, V: float):
		return self.g_L * (V - self.E_L)

	def n_inf(self, V: float):
		return self.alpha_n(V) / (self.alpha_n(V) + self.beta_n(V))

	def m_inf(self, V: float):
		return self.alpha_m(V) / (self.alpha_m(V) + self.beta_m(V))

	def h_inf(self, V: float):
		return self.alpha_h(V) / (self.alpha_h(V) + self.beta_h(V))

	def tau_n(self, V: float):
		return 1 / (self.alpha_n(V) + self.beta_n(V))

	def tau_m(self, V: float):
		return 1 / (self.alpha_m(V) + self.beta_m(V))

	def tau_h(self, V: float):
		return 1 / (self.alpha_h(V) + self.beta_h(V))

	def dXdt(self, X: np.ndarray, t: float):
		V, m, h, n = X

		dVdt = (self.I_inj(t) - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)) / self.C_m
		dmdt = (self.m_inf(V) - m) / self.tau_m(V)
		dhdt = (self.h_inf(V) - h) / self.tau_h(V)
		dndt = (self.n_inf(V) - n) / self.tau_n(V)
		return [dVdt, dmdt, dhdt, dndt]

	def compute_model(self, init_cond: list = None):
		if init_cond is None:
			init_cond = [
				self.V_repos,
				self.m_inf(self.V_repos),
				self.h_inf(self.V_repos),
				self.n_inf(self.V_repos)
			]
		X = np.asarray(odeint(self.dXdt, np.asarray(init_cond), self.time))
		V = X[:, 0]
		m = X[:, 1]
		h = X[:, 2]
		n = X[:, 3]
		return self.time, V, m, h, n

	def Jacobian(self, X):
		return np.array([
			[self.deldV_delVdt(X), self.deldV_delmdt(X), self.deldV_delndt(X), self.deldV_delhdt(X)],
			[self.deldm_delVdt(X), self.deldm_delmdt(X), 0.0, 0.0],
			[self.deldh_delVdt(X), 0.0, self.deldh_delhdt(X), 0.0],
			[self.deldn_delVdt(X), 0.0, 0.0, self.deldn_delndt(X)],
		])

	def deldV_delVdt(self, X):
		V, m, h, n = X
		return (-m*m*m*h*self.g_Na - n*n*n*n*self.g_K - self.g_L) / self.C_m

	def deldV_delmdt(self, X):
		V, m, h, n = X
		return (3*m*m*h*self.g_Na*(self.E_Na - V)) / self.C_m

	def deldV_delndt(self, X):
		V, m, h, n = X
		return (4*n*n*n*self.g_K*(self.E_K - V)) / self.C_m

	def deldV_delhdt(self, X):
		V, m, h, n = X
		return (m*m*m*self.g_Na*(self.E_Na - V)) / self.C_m

	def deldm_delVdt(self, X):
		V, m, h, n = X
		return (1 - m) * self.delalpha_m_delV(V) - m*self.delbeta_m_delV(V)

	def deldm_delmdt(self, X):
		V, m, h, n = X
		return -self.alpha_m(V) - self.beta_m(V)

	def deldn_delVdt(self, X):
		V, m, h, n = X
		return (1 - n) * self.delalpha_n_delV(V) - n*self.delbeta_n_delV(V)

	def deldn_delndt(self, X):
		V, m, h, n = X
		return -self.alpha_n(V) - self.beta_n(V)

	def deldh_delVdt(self, X):
		V, m, h, n = X
		return (1 - h) * self.delalpha_h_delV(V) - h*self.delbeta_h_delV(V)

	def deldh_delhdt(self, X):
		V, m, h, n = X
		return -self.alpha_h(V) - self.beta_h(V)

	def delalpha_m_delV(self, V):
		tau = 10
		binom = V + 40
		c = 0.1
		exp_term = np.exp(-binom/tau)
		return c*((1-exp_term) - (binom/tau)*exp_term) / ((1 - exp_term)**2)

	def delbeta_m_delV(self, V):
		return (-4/18)*np.exp(-(V+65)/18)

	def delalpha_n_delV(self, V):
		tau = 10
		binom = V + 55
		c = 0.01
		exp_term = np.exp(-binom / tau)
		return c * ((1 - exp_term) - (binom / tau) * exp_term) / ((1 - exp_term) ** 2)

	def delbeta_n_delV(self, V):
		return (-0.125/80)*np.exp(-(V+65)/80)

	def delalpha_h_delV(self, V):
		return (-0.07/20)*np.exp(-(V+65)/20)

	def delbeta_h_delV(self, V):
		tau = 10
		binom = V + 35
		exp_term = np.exp(-binom / tau)
		return exp_term / (tau * ((1 + exp_term) ** 2))

	def X_nullcline(self, V, t: float):
		gamma_m = self.alpha_m(V) / self.beta_m(V)
		gamma_h = self.alpha_h(V) / self.beta_h(V)
		gamma_n = self.alpha_n(V) / self.beta_n(V)

		m = gamma_m / (1 + gamma_m)
		h = gamma_h / (1 + gamma_h)
		n = gamma_n / (1 + gamma_n)
		V_nullcline = self.I_inj(t) - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)
		return [V_nullcline, m, h, n]


def display_HHModel(I_inj: callable, t_init: float, t_end: float, t_inter: float):
	model = HHModel(I_inj, t_init=t_init, t_end=t_end, t_inter=t_inter)
	I_inj = np.vectorize(I_inj)
	t, V, m, h, n = model.compute_model()
	n_row = 4
	fig = make_subplots(
		rows=n_row,
		cols=1,
		shared_xaxes=True,
		vertical_spacing=0.003,
		row_heights=[0.3, 0.3, 0.3, 0.1]
	)
	fig.add_trace(
		go.Scatter(
			x=t,
			y=I_inj(t),
			mode='lines',
			name='I_inj(t)'
		),
		row=n_row, col=1
	)
	fig.add_trace(
		go.Scatter(
			x=t,
			y=V,
			mode='lines',
			name='V(t)'
		),
		row=1, col=1
	)
	fig.add_trace(
		go.Scatter(
			x=t,
			y=n + h,
			mode='lines',
			name='n + h'
		),
		row=2, col=1
	)
	fig.add_trace(
		go.Scatter(
			x=t,
			y=m - model.m_inf(V),
			mode='lines',
			name='m - m∞'
		),
		row=3, col=1
	)
	fig.update_layout(
		yaxis4=dict(title='Courant appliqué [μA]', showgrid=False, zeroline=False),
		yaxis3=dict(title='m - m∞ [-]', showgrid=False, zeroline=False),
		yaxis2=dict(title='n + h [-]', showgrid=False, zeroline=False),
		yaxis1=dict(title='Potentiel de la membrane [mV]', showgrid=False, zeroline=False),
		xaxis4=dict(title='Temps [msec]', showgrid=True, zeroline=False, linecolor='black', linewidth=2, ticks=''),
		xaxis3=dict(showgrid=True, zeroline=False, linecolor='black', linewidth=2, ticks=''),
		xaxis2=dict(showgrid=True, zeroline=False, linecolor='black', linewidth=2, ticks=''),
		xaxis1=dict(showgrid=True, zeroline=False, linecolor='black', linewidth=2, ticks='')
	)
	fig.show()


def I_ramp(t: float):
	return (0.1 * t) * (t < 1700)


def I_stairs(current_values: list):
	step_len = 75

	def func(t):
		index = int(t // step_len)
		return current_values[index] if index < len(current_values) else 0

	return func


def I_steps(current_values: list):
	step_len = 75
	inactive_len = 25

	def func(t):
		active = (t % (inactive_len + step_len)) > inactive_len
		index = int(t // (inactive_len + step_len))
		if active:
			return current_values[index] if index < len(current_values) else 0
		else:
			return 0

	return func


def display_eigenvalues_to_I(
		model: NeuroneModelDefault,
		v_min: float,
		v_max: float,
		numtick: int,
		i_max: Union[float, None] = None,
		save: bool = False,
		save_name: str = "eigenvalues.html"
):
	bifurcation_marker_size = 5
	i, v, w = model.get_fixed_point(v_min, v_max, numtick)
	eigen0, eigen1 = model.get_eigenvalues_at_fixed(v)
	bifurcation_I, bifurcation_eigen = get_bifurcation_point(i, eigen0, eigen1)
	if i_max is not None:
		i, eigen0, eigen1 = np.array(i), np.array(eigen0), np.array(eigen1)
		mask = i <= i_max
		i, eigen0, eigen1 = i[mask].tolist(), eigen0[mask].tolist(), eigen1[mask].tolist()
	figure = go.Figure()
	figure.add_trace(
		go.Scatter(
			x=i,
			y=eigen0,
			name='real part eigenvalue 0',
			mode='lines'
		)
	)
	figure.add_trace(
		go.Scatter(
			x=i,
			y=eigen1,
			name='real part eigenvalue 1',
			mode='lines'
		)
	)
	figure.add_trace(
		go.Scatter(
			x=bifurcation_I,
			y=bifurcation_eigen,
			mode='markers',
			marker_size=bifurcation_marker_size + 1,
			marker_color='black',
			name='highlight',
			hoverinfo='skip'
		)

	)
	figure.add_trace(
		go.Scatter(
			x=bifurcation_I,
			y=bifurcation_eigen,
			mode='markers',
			marker_size=bifurcation_marker_size,
			marker_color='orange',
			name='bifurcation',
			hovertemplate='Current : %{x:.4f}'
		)
	)
	for index, bifurcation_current in enumerate(bifurcation_I):
		if index == 0:
			figure.add_annotation(
				text="Bifurcations",
				x=bifurcation_current,
				y=bifurcation_eigen[index],
				showarrow=True,
				arrowwidth=1.5,
				arrowhead=1,
				ax=bifurcation_current,
				ay=bifurcation_eigen[index] + 0.5,
				ayref='y',
				axref='x'

			)
			tailx = bifurcation_current
			taily = bifurcation_eigen[index] + 0.5
		else:
			figure.add_annotation(
				x=bifurcation_current,
				y=bifurcation_eigen[index],
				showarrow=True,
				arrowwidth=1.5,
				arrowhead=1,
				ax=tailx,
				ay=taily,
				ayref='y',
				axref='x'

			)
	figure.update_layout(
		xaxis=dict(title='I'),
		yaxis=dict(title='Eigenvalue')
	)
	if save:
		figure.write_html(save_name)
	figure.show()


if __name__ == '__main__':
	# I = lambda t: 35 * (t > 100) - 35 * (t > 200) + 150 * (t > 300) - 150 * (t > 400)
	# I = I_stairs(list(range(0, 110, 5)))
	I = I_steps(list(range(0, 150, 10)))
	display_HHModel(I, 0, 1800, 0.01)
