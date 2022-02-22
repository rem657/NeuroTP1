import numpy as np
import scipy as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy.integrate import solve_ivp, odeint



class FHNModel:
	def __init__(
			self,
			# t: float = 0.08,
			b: float = 0.8,
			a: float = 0.7,
			t_init: float = 0.0,
			t_end: float = 500.0,
			t_inter: float = 0.01,
			I_inj: callable = lambda t: 0
	):
		self.I_inj = I_inj
		#		self.t = t
		self.b = b
		self.a = a
		self.time = np.arange(t_init, t_end, t_inter)  # (t_init, t_end)

	def dVdt(self, V, w, t):
		return V - ((V ** 3) / 3) - w + self.I_inj(t)

	def dwdt(self, V, w):
		return V + self.a - self.b * w  # *self.t

	def dXdt(self, X: np.ndarray, t: float):
		V, w = X
		dVdt = self.dVdt(V, w, t)
		dwdt = self.dwdt(V, w)
		return [dVdt, dwdt]

	def compute_model(self, v_init: float, w_init: float):
		init_cond = [
			v_init,
			w_init
		             ]
		X: np.ndarray = odeint(self.dXdt, init_cond, self.time)
		V = X[:, 0]
		m = X[:, 1]
		h = X[:, 2]
		n = X[:, 3]
		return self.time, V, m, h, n

	@staticmethod
	def nullcline_V(I: np.ndarray, V: np.ndarray):
		return V - (V ** 3) / 3 + I

	def nullcline_W(self, I: np.ndarray, V: np.ndarray):
		return (V + self.a) / self.b

	def nullcline_V_3D(self, i, v):
		nullcline_V = np.vectorize(self.nullcline_V)
		nullclineV_arr = nullcline_V(i, v)
		return nullclineV_arr

	def nullcline_W_3D(self, i, v):
		nullcline_W = np.vectorize(self.nullcline_W)
		nullclineW_arr = nullcline_W(i, v)
		return nullclineW_arr

	def nullclines_3D(self, i_max: float = 200.0, v_min: float = -100.0, v_max: float = 50.0, numtick=1000):
		i, v = np.mgrid[0:i_max:numtick * 1j, v_min:v_max:numtick * 1j]
		return i, v, self.nullcline_W_3D(i, v), self.nullcline_V_3D(i, v)

	def nullcline_intersect(self, V):  # nullcline_V, nullcline_W, I, V):
		# idx = np.argwhere(np.diff(np.sign(nullcline_V - nullcline_W))).flatten()
		# newI = I.flatten()[idx]
		# newV = V.flatten()[idx]
		# return newI, newV, self.nullcline_W(newI, newV)
		Is = (1 / self.b - 1) * V + (V ** 3) / 3 + self.a / self.b
		Ws = self.nullcline_W(Is, V)
		return Is, V, Ws

	def get_eigenvalues(self, v):
		J = np.array([
			[1 - v ** 2, -1],
			[1, -self.b],
		])
		w, v = np.linalg.eig(J)
		return w, v


# def integrate_system(self, I):


def display3D_phaseplane(i_max: float, v_min: float, v_max: float, I_to_integrate: list = []):
	m = FHNModel()
	numtick = 100
	i, v, nullV, nullW = m.nullclines_3D(i_max, v_min, v_max, numtick)
	colorscale = [
		[0, 'rgb(30, 144, 255)'],
		[1, 'rgb(255, 127, 14)']
	]
	color_V = np.zeros(shape=nullV.shape)
	color_W = np.ones(shape=nullW.shape)
	fig = go.Figure()
	fig.add_trace(
		go.Surface(
			x=v, y=i, z=nullV,
			opacity=0.5,
			name='nullcline V',
			showscale=False,
			cmin=0,
			cmax=1,
			colorscale=colorscale,
			surfacecolor=color_V
		)
	)
	fig.add_trace(
		go.Surface(
			x=v, y=i, z=nullW,
			opacity=0.5,
			name='nullcline W',
			showscale=False,
			cmin=0,
			cmax=1,
			colorscale=colorscale,
			surfacecolor=color_W
		)
	)
	intersect_i, intersect_v, intersect_w = m.nullcline_intersect(np.linspace(v_min, stop=v_max, num=numtick * 10))
	intersect_mask = (intersect_i >= 0) & (intersect_i <= i_max)
	intersect_i, intersect_v, intersect_w = intersect_i[intersect_mask], intersect_v[intersect_mask], intersect_w[
		intersect_mask]
	fig.add_trace(
		go.Scatter3d(
			x=intersect_v.tolist(), y=intersect_i.tolist(), z=intersect_w.tolist(),
			mode='lines',
			line=dict(
				color='purple',
				width=10
			),

		)
	)
	for current in I_to_integrate:
		w = np.linspace(v_min,v_max,)
	fig.update_layout(
		scene=dict(
			xaxis=dict(title='V [mV]'),  # range=[-5, 5]),
			yaxis=dict(title='I injecté'),  # range=[0, 10]),
			zaxis=dict(title='W')  # , range=[-40, 40])
		)
	)
	# fig.write_html('FHNModel.html')
	fig.show()


if __name__ == '__main__':
	I = lambda t: 0.5

	imax = 10
	vmin = -5
	vmax = 5
	display3D_phaseplane(imax, vmin, vmax)
