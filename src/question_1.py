import numpy as np
import scipy as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy.integrate import solve_ivp, odeint
from typing import List, Union, Tuple
from scipy.optimize import fsolve
from scipy.interpolate import interp1d, LinearNDInterpolator


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

	def dVdt(self, V, w, I):
		return V - ((V ** 3) / 3) - w + I

	def dwdt(self, V, w):
		return V + self.a - self.b * w  # *self.t

	def dXdt(self, X: np.ndarray, t: float):
		V, w = X
		dVdt = self.dVdt(V, w, self.I_inj(t))
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
	def nullcline_V(I: Union[np.ndarray, float], V: Union[np.ndarray, float]):
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
		eigenval, eigenvect = np.linalg.eig(J)
		return eigenval, eigenvect

	def get_eigenvalues_at_fixed(self, V: List) -> tuple:
		list_eigenval0 = []
		list_eigenval1 = []
		for potential in V:
			eigenvals, eigenvects = self.get_eigenvalues(potential)
			list_eigenval0.append(np.real(eigenvals[0]))
			list_eigenval1.append(np.real(eigenvals[1]))
		return list_eigenval0, list_eigenval1

	def get_fixed_point(self, v_min: float, v_max: float, numtick: int):
		intersect_i, intersect_v, intersect_w = self.nullcline_intersect(np.linspace(v_min, stop=v_max, num=numtick))
		intersect_mask = intersect_i >= 0  # & (intersect_i <= i_max)
		return intersect_i[intersect_mask], intersect_v[intersect_mask], intersect_w[intersect_mask]


def phaseplane3D(
		figure: go.Figure,
		model: FHNModel,
		numtick: int,
		i_max: float,
		v_min: float,
		v_max: float) -> go.Figure:
	i, v, nullW, nullV = model.nullclines_3D(i_max, v_min, v_max, numtick)
	colorscale = [
		[0, 'rgb(30, 144, 255)'],
		[1, 'rgb(255, 127, 14)']
	]
	color_V = np.zeros(shape=nullV.shape)
	color_W = np.ones(shape=nullW.shape)
	figure.add_trace(
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
	figure.add_trace(
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
	return figure


def nullclineintersect3D(
		figure: go.Figure,
		v_min: float,
		v_max: float,
		i_max: float,
		numtick: int,
		model: FHNModel) -> go.Figure:
	intersect_i, intersect_v, intersect_w = model.nullcline_intersect(np.linspace(v_min, stop=v_max, num=numtick))
	intersect_mask = (intersect_i >= 0) & (intersect_i <= i_max)
	intersect_i, intersect_v, intersect_w = intersect_i[intersect_mask], intersect_v[intersect_mask], intersect_w[
		intersect_mask]
	figure.add_trace(
		go.Scatter3d(
			x=intersect_v.tolist(), y=intersect_i.tolist(), z=intersect_w.tolist(),
			mode='lines',
			line=dict(
				color='purple',
				width=10
			),

		)
	)
	return figure


def integrate_trajectory3D(
		figure: go.Figure,
		v_min: float,
		v_max: float,
		numtick: int,
		model: FHNModel,
		I_to_integrate: float,
		**kwargs,
) -> go.Figure:
	w_min, w_max = np.sort([model.nullcline_V(I_to_integrate, v_min), model.nullcline_V(I_to_integrate, v_max)])
	v, w = np.linspace(v_min, v_max, numtick), np.linspace(w_min, w_max, numtick)
	V, W = np.meshgrid(v, w)
	dvdt, dwdt = model.dVdt(V, W, I_to_integrate), model.dwdt(V, W)
	streamline_fig = ff.create_streamline(v, w, dvdt, dwdt, density=2, arrow_scale=0.6, angle=np.pi / 7)
	# streamline_fig.show()
	streamline = streamline_fig.data[0]
	x, y = streamline['x'], streamline['y']
	figure.add_trace(
		go.Scatter3d(
			name=f'I = {I_to_integrate}',
			x=x,
			y=[I_to_integrate for _ in x],
			z=y,
			mode='lines',
			line=dict(
				width=2,
				color='crimson'
			),
			**kwargs
		)
	)
	return figure


def get_bifurcation_point(I: List, eigen0: List, eigen1: List) -> Tuple[list, list]:
	for eigenval in [eigen0, eigen1]:
		func = interp1d(I, eigenval)
		amax_eigen = np.argmax(eigenval)
		current_max = I[amax_eigen]
		bifurcation_I = fsolve(func, [current_max - 0.05, current_max + 0.05]).tolist()
	bifurcation_eigen = [0 for _ in bifurcation_I]
	return bifurcation_I, bifurcation_eigen


def compute_bifurcation_from_model(model: FHNModel, v_min: float, v_max: float, numtick: int):
	i, v, w = model.get_fixed_point(v_min, v_max, numtick)
	V_I = interp1d(i, v)
	W_I = interp1d(i, w)
	eigen0, eigen1 = model.get_eigenvalues_at_fixed(v)
	return *get_bifurcation_point(i, eigen0, eigen1), V_I, W_I


def integrate_near_bifurcation(
		model: FHNModel,
		figure: go.Figure,
		v_min: float,
		v_max: float,
		numtick: int
):
	di = 0.05
	bifurcation_I, bifurcation_eigen, stablePointVI, stablePointWI = compute_bifurcation_from_model(model, v_min, v_max, numtick)
	labels = []
	current_to_integrate = []
	for bifurcation in np.sort(bifurcation_I):
		labels.append(f'I = {bifurcation - di:.3f}')
		current_to_integrate.append(bifurcation - di)
		integrate_trajectory3D(figure, v_min, v_max, numtick, model, bifurcation - di, visible=False)
		figure.add_trace(
			go.Scatter3d(
				x=stablePointVI(bifurcation - di),
				y=bifurcation - di,
				z=stablePointWI(bifurcation - di),
				marker_color='purple',
				visible=False,
			)
		)

		labels.append(f'bifurcation I = {bifurcation:.3f}')
		current_to_integrate.append(bifurcation)
		integrate_trajectory3D(figure, v_min, v_max, numtick, model, bifurcation, visible=False)
		figure.add_trace(
			go.Scatter3d(
				x=stablePointVI(bifurcation),
				y=bifurcation,
				z=stablePointWI(bifurcation),
				marker_color='purple',
				visible=False,
			)
		)

		labels.append(f'I = {bifurcation + di:.3f}')
		current_to_integrate.append(bifurcation + di)
		integrate_trajectory3D(figure, v_min, v_max, numtick, model, bifurcation + di, visible=False)
		figure.add_trace(
			go.Scatter3d(
				x=stablePointVI(bifurcation + di),
				y=bifurcation + di,
				z=stablePointWI(bifurcation + di),
				marker_color='purple',
				visible=False,
			)
		)
	figure.update_layout(
		updatemenu=None #todo
	)

# Function du devoir
def display3D_phaseplane(
		i_max: float,
		v_min: float,
		v_max: float,
		save: bool = True,
		numtick: int = 100
):
	model = FHNModel()
	figure = go.Figure()
	phaseplane3D(figure, model, numtick, i_max, v_min, v_max)
	nullclineintersect3D(figure, v_min, v_max, i_max, numtick * 10, model)
	integrate_near_bifurcation(model, figure, v_min, v_max, numtick)
	figure.update_layout(
		scene=dict(
			xaxis=dict(title='V [mV]'),  # range=[-5, 5]),
			yaxis=dict(title='I injecté'),  # range=[0, 10]),
			zaxis=dict(title='W')  # , range=[-40, 40])
		)
	)
	if save:
		figure.write_html('FHNModel.html')
	figure.show()


def display_eigenvalues_to_I(v_min: float, v_max: float, numtick: int, save: bool = False):
	bifurcation_marker_size = 5
	model = FHNModel()
	i, v, w = model.get_fixed_point(v_min, v_max, numtick)
	eigen0, eigen1 = model.get_eigenvalues_at_fixed(v)
	bifurcation_I, bifurcation_eigen = get_bifurcation_point(i, eigen0, eigen1)
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
		figure.write_html('eigenvalue.html')
	figure.show()


if __name__ == '__main__':
	I = lambda t: 0.5
	imax = 10
	vmin = -5
	vmax = 3
	# display_eigenvalues_to_I(vmin, vmax, 1000, save=True)
	display3D_phaseplane(imax, vmin, vmax, save=False)
