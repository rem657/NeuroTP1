import scipy as sp
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.optimize import fsolve
from scipy.interpolate import interp1d, LinearNDInterpolator
from NeuroneModelDefault import *





class FHNModel(NeuroneModelDefault):
	def __init__(
			self,
			t_init: float = 0.0,
			t_end: float = 500.0,
			t_inter: float = 0.01,
			I_inj: callable = lambda t: 0.0,
			b: float = 0.8,
			a: float = 0.7,
	):
		super().__init__(t_init, t_end, t_inter, I_inj)
		self.b = b
		self.a = a
		self.Jacobian = lambda v: np.array([
			[1 - v**2, -1],
			[1, -self.b]])

	def dVdt(self, V, w, I):
		return V - ((V ** 3) / 3) - w + I

	def dwdt(self, V, w):
		return V + self.a - self.b * w  # *self.t

	def dXdt(self, X: np.ndarray, t: float):
		V, w = X
		dVdt = self.dVdt(V, w, self.I_inj(t))
		dwdt = self.dwdt(V, w)
		return [dVdt, dwdt]

	def compute_model(self, init_cond: list):
		X = super().compute_model(init_cond)
		V = X[:, 0]
		W = X[:, 1]
		return self.time, V, W

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

	# def get_eigenvalues(self, v):
	# 	eigenval, eigenvect = np.linalg.eig(self.Jacobian(v))
	# 	return eigenval, eigenvect

	def get_fixed_point(self, v_min: float, v_max: float, numtick: int):
		intersect_i, intersect_v, intersect_w = self.nullcline_intersect(np.linspace(v_min, stop=v_max, num=numtick))
		intersect_mask = intersect_i >= 0  # & (intersect_i <= i_max)
		return intersect_i[intersect_mask], intersect_v[intersect_mask], intersect_w[intersect_mask]

	def get_eigenvalues_at_fixed(self, V: list):
		list_eigenval, list_eigenvects = super().get_eigenvalues_at_fixed(list(zip(V)))
		return list_eigenval[0], list_eigenval[1]

	@staticmethod
	def fit_fixed_point(I: list, V: list, W: list):
		V_I = interp1d(I, V)
		W_I = interp1d(I, W)
		return V_I, W_I

	def compute_bifurcation_from_model(self, I: list, V: list):
		eigen0, eigen1 = self.get_eigenvalues_at_fixed(V)
		return get_bifurcation_point(I, eigen0, eigen1)

	def display_model_solution(self, init_cond: Union[list, None]):
		if init_cond is None:
			i, v, w = model.get_fixed_point(-2, 4, 1000)
			stablePointVI, stablePointWI = model.fit_fixed_point(i, v, w)
			init_cond = [stablePointVI(self.I_inj(0)), stablePointWI(self.I_inj(0))]
		time, V, W = self.compute_model(init_cond)
		figure = go.Figure()
		figure.add_trace(
			go.Scatter(
				x=time,
				y=V,
				mode='lines'
			)
		)
		figure.show()


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


def integrate_near_bifurcation(
		model: FHNModel,
		figure: go.Figure,
		v_min: float,
		v_max: float,
		numtick: int
):
	di = 0.05
	i, v, w = model.get_fixed_point(v_min, v_max, numtick)
	stablePointVI, stablePointWI = model.fit_fixed_point(i, v, w)
	bifurcation_I, bifurcation_eigen = model.compute_bifurcation_from_model(i, v)
	current_to_integrate = []
	default_visibility = [False for _ in range(len(bifurcation_I)*12)]
	default_visibility[:3] = [True, True, True]
	steps = [
		dict(
			method="restyle",
			args=[
				{'visible': default_visibility}
			],
			label='None',
			value='None',
		)
	]
	for index, bifurcation in enumerate(np.sort(bifurcation_I)):
		for j in range(3):
			current_value = bifurcation + (j-1)*di
			label = f'I = {current_value:.3f}' if j != 1 else f'bifurcation I = {current_value:.3f}'
			current_to_integrate.append(current_value)
			integrate_trajectory3D(figure, v_min, v_max, numtick, model, current_value, visible=False)
			figure.add_trace(
				go.Scatter3d(
					x=[stablePointVI(current_value)],
					y=[current_value],
					z=[stablePointWI(current_value)],
					marker_color='purple',
					visible=False,
				)
			)
			figure_index = 3 + (index * 3 * len(bifurcation_I)) + 2*j
			visibility_current = [_ for _ in default_visibility]
			visibility_current[figure_index:figure_index+2] = [True, True]
			steps.append(
				dict(
					method="restyle",
					label=label,
					value=current_value,
					args=[
						{'visible': visibility_current}
					]
				)
			)
	# raise Exception
	sliders = [dict(
		active=0,
		pad={"t": 50},
		steps=steps,

	)]
	figure.update_layout(
		sliders=sliders
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


def display_eigenvalues_to_I(
		v_min: float,
		v_max: float,
		numtick: int,
		i_max: Union[float, None] = None,
		save: bool = False
):
	bifurcation_marker_size = 5
	model = FHNModel()
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
		figure.write_html('eigenvalue.html')
	figure.show()


if __name__ == '__main__':
	I = lambda t: 0.967
	imax = 10
	vmin = -3.5
	vmax = 3.5
	model = FHNModel(t_end=500, I_inj=I)
	# model.display_model_solution(None)
	display_eigenvalues_to_I(vmin, vmax, 1000, i_max=7, save=False)
	# display3D_phaseplane(imax, vmin, vmax, save=False)
