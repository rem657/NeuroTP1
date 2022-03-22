import numpy as np
import scipy as sp
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.optimize import fsolve
from scipy.interpolate import interp1d, LinearNDInterpolator
from src.NeuroneModelDefault import *
import seaborn as sns

plot_layout = dict(
	plot_bgcolor='aliceblue',
	paper_bgcolor="white",
	xaxis=dict(
		showgrid=False,
		zeroline=False,
		title_font={'size': 20},
		tickfont=dict(
			size=20
		)
	),
	yaxis=dict(
		showgrid=False,
		zeroline=False,
		title_font={'size': 20},
		tickfont=dict(
			size=20
		)
	),
	legend=dict(
		font=dict(
			size=19
		)
	)
)


class FHNModel(NeuroneModelDefault):
	def __init__(
			self,
			t_init: float = 0.0,
			t_end: float = 500.0,
			t_inter: float = 0.01,
			# I_inj: callable = lambda t: 0.0,
			b: float = 0.8,
			a: float = 0.7,
	):
		super().__init__(t_init, t_end, t_inter)  # , I_inj)
		self.b = b
		self.a = a
		self.Jacobian = lambda v: np.array([
			[1 - v ** 2, -1],
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

	def compute_model(self, init_cond: list, current_func: callable):
		X = super().compute_model(init_cond, current_func)
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
		intersect_mask = intersect_i >= -0.01  # & (intersect_i <= i_max)
		return intersect_i[intersect_mask], intersect_v[intersect_mask], intersect_w[intersect_mask]

	def get_eigenvalues_at_fixed(self, V: list):
		list_eigenval, list_eigenvects = super().get_eigenvalues_at_fixed(list(zip(V)))
		return list_eigenval[0], list_eigenval[1]

	@staticmethod
	def fit_fixed_point(currents: list, V: list, W: list):
		V_I = interp1d(currents, V)
		W_I = interp1d(currents, W)
		return V_I, W_I

	def compute_bifurcation_from_model(self, currents: list, V: list):
		eigen0, eigen1 = self.get_eigenvalues_at_fixed(V)
		return get_bifurcation_point(currents, eigen0, eigen1)

	def display_model_solution(self, init_cond: Union[list, None], current_func: callable):
		if init_cond is None:
			i, v, w = self.get_fixed_point(-2, 4, 1000)
			stablePointVI, stablePointWI = self.fit_fixed_point(i, v, w)
			init_cond = [stablePointVI(current_func(0)), stablePointWI(current_func(0))]
		time, V, W = self.compute_model(init_cond, current_func)
		figure = go.Figure()
		figure.add_trace(
			go.Scatter(
				x=time,
				y=V,
				mode='lines'
			)
		)
		figure.show()

	def integrate_multiple_currents(self, currents: list, initial_conditions_V, initial_conditions_W):
		all_V, all_W = [], []
		for index, i in enumerate(currents):
			current_func = lambda t: i
			time_integrated, V_integrated, W_integrated = self.compute_model(
				[
					initial_conditions_V[index],
					initial_conditions_W[index]
				],
				current_func
			)
			all_V.append(V_integrated)
			all_W.append(W_integrated)
		return time_integrated, all_V, all_W

	def bifurcation_diagram(self, currents: list, initial_conditions_V, initial_conditions_W):
		time_integrated, all_V, all_W = self.integrate_multiple_currents(currents, initial_conditions_V,
		                                                                 initial_conditions_W)
		min_values = []
		max_values = []
		for index in range(len(currents)):
			v_for_current = all_V[index]
			last_v = v_for_current[int(len(v_for_current) / 2):]
			min_values.append(min(last_v))
			max_values.append(max(last_v))
		return min_values, max_values

	def display_bifurcation_diagram(self, currents: np.ndarray, save=False):
		i, v, w = self.get_fixed_point(-2, 4, 5000)
		fixedPointV_I, fixedPointW_I = self.fit_fixed_point(i, v, w)
		initial_conditions_V, initial_conditions_W = fixedPointV_I(currents), fixedPointW_I(currents)
		min_values, max_values = self.bifurcation_diagram(currents, initial_conditions_V + 0.01,
		                                                  initial_conditions_W + 0.01)
		bifurcation_I, bifurcation_eigen = self.compute_bifurcation_from_model(i, v)
		bifurcation_V = fixedPointV_I(bifurcation_I).tolist()
		figure = go.Figure()
		figure.add_trace(
			go.Scatter(
				name='fixed points',
				x=currents,
				mode='lines',
				y=initial_conditions_V,
				marker_color='crimson',
				line_dash='dot'
			)
		)
		linewidth = 2
		figure.add_trace(
			go.Scatter(
				name='Minimum potential',
				x=currents,
				y=min_values,
				mode='lines',
				marker_color='royalblue',
				line_width=linewidth
			)
		)
		figure.add_trace(
			go.Scatter(
				name='Maximum potential',
				x=currents,
				y=max_values,
				mode='lines',
				marker_color='royalblue',
				line_width=linewidth
			)
		)
		bifurcation_marker_size = 4
		figure.add_trace(
			go.Scatter(
				x=bifurcation_I,
				y=bifurcation_V,
				mode='markers',
				marker_size=bifurcation_marker_size + 1,
				marker_color='black',
				name='bifurcation',
				hoverinfo='skip',
				legendgroup='point',
				showlegend=False
			)
		)
		figure.add_trace(
			go.Scatter(
				x=bifurcation_I,
				y=bifurcation_V,
				mode='markers',
				marker_size=bifurcation_marker_size,
				marker_color='orange',
				name='bifurcation',
				hoverinfo='skip',
				legendgroup='point',
			)
		)
		for i in range(len(bifurcation_V)):
			figure.add_annotation(
				text='',
				x=bifurcation_I[i],
				ax=bifurcation_I[i] + ((-1) ** i) * 0.1,
				axref='x',
				y=bifurcation_V[i],
				ay=bifurcation_V[i] + ((-1) ** i) * 0.1,
				ayref='y',
				showarrow=True,
				arrowwidth=2.5,
				arrowhead=2
			)
		figure.update_layout(
			xaxis=dict(
				title='I [μA/cm²]',
				showgrid=False,
				zeroline=False
			),
			yaxis=dict(
				title='V [mV]',
				showgrid=False,
				zeroline=False
			)
		)
		figure.update_layout(plot_layout)
		if save:
			figure.write_html('figures/Q1/1b_bifurcationFHN.html')


def integrate_trajectory3D(
		model: FHNModel,
		figure: go.Figure,
		initial_conditions: List[tuple],
		I_to_integrate: float,
		colorscale='crest',
		**kwargs,
) -> go.Figure:
	palette = sns.color_palette(colorscale, len(initial_conditions))
	current_func = lambda t: I_to_integrate
	for index, init_cond in enumerate(initial_conditions):
		time, V, W = model.compute_model(init_cond, current_func)
		figure.add_trace(
			go.Scatter3d(
				x=[init_cond[0]],
				y=[I_to_integrate],
				z=[init_cond[1]],
				mode='markers',
				marker_color=palette[index],
				hovertext='initial condition'
			)
		)
		figure.add_trace(
			go.Scatter3d(
				name=f'I = {I_to_integrate:.3f} - ({init_cond[0]:.3f}, {init_cond[1]:.3f})',
				x=V,
				y=[I_to_integrate for _ in V],
				z=W,
				mode='lines',
				line=dict(
					width=5,
					color=f'rgb{palette[index]}'
				),
				**kwargs
			)
		)
	return figure


def _integrate_trajectory(
		model: FHNModel,
		initial_conditions: List[tuple],
		I_to_integrate: float,
		colorscale='crest',
		scatter3D: bool = True,
		**kwargs,
) -> Tuple[list, list]:
	palette = sns.color_palette(colorscale, len(initial_conditions))
	current_func = lambda t: I_to_integrate
	list_init_cond_fig = []
	list_trajectory_fig = []
	for index, init_cond in enumerate(initial_conditions):
		time, V, W = model.compute_model(init_cond, current_func)
		init_cond_dict = dict(
			type='scatter',
			x=[init_cond[0]],
			y=[init_cond[1]],
			mode='markers',
			marker=dict(
				color=f'rgb{palette[index]}',
				size=8
			),
			hovertext='initial condition',
			name='initial condition',
			legendgroup=f'({init_cond[0]}, {init_cond[1]})',
			**kwargs
		)
		trajectory_dict = dict(
			name=f'({init_cond[0]:.3f}, {init_cond[1]:.3f})',
			type='scatter',
			x=V,
			y=W,
			mode='lines',
			line=dict(
				width=2,
				color=f'rgb{palette[index]}'
			),
			legendgroup=f'({init_cond[0]}, {init_cond[1]})',
			**kwargs
		)
		if scatter3D:
			init_cond_dict['type'] = 'scatter3d'
			init_cond_dict['z'] = [init_cond[1]]
			init_cond_dict['y'] = [I_to_integrate]

			trajectory_dict['type'] = 'scatter3d'
			trajectory_dict['z'] = W
			trajectory_dict['y'] = [I_to_integrate for _ in V]
		list_init_cond_fig.append(init_cond_dict)
		list_trajectory_fig.append(trajectory_dict)
	return list_init_cond_fig, list_trajectory_fig


def integrate_trajectory(
		model: FHNModel,
		figure: go.Figure,
		initial_conditions: List[tuple],
		I_to_integrate: float,
		colorscale='crest',
		scatter3D: bool = True,
		**kwargs,
) -> go.Figure:
	list_init_cond_fig, list_trajectory_fig = _integrate_trajectory(model, initial_conditions, I_to_integrate,
	                                                                colorscale, scatter3D, **kwargs)
	figure.add_traces(
		list_init_cond_fig + list_trajectory_fig
	)
	return figure


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


def phaseplane2D(
		figure: go.Figure,
		model: FHNModel,
		I: float,
		V: np.ndarray) -> go.Figure:
	nullV = model.nullcline_V(I, V)
	nullW = model.nullcline_W(I, V)

	figure.add_trace(
		go.Scatter(
			x=V.tolist(), y=nullV.tolist(),
			name='nullcline V',
			showlegend=True,
			mode='lines',
			marker_color='rgb(30, 144, 255)',
			line_width=5
		)
	)
	figure.add_trace(
		go.Scatter(
			x=V.tolist(), y=nullW.tolist(),
			name='nullcline W',
			showlegend=True,
			mode='lines',
			marker_color='rgb(255, 127, 14)',
			line_width=5
		)
	)
	return figure


def streamline_trajectories(
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


def make_trajectories_near_bifurcation(
		model: FHNModel,
		figure: go.Figure,
):
	di = 0.01
	i, v, w = model.get_fixed_point(-2, 2, 500)
	stablePointVI, stablePointWI = model.fit_fixed_point(i, v, w)
	bifurcation_I, bifurcation_eigen = model.compute_bifurcation_from_model(i, v)
	nb_init_cond = 11
	# default_visibility = [False for _ in range(len(bifurcation_I) * nb_init_cond * 3 * 2)]
	# default_visibility[:3] = [True, True, True]
	steps = [
		# dict(
		# 	method="restyle",
		# 	args=[
		# 		{'visible': default_visibility}
		# 	],
		# 	label='None',
		# 	value='None',
		# )
	]
	nullclineVRange = np.linspace(-4, 4, 500)
	for index, bifurcation in enumerate(np.sort(bifurcation_I)):
		for j in range(3):
			current_value = bifurcation + (j - 1) * di
			null_v = model.nullcline_V(current_value, nullclineVRange)
			null_w = model.nullcline_W(current_value, nullclineVRange)
			label = f'I = {current_value:.3f}' if j != 1 else f'bifurcation I = {current_value:.3f}'
			fixedV, fixedW = stablePointVI(current_value), stablePointWI(current_value)
			initial_conditions = [[fixedV + (itera * 1.4 - 2.8), fixedW + (itera * 6 - 12)] for itera in
			                      range(nb_init_cond - 4)] + \
			                     [[fixedV + (((-1) ** itera1) * 3.5), fixedW] for itera1 in range(2)] + \
			                     [[nullclineVRange[indexnull * (-1)], null_w[indexnull * (-1)]] for indexnull in
			                      range(2)] + \
			                     [[fixedV + 0.02, fixedW + 0.1]]
			init_v, init_w = tuple(zip(*initial_conditions))
			sorter = np.argsort(init_w)
			initial_conditions = np.array(initial_conditions)[sorter].tolist()
			if (index == 0) and (j == 0):
				phaseplane2D(figure, model, current_value, nullclineVRange)
				integrate_trajectory(model, figure, initial_conditions, current_value, scatter3D=False)
				figure.add_trace(
					go.Scatter(
						x=[fixedV],
						y=[fixedW],
						mode='markers',
						name='Point fixe',
						marker=dict(
							color='purple',
							size=8
						)
					)
				)
			list_init_cond_fig, list_trajectory_fig = _integrate_trajectory(model,
			                                                                initial_conditions,
			                                                                current_value,
			                                                                scatter3D=False,
			                                                                visible=True)
			all_fig = list_init_cond_fig + list_trajectory_fig
			# figure_index = 3 + (index * 3 * nb_init_cond * 2) + (nb_init_cond * j * 2)
			# visibility_current = [_ for _ in default_visibility]
			# visibility_current[figure_index:figure_index + 2*nb_init_cond] = [True, True]
			steps.append(
				dict(
					method="restyle",
					label=label,
					value=current_value,
					args=[
						{
							'x': [nullclineVRange.tolist(), nullclineVRange.tolist()] + [fig['x'] for fig in
							                                                             all_fig] + [[fixedV]],
							'y': [null_v.tolist(), null_w.tolist()] + [fig['y'] for fig in all_fig] + [[fixedW]],
							'name': ['nullcline V', 'nullcline W'] + [fig['name'] for fig in all_fig] + ['Point fixe']
						}
					]
				)
			)
	sliders = [dict(
		active=0,
		pad={"t": 50},
		steps=steps,

	)]
	figure.update_layout(
		sliders=sliders
	)


def get_bifurcation_point(I: List, eigen0: List, eigen1: List) -> Tuple[list, list]:
	bifurcation_I = []
	for eigenval in [eigen0, eigen1]:
		func = interp1d(I, eigenval)
		amax_eigen = np.argmax(eigenval)
		current_max = I[amax_eigen]
		bifurcation_I += fsolve(func, [current_max - 0.05, current_max + 0.05]).tolist()
	bifurcation_I = list(set(bifurcation_I))
	bifurcation_eigen = [0 for _ in bifurcation_I]
	return bifurcation_I, bifurcation_eigen


def integrate_near_bifurcation(
		model: FHNModel,
		figure: go.Figure,
		v_min: float,
		v_max: float,
		numtick: int
):
	di = 0.01
	i, v, w = model.get_fixed_point(v_min, v_max, numtick)
	stablePointVI, stablePointWI = model.fit_fixed_point(i, v, w)
	bifurcation_I, bifurcation_eigen = model.compute_bifurcation_from_model(i, v)
	current_to_integrate = []
	default_visibility = [False for _ in range(len(bifurcation_I) * 12)]
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
			current_value = bifurcation + (j - 1) * di
			label = f'I = {current_value:.3f}' if j != 1 else f'bifurcation I = {current_value:.3f}'
			current_to_integrate.append(current_value)
			integrate_trajectory3D(model, figure, [], current_value, visible=False)
			figure.add_trace(
				go.Scatter3d(
					x=[stablePointVI(current_value)],
					y=[current_value],
					z=[stablePointWI(current_value)],
					marker_color='purple',
					visible=False,
				)
			)
			figure_index = 3 + (index * 3 * len(bifurcation_I)) + 2 * j
			visibility_current = [_ for _ in default_visibility]
			visibility_current[figure_index:figure_index + 2] = [True, True]
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
	# integrate_near_bifurcation(model, figure, v_min, v_max, numtick)
	figure.update_layout(
		scene=dict(
			xaxis=dict(title='V [mV]'),  # range=[-5, 5]),
			yaxis=dict(title='I injecté'),  # range=[0, 10]),
			zaxis=dict(title='W')  # , range=[-40, 40])
		)
	)
	if save:
		figure.write_html('figures/Q1/1b_PhasePlanFHN.html')
	else:
		figure.show()


def Q1display_eigenvalues_to_I(
		v_min: float,
		v_max: float,
		numtick: int,
		i_max: Union[float, None] = None,
		save: bool = False
):
	bifurcation_marker_size = 13
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
			name='Re{λ<sub>0</sub>}',
			mode='lines',
			line=dict(
				width=8
			)
		)
	)
	figure.add_trace(
		go.Scatter(
			x=i,
			y=eigen1,
			name='Re{λ<sub>1</sub>}',
			mode='lines',
			line=dict(
				width=8
			)
		)
	)
	figure.add_trace(
		go.Scatter(
			x=bifurcation_I,
			y=bifurcation_eigen,
			mode='markers',
			marker_size=bifurcation_marker_size + 3,
			marker_color='black',
			name='highlight',
			hoverinfo='skip',
			legendgroup='point',
			showlegend=False
		)

	)
	figure.add_trace(
		go.Scatter(
			x=bifurcation_I,
			y=bifurcation_eigen,
			mode='markers',
			marker_size=bifurcation_marker_size,
			marker_color='orange',
			name='Bifurcations',
			hovertemplate='Current : %{x:.4f}',
			legendgroup='point'
		)
	)
	tailx = bifurcation_I[0]
	taily = bifurcation_eigen[0] + 0.5
	for index, bifurcation_current in enumerate(bifurcation_I):
		if index == 0:
			figure.add_annotation(
				text="Bifurcations",
				font=dict(size=20),
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
		xaxis=dict(title='I [μA/cm²]'),
		yaxis=dict(title='Re{λ}'),
		legend=dict(
			bordercolor="Black",
			borderwidth=1,
			yanchor='bottom',
			xanchor='left',
			x=0.01,
			y=0.01
		)
	)
	figure.update_layout(plot_layout)
	figure.update_yaxes(
		zeroline=True,
		zerolinecolor='black'
	)
	if save:
		figure.write_html('figures/Q1/1b_eigenvalueFHN.html')
	else:
		figure.show()


def display_trajectories(
		save: bool = True
):
	model = FHNModel()
	figure = go.Figure()
	make_trajectories_near_bifurcation(model, figure)
	figure.update_layout(**plot_layout)
	figure.update_xaxes(
		title_text='V [mV]',
		range=[-4, 4]
	)
	figure.update_yaxes(
		title_text='W [-]',
	)
	if save:
		figure.write_html('figures/Q1/1a_orbitesFHN.html')
	else:
		figure.show()


def bifurcation_diagram():
	model = FHNModel()
	model.display_bifurcation_diagram(np.linspace(0.5, 1.2, num=800), save=True)


if __name__ == '__main__':
	I = lambda t: 0.967
	imax = 10
	vmin = -3.5
	vmax = 3.5
	# bifurcation_diagram()
	# model.display_model_solution(None,I)
	# display_eigenvalues_to_I(vmin, vmax, 1000, i_max=7, save=True)
	# display3D_phaseplane(imax, vmin, vmax, save=False)
	display_trajectories(False)
