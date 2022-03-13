import os

import numpy as np
import tqdm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from pythonbasictools.multiprocessing import apply_func_multiprocess
from src.ifunc import IConst, IFunc, ISin, ISteps
import itertools


class CoupleHH:
	def __init__(
			self,
			C_m: float = 1.0,
			g_Na: float = 120.0,
			g_K: float = 36.0,
			g_L: float = 0.3,
			E_Na: float = 50.0,
			E_K: float = -77.0,
			E_L: float = -54.387,
			E_syn: np.ndarray = np.array([0, -80]),
			tau_syn: np.ndarray = np.array([20, 20]),
			weights: np.ndarray = np.array([0.0, 0.0]),
	):
		self.V_reset = -65.0
		self.C_m = C_m  # capacitance, uF/cm^2
		self.g_Na = g_Na  # conductance Sodium, mS/cm^3
		self.g_K = g_K  # conductance Potassium, mS/cm^3
		self.g_L = g_L  # conductance leak, mS/cm^3
		self.E_Na = E_Na  # Potentiel Nernst Sodium, mV
		self.E_K = E_K  # Potentiel Nernst Potassium, mV
		self.E_L = E_L  # Potentiel Nernst leak, mV
		self.E_syn = E_syn
		self.tau_syn = tau_syn
		self.weights = weights

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

	def run(
			self,
			T: int = 80,
			I_in_func: IFunc = IConst(0.0),
			start_with_spike: bool = False,
			threshold=None,
			**kwargs
	):
		dt = kwargs.get('dt', 1e-3)
		time_steps = int(T / dt)
		input_weights = kwargs.get("input_weights", np.array([1.0, *np.zeros(len(self.weights)-1)]))
		I_Na = np.zeros((time_steps, *self.weights.shape), dtype=float)
		I_K = np.zeros((time_steps, *self.weights.shape), dtype=float)
		I_L = np.zeros((time_steps, *self.weights.shape), dtype=float)
		I_in = np.zeros((time_steps, *self.weights.shape), dtype=float)
		I_syn = np.zeros((time_steps, *self.weights.shape), dtype=float)
		V = np.zeros((time_steps, *self.weights.shape), dtype=float)
		V[0] = self.V_reset * np.ones_like(self.weights)
		m = self.m_inf(V[0])
		n = self.n_inf(V[0])
		h = self.h_inf(V[0])
		spikes = np.zeros((time_steps, *self.weights.shape), dtype=int)
		if start_with_spike:
			spikes[0] = np.ones_like(self.weights, dtype=int)
		dt_spikes = (self.tau_syn/dt) * (1-spikes[0])
		g_syn = np.zeros((time_steps, *self.weights.shape), dtype=float)
		g_syn[0] = np.asarray(self.weights) * spikes[0]
		if threshold is None:
			threshold = 0.0
			# print(f"{threshold = }")

		t_range = range(1, time_steps)
		if kwargs.get('verbose', False):
			t_range = tqdm.tqdm(t_range)

		for t in t_range:
			I_Na[t] = (m ** 3) * h * self.g_Na * (self.E_Na - V[t-1])
			I_K[t] = (n ** 4) * self.g_K * (self.E_K - V[t-1])
			I_L[t] = self.g_L * (self.E_L - V[t-1])
			I_in[t] = I_in_func(t*dt) * input_weights

			g_syn_others = np.sum(g_syn[t-1]) - g_syn[t-1]
			E_syn_others = np.sum(self.E_syn) - self.E_syn
			V_others = np.sum(V[t-1]) - V[t-1]
			I_syn[t] = g_syn_others * (E_syn_others - V_others)

			V[t] = V[t - 1] + dt * ((I_Na[t] + I_K[t] + I_L[t] + I_syn[t] + I_in[t]) / self.C_m)
			m += dt * ((1 - m) * self.alpha_m(V[t]) - m * self.beta_m(V[t]))
			n += dt * ((1 - n) * self.alpha_n(V[t]) - n * self.beta_n(V[t]))
			h += dt * ((1 - h) * self.alpha_h(V[t]) - h * self.beta_h(V[t]))
			spikes[t] = (V[t] >= threshold).astype(int)
			dt_spikes = (1 - spikes[t]) * (dt_spikes + 1)
			g_syn[t] = g_syn[t-1] * (1 - dt / self.tau_syn) + self.weights * spikes[t]
		return dict(
			spikes=spikes,
			V=V,
			T=T,
			dt=dt,
			threshold=threshold,
			g_syn=g_syn,
			I_Na=I_Na,
			I_K=I_K,
			I_L=I_L,
			I_in=I_in,
			I_syn=I_syn,
		)

	def show_weights_in_func_of_g_syn(self, dt=1e-3):
		fig = plt.figure(figsize=(10, 8))
		g_list = [self.g_Na, self.g_K, self.g_L]
		g_syn = np.linspace(0, 2*max(g_list))
		weights = g_syn[:, np.newaxis] - g_syn[:, np.newaxis] * (1 - dt / self.tau_syn)
		w_g_func_list = [interp1d(g_syn, weights[:, i]) for i in range(weights.shape[-1])]
		print(f"{g_syn.shape = }, {weights.shape = }")
		plt.plot(g_syn, weights[:, 0], label="Exc")
		plt.plot(g_syn, weights[:, 1], label="Ihn")
		for g_name, g in zip(["g_Na", "g_K", "g_L"], g_list):
			plt.scatter(g*np.ones_like(self.weights), [f(g) for f in w_g_func_list], label=g_name)
		plt.xlabel("$g_{syn}$ [$mS/cm^3$]")
		plt.ylabel("weights [-]")
		plt.legend()
		plt.show()

	def get_weights_space(self, alpha=1.0, d=100, **kwargs):
		dt = kwargs.get('dt', 1e-3)
		dt = 1e-3
		g_list = [self.g_Na, self.g_K, self.g_L]
		g_syn = np.linspace(0, alpha * np.max(g_list), num=d)
		weights = g_syn[:, np.newaxis] - g_syn[:, np.newaxis] * (1 - dt / self.tau_syn)
		return weights

	def get_stable_currents(self):
		m = self.m_inf(self.V_reset)
		n = self.n_inf(self.V_reset)
		h = self.h_inf(self.V_reset)
		I_Na = (m ** 3) * h * self.g_Na * self.E_Na
		I_K = (n ** 4) * self.g_K * self.E_K
		I_L = self.g_L * self.E_L
		return dict(I_Na=I_Na, I_K=I_K, I_L=I_L)

	def show_weights_exploration(self, k=5, **kwargs):
		n_names = ["Exc", "Inh"]
		if 'weights' in kwargs:
			weights = kwargs['weights']
		else:
			weights = self.get_weights_space(alpha=1.0, d=k, **kwargs)
			weights[:, -1] = 1.0 * weights[:, -1][::-1]
		out_list = apply_func_multiprocess(show_weights_exploration_worker, [(self, wi, kwargs) for wi in weights])
		fig, axes = plt.subplots(nrows=k, ncols=weights.shape[-1]+1, sharex='all', figsize=(16, 10))
		for i, [ax_V, *axes_I_list] in enumerate(axes):
			out = out_list[i]
			for j in range(out['V'].shape[-1]):
				x = np.arange(0, out['T'], out['dt'])
				ax_V.plot(x, out['V'][:, j], label=f"{n_names[j]}")
				if j == 0:
					ax_V.plot(x, out['threshold']*np.ones_like(x), label=f"threshold", c='k')
			title = "weights: ["
			for w in weights[i]:
				title += f'{w:.2e}, '
			title += "]"
			ax_V.set_title(title)

			I_list = [
				# "I_Na",
				# "I_K",
				"I_L",
				"I_in",
				"I_syn",
			]
			for j, ax_I in enumerate(axes_I_list):
				for I_idx, I_name in enumerate(I_list):
					x = np.arange(0, out['T'], out['dt'])
					ax_I.plot(x, out[I_name][:, j], label=f"{I_name}")
			if i == 0:
				ax_V.legend()
				[ax_I.legend() for ax_I in axes_I_list]
				[ax_I.set_title(f"{n_names[j]}") for j, ax_I in enumerate(axes_I_list)]
			if i == k//2:
				ax_V.set_ylabel("V [mV]")
				[ax_I.set_ylabel(f"I [$\mu A / cm^2$]") for j, ax_I in enumerate(axes_I_list)]
			if i == len(axes) - 1:
				axes_I_list[0].set_xlabel("T [ms]")

		fig.tight_layout(pad=2.0)
		plt.savefig(f"figures/q3a_{kwargs['I_in_func'].name}.png", dpi=300)
		# plt.show()
		plt.close(fig)

	def show_spike_freq_by_I(self, I_space=np.linspace(0.0, 20.0, num=10), **kwargs):
		n_names = ["Exc", "Inh"]
		freqs = []
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
		# p_bar = tqdm.tqdm(total=len(I_space))
		out_list = apply_func_multiprocess(
			show_spike_freq_by_I_worker, [(self, i, kwargs) for i_idx, i in enumerate(I_space)]
		)
		for i_idx, i in enumerate(I_space):
			out = out_list[i_idx]
			spike_counts = np.sum(np.abs(np.diff(out["spikes"], axis=0)), axis=0) / 2
			spike_freqs = spike_counts / out['T']
			freqs.append(spike_freqs)
		# 	p_bar.update()
		# p_bar.close()
		freqs = np.asarray(freqs)
		# for j in range(freqs.shape[-1]):
		# 	ax.plot(I_space, freqs[:, j], label=n_names[j])
		ax.plot(I_space, freqs[:, 0], label=n_names[0])
		ax.set_xlabel("I [$\mu A / cm^2$]")
		ax.set_ylabel("Spiking frequency [Hz]")
		ax.legend()
		fig.tight_layout(pad=2.0)
		plt.savefig(f"figures/q3b1.png", dpi=300)
		# plt.show()
		plt.close(fig)


def show_spike_freq_by_I_worker(model, i, kwargs):
	kwargs['I_in_func'] = IConst(i)
	out = model.run(**kwargs)
	return out


def show_weights_exploration_worker(model, weights, kwargs):
	model.weights = weights
	out = model.run(**kwargs)
	return out


def question_3_a():
	T, dt = 160, 1e-2
	model = CoupleHH()
	model.show_weights_exploration(T=T, dt=dt, k=5, I_in_func=IConst(3.0))
	for p in [10, 18, 20, 21, 22, 30]:
		model.show_weights_exploration(T=T, dt=dt, k=5, I_in_func=ISin(p, 1.6))
	model.show_weights_exploration(T=T, dt=dt, k=5, I_in_func=ISteps(2.1 * np.ones(10), 10, 10, alt=False))
	model.show_weights_exploration(T=T, dt=dt, k=5, I_in_func=ISteps(1.2 * np.ones(10), 10, 10, alt=True))
	model.show_weights_exploration(T=T, dt=dt, k=5, I_in_func=IConst(10))


def question_3_b_1():
	T, dt = 160, 1e-2
	model = CoupleHH()
	model.show_spike_freq_by_I(T=T, dt=dt, I_space=np.linspace(0.0, 50.0, num=1_000))


def question_3_b_2_worker(model, wi, kwargs):
	model.weights = np.array([wi, 0.0])
	out = model.run(**kwargs)
	return out


def question_3_b_2():
	model = CoupleHH()
	n_names = ["Exc", "Inh"]
	V_max_list = []
	V_min_list = []
	g_syn_list = []
	x_list = []
	kwargs = dict(d=1_000, I_in_func=IConst(3.0), dt=1e-2, T=160, k=5)
	w1_space = model.get_weights_space(**kwargs)[:, 0]
	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
	out_list = apply_func_multiprocess(question_3_b_2_worker, [(model, wi, kwargs) for wi in w1_space])
	for out_idx, out in enumerate(out_list):
		V_max_list.append(np.max(out['V'][:, -1]))
		V_min_list.append(np.min(out['V'][:, -1]))
		g_syn_list.append(out['g_syn'])
		x_list.append(np.arange(0, out['T'], out['dt']))
		if out_idx == 0:
			axes[0].plot(w1_space, out['threshold'] * np.ones_like(w1_space), label=f"threshold", c='k')
	V_max_list = np.asarray(V_max_list)
	V_min_list = np.asarray(V_min_list)
	axes[0].plot(w1_space, V_max_list, label="max($V_{Inh}$)")
	axes[0].plot(w1_space, V_min_list, label="min($V_{Inh}$)")
	axes[0].set_xscale('log')
	axes[0].set_xlabel("$w_1$ [-]")
	axes[0].set_ylabel("Potential [mV]")
	axes[0].legend()

	# Show g_syn
	w_0_idx = np.argmin(np.abs(V_max_list))
	k = kwargs.get('k', 5)
	w_indexes = list(np.random.randint(0, w_0_idx, size=(k - 1) // 2)) if w_0_idx > 0 else []
	w_indexes.extend(list(np.random.randint(w_0_idx + 1, len(w1_space), size=k - 1 - len(w_indexes))))
	w_indexes = np.append(w_indexes, w_0_idx)
	w_indexes_argsort = np.argsort(w_indexes)
	for i, w_idx in enumerate(np.sort(w_indexes)):
		axes[1].plot(x_list[w_idx], g_syn_list[w_idx][:, 0], label=f"$w_1$ = {w1_space[w_idx]:.3e}")
	legend = axes[1].legend()
	legend.get_texts()[w_indexes_argsort[-1]].set_color("red")
	axes[1].set_xlabel("T [ms]")
	axes[1].set_ylabel("$g_{syn, 1}$ [$mS$/$cm^3$]")

	fig.tight_layout(pad=2.0)
	plt.savefig(f"figures/q3b2_V_minmax_gsyn.png", dpi=300)
	# plt.show()
	plt.close(fig)

	# Show g_syn
	w_0_idx = np.argmin(np.abs(V_max_list))
	k = kwargs.get('k', 5)
	w_indexes = list(np.random.randint(0, w_0_idx, size=(k - 1) // 2)) if w_0_idx > 0 else []
	w_indexes.extend(list(np.random.randint(w_0_idx+1, len(w1_space), size=k-1-len(w_indexes))))
	w_indexes = np.sort(np.append(w_indexes, w_0_idx))
	fig, axes = plt.subplots(nrows=k, ncols=1, figsize=(10, 8))
	for i in range(k):
		for j in range(g_syn_list[w_indexes[i]].shape[-1]):
			axes[i].plot(x_list[w_indexes[i]], g_syn_list[w_indexes[i]][:, j], label=n_names[j])
		axes[i].set_title(f"$w_1$ = {w1_space[w_indexes[i]]:.3e}")
		if i == k-1:
			axes[i].set_xlabel("T [ms]")
		if i == k//2:
			axes[i].set_ylabel("$g_{syn}$ [$mS$/$cm^3$]")
	axes[0].legend()
	fig.tight_layout(pad=2.0)
	plt.savefig(f"figures/q3b2_g_syn.png", dpi=300)
	# plt.show()
	plt.close(fig)
	return dict(w_exc_spike=w1_space[w_0_idx])


def question_3_b_3_worker(model, w1, w2, kwargs):
	model.weights = np.array([w1, w2])
	out = dict(err=0)
	try:
		out.update(model.run(**kwargs))
	except RuntimeWarning:
		print("RuntimeWarning Fuck you")
		out['err'] = 1
	return out


def mean_free_path(hit_path: np.ndarray, reverse=True):
	not_hit = np.logical_not(hit_path.astype(bool)) if reverse else hit_path.astype(bool)
	times = [sum(1 for _ in group) for key, group in itertools.groupby(not_hit) if key]
	return np.mean(times)


def question_3_b_3(out_question_2_b_2: dict = None):
	model = CoupleHH()
	n_names = ["Exc", "Inh"]
	V_max_list = []
	V_min_list = []
	V_list = []
	g_syn_list = []
	freqs = []
	mean_free_path_list = []
	x_list = []
	kwargs = dict(d=1_000, I_in_func=IConst(5.0), dt=1e-2, T=160, k=5)
	w_space = model.get_weights_space(**kwargs)
	w1 = out_question_2_b_2["w_exc_spike"] * 1.1
	# w1 = np.quantile(w_space[:, 0], 0.9)
	# w1 = np.mean(w_space[:, 0])
	w2_space = w_space[:, -1]
	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
	out_list = apply_func_multiprocess(question_3_b_3_worker, [(model, w1, wi, kwargs) for wi in w2_space])
	for out_idx, out in enumerate(out_list):
		if out['err']:
			continue
		V_list.append(out['V'])
		V_max_list.append(np.max(out['V'], axis=0))
		V_min_list.append(np.min(out['V'], axis=0))
		g_syn_list.append(out['g_syn'])
		spike_counts = np.sum(np.abs(np.diff(out["spikes"], axis=0)), axis=0) / 2
		spike_freqs = spike_counts / out['T']
		freqs.append(spike_freqs)
		mean_free_paths = [mean_free_path(out["spikes"][:, j])*out['dt'] for j in range(out["spikes"].shape[-1])]
		mean_free_path_list.append(mean_free_paths)
		x_list.append(np.arange(0, out['T'], out['dt']))
		# if out_idx == 0:
		# 	axes[0].plot(w2_space, out['threshold'] * np.ones_like(w2_space), label=f"threshold", c='k')
	V_max_list = np.asarray(V_max_list)
	V_min_list = np.asarray(V_min_list)
	V_list = np.asarray(V_list)
	freqs = np.asarray(freqs)
	mean_free_path_list = np.asarray(mean_free_path_list)
	# j = 0
	# axes[0].plot(w2_space, V_max_list[:, j], label="max($V_{"+n_names[j]+"}$)")
	# axes[0].plot(w2_space, V_min_list[:, j], label="min($V_{"+n_names[j]+"}$)")
	# axes[0].set_xscale('log')
	axes[0].set_xlabel("$w_2$ [-]")
	# axes[0].set_ylabel("Potential [mV]")
	# for j in range(freqs.shape[-1]):
	# 	axes[0].plot(w2_space, freqs[:, j]*1_000, label="${" + n_names[j] + "}$")
	# axes[0].set_ylabel("Spiking rate [Hz]")
	for j in range(freqs.shape[-1]):
		axes[0].plot(w2_space, mean_free_path_list[:, j], label="${" + n_names[j] + "}$")
	axes[0].set_ylabel("Mean free path [ms]")
	axes[0].set_title(f"$w_1$={w1:.3e}")
	axes[0].legend()

	# Show g_syn
	w_0_idx = np.argmin(np.abs(V_max_list[:, -1]))
	k = kwargs.get('k', 5)
	w_indexes = list(np.random.randint(0, w_0_idx, size=(k - 1) // 2)) if w_0_idx > 0 else []
	w_indexes.extend(list(np.random.randint(w_0_idx + 1, len(w2_space), size=k - 1 - len(w_indexes))))
	w_indexes = np.append(w_indexes, w_0_idx)
	w_indexes_argsort = np.argsort(w_indexes)
	for i, w_idx in enumerate(np.sort(w_indexes)):
		axes[1].plot(x_list[w_idx], V_list[w_idx][:, 0], label=f"$w_2$ = {w2_space[w_idx]:.3e}")
	legend = axes[1].legend()
	legend.get_texts()[w_indexes_argsort[-1]].set_color("red")
	axes[1].set_xlabel("T [ms]")
	# axes[1].set_ylabel("$g_{syn, 2}$ [$mS$/$cm^3$]")
	axes[1].set_ylabel("$V_{1}$ [mV]")

	fig.tight_layout(pad=2.0)
	plt.savefig(f"figures/q3b3_spiking_rate_{kwargs['I_in_func'].name}.png", dpi=300)
	# plt.show()
	plt.close(fig)


if __name__ == '__main__':
	os.makedirs("figures/", exist_ok=True)
	plt.rcParams.update({'font.size': 12})
	# CoupleHH().show_weights_in_func_of_g_syn()
	question_3_a()
	question_3_b_1()
	question_3_b_2_dict = question_3_b_2()
	question_3_b_3(question_3_b_2_dict)



