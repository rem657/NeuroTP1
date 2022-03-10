import numpy as np
import tqdm
from scipy.interpolate import interp1d

from src.ifunc import IConst, IFunc, ISin, ISteps


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
			dt: float = 0.01,
			I_in_func: IFunc = IConst(0.0),
			start_with_spike: bool = False,
			threshold=None,
	):
		time_steps = int(T / dt)
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
		print(f"{m = }, {n = }, {h = }")
		spikes = np.zeros((time_steps, *self.weights.shape), dtype=int)
		if start_with_spike:
			spikes[0] = np.ones_like(self.weights, dtype=int)
		dt_spikes = (self.tau_syn/dt) * (1-spikes[0])
		g_syn = np.asarray(self.weights) * spikes[0]
		if threshold is None:
			threshold = 0.0
			print(f"{threshold = }")

		for t in tqdm.tqdm(range(1, time_steps)):
			I_Na[t] = (m ** 3) * h * self.g_Na * (self.E_Na - V[t-1])
			I_K[t] = (n ** 4) * self.g_K * (self.E_K - V[t-1])
			I_L[t] = self.g_L * (self.E_L - V[t-1])
			I_in[t] = I_in_func(t*dt)

			g_syn_others = np.sum(g_syn) - g_syn
			E_syn_others = np.sum(self.E_syn) - self.E_syn
			V_others = np.sum(V[t-1]) - V[t-1]
			I_syn[t] = g_syn_others * (E_syn_others - V_others)

			V[t] = V[t - 1] + dt * ((I_Na[t] + I_K[t] + I_L[t] + I_syn[t] + I_in[t]) / self.C_m)
			m += dt * ((1 - m) * self.alpha_m(V[t]) - m * self.beta_m(V[t]))
			n += dt * ((1 - n) * self.alpha_n(V[t]) - n * self.beta_n(V[t]))
			h += dt * ((1 - h) * self.alpha_h(V[t]) - h * self.beta_h(V[t]))
			spikes[t] = (V[t] >= threshold).astype(int)
			dt_spikes = (1 - spikes[t]) * (dt_spikes + 1)
			g_syn = g_syn * (1 - dt / self.tau_syn) + self.weights * spikes[t]
		return dict(
			spikes=spikes,
			V=V,
			T=T,
			dt=dt,
			threshold=threshold,
			I_Na=I_Na,
			I_K=I_K,
			I_L=I_L,
			I_in=I_in,
			I_syn=I_syn
		)

	def show_weights_in_func_of_g_syn(self):
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

	def get_weights_space(self, alpha=1.0, d=100):
		g_list = [self.g_Na, self.g_K, self.g_L]
		g_syn = np.linspace(0, alpha * max(g_list), num=d)
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
		weights = self.get_weights_space(alpha=1.0, d=k)
		weights[:, -1] = 1.0*weights[:, -1][::-1]
		fig, axes = plt.subplots(nrows=k, ncols=weights.shape[-1]+1, sharex='all', figsize=(16, 10))
		for i, [ax_V, *axes_I_list] in enumerate(axes):
			model.weights = weights[i]
			out = self.run(**kwargs)
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
				[ax_I.set_title(f"{n_names[j]}") for j, ax_I in enumerate(axes_I_list)]
			if i == k//2:
				ax_V.set_ylabel("V [mV]")
				[ax_I.set_ylabel(f"I [mA]") for j, ax_I in enumerate(axes_I_list)]
			if i == len(axes) - 1:
				ax_V.legend()
				[ax_I.legend() for ax_I in axes_I_list]
				axes_I_list[0].set_xlabel("T [ms]")

		fig.tight_layout(pad=2.0)
		plt.savefig(f"figures/q3a_{kwargs['I_in_func'].name}.png", dpi=300)
		plt.show()


if __name__ == '__main__':
	import matplotlib.pyplot as plt
	import pprint
	T, dt = 160, 1e-3
	model = CoupleHH(weights=np.array([0.3, -0.5]))
	# model.show_weights_in_func_of_g_syn()
	# model.show_weights_exploration(T=T, k=5, I_in_func=ISteps(1.2*np.ones(10), 10, 10, alt=True, name="ISteps_alt"))
	model.show_weights_exploration(T=T, k=5, I_in_func=IConst(10))
	pprint.pprint(model.get_stable_currents(), indent=4)
	model.weights = np.max(model.get_weights_space(), axis=0) * np.array([1., -1.])
	print(f"{model.weights = }")
	# I_step_func = lambda t: np.array([I_steps(10*np.ones(10), step_len=10, inactive_len=10)(t), 0.0])

	# out = model.run(T=T, dt=dt, I_in_func=I_step_func)
	# for i in range(out['V'].shape[-1]):
	# 	x = np.arange(0, T, dt)
	# 	plt.plot(x, out['V'][:, i], label=f"$n_{i}$")
	# 	x_indexes = np.argwhere(out['spikes'][:, i] > 0)
	# 	plt.scatter(x[x_indexes], out['V'][:, i][x_indexes], label=f'spikes $n_{i}$')
	# plt.legend()
	# plt.xlabel("T [ms]")
	# plt.ylabel("V [mV]")
	# plt.show()

	# plt.imshow(np.transpose(out['spikes']))
	# plt.colorbar()
	# plt.show()


