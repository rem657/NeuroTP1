import numpy as np


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
			weights: np.ndarray = np.array([0.1, -0.1]),
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

	def run(self, T: int, dt: float = 0.01, I_in_func=lambda t: 0.0):
		time_steps = int(T / dt)
		V = np.zeros((time_steps, *self.weights.shape), dtype=float)
		g_syn = np.zeros_like(self.weights)
		m = np.zeros_like(self.weights)
		n = np.zeros_like(self.weights)
		h = np.zeros_like(self.weights)
		spikes = np.zeros((time_steps, *self.weights.shape), dtype=int)

		V[0] = self.V_reset * np.ones_like(self.weights)
		spikes[0] = np.ones_like(self.weights, dtype=int)
		for t in range(1, time_steps):
			I_Na = (m ** 3) * h * self.g_Na * (self.E_Na - V[t])
			I_K = (n ** 4) * self.g_K * (self.E_K - V[t])
			I_L = self.g_L * (self.E_L - V[t])

			g_syn_others = np.sum(g_syn) - g_syn
			E_syn_others = np.sum(self.E_syn) - self.E_syn
			I_syn = g_syn_others * (E_syn_others - V[t])

			V[t] = V[t - 1] + dt * ((I_Na + I_K + I_L + I_syn + I_in_func(t)) / self.C_m)
			m += dt * ((1 - m) * self.alpha_m(V[t]) - m * self.beta_m(V[t]))
			n += dt * ((1 - n) * self.alpha_n(V[t]) - n * self.beta_n(V[t]))
			h += dt * ((1 - h) * self.alpha_h(V[t]) - h * self.beta_h(V[t]))
			spikes[t] = (V[t] >= 0.0).astype(int)  # TODO: rafiner
			g_syn = g_syn * (1 - dt / self.tau_syn) + self.weights * spikes[t]

		return dict(spikes=spikes, V=V)


if __name__ == '__main__':
	import matplotlib.pyplot as plt

	model = CoupleHH(weights=np.array([0.5, -0.5]))
	out = model.run(T=1, dt=0.01)
	for i in range(out['V'].shape[-1]):
		plt.plot(out['V'][:, i], label=f"$n_{i}$")
	plt.legend()
	plt.show()
