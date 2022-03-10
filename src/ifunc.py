import numpy as np


class IFunc:
	def __init__(self, name):
		self.name = name

	def __call__(self, t_ms: float):
		raise NotImplementedError()


class IConst(IFunc):
	def __init__(self, value: float, name="IConst"):
		super(IConst, self).__init__(name)
		self.value = value

	def __call__(self, t_ms: float):
		return self.value


class ISteps(IFunc):
	def __init__(self, current_values: np.ndarray, step_len=20, inactive_len=20, alt=False, name="ISteps"):
		super(ISteps, self).__init__(name)
		self.current_values = current_values
		self.step_len = step_len
		self.inactive_len = inactive_len
		self.alt = alt
		self._sign = 1

	def __call__(self, t_ms: float):
		active = (t_ms % (self.inactive_len + self.step_len)) > self.inactive_len
		index = int(t_ms // (self.inactive_len + self.step_len)) % len(self.current_values)
		if active:
			value = self.current_values[index]
		else:
			value = -self.current_values[index] if self.alt else 0.0
		return value


class ISin(IFunc):
	def __init__(self, period: float, amplitude: float = 1.0):
		super(ISin, self).__init__(f"ISin_p{str(period).replace('.', '_')}_a{str(amplitude).replace('.', '_')}")
		self.period = period
		self.amplitude = amplitude

	def __call__(self, t_ms: float):
		return self.amplitude*np.sin(2*np.pi*t_ms / self.period)





