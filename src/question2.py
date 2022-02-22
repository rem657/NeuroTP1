# -C_m dV_m/dt = g_K(V_M - E_K) + g_Na(V_M - E_Na) + g_L(V_M - E_L)

# dn/dt = α_n(V)(1 - n) - β_n(V)n
# dm/dt = α_m(V)(1 - m) - β_m(V)m
# dh/dt = α_h(V)(1 - h) - β_h(V)h
from scipy.integrate import solve_ivp, odeint
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class HHModel:
    def __init__(self,
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
        return self.alpha_n(V)/(self.alpha_n(V) + self.beta_n(V))

    def m_inf(self, V: float):
        return self.alpha_m(V)/(self.alpha_m(V) + self.beta_m(V))

    def h_inf(self, V: float):
        return self.alpha_h(V)/(self.alpha_h(V) + self.beta_h(V))

    def tau_n(self, V: float):
        return 1 / (self.alpha_n(V) + self.beta_n(V))

    def tau_m(self, V: float):
        return 1 / (self.alpha_m(V) + self.beta_m(V))

    def tau_h(self, V: float):
        return 1 / (self.alpha_h(V) + self.beta_h(V))

    def dXdt(self, X: np.ndarray, t: float):
        V, m, h, n = X

        dVdt = (self.I_inj(t) - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)) / self.C_m
        dmdt = (self.m_inf(V) - m)/self.tau_m(V)
        dhdt = (self.h_inf(V) - h)/self.tau_h(V)
        dndt = (self.n_inf(V) - n)/self.tau_n(V)
        return [dVdt, dmdt, dhdt, dndt]

    def compute_model(self):
        init_cond = [self.V_repos,
                     self.m_inf(self.V_repos),
                     self.h_inf(self.V_repos),
                     self.n_inf(self.V_repos)]
        X: np.ndarray = odeint(self.dXdt, init_cond, self.time)
        V = X[:, 0]
        m = X[:, 1]
        h = X[:, 2]
        n = X[:, 3]
        return self.time, V, m, h, n


def display_HHModel(I_inj: callable, t_init: float, t_end: float, t_inter: float):
    model = HHModel(I_inj, t_init=t_init, t_end=t_end, t_inter=t_inter)
    I_inj = np.vectorize(I_inj)
    t, V, m, h, n = model.compute_model()
    n_row = 4
    fig = make_subplots(rows=n_row,
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
            y=m-model.m_inf(V),
            mode='lines',
            name='m - m∞'
        ),
        row=3, col=1
    )
    fig.update_layout(yaxis4=dict(title='Courant appliqué [μA]', showgrid=False, zeroline=False),
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
    return (0.1*t)*(t < 1700)


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


if __name__ == '__main__':
    # I = lambda t: 35 * (t > 100) - 35 * (t > 200) + 150 * (t > 300) - 150 * (t > 400)
    # I = I_stairs(list(range(0, 110, 5)))
    I = I_steps(list(range(0, 150, 10)))
    display_HHModel(I, 0, 1800, 0.01)
