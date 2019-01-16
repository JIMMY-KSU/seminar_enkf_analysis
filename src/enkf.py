import numpy as np
from scipy.integrate import ode


class ContEnKF:
    def __init__(self, config):
        u0 = config['initial_ensemble']
        y = config['data']
        noise_var = config['noise_var']
        g = config['model']
        self.until = config['integrate_until']
        self.step_size = config['step_size']

        self.ensemble_size = u0.shape[0]
        self.param_dim = u0.shape[1]
        self.data_dim = y.shape[0]
        self.path = np.zeros(0)

        self.u0 = u0.reshape(self.ensemble_size * self.param_dim)
        self.y = y
        self.g = g
        self.noise_var = noise_var

    @staticmethod
    def __mean_kro(x, y):
        C = np.zeros((x.shape[1], y.shape[1]))
        for i in range(x.shape[0]):
            C += x[i:i+1,:].T.dot(y[i:i+1, :])
        return C/x.shape[0]

    def __rhs(self, u):
        u = u.reshape((self.ensemble_size, self.param_dim))
        du = np.zeros_like(u)

        p = self.g(u)

        p_bar = np.average(p, axis=0).reshape((1, self.data_dim))
        u_bar = np.average(u, axis=0).reshape((1, self.param_dim))

        C_up = self.__mean_kro(u - u_bar, p - p_bar) / self.noise_var

        for j in range(u.shape[0]):
            du[j, :] = C_up.dot(self.y - p[j, :]).reshape(self.param_dim)

        return du.reshape(self.ensemble_size * self.param_dim)

    def compute(self):
        n_steps = int(self.until / self.step_size)
        path = np.zeros((n_steps+1, self.ensemble_size, self.param_dim))
        path[0, :, :] = self.u0.reshape((self.ensemble_size, self.param_dim))

        def rhs(_, u): return self.__rhs(u)

        ivp = ode(rhs).set_integrator('dopri5')
        ivp.set_initial_value(self.u0, 0)

        for i in range(1, n_steps+1):
            ivp.integrate(ivp.t + self.step_size)
            path[i, :, :] = ivp.y.reshape((self.ensemble_size, self.param_dim))

        self.path = path
        return path[-1, :, :]
