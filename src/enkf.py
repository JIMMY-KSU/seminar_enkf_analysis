import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import ode


class ContEnKF:
    def __init__(self, config):
        u0 = config['initial_ensemble']
        y = config['data']
        noise_var = config['noise_var']
        g = config['model']
        self.until = config['integrate_until']
        self.time_step = config['time_step_size']

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
        n_steps = int(self.until / self.time_step)
        path = np.zeros((n_steps+1, self.ensemble_size, self.param_dim))
        path[0, :, :] = self.u0.reshape((self.ensemble_size, self.param_dim))

        def rhs(_, u): return self.__rhs(u)

        ivp = ode(rhs).set_integrator('dopri5')
        ivp.set_initial_value(self.u0, 0)

        for i in range(1, n_steps+1):
            ivp.integrate(ivp.t + self.time_step)
            path[i, :, :] = ivp.y.reshape((self.ensemble_size, self.param_dim))

        self.path = path
        return path[-1, :, :]

    def __compute_running(self, f):
        bars = np.zeros(self.path.shape[0])
        mins = np.zeros(self.path.shape[0])
        maxs = np.zeros(self.path.shape[0])

        for i in range(self.path.shape[0]):
            e = np.linalg.norm(f(i), 2, axis=1)
            bars[i] = e.mean()
            mins[i] = np.min(e)
            maxs[i] = np.max(e)

        return bars, mins, maxs

    def plot_mean_convergence(self):
        # compute the statistics
        u_bars = self.path.mean(axis=1)
        e_bars, e_min, e_max = self.__compute_running(lambda i: self.path[i, :, :] - u_bars[i])

        # and plot them
        x = np.linspace(self.time_step, self.until, self.path.shape[0])

        plt.title('Convergence to the ensemble mean (ensemble size = %d)' % self.ensemble_size)
        plt.loglog(x, (e_bars[0]*x[0]**0.5)/x**0.5, 'b--', label="O(1/sqrt(t)")
        plt.loglog(x, e_bars, c='r', label="E[e(t)]")
        plt.fill_between(x, e_max, e_min, color='r', alpha=0.1, label='spread E[e(t)]')

        plt.legend()

    def plot_value_convergence(self, target_value):
        # compute the projected error
        r_bars, r_min, r_max = self.__compute_running(lambda i: self.path[i, :, :] - target_value)

        # and plot them
        x = np.linspace(self.time_step, self.until, self.path.shape[0])

        plt.title('Convergence of the projected error (ensemble size = %d)' % self.ensemble_size)
        plt.loglog(x, (r_bars[0]*x[0]**0.5)/x**0.5, 'b--', label="O(1/sqrt(t)")
        plt.loglog(x, r_bars, c='r', label="E[r(t)]")
        plt.fill_between(x, r_max, r_min, color='r', alpha=0.1, label='spread E[r(t)]')

        plt.legend()

    def plot_animation(self, target_value, plotted_dims, plot):
        fig, ax = plot
        ix, iy = plotted_dims

        w = max(np.max(self.path[0, :, ix]) - np.min(self.path[0, :, ix]),
                np.max(np.abs(target_value[ix] - self.path[0, :, ix])))

        h = max(np.max(self.path[0, :, iy]) - np.min(self.path[0, :, iy]),
                np.max(np.abs(target_value[iy] - self.path[0, :, iy])))

        def init():
            plt_particles.set_data([], [])
            plt_truth.set_data(target_value[ix:ix+1], target_value[iy:iy+1])
            plt_mean.set_data([], [])
            plt_paths.set_data([], [])

            ax.set_xlim((target_value[ix]-w, target_value[ix]+w))
            ax.set_ylim((target_value[iy]-h, target_value[iy]+h))
            # ax.set_xlim((np.min(self.path[0, :, ix]), np.max(self.path[0, :, ix])))
            # ax.set_ylim((np.min(self.path[0, :, iy]), np.max(self.path[0, :, iy])))

            return plt_particles, plt_mean, plt_paths, plt_truth

        def animate(i):
            u_bar = self.path[i, :, plotted_dims].mean(axis=1).reshape((1, 2))
            plt_particles.set_data(self.path[i, :, ix], self.path[i, :, iy])
            plt_mean.set_data(u_bar[:, ix], u_bar[:, iy])

            return plt_particles, plt_mean, plt_paths, plt_truth

        plt_particles, = ax.plot([], [], 'bo', ms=6)
        plt_truth, = ax.plot([], [], 'rx', ms=6)
        plt_mean, = ax.plot([], [], 'go', ms=6)
        plt_paths, = ax.plot([], [], 'b--', lw=1)

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=self.path.shape[0], interval=60,
                                       repeat_delay=1000, blit=True)

        return anim
