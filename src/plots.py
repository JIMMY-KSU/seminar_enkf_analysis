from src.problem import *
from numpy.random import normal
from IPython.core.display import HTML
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def load_style():
    plt.rc('font', size=14)
    return HTML(""".output {
            display: flex;
            align-items: center;
            text-align: center;
        }
    """)


def plot_truth(u_true, measurement_points, y):
    x = np.linspace(0, 1, u_true.shape[0]*10)
    plt.figure(figsize=(15, 8))
    plt.rc('font', size=14)
    plt.grid()
    plt.title("Truth $u^\\dagger$ and observations")
    plt.xlabel("x")
    plt.ylabel("$u^\\dagger(x)$")
    plt.scatter(measurement_points, y, color='orange', marker='o', label='Observations $y$')
    plt.plot(x, eval_gp(u_true, x), color='black', label='$u^\\dagger$')
    plt.legend()


def plot_gps(ks):
    x = np.linspace(0, 1, 1000)

    plt.figure(figsize=(20, 5))
    for i, K in enumerate(ks):
        plt.subplot(1, len(ks), i+1)
        plt.title("K=%d expansion terms" % K)

        gca = plt.gca()
        gca.axes.xaxis.set_ticklabels([])
        gca.axes.yaxis.set_ticklabels([])
        for j in range(5):
            plt.plot(eval_gp(normal(size=K), x), c='gray', alpha=0.5)


def __compute_running(enkf, f):
    bars = np.zeros(enkf.path.shape[0])
    mins = np.zeros(enkf.path.shape[0])
    maxs = np.zeros(enkf.path.shape[0])

    for i in range(enkf.path.shape[0]):
        e = np.linalg.norm(f(i), 2, axis=1)
        bars[i] = e.mean()
        mins[i] = np.min(e)
        maxs[i] = np.max(e)

    return bars, mins, maxs


def plot_mean_convergence(enkf):
    # compute the statistics
    u_bars = enkf.path.mean(axis=1)
    e_bars, e_min, e_max = __compute_running(enkf, lambda i: enkf.path[i, :, :] - u_bars[i])

    # and plot them
    x = np.linspace(1e-4, enkf.until, enkf.path.shape[0])

    plt.grid()
    plt.title('Convergence to the ensemble mean (ensemble size = %d)' % enkf.ensemble_size)
    plt.loglog(x, (e_bars[0]*x[0]**0.5)/x**0.5, 'b--', label="O(1/sqrt(t)")
    plt.loglog(x, e_bars, c='r', label="E[e(t)]")
    plt.fill_between(x, e_max, e_min, color='r', alpha=0.1, label='spread E[e(t)]')

    plt.legend()


def plot_residual_convergence(enkf, g, data):
    # compute the projected error
    r_bars, r_min, r_max = __compute_running(enkf, lambda i: g(enkf.path[i, :, :]) - data)

    # and plot them
    x = np.linspace(1e-4, enkf.until, enkf.path.shape[0])

    plt.grid()
    plt.title('Convergence of the projected residuals (ensemble size = %d)' % enkf.ensemble_size)
    plt.loglog(x, (r_bars[0]*x[0]**0.5)/x**0.5, 'b--', label="O(1/sqrt(t)")
    plt.loglog(x, r_bars, c='r', label="E[r(t)]")
    plt.fill_between(x, r_max, r_min, color='r', alpha=0.1, label='spread E[Ar(t)]')

    plt.legend()


def plot_animation(enkf, target_value, plotted_dims):
    fig, ax = plt.subplots(figsize=(8,8))

    ix, iy = plotted_dims

    w = max(np.max(enkf.path[0, :, ix]) - np.min(enkf.path[0, :, ix]),
            np.max(np.abs(target_value[ix] - enkf.path[0, :, ix])))

    h = max(np.max(enkf.path[0, :, iy]) - np.min(enkf.path[0, :, iy]),
            np.max(np.abs(target_value[iy] - enkf.path[0, :, iy])))

    def init():
        plt_particles.set_data([], [])
        plt_truth.set_data(target_value[ix:ix+1], target_value[iy:iy+1])
        plt_mean.set_data([], [])

        ax.set_xlim((target_value[ix]-w, target_value[ix]+w))
        ax.set_ylim((target_value[iy]-h, target_value[iy]+h))

        plt.legend(loc='upper left')
        plt.xlabel('u_%d' % ix)
        plt.ylabel('u_%d' % iy)

        return plt_particles, plt_mean, plt_truth

    def animate(i):
        u_bar = enkf.path[i, :, plotted_dims].mean(axis=1).reshape((1, 2))
        plt_particles.set_data(enkf.path[i, :, ix], enkf.path[i, :, iy])
        plt_mean.set_data(u_bar[:, 0], u_bar[:, 1])

        return plt_particles, plt_mean, plt_truth

    plt_particles, = ax.plot([], [], 'b.', ms=6, label='$\\{u^{(j)}\\}_{j=1}^J$')
    plt_truth, = ax.plot([], [], 'rx', ms=6, label='$u^\\dagger$')
    plt_mean, = ax.plot([], [], 'go', ms=6, label='$\\overline{u}$')

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=enkf.path.shape[0], interval=100,
                                   repeat=False, blit=True)

    return HTML(anim.to_html5_video())


def plot_enkf_vs_truth(enkf, u_true, measurement_points, y):
    x = np.linspace(0, 1, u_true.shape[0]*10)
    u_bar = enkf.path[-1, :, :].mean(axis=0)

    plt.rc('font', size=14)
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 1, 1)
    plt.grid()
    plt.plot(x, eval_gp(u_true, x), color='black', label='$u^\\dagger$')
    plt.plot(x, eval_gp(u_bar, x), c='red', label='EnKF estimate ($J = %d$)' % enkf.path.shape[1])

    plt.legend()

    plt.subplot(2, 1, 2)
    plt.grid()
    plt.scatter(measurement_points, y, color='black', marker='o', label='Observations $y$')
    plt.scatter(measurement_points, eval_gp(u_bar, measurement_points), color='red', marker='x', label='')

    plt.legend()


def plot_multi_enkf_vs_truth(enkfs, u_true, colors):

    x = np.linspace(0, 1, u_true.shape[0]*10)

    plt.figure(figsize=(15, 5))
    plt.grid()
    plt.rc('font', size=14)

    for i, enkf in enumerate(enkfs):
        u_bar = enkf.path[-1, :, :].mean(axis=0)
        plt.plot(x, eval_gp(u_bar, x), '--', color=colors[i], label='EnKF est. (J = %d)' % enkf.path.shape[1])

    plt.plot(x, eval_gp(u_true, x), color='black', label='$u^\\dagger$')
    plt.legend()


def plot_convergence(enkf, g, y):
    plt.figure(figsize=(20,5))
    plt.grid()
    plt.rc('font', size=14)
    plt.subplot(1, 2, 1)
    plot_mean_convergence(enkf)

    plt.subplot(1, 2, 2)
    plot_residual_convergence(enkf, g, y)


def plot_multi_convergence(enkfs, g, y, colors):
    plt.figure(figsize=(20,5))
    plt.rc('font', size=14)
    plt.subplot(1, 2, 2)
    plt.grid()
    plt.title('Convergence of the projected residuals')
    x = np.linspace(1e-4, enkfs[0].until, enkfs[0].path.shape[0])
    plt.loglog(x, 1/x**0.5, '--', color='grey', label="O(1/sqrt(t)")
    plt.xlabel("t")
    for i, enkf in enumerate(enkfs):
        r_bars, r_min, r_max = __compute_running(enkf, lambda i: g(enkf.path[i, :, :]) - y)
        plt.loglog(x, r_bars, color=colors[i], label="E[Ar(t)] for J = %d" % enkf.ensemble_size)
        plt.fill_between(x, r_max, r_min, color=colors[i], alpha=0.1, label='spread E[Ar(t)] for J = %d' % enkf.ensemble_size)
    plt.legend()

    plt.subplot(1, 2, 1)
    plt.grid()
    plt.title('Convergence to the ensemble mean')
    plt.xlabel("t")
    plt.loglog(x, 1/x**0.5, '--', color='grey', label="O(1/sqrt(t)")
    for i, enkf in enumerate(enkfs):
        u_bars = enkf.path.mean(axis=1)
        e_bars, e_min, e_max = __compute_running(enkf, lambda j: enkf.path[j, :, :] - u_bars[j])
        plt.loglog(x, e_bars, c=colors[i], label='E[e(t)] for J = %d' % enkf.ensemble_size)
        plt.fill_between(x, e_max, e_min, color=colors[i], alpha=0.1, label='spread E[e(t)] for J = %d' % enkf.ensemble_size)
    plt.legend()
