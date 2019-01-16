import numpy as np
from numpy.random import normal


# Initialize a linear inverse problem over the space of smooth [0, 1] -> R functions and zero boundary
# conditions with a N(0, L^-1) prior on the space of functions -- where L is the Laplace operator.
#
# Since the space of functions is of infinite dimensions, we discretize the problem by considering the subspace
# spanned by the largest `n_params` eigenfunction of the inverse laplace operator.
#
def prep_problem(n_params, measurement_points, error_std):
    data_dim = len(measurement_points)
    design_matrix = np.zeros((data_dim, 2*n_params))

    for n in range(1, 1 + 2*n_params):
        design_matrix[:, n-1] = np.sqrt(2) * np.sin(measurement_points * np.pi * n) / (np.pi * n)

    def fm(u):
        n_expansions = u.shape[1]
        return np.dot(u, design_matrix[:, :n_expansions].T)

    real_param = normal(size=n_params)
    y = fm(real_param.reshape(1, n_params)) + np.sqrt(error_std) * normal(size=data_dim)

    return fm, real_param, y.reshape((data_dim,))


def eval_gp(params, locs):
    n_params = params.shape[0]
    y = np.zeros_like(locs)

    for n in range(1, 1 + n_params):
        y += np.sqrt(2) * np.sin(locs * np.pi * n) / (np.pi * n) * params[n-1]

    return y


def make_initial_ensemble(ensemble_size, mixed=False):
    if not mixed:
        initial_ensemble = np.zeros((ensemble_size, ensemble_size))
        np.fill_diagonal(initial_ensemble, normal(size=ensemble_size))
        return initial_ensemble
    else:
        return normal(size=ensemble_size*ensemble_size).reshape((ensemble_size,ensemble_size))
