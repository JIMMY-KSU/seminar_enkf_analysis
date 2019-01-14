import numpy as np
import fenics as fe
from numpy.random import normal


class GaussianProcessProblem:
    # Initialize a linear inverse problem over the space of smooth [0, 1] -> R functions and zero boundary
    # conditions with a N(0, L^-1) prior on the space of functions -- where L is the Laplace operator.
    #
    # Since the space of functions is of infinite dimensions, we discretize the problem by considering the subspace
    # spanned by the largest `n_params` eigenfunction of the inverse laplace operator.
    #
    @staticmethod
    def prep_problem(n_params, measurement_points, error_std):
        data_dim = len(measurement_points)
        design_matrix = np.zeros((data_dim, n_params))

        for n in range(1, 1 + n_params):
            design_matrix[:, n-1] = np.sqrt(2) * np.sin(measurement_points * np.pi * n) * np.power(np.pi*n, -2)

        def fm(u):
            return np.dot(u, design_matrix.T)

        real_param = normal(size=n_params)
        y = GaussianProcessProblem.eval(real_param, measurement_points) + np.sqrt(error_std) * normal(size=data_dim)

        return fm, real_param, y

    @staticmethod
    def eval(params, locs):
        n_params = params.shape[0]
        y = np.zeros_like(locs)

        for n in range(1, 1 + n_params):
            y += np.sqrt(2) * np.sin(locs * np.pi * n) * np.power(np.pi * n, -2) * params[n-1]

        return y


class PoissonUnknownRHS:


    @staticmethod
    def solve_pde(params):
        level = 50
        mesh = fe.UnitIntervalMesh.create(level)

        V = fe.FunctionSpace(mesh, 'P', 1)
        bc = fe.DirichletBC(V, fe.Constant(0), lambda _,ob: ob)

        u = fe.TrialFunction(V)
        v = fe.TestFunction(V)

        codes = ['p_%d * sqrt(2) * sin(x[0]*pi*%d)/%d/pi/pi' % (n-1, n, n**2) for n in range(1, 1 + params.shape[0])]

        cpp_args = dict()
        for n in range(params.shape[0]): cpp_args['p_%d' % n] = params[n]

        f = fe.Expression(' + '.join(codes), **cpp_args, degree=1)

        a = fe.dot(fe.grad(u), fe.grad(v)) * fe.dx
        L = f * v * fe.dx

        u = fe.Function(V)
        fe.solve(a == L, u, bc)

        return u

    @staticmethod
    def prep_problem(n_params, measurement_points, error_std):
        data_dim = len(measurement_points)

        def fm(params):
            res = np.zeros((params.shape[0], data_dim))
            for i in range(params.shape[0]):
                res[i, :] = PoissonUnknownRHS.eval(params[i, :], measurement_points)
            return res

        real_param = 100*normal(size=n_params)
        y = PoissonUnknownRHS.eval(real_param, measurement_points) + np.sqrt(error_std) * normal(size=data_dim)

        return fm, real_param, y

    @staticmethod
    def eval(params, locs):
        u = PoissonUnknownRHS.__solve_pde(params)
        return np.array([u(x) for x in locs])
