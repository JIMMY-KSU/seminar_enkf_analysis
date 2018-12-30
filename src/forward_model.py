import fenics as fe


def make_source_term(sources):
    code = "w*exp(-pow(x[0] - p, 2)/0.001)"
    f = fe.Constant(0)
    for (weight, position) in sources:
        f += fe.Expression(code, degree=1, p=position, w=weight / W)

    return f


def solve_heat_eq(sources):
    level = 50
    mesh = fe.UnitIntervalMesh.create(level)

    V = fe.FunctionSpace(mesh, 'P', 1)

    def boundary(x, on_boundary):
        return on_boundary

    bc = fe.DirichletBC(V, fe.Constant(0), boundary)

    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)

    f = make_source_term(sources)

    a = fe.dot(fe.grad(u), fe.grad(v)) * fe.dx
    L = f * v * fe.dx

    u = fe.Function(V)
    fe.solve(a == L, u, bc)

    return u