from forward_model import *
import matplotlib as plt
import fenics as fe

u = solve_heat_eq([1, 1])
fe.plot(u)

