import dolfin as df
import fenics_helpers as fh
import matplotlib.pyplot as plt

def l_panel_mesh(lx, ly, refinement=5, show=False):
    """
    Creates something like
     ly +---------+
        |    ^y   |
        |    |    |
        |    0->--+
        |    | x
        |    |
    -ly +----+
       -lx       lx
        out of triangles
    """
    mesh = df.Mesh()

    e = df.MeshEditor()
    e.open(mesh, "triangle", 2, 2)
    e.init_vertices(8)
    e.add_vertex(0, [0, 0])
    e.add_vertex(1, [lx, 0])
    e.add_vertex(2, [lx, ly])
    e.add_vertex(3, [0, ly])
    e.add_vertex(4, [-lx, ly])
    e.add_vertex(5, [-lx, 0])
    e.add_vertex(6, [-lx, -ly])
    e.add_vertex(7, [0, -ly])

    e.init_cells(6)
    e.add_cell(0, [0, 1, 3])
    e.add_cell(1, [1, 2, 3])
    e.add_cell(2, [0, 3, 4])
    e.add_cell(3, [0, 4, 5])
    e.add_cell(4, [0, 5, 7])
    e.add_cell(5, [7, 5, 6])

    e.close
    mesh.order()

    for _ in range(refinement):
        mesh = df.refine(mesh)

    return mesh


def max(a, b):
    return 0.5 * (a + b + abs(a - b))

class Mat:
    E= 20000
    nu = 0.2
    ft = 4.
    Gf = 0.01
    alpha = 0.99

def omega(k):
    k0 = Mat.ft / Mat.E
    beta = Mat.ft / Mat.Gf
    exponent = beta * (k0 - k)
    omega = 1.0 - k0 / k* (1.0 - Mat.alpha + Mat.alpha * df.exp(exponent))
    return df.conditional(df.lt(k, k0), 0, omega)

class ModifiedMises:
    """ Modified von Mises equivalent strain, see
    de Vree et al., 1995, "Comparison of Nonlocal Approaches in
    Continuum Damage Mechanics"

    Invariants from https://en.wikipedia.org/wiki/Cauchy_stress_tensor

    The members T1, T2, T3 correspond to the term 1,2,3 in the equation
    """
    def __init__(self, k, nu):
        self.k = k
        self.nu = nu
        self.T1 = (k - 1) / (2. * k * (1. - 2. * nu))
        self.T2 = (k - 1) / (1. - 2. * nu)
        self.T3 = 12. * k / ((1. + nu) * (1. + nu))

    def __call__(self, eps):
        I1 = df.tr(eps)
        J2 = 0.5 * df.tr(df.dot(eps, eps)) - 1. / 6. * df.tr(eps) * df.tr(eps)

        A = (self.T2 * I1) ** 2 + self.T3 * J2
        A_pos = max(A, 1.e-14)

        return self.T1 * I1 + df.sqrt(A_pos) / (2. * self.k)





# class LoadDisplacementCurve:
#     def __init__(self, model, boundary, marker=6174):
#         mesh = model.mesh
#         self.model = model
#         boundary_markers = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
#
#         boundary.mark(boundary_markers, marker)
#         ds = df.Measure("ds", domain=mesh, subdomain_data=boundary_markers)
#
#         n = df.FacetNormal(mesh)
#         self.boundary_load = df.dot(model.traction(n), n) * ds(marker)
#         self.boundary_disp = df.dot(model.d, n) * ds(marker)
#         self.area = df.assemble(1.0 * ds(marker))
#
#         self.load = []
#         self.disp = []
#         self.ts = []
#
#         self.plot = None
#
#     def __call__(self, t):
#         load = df.assemble(self.boundary_load)
#         disp = df.assemble(self.boundary_disp) / self.area
#         self.load.append(load)
#         self.disp.append(disp)
#         self.ts.append(t)
#         if self.plot:
#             self.plot(disp, load)
#
#     def integrate(self):
#         return trapz(self.load, self.disp)
#
#     def show(self, fmt="-rx"):
#         from fenics_helpers.plotting import AdaptivePlot
#
#         self.plot = AdaptivePlot(fmt)
#
#     def keep_plot(self):
#         self.plot.keep()


mesh = l_panel_mesh(10, 10, refinement=5)
order = 1

# displacement field u
W = df.VectorFunctionSpace(mesh, "P", order)
u = df.Function(W)

# history data field k
W_k = df.FunctionSpace(mesh, "P", order)
k = df.Function(W_k)

# strain norm definition
eeq = ModifiedMises(1., Mat.nu)

# constitutive law stress = (1. - omega) * C(E, nu) * strain
strain = df.sym(df.grad(u))
lmbda = Mat.E * Mat.nu / (1. + Mat.nu) / (1. - 2. * Mat.nu)
mu = Mat.E / 2. / (1. + Mat.nu)
stress = 2 * mu * strain + lmbda * df.tr(strain) * df.Identity(2)

def update_history():
    k.assign(df.project(max(k, eeq(strain))))

# define boundary conditions using fh.boundary helpers
bot = fh.boundary.plane_at(0, "y") 
right = fh.boundary.plane_at(0, "x") 

bc_top_expr=df.Expression("du * t", du=0.0002, t=0, degree=0)

bc_bot = df.DirichletBC(W, [0, 0], bot)
bc_top = df.DirichletBC(W.sub(1), bc_top_expr, right)
bcs = [bc_bot, bc_top]


# define the potential and its derivatives
df.parameters["form_compiler"]["quadrature_degree"] = 3
v = df.TestFunction(W)
dd = df.TrialFunction(W)

Psi0 = 0.5 * df.inner(stress, strain) # elastic potential
Psi = (1. - omega(k)) * Psi0 *df.dx # damaged elastic potential

F = df.derivative(Psi, u, v)
J = df.derivative(F, u, dd)

problem = df.NonlinearVariationalProblem(F, u, bcs, J=J)

# configure the solver.
# The error_on_nonconvergence is crucial for the fh.timestepping.TimeStepper !
solver = df.NonlinearVariationalSolver(problem)
solver.parameters["nonlinear_solver"] = "snes"
solver.parameters["snes_solver"]["error_on_nonconvergence"] = False
solver.parameters["snes_solver"]["line_search"] = "bt"
solver.parameters["snes_solver"]["linear_solver"] = "mumps"
solver.parameters["snes_solver"]["maximum_iterations"] = 10

# ld = LoadDisplacementCurve(problem, right)
# ld.show()

# solver = problem.get_solver([bc_bot, bc_top])

def solve(t, dt):
    """
    `t` and `dt` are provided by the fh.TimeStepper
    """
    bc_top_expr.t=t
    return solver.solve()

def pp(t):
    """
    This is called after every successful solve
    """
    update_history()
    # plt.plot(problem.u.vector()[:])
    # ld(t)

fh.timestepping.TimeStepper(solve, pp, u=u).adaptive(2)
# ld.keep_plot()

