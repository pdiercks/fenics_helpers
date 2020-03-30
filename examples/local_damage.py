import numpy as np
import dolfin as df
import fenics_helpers as fh


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
    e.add_cell(2, [0, 3, 5])
    e.add_cell(3, [5, 3, 4])
    e.add_cell(4, [0, 5, 7])
    e.add_cell(5, [7, 5, 6])

    e.close
    mesh.order()

    for _ in range(refinement):
        mesh = df.refine(mesh)

    if show:
        df.plot(mesh)
        import matplotlib.pyplot as plt

        plt.show()

    return mesh


def max(a, b):
    return 0.5 * (a + b + abs(a - b))


class Mat:
    E = 20000
    nu = 0.20
    ft = 4.0
    Gf = 0.2
    alpha = 0.99


def omega(k):
    k0 = Mat.ft / Mat.E
    beta = Mat.ft / Mat.Gf
    exponent = beta * (k0 - k)
    omega = 1.0 - k0 / k * (1.0 - Mat.alpha + Mat.alpha * df.exp(exponent))
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
        self.T1 = (k - 1) / (2.0 * k * (1.0 - 2.0 * nu))
        self.T2 = (k - 1) / (1.0 - 2.0 * nu)
        self.T3 = 12.0 * k / ((1.0 + nu) * (1.0 + nu))

    def __call__(self, eps):
        I1 = df.tr(eps)
        J2 = 0.5 * df.tr(df.dot(eps, eps)) - 1.0 / 6.0 * df.tr(eps) * df.tr(eps)

        A = (self.T2 * I1) ** 2 + self.T3 * J2
        A_pos = max(A, 1.0e-14)

        return self.T1 * I1 + df.sqrt(A_pos) / (2.0 * self.k)


class Problem:
    def __init__(self, mat, mesh, order=1):
        self.mat = mat
        self.mesh = mesh

        self.W = df.VectorFunctionSpace(mesh, "P", order)
        self.W_k = df.FunctionSpace(mesh, "P", order)

        self.d = df.Function(self.W)
        # self.v = df.TestFunction(self.W)

        self.k = df.Function(self.W_k)

        self.eeq = ModifiedMises(10, mat.nu)

    def eps(self, d=None):
        if d is None:
            d = self.d
        return df.sym(df.grad(d))

    def sigma(self):
        eps, E, nu = self.eps(), self.mat.E, self.mat.nu
        lmbda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu)
        mu = E / 2.0 / (1.0 + nu)

        return 2 * mu * eps + lmbda * df.tr(eps) * df.Identity(2)

    def traction(self, n):
        return df.dot((1.0 - omega(self.k)) * self.sigma(), n)

    def kappa(self):
        return max(self.k, self.eeq(self.eps(self.d)))

    def update(self):
        self.k.assign(df.project(self.kappa()))

    def get_solver(self, bcs):
        df.parameters["form_compiler"]["quadrature_degree"] = 2

        v = df.TestFunction(self.W)
        F = df.inner(self.eps(v), (1.0 - omega(self.kappa())) * self.sigma()) * df.dx
        J = df.derivative(F, self.d)

        problem = df.NonlinearVariationalProblem(F, self.d, bcs, J=J)
        solver = df.NonlinearVariationalSolver(problem)
        solver.parameters["nonlinear_solver"] = "snes"
        solver.parameters["symmetric"] = True
        solver.parameters["snes_solver"]["error_on_nonconvergence"] = False
        solver.parameters["snes_solver"]["line_search"] = "bt"
        solver.parameters["snes_solver"]["linear_solver"] = "mumps"
        solver.parameters["snes_solver"]["maximum_iterations"] = 10
        solver.parameters["snes_solver"]["report"] = False

        return solver


class LoadDisplacementCurve:
    def __init__(self, model, boundary, marker=6174, direction=None):
        """
        Postprocessing class to track the tractions at `boundary` (actually 
        tractions . direction) and the displacements at the boundary.

        Each update to the plot can visualized using 
            fenics_helpers.plotting.AdaptivePlot
        """
        mesh = model.mesh
        self.model = model
        boundary_markers = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

        boundary.mark(boundary_markers, marker)
        ds = df.Measure("ds", domain=mesh, subdomain_data=boundary_markers)

        n = df.FacetNormal(mesh)
        if direction is None:
            direction = n
        self.boundary_load = df.dot(model.traction(n), direction) * ds(marker)
        self.boundary_disp = df.dot(model.d, direction) * ds(marker)
        self.area = df.assemble(1.0 * ds(marker))

        self.load = []
        self.disp = []

        self.plot = None

    def __call__(self, t):
        load = df.assemble(self.boundary_load)
        disp = df.assemble(self.boundary_disp) / self.area
        self.load.append(load)
        self.disp.append(disp)
        if self.plot is not None:
            self.plot(disp, load)

    def show(self, fmt="-rx"):
        self.plot = fh.plotting.AdaptivePlot(fmt)


class Plotter:
    def __init__(self, model, filename="out.xdmf"):
        self.model = model
        self.plot = df.XDMFFile(filename)
        self.plot.parameters["functions_share_mesh"] = True

        self.model.d.rename("u", "u")
        self.model.k.rename("kappa", "kappa")

    def __call__(self, t):
        self.plot.write(self.model.d, t)
        self.plot.write(self.model.k, t)


problem = Problem(Mat(), l_panel_mesh(10, 10, refinement=4), order=1)

# =====================================================
#   Select boundaries via the fh.boundary module
# =====================================================

bot = fh.boundary.plane_at(-10, "y")
right = fh.boundary.plane_at(10, "x")

bc_top_expr = df.Expression("du * t", du=0.5, t=0, degree=0)

bc_bot = df.DirichletBC(problem.W, [0, 0], bot)
bc_top_x = df.DirichletBC(problem.W.sub(0), 0, right)
bc_top_y = df.DirichletBC(problem.W.sub(1), bc_top_expr, right)

ld = LoadDisplacementCurve(problem, right, direction=df.Constant((0, 1)))
ld.show()
ld.plot.ax.set_ylabel("load [N]")
ld.plot.ax.set_xlabel("displacement [mm]")

plot_2d = Plotter(problem)

solver = problem.get_solver([bc_bot, bc_top_y])

# =====================================================
#   Define solve() and pp() to use by the
#     fh.timestepping module
# =====================================================


def solve(t, dt):
    bc_top_expr.t = t
    return solver.solve()


def pp(t):
    problem.update()
    ld(t)
    plot_2d(t)


ts = fh.timestepping.TimeStepper(solve, pp, u=problem.d)
ts.increase_num_iter = 7
ts.adaptive(1.0)
ld.plot.keep()
