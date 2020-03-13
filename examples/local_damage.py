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


class LoadDisplacementCurve:
    def __init__(self, model, boundary, marker=6174, direction=None):
        mesh = model.mesh
        self.model = model
        boundary_markers = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

        boundary.mark(boundary_markers, marker)
        ds = df.Measure("ds", domain=mesh, subdomain_data=boundary_markers)

        n = df.FacetNormal(mesh)
        if direction is None:
            direction = n
        self.boundary_load = df.dot(model.traction(n), direction) * ds(marker)
        self.boundary_disp = df.dot(model.u, direction) * ds(marker)
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

class Problem:
    def __init__(self, mat, mesh, order=1):
        self.mat = mat
        self.mesh = mesh

        self.W_k = df.FunctionSpace(mesh, "P", order)
        self.k = df.Function(self.W_k)

        self.W = df.VectorFunctionSpace(mesh, "P", order)
        self.u = df.Function(self.W)

        self.eeq = ModifiedMises(1, mat.nu)

    def eps(self):
        return df.sym(df.grad(self.u))
    
    def sigma(self):
        eps = self.eps()
        E = self.mat.E
        nu = self.mat.nu
        lmbda = E * nu / (1. + nu) / (1. - 2. * nu)
        mu = E / 2. / (1. + nu)

        return 2 * mu * eps + lmbda * df.tr(eps) * df.Identity(self.mesh.geometric_dimension())

    def traction(self, n):
        return df.dot((1.-omega(self.k))*self.sigma(), n)

    def update(self):
        self.k.assign(df.project(max(self.k, self.eeq(self.eps()))))

    def get_solver(self, bcs):

        df.parameters["form_compiler"]["quadrature_degree"] = 3
        # elastic potential
        Psi0 = 0.5 * df.inner(self.sigma(), self.eps()) 
        # damaged elastic potential
        Psi = (1. - omega(self.k)) * Psi0 *df.dx 

        F = df.derivative(Psi, self.u, df.TestFunction(self.W))
        J = df.derivative(F, self.u, df.TrialFunction(self.W))

        problem = df.NonlinearVariationalProblem(F, self.u, bcs, J=J)
        solver = df.NonlinearVariationalSolver(problem)
        solver.parameters["nonlinear_solver"] = "snes"
        solver.parameters["snes_solver"]["error_on_nonconvergence"] = False
        solver.parameters["snes_solver"]["line_search"] = "bt"
        solver.parameters["snes_solver"]["linear_solver"] = "mumps"
        solver.parameters["snes_solver"]["maximum_iterations"] = 10

        return solver


problem = Problem(Mat(), l_panel_mesh(10, 10))

bot = fh.boundary.plane_at(0, "y") 
right = fh.boundary.plane_at(0, "x") 

bc_top_expr=df.Expression("du * t", du=0.02, t=0, degree=0)

bc_bot = df.DirichletBC(problem.W, [0, 0], bot)
bc_top = df.DirichletBC(problem.W.sub(1), bc_top_expr, right)

ld = LoadDisplacementCurve(problem, right, direction=df.Constant((0, 1)))
ld.show()

solver = problem.get_solver([bc_bot, bc_top])

def solve(t, dt):
    bc_top_expr.t=t
    return solver.solve()

def pp(t):
    problem.update()
    # plt.plot(problem.u.vector()[:])
    ld(t)

fh.timestepping.TimeStepper(solve, pp, u=problem.u).adaptive(2)
ld.plot.keep()

