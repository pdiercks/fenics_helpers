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
    e.add_cell(2, [0, 3, 4])
    e.add_cell(3, [0, 4, 5])
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
    E= 20000
    nu = 0.15
    ft = 4.
    Gf = 0.05
    alpha = 0.99
    l = 0.05

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

class Problem:
    def __init__(self, mat, mesh, order=1):
        self.mat = mat
        self.mesh = mesh

        vp = df.VectorElement("P", mesh.ufl_cell(), order)
        p = df.FiniteElement("P", mesh.ufl_cell(), order)

        mixed_element = vp * p
        self.W = df.FunctionSpace(mesh, mixed_element)
        self.W_d, self.W_e = self.W.split()
        self.W_k = df.FunctionSpace(mesh, "P", order)

        self.u = df.Function(self.W)
        self.d, self.e = df.split(self.u)
        self.v = df.TestFunction(self.W)
        self.v_d, self.v_e = df.TestFunctions(self.W)

        self.k = df.Function(self.W_k)

        self.eeq = ModifiedMises(10, mat.nu)

    def eps(self, d=None):
        if d is None:
            d = self.d
        return df.sym(df.grad(d))
    
    def sigma(self):
        eps, E, nu = self.eps(), self.mat.E, self.mat.nu
        lmbda = E * nu / (1. + nu) / (1. - 2. * nu)
        mu = E / 2. / (1. + nu)

        return 2 * mu * eps + lmbda * df.tr(eps) * df.Identity(self.mesh.geometric_dimension())

    def traction(self, n):
        return df.dot((1.-omega(self.k))*self.sigma(), n)

    def kappa(self):
        return max(self.k, self.e)

    def update(self):
        d, e = self.u.split(deepcopy=True)
        new_k = np.maximum(self.k.vector().get_local(), e.vector().get_local())
        self.k.vector().set_local(new_k)
        
    def get_solver(self, bcs):

        df.parameters["form_compiler"]["quadrature_degree"] = 2
        r_d = df.inner(self.eps(self.v_d), (1.-omega(self.kappa())) * self.sigma()) * df.dx
        r_e = self.v_e * (self.e - self.eeq(self.eps())) * df.dx + df.dot(df.grad(self.v_e), Mat.l**2 * df.grad(self.e)) * df.dx
        
        F = r_d + r_e
        J = df.derivative(F, self.u)

        problem = df.NonlinearVariationalProblem(F, self.u, bcs, J=J)
        solver = df.NonlinearVariationalSolver(problem)
        solver.parameters["nonlinear_solver"] = "snes"
        solver.parameters["snes_solver"]["error_on_nonconvergence"] = False
        solver.parameters["snes_solver"]["line_search"] = "bt"
        solver.parameters["snes_solver"]["linear_solver"] = "mumps"
        solver.parameters["snes_solver"]["maximum_iterations"] = 10

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
    def __init__(self, model, filename=None):
        self.model = model
        if filename is None:
            filename = "out.xdmf"
        self.plot = df.XDMFFile(filename)
        self.plot.parameters["functions_share_mesh"] = True

    def __call__(self, t):
        self.plot.write(self.model.u, t)
        self.plot.write(self.model.k, t)

problem = Problem(Mat(), l_panel_mesh(1, 1, refinement=4), order=2)

bot = fh.boundary.plane_at(-1, "y") 
right = fh.boundary.plane_at(1, "x") 

bc_top_expr=df.Expression("du * t", du=0.05, t=0, degree=0)

bc_bot = df.DirichletBC(problem.W_d, [0, 0], bot)
bc_top_x = df.DirichletBC(problem.W_d.sub(0), 0, right)
bc_top_y = df.DirichletBC(problem.W_d.sub(1), bc_top_expr, right)

ld = LoadDisplacementCurve(problem, right, direction=df.Constant((0, 1)))
ld.show()
plot_2d = Plotter(problem)

solver = problem.get_solver([bc_bot, bc_top_y])

def solve(t, dt):
    bc_top_expr.t=t
    return solver.solve()

def pp(t):
    problem.update()
    ld(t)
    plot_2d(t)

ts = fh.timestepping.TimeStepper(solve, pp, u=problem.u)
ts.increase_num_iter = 6
ts.adaptive(1)
ld.plot.keep()

