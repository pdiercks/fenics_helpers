import unittest
import dolfin as d
from fenics_helpers import rk


class TestRungeKuttaWithHeatEq(unittest.TestCase):
    def setUp(self):
        d.set_log_level(d.LogLevel.WARNING)
        N = 50
        order = 2
        tF = 0.10
        # Dirichlet boundary characteristic time
        tau = tF / 10.0
        # time step
        h = 0.001

        self.num_steps = round(tF / h)

        # Mesh and Function space
        mesh = d.UnitIntervalMesh(N)
        V = d.FunctionSpace(mesh, "P", order)
        w = d.TestFunction(V)
        v = d.TrialFunction(V)

        # Initial conditions chosen such that the wave travels to the right
        uInit = d.Expression(
            "((1./3 < x[0]) && (x[0] < 2./3)) ? 0.5*(1-cos(2*pi*3*(x[0]-1./3))) : 0.",
            degree=2,
        )

        u = d.interpolate(uInit, V)  # values

        # Dirichlet boundary on the right with its derivatives
        g = d.Expression(
            "(t < total) ? 0.4*(1.-cos(2*pi*t/total))/2. : 0.",
            degree=2,
            t=0.0,
            total=tau,
        )
        dg = d.Expression(
            "(t < total) ? 0.4*(pi/total) * sin(2*pi*t/total) : 0.",
            degree=2,
            t=0.0,
            total=tau,
        )

        def updateBC(t):
            g.t = t
            dg.t = t

        def right(x, on_boundary):
            return on_boundary and d.near(x[0], 1.0)

        bc0 = d.DirichletBC(V, g, right)
        bc1 = d.DirichletBC(V, dg, right)
        self.bc = [bc0, bc1]

        L1 = -d.inner(d.grad(w), d.grad(u)) * d.dx
        L2 = w * v * d.dx
        self.parameters = (L1, L2, u, h, updateBC)

    def test_ERK(self):
        timeStepper = rk.ERKo1(rk.Euler, *self.parameters)
        for step in range(self.num_steps):
            timeStepper.do_step(bc=self.bc)

    def test_DIRK(self):
        timeStepper = rk.DIRKo1(rk.ImplicitEuler, *self.parameters)
        for step in range(self.num_steps):
            timeStepper.do_step(bc=self.bc)


class TestRungeKuttaWithWaveEq(unittest.TestCase):
    def setUp(self):
        d.set_log_level(d.LogLevel.WARNING)
        N = 100
        order = 2
        tF = 0.01
        tau = 0.2  # Dirichlet boundary characteristic time
        h = 0.001  # time step

        self.num_steps = round(tF / h)

        # Mesh and Function space
        mesh = d.UnitIntervalMesh(N)
        V = d.FunctionSpace(mesh, "P", order)
        w = d.TestFunction(V)
        a = d.TrialFunction(V)

        # Initial conditions chosen such that the wave travels to the right
        uInit = d.Expression(
            "((1./3 < x[0]) && (x[0] < 2./3)) ? 0.5*(1-cos(2*pi*3*(x[0]-1./3))) : 0.",
            degree=2,
        )
        vInit = d.Expression(
            "((1./3 < x[0]) && (x[0] < 2./3)) ? -pi*3 * (sin(2*pi*3*(x[0]-1./3))) : 0.",
            degree=2,
        )

        u = d.interpolate(uInit, V)  # values
        v = d.interpolate(vInit, V)  # velocities

        f = d.Expression(
            "(t < total) ? sin(2*pi*t/total) : 0.", degree=2, t=0.0, total=tF * 0.1
        )

        # Dirichlet boundary on the right with its derivatives
        g = d.Expression(
            "(t < total) ? 0.4*(1.-cos(2*pi*t/total))/2. : 0.",
            degree=2,
            t=0.0,
            total=tau,
        )
        dg = d.Expression(
            "(t < total) ? 0.4*(pi/total) * sin(2*pi*t/total) : 0.",
            degree=2,
            t=0.0,
            total=tau,
        )
        ddg = d.Expression(
            "(t < total) ? 0.4*(pi/total)*(2*pi/total) * cos(2*pi*t/total) : 0.",
            degree=2,
            t=0.0,
            total=tau,
        )

        def updateBC(t):
            f.t = t
            g.t = t
            dg.t = t
            ddg.t = t

        def right(x, on_boundary):
            return on_boundary and d.near(x[0], 1.0) 

        bc0 = d.DirichletBC(V, g, right)
        bc1 = d.DirichletBC(V, dg, right)
        bc2 = d.DirichletBC(V, ddg, right)
        self.bc = [bc0, bc1, bc2]

        L1 = -d.inner(d.grad(w), d.grad(u)) * d.dx + w * f * d.ds
        L2 = w * a * d.dx

        self.parameters = (L1, L2, u, v, h, updateBC)

    def test_ERK(self):
        timeStepper = rk.ERKo2(rk.ERK4Classic, *self.parameters)
        for step in range(self.num_steps):
            timeStepper.do_step(bc=self.bc)

    def test_DIRK(self):
        timeStepper = rk.DIRKo2(rk.CrankNicholson, *self.parameters)
        for step in range(self.num_steps):
            timeStepper.do_step(bc=self.bc)


if __name__ == "__main__":
    unittest.main()
