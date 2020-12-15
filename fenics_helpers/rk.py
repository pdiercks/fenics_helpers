# -*- coding: utf-8 -*-
"""Runge Kutta methods."""
import numpy as np
from numpy import array
from numpy.lib.scimath import sqrt

import dolfin as d
import ufl


def _change_stepsize(h, err, q, f=0.8, fmin=0.1, fmax=5.0, stepMin=0, stepMax=np.Inf):
    """
    Args:
        h: current stepsize
        err: error indicator estimatedError/allowedError
        q: theoretical convergence order
        fmin: minimal relative change
        fmax: maximal relative change
        f: safety factor: hnew = f*hoptimal
        stepMin: minimal absolute stepsize
        stepMin: maximal absolute stepsize
    Returns:
        float: changed stepsize
    """
    hopt = h * (1.0 / err) ** (1.0 / (q + 1.0))
    hmin = max(fmin * hopt, stepMin)
    hmax = min(fmax * hopt, stepMax)
    hnew = max(min(hmax, f * hopt), hmin)
    return hnew


class RKo1:
    """
    An explicit Runge Kutta method for differential equations
    of 1st order (e.g. the heat equation) base class
    """

    def __init__(self, bt, L1, L2, u, h, update=lambda t: None):
        """
        Args:
            bt: Butcher tableau
            L1: Right hand side of the equation
               expected to be a 1-form
            L2: Left hand side of the equation
               expected to be a 2-form
            u: Function representing the values
            h: stepSize
            update: a function f(t) that updates relevant expressions
        """
        self.bt = bt

        # error control for embedded methods
        try:
            self.b_corrector = bt.b_corrector
        except AttributeError:
            self.b_corrector = None
        self.order = bt.order

        self.num_stages = len(bt.c)
        self.u = u
        self.h = d.Constant(h)
        self.update = update
        self.t = 0.0

    def solve_stages(self, bc):
        """
        Different for implicit and explicit
        """
        raise Exception("To be implemented in derived classes")

    def next_step(self):
        """
        Returns next step from members.
        """
        h = self.h.values()[0]
        x1 = self.u.vector()
        for j in range(self.num_stages):
            x1 += h * self.bt.b[j] * self.ll[j].vector()
        return x1

    def next_step_with_error_estimate(self):
        """
        Returns next step from members.

        Uses embedded method for error estimation.
        """
        h = self.h.values()[0]
        x1 = 0 + self.u.vector()
        x1_corr = 0 + self.u.vector()
        for j in range(self.num_stages):
            x1 += h * self.bt.b[j] * self.ll[j].vector()
            x1_corr += h * self.bt.b_corrector[j] * self.ll[j].vector()
        # Compare solution with corrector
        error = d.Vector(x1 - x1_corr)
        error.abs()

        return x1, error

    def try_adaptive_step(self, tolA, tolR, bc=None):
        """
        Tries a RK step advancing values and velocities from self.t -> self.t+h.

        Note: Changes the stepsize.
        """
        self.solve_stages(bc)
        h = self.h.values()[0]
        x1, eX = self.next_step_with_error_estimate()

        s1 = tolA + np.absolute(self.u.vector()[:]) * tolR
        s2 = tolA + np.absolute(x1[:]) * tolR

        e = eX[:]
        err = max((e / s1).max(), (e / s2).max())

        if err <= 1.0:  # Step accepted
            step_accepted = True
            # Advance solution
            self.u.vector().set_local(x1.get_local())
            self.t += h
        else:
            step_accepted = False

        # Change stepsize
        self.h.assign(_change_stepsize(h, err, max(self.order)))

        return step_accepted

    def do_step(self, bc=None):
        """
        Does a RK step advancing values and velocities from t -> t+h.
        """
        self.solve_stages(bc)
        x1 = self.next_step()
        self.u.vector().set_local(x1.get_local())
        self.t += self.h.values()[0]


class DIRKo1(RKo1):
    """
    An diagonally implicit Runge Kutta method for differential equations
    of 1st order, e.g. the heat equation.
    """

    @staticmethod
    def create_intermediate_forms(bt, L1, L2, u, h):
        num_stages = len(bt.c)

        test_func = L2.arguments()[0]
        trial_func = L2.arguments()[1]
        V = test_func.function_space()
        ll = [d.Function(V) for i in range(num_stages)]
        LL = []

        for stage in range(num_stages):
            xs = u
            for j in range(num_stages):
                xs += h * bt.A[stage, j] * ll[j]
            Ls1 = ufl.replace(L1, {u: xs})
            Ls2 = ufl.replace(L2, {u: xs, trial_func: ll[stage]})
            Ls = Ls1 - Ls2
            LL.append(Ls)
        return [LL, ll]

    def __init__(self, bt, L1, L2, u, h, update=lambda t: None):
        """
        Args:
            bt: Butcher tableau
            L1: Right hand side of the equation
               expected to be a 1-form
            L2: Left hand side of the equation
               expected to be a 2-form
            u: Function representing the values
            h: stepSize
            update: a function f(t) that updates relevant expressions
        """
        # **** Test if explicit method (bt lower triangular) ****
        rows, columns = np.shape(bt.A)
        if not rows == columns:
            raise Exception("Butcher Table A has wrong shape")
        for i in range(rows):
            for j in range(i + 2, columns):
                if not bt.A[i, j] == 0.0:
                    raise Exception("Only diagonally implicit methods supported.")
        # **** DIRK method. Good. ****
        super(DIRKo1, self).__init__(bt, L1, L2, u, h, update=update)
        self.LL, self.ll = self.create_intermediate_forms(bt, L1, L2, u, self.h)

    def solve_stages(self, bc):
        """
        Solves intermediate steps.

        Uses a full mass matrix constructed in init.
        """
        # TODO don't use full mass matrix
        for stage in range(self.num_stages):
            ti = self.t + self.bt.c[stage] * self.h.values()[0]
            self.update(ti)

            try:
                l = len(bc)
            except TypeError:
                l = 0
                bcs = bc
            if l == 2:
                bcDict0 = bc[0].get_boundary_values()
                bcs = bc[1]
                for k in bcDict0.keys():
                    self.u.vector()[k] = bcDict0[k]

            d.solve(self.LL[stage] == 0, self.ll[stage], bcs=bcs)


class ERKo1(RKo1):
    """
    An explicit Runge Kutta method for differential equations
    of 1st order, e.g. the heat equation.
    """

    @staticmethod
    def create_intermediate_forms(bt, L1, L2, u, h):
        num_stages = len(bt.c)

        test_func = L1.arguments()[0]
        V = test_func.function_space()
        ll = [d.Function(V) for i in range(num_stages)]
        LL = []

        for stage in range(num_stages):
            xs = u
            for j in range(num_stages):
                xs += h * bt.A[stage, j] * ll[j]
            Ls = ufl.replace(L1, {u: xs})
            LL.append(Ls)
        return [LL, ll]

    def __init__(self, bt, L1, L2, u, h, update=lambda t: None):
        """
        Args:
            bt: Butcher tableau
            L1: Right hand side of the equation
               expected to be a 1-form
            L2: Left hand side of the equation
               expected to be a 2-form
            u: Function representing the values
            h: stepSize
            update: a function f(t) that updates relevant expressions
        """
        # **** Test if explicit method (bt lower triangular) ****
        rows, columns = np.shape(bt.A)
        if not rows == columns:
            raise Exception("Butcher Table A has wrong shape")
        for i in range(rows):
            for j in range(i + 1, columns):
                if not bt.A[i, j] == 0.0:
                    raise Exception("Only explicit methods supported.")
        # **** Explicit method. Good. ****
        super(ERKo1, self).__init__(bt, L1, L2, u, h, update=update)
        self.LL, self.ll = self.create_intermediate_forms(bt, L1, L2, u, self.h)
        self.L2 = L2
        self.b = None

    def solve_stages(self, bc):
        """
        Solves intermediate steps.

        Uses a full mass matrix constructed in init.
        """
        # TODO don't use full mass matrix
        for stage in range(self.num_stages):
            ti = self.t + self.bt.c[stage] * self.h.values()[0]
            self.update(ti)

            try:
                l = len(bc)
            except TypeError:
                l = 0
                bcs = bc
            if l == 2:
                bcDict0 = bc[0].get_boundary_values()
                bcs = bc[1]
                for k in bcDict0.keys():
                    self.u.vector()[k] = bcDict0[k]

            rhs = self.LL[stage]
            self.b = d.assemble(rhs, tensor=self.b)
            if bcs:
                bcs.apply(self.b)
            try:
                self.solver.solve(self.A, self.ll[stage].vector(), self.b)
            except AttributeError:
                self.A = d.assemble(self.L2)
                if bcs:
                    bcs.apply(self.b)
                self.solver = d.LUSolver(self.A, method="mumps")
                self.solver.parameters["symmetric"] = True
                self.solver.solve(self.A, self.ll[stage].vector(), self.b)


class RKo2:
    """
    Runge Kutta methods for differential equations
    of 2nd order in time (e.g. the wave equation) base  class
    """

    def __init__(self, bt, L1, L2, u, v, h, update=lambda t: None):
        """
        Args:
            bt: Butcher tableau
            L1: Right hand side of the equation
               expected to be a 1-form
            L2: Left hand side of the equation
               expected to be a 2-form
            u: Function representing the values
            v: Function representing the velocities
            h: stepSize
            update: a function f(t) that updates relevant expressions
        """
        # create 2nd order Butcher tableau
        self.bt_a1 = bt.A
        self.bt_a2 = np.dot(bt.A, bt.A)
        self.bt_b1 = bt.b
        self.bt_c = bt.c
        self.bt_b2 = np.dot(bt.b, bt.A)

        # error control for embedded methods
        try:
            b_corr = bt.b_corrector
            self.b_corrector1 = b_corr
            self.b_corrector2 = np.dot(b_corr, bt.A)
        except AttributeError:
            self.b_corrector1 = None
            self.b_corrector2 = None
        self.order = bt.order

        self.num_stages = len(bt.c)
        self.u = u
        self.v = v
        self.h = d.Constant(h)
        self.update = update
        self.t = 0.0

    def solve_stages(self, bc):
        """
        Different for implicit and explicit
        """
        raise Exception("To be implemented in derived classes")

    def next_step(self):
        """
        Returns next step.
        """
        h = self.h.values()[0]

        u = self.u.vector()
        v = self.v.vector()

        x1 = u + np.sum(self.bt_b1) * h * v
        y1 = 0 + v
        for j in range(self.num_stages):
            x1 += h ** 2 * self.bt_b2[j] * self.ll[j].vector()
            y1 += h * self.bt_b1[j] * self.ll[j].vector()
        return x1, y1

    def next_step_with_error_estimate(self):
        """
        Returns next step.

        Uses embedded method for error estimation.
        """
        u = d.Vector(self.u.vector())
        v = d.Vector(self.v.vector())

        h = self.h.values()[0]
        # Compute solution including embedded step
        x1 = u + np.sum(self.bt_b1) * h * v
        y1 = 0 + v
        x1_corr = u + np.sum(self.b_corrector1) * h * v
        y1_corr = 0 + v
        for j in range(self.num_stages):
            x1 += h ** 2 * self.bt_b2[j] * self.ll[j].vector()
            y1 += h * self.bt_b1[j] * self.ll[j].vector()
            x1_corr += h ** 2 * self.b_corrector2[j] * self.ll[j].vector()
            y1_corr += h * self.b_corrector1[j] * self.ll[j].vector()
        # Compare solution with corrector
        errorX = d.Vector(x1 - x1_corr)
        errorY = d.Vector(y1 - y1_corr)

        errorX.abs()
        errorY.abs()

        return x1, y1, errorX, errorY

    def try_adaptive_step(self, tolA, tolR, bc=None):
        """
        Tries a RK step advancing values and velocities from self.t -> self.t+h.

        Note: Changes the stepsize.
        """
        self.solve_stages(bc)
        h = self.h.values()[0]
        x1, y1, eX, eY = self.next_step_with_error_estimate()

        allowedX1 = tolA + np.absolute(self.u.vector()[:]) * tolR
        allowedY1 = tolA + np.absolute(self.v.vector()[:]) * tolR
        allowedX2 = tolA + np.absolute(x1[:]) * tolR
        allowedY2 = tolA + np.absolute(y1[:]) * tolR

        e = np.r_[eX[:], eY[:]]
        s1 = np.r_[allowedX1, allowedY1]
        s2 = np.r_[allowedX2, allowedY2]
        err = max((e / s1).max(), (e / s2).max())

        if err <= 1.0:  # Step accepted
            step_accepted = True
            # Advance solution
            self.u.vector().set_local(x1.get_local())
            self.v.vector().set_local(y1.get_local())
            self.t += h
        else:
            step_accepted = False

        # Change stepsize
        self.h.assign(_change_stepsize(h, err, max(self.order)))

        return step_accepted

    def do_step(self, bc=None):
        """
        Does a RK step advancing values and velocities from t -> t+h.
        """
        self.solve_stages(bc)
        x1, y1 = self.next_step()
        self.u.vector().set_local(x1.get_local())
        self.v.vector().set_local(y1.get_local())
        self.t += self.h.values()[0]


class ERKo2(RKo2):
    """
    An explicit Runge Kutta method for differential equations
    of 2nd order, e.g. the wave equation.
    """

    @staticmethod
    def create_intermediate_forms(bt, L1, L2, u, v, h):
        num_stages = len(bt.c)
        bt_a1 = bt.A
        bt_a2 = np.dot(bt.A, bt.A)

        test_func = L1.arguments()[0]
        V = test_func.function_space()
        ll = [d.Function(V) for i in range(num_stages)]
        LL = []

        for stage in range(num_stages):
            xs = u + np.sum(bt_a1, axis=1)[stage] * h * v
            ys = v
            for i in range(num_stages):
                xs += h ** 2 * bt_a2[stage, i] * ll[i]
                ys += h * bt_a1[stage, i] * ll[i]
            Ls = ufl.replace(L1, {u: xs, v: ys})
            LL.append(Ls)
        return [LL, ll]

    def __init__(self, bt, L1, L2, u, v, h, update=lambda t: None):
        """
        Args:
            bt: Butcher tableau
            L1: Right hand side of the equation
               expected to be a 1-form
            L2: Left hand side of the equation
               expected to be a 2-form
            u: Function representing the values
            v: Function representing the velocities
            h: stepSize (needed for adaptivity)
            update: a function f(t) that updates relevant expressions
        """
        # **** Test if explicit method (bt lower triangular) ****
        rows, columns = np.shape(bt.A)
        if not rows == columns:
            raise Exception("Butcher Table A has wrong shape")
        for i in range(rows):
            for j in range(i + 1, columns):
                if not bt.A[i, j] == 0.0:
                    raise Exception("Only explicit methods supported.")
        # **** Explicit method. Good. ****
        super(ERKo2, self).__init__(bt, L1, L2, u, v, h, update=update)
        self.LL, self.ll = self.create_intermediate_forms(bt, L1, L2, u, v, self.h)
        self.L2 = L2
        self.b = None

    def solve_stages(self, bc):
        """
        Solves intermediate steps.

        Uses a full mass matrix constructed in init.
        """
        # TODO don't use full mass matrix
        for stage in range(self.num_stages):
            ti = self.t + self.bt_c[stage] * self.h.values()[0]
            self.update(ti)

            try:
                l = len(bc)
            except TypeError:
                l = 0
                bcs = bc
            if l == 3:
                bcDict0 = bc[0].get_boundary_values()
                bcDict1 = bc[1].get_boundary_values()
                bcs = bc[2]
                for k in bcDict0.keys():
                    self.u.vector()[k] = bcDict0[k]
                for k in bcDict1.keys():
                    self.v.vector()[k] = bcDict1[k]

            rhs = self.LL[stage]
            self.b = d.assemble(rhs, tensor=self.b)
            if bcs:
                bcs.apply(self.b)
            try:
                self.solver.solve(self.A, self.ll[stage].vector(), self.b)
            except AttributeError:
                self.A = d.assemble(self.L2)
                if bcs:
                    bcs.apply(self.A)
                self.solver = d.LUSolver(self.A, method="mumps")
                self.solver.parameters["symmetric"] = True
                self.solver.solve(self.A, self.ll[stage].vector(), self.b)


class DIRKo2(RKo2):
    """
    An diagonally implict Runge Kutta method for differential equations
    of 2nd order, e.g. the wave equation.
    """

    @staticmethod
    def create_intermediate_forms(bt, L1, L2, u, v, h):
        num_stages = len(bt.c)
        bt_a1 = bt.A
        bt_a2 = np.dot(bt.A, bt.A)

        L = L1 - L2
        test_func = L.arguments()[0]
        trial_func = L.arguments()[1]
        V = test_func.function_space()
        ll = [d.Function(V) for i in range(num_stages)]
        LL = []

        for stage in range(num_stages):
            xs = u + np.sum(bt_a1, axis=1)[stage] * h * v
            ys = 1 * v
            for i in range(num_stages):
                xs += h ** 2 * bt_a2[stage, i] * ll[i]
                ys += h * bt_a1[stage, i] * ll[i]
            Ls = ufl.replace(L, {u: xs, v: ys, trial_func: ll[stage]})
            LL.append(Ls)
        return [LL, ll]

    def __init__(self, bt, L1, L2, u, v, h, update=lambda t: None):
        """
        Args:
            bt: Butcher tableau
            L1: Right hand side of the equation
               expected to be a 1-form
            L2: Left hand side of the equation
               expected to be a 2-form
            u: Function representing the values
            v: Function representing the velocities
            h: stepSize (needed for adaptivity)
            update: a function f(t) that updates relevant expressions
        """
        # **** Test if explicit method (bt lower triangular) ****
        rows, columns = np.shape(bt.A)
        if not rows == columns:
            raise Exception("Butcher Table A has wrong shape")
        for i in range(rows):
            for j in range(i + 2, columns):
                if not bt.A[i, j] == 0.0:
                    raise Exception("Only diagonally implicit methods supported.")
        # **** DIRK method. Good. ****
        super(DIRKo2, self).__init__(bt, L1, L2, u, v, h, update=update)
        self.LL, self.ll = self.create_intermediate_forms(bt, L1, L2, u, v, self.h)

    def solve_stages(self, bc):
        """
        Solves intermediate steps.

        Uses a full mass matrix constructed in init.

        Args:
            bc: boundary conditions
        """
        for stage in range(self.num_stages):
            ti = self.t + self.bt_c[stage] * self.h.values()[0]
            self.update(ti)

            try:
                l = len(bc)
            except TypeError:
                l = 0
                bcs = bc
            if l == 3:
                bcDict0 = bc[0].get_boundary_values()
                bcDict1 = bc[1].get_boundary_values()
                bcs = bc[2]
                for k in bcDict0.keys():
                    self.u.vector()[k] = bcDict0[k]
                for k in bcDict1.keys():
                    self.v.vector()[k] = bcDict1[k]

            d.solve(self.LL[stage] == 0, self.ll[stage], bcs=bcs)


# Butcher tables
# (keep their format)
# fmt: off
# pylint: disable=bad-whitespace,line-too-long

class ERK4Classic:
    A = array([[0.,   0.,   0.,   0.],
               [0.5,  0.,   0.,   0.],
               [0.,   0.5,  0.,   0.],
               [0.,   0.,   1.,   0.]])
    b = array([1./6.,1./3.,1./3.,1./6.])
    c = array([0.   ,0.5  ,0.5  ,1.   ])
    order = 4

class ERK4_38:
    A = array([[0.,     0.,   0.,   0.],
               [ 1./3,  0.,   0.,   0.],
               [-1./3., 1.,   0.,   0.],
               [ 1.,   -1.,   1.,   0.]])
    b = array([1./8.,3./8.,3./8.,1./8.])
    c = array([0.   ,1./3  ,2./3  ,1.   ])
    order = 4

class ERK3Heun:
    A = array([[0.,     0.,    0.],
               [1./3,   0.,    0.],
               [0.,     2./3,  0.]])
    b = array([ 1./4,   0.,    3./4])
    c = array([0.   ,1./3,     2./3])
    order = 3

class BogackiShampine32:
    A = array([[0.,     0.,   0.,   0.],
               [0.5,    0.,   0.,   0.],
               [0. ,  0.75,   0.,   0.],
               [2./9, 1./3, 4./9,   0.]])
    b = array([ 2./9, 1./3, 4./9,   0.])
    b_corrector \
      = array([7./24, 1./4, 1./3, 1./8])
    c = array([0.   ,0.5, 3./4, 1.])
    order = [3,2]

class ERK2:
    A = array([[0.,   0.],
               [0.5,  0.]])
    b = array([ 0.,   1.])
    c = array([0.   ,0.5])
    order = 2

class Euler:
    A = array([[0.]])
    b = array([ 1.])
    c = array([0.])
    order = 1

class Fehlberg45:
    """
    Embedded RK order 4; corrector step has order 5.

    Note: from Hairer and Wanner 1
    """
    A = array([[0.,         0.,          0.,          0.,         0.,      0.],
               [1./4,       0.,          0.,          0.,         0.,      0.],
               [3./32,      9./32,       0.,          0.,         0.,      0.],
               [1932./2197, -7200./2197, 7296./2197,  0.,         0.,      0.],
               [439./216,   -8.,         3680./513,   -845./4104, 0.,      0.],
               [-8./27,     2.,          -3544./2565, 1859./4104, -11./40, 0.]])
    b =          array([ 25./216, 0., 1408./2565,   2197./4104,    -1./5,  0.   ])
    b_corrector = array([16./135,  0., 6656./12825, 28561./56430, -9./50, 2./55])
    c = array([0., 1./4, 3./8, 12./13, 1., 1./2])
    order = [4,5]

class Fehlberg78:
    """
    Embedded RK order 7; corrector step has order 8.

    Note: from Hairer and Wanner 1
    """
    A = array([[0.,         0.,          0.,          0.,          0.,       0.,          0.,     0.,       0.,     0.,          0.,          0., 0.],
               [2./27,      0.,          0.,          0.,          0.,       0.,          0.,     0.,       0.,     0.,          0.,          0., 0.],
               [1./36,   1./12,          0.,          0.,          0.,       0.,          0.,     0.,       0.,     0.,          0.,          0., 0.],
               [1./24,      0.,       1./8 ,          0.,          0.,       0.,          0.,     0.,       0.,     0.,          0.,          0., 0.],
               [5./12,      0.,     -25./16,      25./16,          0.,       0.,          0.,     0.,       0.,     0.,          0.,          0., 0.],
               [1./20,      0.,          0.,        1./4,        1./5,       0.,          0.,     0.,       0.,     0.,          0.,          0., 0.],
               [-25./108.,  0.,          0.,    125./108,     -65./27,  125./54,          0.,     0.,       0.,     0.,          0.,          0., 0.],
               [31./300.,   0.,          0.,          0.,     61./225,    -2./9,     13./900,     0.,       0.,     0.,          0.,          0., 0.],
               [2.,         0.,          0.,      -53./6,     704./45,  -107./9,      67./90,     3.,       0.,     0.,          0.,          0., 0.],
               [-91./108.,  0.,          0.,      23./108,  -976./135,  311./54,     -19./60,  17./6,   -1./12,     0.,          0.,          0., 0.],
               [2383./4100, 0.,          0.,    -341./164, 4496./1025, -301./82,  2133./4100, 45./82,  45./164, 18./41,          0.,          0., 0.],
               [3./205,     0.,          0.,           0.,         0.,   -6./41,     -3./205, -3./41,    3./41,  6./41,          0.,          0., 0.],
               [-1777./4100, 0.,          0.,    -341./164, 4496./1025, -289./82,  2193./4100, 51./82,  33./164, 12./41,          0.,          1., 0.]])
    b =  array([41./840,    0.,          0.,          0.,          0.,  34./105,       9./35,  9./35,   9./280, 9./280,     41./840,          0., 0.])
    b_corrector \
      =   array([0.,        0.,          0.,          0.,          0.,  34./105,       9./35,  9./35,   9./280, 9./280,           0.,   41./840, 41./840])
    c = array([0.,      2./27.,        1./9,        1./6,       5./12,     1./2,        5./6,   1./6,     2./3,   1./3,           1.,          0., 1.])
    order = [7,8]

class DVERK65:
    """
    Embedded RK order 6; corrector step has order 5.

    Note: from Hairer and Wanner 1
    """
    A = array([[0.,                0.,           0.,          0.,        0.,           0.,     0.,     0.],
               [1./6,              0.,           0.,          0.,        0.,           0.,     0.,     0.],
               [4./75,         16./75,           0.,          0.,        0.,           0.,     0.,     0.],
               [5./6.,          -8./3,         5./2,          0.,        0.,           0.,     0.,     0.],
               [-165./64,       55./6,     -425./64,      85./96,        0.,           0.,     0.,     0.],
               [12./5,            -8.,    4015./612,     -11./36,  88./255.,           0.,     0.,     0.],
               [-8263./15000, 124./75,    -643./680,    -81./250, 2484./10625,         0.,     0.,     0.],
               [3501./1720,  -300./43, 297275./52632,   -319./2322, 24068./84065,      0.,  3850./26703.,   0.]])
    b = array([3./40,              0.,     875./2244,       23./72,  264./1955,        0., 125./11592,     43./616])
    b_corrector \
      = array([13./160,            0.,    2375./5984,        5./16,   12./85,     3./44,     0.,     0.])
    c = array([0.,               1./6,         4./15,         2./3,     5./6,        1.,  1./15,     1.])
    order = [6,5]

class DOPRI54:
    """
    Embedded RK order 5; corrector step has order 4, uses FSAL (first same as last).

    Note: from Hairer and Wanner 1
    """
    A = array([[0.,          0.,           0.,          0.,        0.,           0.,     0.],
               [1./5,        0.,           0.,          0.,        0.,           0.,     0.],
               [3./40,       9./40,        0.,          0.,        0.,           0.,     0.],
               [44./45,      -56./15.,     32./9,       0.,        0.,           0.,     0.],
               [19372./6561, -25360./2187, 64448./6561, -212./729, 0.,           0.,     0.],
               [9017./3168,  -355./33,     46732./5247, 49./176,   -5103./18656, 0.,     0.],
               [35./384,     0.,           500./1113,   125./192., -2187./6784,  11./84, 0.]
               ])
    b = array([35./384, 0., 500./1113, 125./192, -2187./6784, 11./84, 0.])
    b_corrector = ([5179./57600, 0., 7571./16695, 393./640, -92097./339200, 187./2100, 1./40])
    c = array([0., 1./5, 3./10, 4./5, 8./9, 1., 1.])
    order = [5,4]


class Ny4:
    """
    Explicit Nystrom method.

    For use with second order explicit ODE without velocity dependence.
    Note: from Hairer and Wanner 1
    """
    A1 = None
    A2 = array([[0.,   0.,  0.],
                [1./8.,0.,  0.],
                [0.,   0.5, 0.]])
    b1 = array([1./6, 2./3., 1./6.])
    b2 = array([1./6, 1./3., 0.   ])
    c = array([0.,   1./2., 1.   ])
    order = 4

class Ny5:
    """
    Explicit Nystrom method.

    For use with second order explicit ODE without velocity dependence.
    Note: from Hairer and Wanner 1
    """
    A1 = None
    A2 = array([[ 0.,     0.,    0.,    0.],
                [ 1./50., 0.,    0.,    0.],
                [-1./27,  7./27, 0.,    0.],
                [0.3,-2./35,     9./35, 0.]])
    b1 = array([14.,125.,162.,35])/336.
    b2 = array([14.,100.,54.,0.])/336.
    c = array([0.,0.2,2./3.,1.])
    order = 5

class ImplicitEuler:
    A = array([[1.]])
    b = array([ 1.])
    c = array([1.])
    order = 1

class ImplicitMidpoint:
    A = array([[1./2]])
    b = array([ 1.])
    c = array([1./2])

class CrankNicholson:
    A = array([[0.,     0.],
               [1./2,   1./2]])
    b = array([ 1./2,   1./2])
    c = array([ 0.,     1.])
    order = 2

class SDIRK3:
    """
    Order 3 SDIRK method.
    """
    # two choices of g are possible
    g = (3.+sqrt(3.))/6
    #g = (3.-sqrt(3.))/6
    A = array([[g,       0.],
               [1.-2.*g, g ]
               ])
    b = array([ 1./2, 1./2])
    c = array([g, 1.-g])
    order = 3

class HammerHollingsworth4:
    """
    IRK, Gauss-Legendre based.
    """
    A = array([[1./4,                 1./4 - sqrt(3.)/6.],
               [1./4 + sqrt(3.)/6.,  1./4               ]])
    b = array([ 1./2, 1./2])
    c = array([1./2 - sqrt(3.)/6., 1./2 + sqrt(3.)/6.])
    order = 4

class KuntzmannButcher6:
    """
    IRK, Gauss-Legendre based.
    """
    A = array([[5./36,                2./9 - sqrt(15.)/15.,      5./36 - sqrt(15.)/30],
               [5./36 + sqrt(15.)/24, 2./9                 ,      5./36 - sqrt(15.)/24],
               [5./36 + sqrt(15.)/30, 2./9 + sqrt(15.)/15  ,      5./36]])
    b = array([ 5./18,   4./9.,    5./18])
    c = array([1./2 - sqrt(15.)/10., 1./2, 1./2 + sqrt(15.)/10])
    order = 6

class RadauIA3:
    """
    IRK, based on Radau (left) quadrature.
    """
    A = array([[1./4   ,     - 1./4],
               [1./4   ,       5./12]])
    b = array([ 1./4   ,       3./4])
    c = array([0.,         2./3])
    order = 3

class RadauIIA3:
    """
    IRK, based on Radau (right) quadrature.
    """
    A = array([[5./12   ,     - 1./12],
               [3./4   ,       1./4]])
    b = array([ 3./4   ,       1./4])
    c = array([1./3,         1.])
    order = 3

class RadauIA5:
    """
    IRK, based on Radau (left) quadrature.
    """
    A = array([[1./9      ,     (-1. - sqrt(6))/18.     ,  (-1. + sqrt(6))/18.],
               [1./9      ,     (88. +7*sqrt(6))/360.     ,  (88. - 43* sqrt(6))/360.],
               [1./9      ,     (88. +43*sqrt(6))/360.     ,  (88. - 7*sqrt(6))/360.]])
    b = array([1./9      ,     (16. +sqrt(6))/36.     ,  (16. - sqrt(6))/36.])
    c = array([   0      ,     (6. - sqrt(6))/10.     ,  (6. + sqrt(6))/10.])
    order = 5

class RadauIIA5:
    """
    IRK, based on Radau (right) quadrature.
    """
    A = array([[(88. - 7*sqrt(6))/360.      ,     (296. - 169*sqrt(6))/1800.     ,  (-2. + 3*sqrt(6))/225.],
               [(296. + 169*sqrt(6))/1800.      ,     (88. + 7*sqrt(6))/360.     ,  (-2. - 3*sqrt(6))/225.],
               [(16. - sqrt(6))/36.      ,     (16. + sqrt(6))/36.     ,  1./9]])
    b =  array([(16. - sqrt(6))/36.      ,     (16. + sqrt(6))/36.     ,  1./9])
    c = array([(4. - sqrt(6))/10.      ,     (4. + sqrt(6))/10.     ,  1.])
    order = 5

class LobattoIIIA2:
    """
    IRK, based on Lobatto quadrature.
    """
    A = array([[0.     ,     0.],
               [0.5    ,     0.5]])
    b =  array([0.5    ,     0.5])
    c =  array([0.    ,      1.])
    order = 2

class LobattoIIIB2:
    """
    IRK, based on Lobatto quadrature.
    """
    A = array([[0.5     ,     0.],
               [0.5    ,      0.]])
    b =  array([0.5    ,     0.5])
    c =  array([0.    ,      1.])
    order = 2

class LobattoIIIA4:
    """
    IRK, based on Lobatto quadrature.
    """
    A = array([[0.     ,     0.      ,   0.],
               [5./24. ,   1./3      , -1./24],
               [1./6. ,   2./3      ,  1./6]])
    b =  array([1./6. ,   2./3      ,  1./6])
    c =  array([0.    ,    1./2    ,   1.])
    order = 4

class LobattoIIIB4:
    """
    IRK, based on Lobatto quadrature.
    """
    A = array([[1./6.     ,     -1./6      ,   0.],
               [1./6. ,   1./3      , 0],
               [1./6. ,   5./6      , 0]])
    b =  array([1./6. ,   2./3      ,  1./6])
    c =  array([0.    ,    1./2    ,   1.])
    order = 4
