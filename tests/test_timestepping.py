import random
import unittest
import dolfin
import numpy as np
from hypothesis import given, reproduce_failure
import hypothesis.strategies as st


from context import fenics_helpers


class Solver:
    def __init__(self):
        self.t = 0
        self.past_it = {}

    def __call__(self, t):
        dt = t - self.t
        key = (t, dt)
        value = self.past_it.get(key)
        if value:
            return 3, value

        value = random.choice([True, True, False])
        self.past_it[key] = value
        if value:
            self.t = t

        return 3, value


class TestEquidistant(unittest.TestCase):
    def setUp(self):
        solve = lambda t: (3, True)
        pp = lambda t: None
        self.equidistant = fenics_helpers.timestepping.Equidistant(
            solve, pp
        )

    def test_run(self):
        self.assertTrue(self.equidistant.run(5.0, 1.0))
    
    def test_run_failed(self):
        self.equidistant._solve = lambda t: (3, False)
        self.assertFalse(self.equidistant.run(100.0, 1.0, show_bar=True))
    
    def test_invalid_solve(self):
        self.equidistant._solve = lambda t: (3, "False")
        self.assertRaises(Exception, self.equidistant.run, 5.0, 1.0)

    @given(st.lists(st.floats(0.0, 5.0)))
    def test_checkpoints(self, checkpoints):
        visited_timesteps = []
        pp = lambda t: visited_timesteps.append(t)
        self.equidistant._post_process = pp
        self.assertTrue(self.equidistant.run(5.0, 1.0, checkpoints=checkpoints))
        for checkpoint in checkpoints:
            self.assertIn(checkpoint, visited_timesteps)

        self.assertIn(0., visited_timesteps)
        self.assertIn(5., visited_timesteps)

    @given(st.lists(st.floats(max_value=0.0, exclude_max=True), min_size=1))
    def test_too_low_checkpoints(self, checkpoints):
        with self.assertRaises(RuntimeError) as cm:
            self.equidistant.run(5.0, 1.0, checkpoints=checkpoints)

    @given(st.lists(st.floats(min_value=5.0, exclude_min=True), min_size=1))
    def test_too_high_checkpoints(self, checkpoints):
        with self.assertRaises(RuntimeError) as cm:
            self.equidistant.run(5.0, 1.0, checkpoints=checkpoints)


class TestAdaptive(unittest.TestCase):
    def setUp(self):
        s = Solver()
        solve = lambda t: (3, random.choice([True, True, True, True, False]))
        pp = lambda t: None
        u = dolfin.Function(dolfin.FunctionSpace(dolfin.UnitIntervalMesh(10), "P", 1))
        # self.adaptive = fenics_helpers.timestepping.Adaptive(solve, pp, u)
        self.adaptive = fenics_helpers.timestepping.Adaptive(s, pp, u)

    def test_run(self):
        self.assertTrue(self.adaptive.run(1.5, 1.0))
    
    def test_invalid_solve_first_str(self):
        self.adaptive._solve = lambda t: ("3", True)
        self.assertRaises(Exception, self.adaptive.run, 1.5, 1.0)

    def test_invalid_solve_first_bool(self):
        # You may switch arguments and put the bool first. We have to handle
        # this case since booleans are some kind of subclass of int.
        # So False > 5 will not produce errors.
        self.adaptive._solve = lambda t: (False, True)
        self.assertRaises(Exception, self.adaptive.run, 1.5, 1.0)

    def test_invalid_solve_second(self):
        self.adaptive._solve = lambda t: (3, "False")
        self.assertRaises(Exception, self.adaptive.run, 1.5, 1.0)


    @given(st.lists(st.floats(0.0, 1.5)))
    def test_checkpoints(self, checkpoints):
        visited_timesteps = []
        pp = lambda t: visited_timesteps.append(t)
        self.adaptive._post_process = pp
        self.assertTrue(self.adaptive.run(1.5, dt=1.0, checkpoints=checkpoints))
        eps = np.finfo(float).eps # eps float
        for checkpoint in checkpoints:
            self.assertTrue(np.isclose(visited_timesteps, checkpoint, atol=eps).any())
       
        self.assertTrue(np.isclose(visited_timesteps, 0, atol=eps).any())
        self.assertTrue(np.isclose(visited_timesteps, 1.5, atol=eps).any())

    @given(st.lists(st.floats(max_value=0.0, exclude_max=True), min_size=1))
    def test_too_low_checkpoints(self, checkpoints):
        with self.assertRaises(RuntimeError) as cm:
            self.adaptive.run(1.5, dt=1.0, checkpoints=checkpoints)

    @given(st.lists(st.floats(min_value=1.5, exclude_min=True), min_size=1))
    def test_too_high_checkpoints(self, checkpoints):
        with self.assertRaises(RuntimeError) as cm:
            self.adaptive.run(1.5, dt=1.0, checkpoints=checkpoints)


if __name__ == "__main__":
    unittest.main()
