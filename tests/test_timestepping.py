import random
import unittest
import dolfin
import numpy as np
from hypothesis import given, reproduce_failure
import hypothesis.strategies as st


from context import fenics_helpers


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
        self.assertFalse(self.equidistant.run(5.0, 1.0))
    
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
            self.assertTrue(np.isclose(visited_timesteps, checkpoint, atol=1e-6).any())

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
        solve = lambda t: (3, random.choice([True, True, True, True, False]))
        pp = lambda t: None
        u = dolfin.Function(dolfin.FunctionSpace(dolfin.UnitIntervalMesh(10), "P", 1))
        self.adaptive = fenics_helpers.timestepping.Adaptive(solve, pp, u)

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
        for checkpoint in checkpoints:
            self.assertTrue(np.isclose(visited_timesteps, checkpoint, atol=self.adaptive.dt_min).any())

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
