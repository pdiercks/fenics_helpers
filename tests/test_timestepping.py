import random
import unittest
import dolfin
import numpy as np
from hypothesis import given, reproduce_failure
import hypothesis.strategies as st


from context import fenics_helpers

class DeterministicSolve:
    """
    Returns the same (randomly created) boolean for the same combination of
    (t, dt).
    Keeps track of how many times the same value was requested to find
    infinite loops.
    """
    def __init__(self):
        self.memory = {}
        self.t = 0.
        self.same_value_counter = 0
        random.seed(6174)

    def set(self, dt, value):
        self.memory[(self.t, dt)] = value
    
    def get(self, dt):
        return self.memory.get((self.t, dt))

    def __call__(self, t):
        dt = t - self.t
        value = self.get(dt)

        if self.same_value_counter > 100:
            raise RuntimeError("Same value requested more than 100 times.")

        if value is not None:
            self.same_value_counter += 1
            return 3, value

        self.same_value_counter = 0

        value = random.choice([True, True, True, True, False])
        self.set(dt, value)
        if value == True:
            self.t += dt

        return 3, value


class TestEquidistant(unittest.TestCase):
    def setUp(self):
        solve = lambda t: (3, True)
        pp = lambda t: None
        self.equidistant = fenics_helpers.timestepping.Equidistant(solve, pp)

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

        self.assertIn(0.0, visited_timesteps)
        self.assertIn(5.0, visited_timesteps)

        # check for duplicates
        self.assertEqual(len(visited_timesteps), np.unique(visited_timesteps).size)

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
        solve = DeterministicSolve()
        # solve = lambda t: (3, random.choice([True, True, True, True, False]))
        pp = lambda t: None
        u = dolfin.Function(dolfin.FunctionSpace(dolfin.UnitIntervalMesh(10), "P", 1))
        self.adaptive = fenics_helpers.timestepping.Adaptive(solve, pp, u)

    def test_run(self):
        self.assertTrue(self.adaptive.run(1.5, 1.0))

    def test_run_nicely(self):
        visited_timesteps = []
        pp = lambda t: visited_timesteps.append(t)
        self.adaptive._post_process = pp
        self.adaptive._solve = lambda t: (7, True)
        self.assertTrue(self.adaptive.run(1.0))
        self.assertAlmostEqual(visited_timesteps[0], 0)
        self.assertAlmostEqual(visited_timesteps[-1], 1)

    def test_checkpoint_step_fails(self):
        cps = [0.5]
        self.assertEqual(self.adaptive._solve.t, 0)
        self.adaptive.dt_max = 1.
        # first time step 0.6 is bigger than the first checkpoint. 
        # So dt --> 0.5, dt0--> 0.6
        self.adaptive._solve.set(cps[0], False)
        self.adaptive.run(1.0, checkpoints=cps)


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

        eps = np.finfo(float).eps  # eps float
        for checkpoint in checkpoints:
            self.assertTrue(np.isclose(visited_timesteps, checkpoint, atol=eps).any())

        self.assertTrue(np.isclose(visited_timesteps, 0, atol=eps).any())
        self.assertTrue(np.isclose(visited_timesteps, 1.5, atol=eps).any())

        # check for duplicates
        self.assertEqual(len(visited_timesteps), np.unique(visited_timesteps).size)

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
