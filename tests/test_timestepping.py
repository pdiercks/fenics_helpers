import random
import unittest
import dolfin
import numpy as np

from context import fenics_helpers


class TestEquidistant(unittest.TestCase):
    def setUp(self):
        solve = lambda t: (3, True)
        pp = lambda t: None
        self.equidistant = fenics_helpers.timestepping.EquidistantTimeStepping(
            solve, pp
        )

    def test_equidistant(self):
        self.assertTrue(self.equidistant.run(100.0, 1.0))

    def test_equidistant_checkpoints(self):
        visited_timesteps = []
        pp = lambda t: visited_timesteps.append(t)
        self.equidistant._post_process = pp
        self.assertTrue(self.equidistant.run(5.0, 1.0, checkpoints=[2.5, 4.5]))
        self.assertListEqual(
            visited_timesteps, [0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 4.5, 5.0]
        )


class TestAdaptive(unittest.TestCase):
    def setUp(self):
        solve = lambda t: (3, random.choice([True, True, True, False]))
        pp = lambda t: None
        u = dolfin.Function(dolfin.FunctionSpace(dolfin.UnitIntervalMesh(10), "P", 1))
        self.adaptive = fenics_helpers.timestepping.AdaptiveTimeStepping(solve, pp, u)

    def test_adaptive(self):
        self.assertTrue(self.adaptive.run(1.5, 1.0))

    def test_adaptive_checkpoints(self):
        visited_timesteps = []
        pp = lambda t: visited_timesteps.append(t)
        self.adaptive._post_process = pp
        self.assertTrue(self.adaptive.run(1.5, dt0=1.0, checkpoints=[1.05, 1.25]))
        self.assertTrue(np.count_nonzero(np.isin(visited_timesteps, [1.05, 1.25])) == 2)


if __name__ == "__main__":
    unittest.main()
