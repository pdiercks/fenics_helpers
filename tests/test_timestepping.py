from context import fenics_helpers
import unittest

class TestTimeStepping(unittest.TestCase):

    def test_equidistant(self):
        solve = lambda t: (3, True)
        pp = lambda t: None
        equidistant = fenics_helpers.timestepping.EquidistantTimeStepping(solve, pp)
        self.assertTrue(equidistant.run(100.0, 1.0))

if __name__ == "__main__":
    unittest.main()
