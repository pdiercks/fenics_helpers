import unittest
from context import fenics_helpers
from fenics_helpers import boundary

class TestBoundary(unittest.TestCase):
    def test_plane_at(self):
        b = boundary.plane_at(0, "Y")
        self.assertTrue(b.inside([0,0], True))
        self.assertTrue(b.inside([42,0,42], True))
        self.assertFalse(b.inside([0,1], True))
        self.assertFalse(b.inside([0,0], False))
       
        with self.assertRaises(Exception):
            b.inside([0], True)

    def test_within_range_1D(self):
        b = boundary.within_range(-5, -3)
        self.assertTrue(b.inside([-4], True))
        self.assertFalse(b.inside([42], True))

    def test_within_range_3D(self):
        with self.assertRaises(Exception):
            b = boundary.within_range([0,0], [1,1,1])
        with self.assertRaises(Exception):
            b = boundary.within_range([0,0,0], [1,1])
        with self.assertRaises(Exception):
            b = boundary.within_range(0, [1,1])

        b = boundary.within_range([0,1,0], [1,0,1])
        self.assertTrue(b.inside([0.5, 0.5, 0.5], True))
        self.assertFalse(b.inside([1.5, 0.5, 0.5], True))
        self.assertFalse(b.inside([0.5, 1.5, 0.5], True))
        self.assertFalse(b.inside([0.5, 0.5, 1.5], True))

    def test_point_at(self):
        b = boundary.point_at([0,0])
        self.assertTrue(b.inside([0,0], True))
        self.assertFalse(b.inside([0,0], False))
        self.assertFalse(b.inside([1,0], True))

if __name__ == "__main__":
    unittest.main()
