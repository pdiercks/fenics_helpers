from context import fenics_helpers as fh
import unittest
import dolfin
import numpy as np


class TestDofExtraction(unittest.TestCase):
    def setUp(self):
        mesh = dolfin.UnitIntervalMesh(100)
        P1 = dolfin.FiniteElement("P", dolfin.interval, 1)
        fs = dolfin.FunctionSpace(mesh, P1)
        self.u = dolfin.Function(fs)

        mixed_element = dolfin.MixedElement([P1, P1, P1])
        mixed_fs = dolfin.FunctionSpace(mesh, mixed_element)
        self.mixed_u = dolfin.Function(mixed_fs)

    def test_function(self):
        u_vec = fh.extract_dof_values(self.u)
        self._is_numpy_array_of_size(u_vec, 101)

    def test_subspace_function(self):
        u1, u2, u3 = self.mixed_u.split()
        u2_values = fh.extract_dof_values(u2)
        self._is_numpy_array_of_size(u2_values, 101)

    def test_expression(self):
        my_expr = self.u ** 2 + self.u + dolfin.sin(self.u) + 73.0
        my_expr_values = fh.extract_dof_values(my_expr)
        self._is_numpy_array_of_size(my_expr_values, 101)

    def test_vector_expression(self):
        my_expr = dolfin.as_vector([self.u ** 2, self.u, dolfin.sin(self.u)])
        my_expr_values = fh.extract_dof_values(my_expr)
        self._is_numpy_array_of_size(my_expr_values, 3 * 101)

    def _is_numpy_array_of_size(self, array, size):
        self.assertIsInstance(array, np.ndarray)
        self.assertEqual(array.size, size)


if __name__ == "__main__":
    unittest.main()
