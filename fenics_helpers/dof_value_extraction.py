import dolfin
from ufl.core.expr import Expr


def _extract_dof_from_function(f):
    # Functions have direct access to their DoF values
    try:
        values = f.vector().get_local()
    # subfunctions need to be interpolated
    except RuntimeError:
        fspace = f.function_space()
        try:
            fspace = fspace.collapse()
        except RuntimeError:
            pass
        values = dolfin.interpolate(f, fspace).vector().get_local()

    return values


def extract_dof_values(expression):
    if isinstance(expression, dolfin.Function):
        return _extract_dof_from_function(expression)
    if isinstance(expression, Expr):
        return dolfin.project(expression).vector().get_local()
    else:
        raise RuntimeError("Can't extract values from {}".format(type(expression)))
