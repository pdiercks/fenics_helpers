from dolfin import SubDomain, near


def plane_at(coordinate, dim=0, eps=1.0e-10):
    """
    Creates a dolfin.SubDomain that only contains boundaries
    that lay on the plane where the `dim`-th dimension equals
    `coordinate` (within a tolerance `eps`).
    """
    if dim in ["x", "X"]:
        dim = 0
    if dim in ["y", "Y"]:
        dim = 1
    if dim in ["z", "Z"]:
        dim = 2

    assert dim in [0, 1, 2]

    class B(SubDomain):
        def inside(self, x, on_boundary):
            assert (
                len(x) > dim
            ), "You cannot constrain dimension {} in a {}D problem.".format(dim, len(x))
            return on_boundary and near(x[dim], coordinate, eps)

    return B()


def to_floats(values):
    floats = []
    try:
        for v in values:
            floats.append(float(v))
    except TypeError as e:
        floats = [float(values)]

    return floats


def within_range(start, end, eps=1.0e-10):
    """
    Creates a dolfin.SubDomain that only contains boundaries
    that lay within `start` << x << `end`. `start` and `end`
    must match the spatial dimension and may contain multiple
    values. 
    """
    start = to_floats(start)
    end = to_floats(end)

    # adjust the values such that start < end for all dimensions
    assert len(start) == len(end)
    for i in range(len(start)):
        if start[i] > end[i]:
            start[i], end[i] = end[i], start[i]

    class B(SubDomain):
        def inside(self, x, on_boundary):
            if not on_boundary:
                return False

            assert len(start) == len(x)

            for i in range(len(x)):
                if not start[i] - eps < x[i] < end[i] + eps:
                    return False

            return True

    return B()


def point_at(p, eps=1.0e-10):
    """
    Creates a dolfin.SubDomain that only contains boundary points that 
    are near `p` (within a tolerance `eps`).

    Corresponding FEniCS DirichletBCs have to be applied with
    method="pointwise", which somehow passes "on_boundary=False". Thus, we
    ignore the on_boundary argument.
    """

    class B(SubDomain):
        def inside(self, x, on_boundary):
            assert len(p) == len(x)
            return all(near(x_i, p_i, eps) for x_i, p_i in zip(x, p))

    return B()
