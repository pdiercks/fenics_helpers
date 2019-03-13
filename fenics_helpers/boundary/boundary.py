from dolfin import SubDomain

def plane_at(coordinate, dim=0, eps=1.e-10):
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
            return on_boundary and near(x[dim], value, eps)

    return B()



def within_range(start, end, eps=1.0e-10):
    """
    Creates a dolfin.SubDomain that only contains boundaries
    that lay within `start` << x << `end`. `start` and `end`
    must match the spatial dimension and may contain multiple
    values. 
    """
    if isnumber(start):
        start = [start]
    
    if isnumber(end):
        start = [end]

    assert len(start) == len(end)
    for i in range(len(start)):
        assert start[i] <= end[i]

    class B(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            b = on_boundary

            assert len(start) == len(end) == len(x)
            for i in range(len(x)):
                b = b and start[i] - eps < x[i] < end[i] + eps
            return b

    return B()

def point_at(p, eps=1.0e-10):
    """
    Creates a dolfin.SubDomain that only contains boundary points that 
    are near `p` (within a tolerance `eps`).
    """
    return within_range(p, p, eps)

