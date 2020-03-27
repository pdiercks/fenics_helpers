# fenics_helpers

[![Build Status](https://travis-ci.org/nutofem/fenics_helpers.svg?branch=master)](https://travis-ci.org/nutofem/fenics_helpers)
[![CircleCI](https://circleci.com/gh/nutofem/fenics_helpers/tree/master.svg?style=svg)](https://circleci.com/gh/nutofem/fenics_helpers/tree/master)

Little Helper classes that come up repeatedly when writing solvers in FEniCS


# Example: Local damage model

![local damage plot](examples/kappa_plot.png)

The [local damage](examples(local_damage.py) example illustrates the use of `.boundary` to conveniently select the boundaries like

~~~py
bot = fh.boundary.plane_at(-10, "y") 
right = fh.boundary.plane_at(10, "x") 
~~~

and `.timestepping` is used to reduce the time step size only at critical points in the simulation to accelerate the time integration.

![stuff](examples/load_displacement_curve.png)

