import time
import math

from context import fenics_helpers
from fenics_helpers.plotting import AdaptivePlot


def single_plot():
    p = AdaptivePlot("-gx", show_head=False)
    p.ax.set_xlabel("iteration")
    p.ax.set_ylabel("time [ms]")
    dt = 0.0
    for i in range(1000):
        t = time.time()
        p(i, 1000 * dt)
        dt = time.time() - t


def combined_plot():
    p1 = AdaptivePlot("-gx")
    p2 = AdaptivePlot("-rx", ax=p1.ax.twinx())

    for i in range(200):
        t = i / 10.0
        p1(t, math.sin(t))
        p2(t, math.cos(t))

    p1.keep()


if __name__ == "__main__":
    # single_plot()
    combined_plot()
