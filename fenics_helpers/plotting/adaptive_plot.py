import matplotlib.pyplot as plt
import numpy as np


class AdaptivePlot:
    def __init__(self, fmt="-bx", ax=None):
        plt.ion()  # interactive on
        if ax is None:
            ax = plt.gca()

        self._line, = ax.plot([], [], fmt)

    def __call__(self, x, y):
        """
        Add the point (x,y) to the plot and rescales the axes.
        """
        xs, ys = self._line.get_data()
        self._line.set_data(np.append(xs, x), np.append(ys, y))
        self.ax.relim()
        self.ax.autoscale_view(True, True, True)
        self.ax.figure.canvas.flush_events()

    def keep(self):
        """
        Deactivates the interactive mode to show the plot.        
        """
        plt.ioff()  # interactive off
        plt.show()

    @property
    def ax(self):
        return self._line.axes
