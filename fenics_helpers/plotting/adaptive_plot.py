import matplotlib.pyplot as plt


class AdaptivePlot:
    def __init__(self, fmt="-bx", ax=None):
        plt.ion()  # interactive on
        if ax is None:
            ax = plt.gca()

        self._line, = ax.plot([0.0], [0.0], fmt)
        self._xs = []
        self._ys = []

    def __call__(self, x, y):
        """
        Add the point (x,y) to the plot and rescales the axes.
        """
        self._xs.append(x)
        self._ys.append(y)
        self._line.set_data(self._xs, self._ys)
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
