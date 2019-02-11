import numpy as np
from dolfin import Function, info
import math


def term_color(color_name):
    if color_name == "red":
        return "\033[31m"
    if color_name == "green":
        return "\033[32m"


def colored(msg, color_name):
    return term_color(color_name) + msg + "\033[m"


class ProgressInfo:
    def __init__(self, t_start, t_end):
        return

    def iteration_info(self, t, dt, iterations):
        return "at t = {:8.5f} after {:2} iteration(s) with dt = {:8.5f}.".format(
            t, iterations, dt
        )

    def success(self, t, dt, iterations):
        msg = "Convergence " + self.iteration_info(t, dt, iterations)
        info(colored(msg, "green"))

    def error(self, t, dt, iterations):
        msg = "No convergence " + self.iteration_info(t, dt, iterations)
        info(colored(msg, "red"))


class ProgressBar:
    def __init__(self, t_start, t_end):
        from tqdm import tqdm

        fmt = "{l_bar}{bar}{rate_fmt}"
        self._pbar = tqdm(total=t_end - t_start, ascii=True, bar_format=fmt)

    def success(self, t, dt, iterations):
        self._pbar.update(dt)
        self._pbar.set_description("dt = {:8.5f}".format(dt))

    def error(self, t, dt, iterations):
        return

    def __del__(self):
        self._pbar.close()


def get_progress(t_start, t_end, show_bar):
    return ProgressBar(t_start, t_end) if show_bar else ProgressInfo(t_start, t_end)


class CheckPoints:
    def __init__(self, points, t_start, t_end):
        self.points = np.sort(np.array(points))

        if self.points.max() > t_end or self.points.min() < t_start:
            raise RuntimeError("Checkpoints outside of integration range.")

    def _first_checkpoint_within(self, t0, t1):
        id_range = (self.points > t0) & (self.points < t1)
        points_within_dt = self.points[id_range]
        if points_within_dt.size != 0:
            for point_within_dt in points_within_dt:
                if not math.isclose(point_within_dt, t0):
                    return point_within_dt

    def timestep(self, t, dt):
        """
        Searches a checkpoint t_check within [t, t+dt]. Picks the one
        with the lowest time if multiple are found. 
        If there is such a point, the dt = t_check - t is returned.
        If nothing is found, the unmodified dt is returned.
        """
        t_check = self._first_checkpoint_within(t, t + dt)
        if t_check:
            return t_check - t
        return dt


class Adaptive:
    def __init__(self, solve, post_process, u):
        assert isinstance(u, Function)
        self.dt_min = 1.0e-6
        self.dt_max = 0.1
        self.decrease_factor = 0.5
        self.increase_factor = 1.5
        self.increase_num_iter = 4
        self._solve = solve
        self._post_process = post_process
        self._u = u

    def run(self, t_end, t_start=0.0, dt=None, checkpoints=[], show_bar=False):
        if dt is None:
            dt = self.dt_max

        u_prev = self._u.copy(deepcopy=True)
        t = t_start
        self._post_process(t)

        progress = get_progress(t_start, t_end, show_bar)

        # Checkpoints are reached exactly. So we add t_end to the checkpoints.
        checkpoints = np.append(np.array(checkpoints), t_end)
        checkpoints = CheckPoints(checkpoints, t_start, t_end)

        dt0 = dt
        while t < t_end and not math.isclose(t, t_end):
            dt = checkpoints.timestep(t, dt0)
            # We keep track of two time steps. dt0 is the time step that
            # ignores the checkpoints. This is the one that is adapted upon
            # fast/no convergence. dt is smaller than dt0
            assert dt <= dt0
            # and coveres checkpoints.

            t += dt

            num_iter, converged = self._solve(t)
            assert isinstance(converged, bool)
            assert type(num_iter) == int  # isinstance(False, int) is True...

            if converged:
                progress.success(t, dt, num_iter)
                u_prev.assign(self._u)
                self._post_process(t)

                # increase the time step for fast convergence
                if dt == dt0 and num_iter < self.increase_num_iter and dt < self.dt_max:
                    dt0 *= self.increase_factor
                    dt0 = min(dt0, self.dt_max)
                    if not show_bar:
                        info("Increasing time step to dt = {}.".format(dt0))

            else:
                progress.error(t, dt, num_iter)

                self._u.assign(u_prev)
                t -= dt

                dt0 *= self.decrease_factor
                if not show_bar:
                    info("Reduce time step to dt = {}.".format(dt0))
                if dt0 < self.dt_min:
                    info("Abort since dt({}) < dt_min({})".format(dt0, self.dt_min))
                    return False
        return True


class Equidistant:
    def __init__(self, solve, post_process):
        self._solve = solve
        self._post_process = post_process

    def run(self, t_end, dt, t_start=0.0, checkpoints=[], show_bar=False):
        progress = get_progress(t_start, t_end, show_bar)

        if checkpoints:  # only for range checking 
            CheckPoints(checkpoints, t_start, t_end)
        points_in_time = np.append(np.arange(t_start, t_end, dt), t_end)
        points_in_time = np.append(points_in_time, checkpoints)

        for t in np.sort(np.unique(points_in_time)):
            num_iter, converged = self._solve(t)
            assert isinstance(converged, bool)

            if converged:
                progress.success(t, dt, num_iter)
                self._post_process(t)
            else:
                progress.error(t, dt, num_iter)
                return False
        return True


class TimeStepping(Adaptive):
    def adaptive(self, t_end, t_start=0.0, dt=None, checkpoints=[], show_bar=False):
        return self.run(t_end, t_start, dt, checkpoints, show_bar)

    def equidistant(self, t_end, dt, t_start=0.0, checkpoints=[], show_bar=False):
        e = Equidistant(self._solve, self._post_process)
        return e.run(t_end, dt, t_start, checkpoints, show_bar)
