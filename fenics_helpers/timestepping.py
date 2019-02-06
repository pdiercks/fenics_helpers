import numpy as np
from dolfin import Function, info


class Progress:
    def __init__(self, t_start, t_end, show_bar):
        self._show_bar = show_bar

        self.TGREEN = "\033[32m"
        self.TRED = "\033[31m"
        self.TEND = "\033[m"

        if self._show_bar:
            from tqdm import tqdm

            self._pbar = tqdm(
                total=t_end - t_start, ascii=True, bar_format="{l_bar}{bar}{rate_fmt}"
            )

    def success(self, t, dt, iterations):
        if self._show_bar:
            self._pbar.update(dt)
            self._pbar.set_description("dt = {:8.5f}".format(dt))
        else:
            info(
                self.TGREEN
                + "Convergence at t = {:8.5f} after {:2} iteration(s) with dt = {:8.5f}.".format(
                    t, iterations, dt
                )
                + self.TEND
            )

    def error(self, t, dt):
        if not self._show_bar:
            info(
                self.TRED
                + "No convergence at t = {:8.5f} with dt = {:8.5f}.".format(t, dt)
                + self.TEND
            )

    def __del__(self):
        if self._show_bar:
            self._pbar.close()


class AdaptiveTimeStepping:
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

    def run(self, t_end, t_start=0.0, dt0=None, checkpoints=[], show_bar=False):
        if dt0 is None:
            dt0 = self.dt_max

        dt = dt0

        u_prev = self._u.copy(deepcopy=True)
        t = t_start

        progress = Progress(t_start, t_end, show_bar)

        checkpoints = np.array(checkpoints)
        if checkpoints.size != 0:
            if checkpoints.max() > t_end or checkpoints.min() < t_start:
                raise RuntimeError("Checkpoints outside of integration range.")
            checkpoints = np.sort(np.unique(np.around(checkpoints, 6)))
            if np.any(checkpoints < t_start + self.dt_min):
                remaining = checkpoints >= t_start + self.dt_min
                checkpoints = checkpoints[remaining]
                self._post_process(t)


        while t < t_end:
            if checkpoints.size != 0 and checkpoints[0] < t + dt:
                dt = checkpoints[0] - t
                t = checkpoints[0]
            else:
                t += dt

            num_iter, converged = self._solve(t)
            if converged:
                progress.success(t, dt, num_iter)
                u_prev.assign(self._u)
                self._post_process(t)

                if checkpoints.size != 0 and t == checkpoints[0]:
                    checkpoints = np.delete(checkpoints, 0)

                # increase the time step for fast convergence
                if num_iter < self.increase_num_iter and dt < self.dt_max:
                    dt *= self.increase_factor
                    dt = min(dt, self.dt_max)
                    if not show_bar:
                        info("Increasing time step to dt = {}.".format(dt))

                # adjust dt to end at t_end
                if t + dt > t_end:
                    dt = t_end - t
                    if dt < self.dt_min:
                        return True
            else:
                progress.error(t, dt)

                self._u.assign(u_prev)
                t -= dt
                dt *= self.decrease_factor
                if not show_bar:
                    info("Reduce time step to dt = {}.".format(dt))
                if dt < self.dt_min:
                    info("Abort since dt({}) < dt_min({})".format(dt, self.dt_min))
                    return False
        return True


class EquidistantTimeStepping:
    def __init__(self, solve, post_process):
        self._solve = solve
        self._post_process = post_process

    def run(self, t_end, dt, t_start=0.0, checkpoints=[], show_bar=False):
        progress = Progress(t_start, t_end, show_bar)

        points_in_time = np.hstack((np.arange(t_start, t_end, dt), t_end))
        checkpoints = np.unique(np.sort(checkpoints))
        if checkpoints.size != 0:
            if checkpoints.max() > t_end or checkpoints.min() < t_start:
                raise RuntimeError("Checkpoints outside of integration range.")
            if np.isclose(checkpoints, t_start).any():
                remaining = np.logical_not(np.isclose(checkpoints, t_start))
                checkpoints = checkpoints[remaining]
            points_in_time = np.hstack((points_in_time, checkpoints))

        for t in np.sort(points_in_time):
            num_iter, converged = self._solve(t)
            if converged:
                progress.success(t, dt, num_iter)
                self._post_process(t)
            else:
                progress.error(t, dt, num_iter)
                return False
        return True
