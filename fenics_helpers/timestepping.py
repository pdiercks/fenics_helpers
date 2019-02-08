import numpy as np
from dolfin import Function, info


class Progress:
    def __init__(self, t_start, t_end, show_bar):
        self._show_bar = show_bar

        if self._show_bar:
            from tqdm import tqdm

            fmt = "{l_bar}{bar}{rate_fmt}"
            self._pbar = tqdm(total=t_end - t_start, ascii=True, bar_format=fmt)

    def _green(self, msg):
        info("\033[32m" + msg + "\033[m")

    def _red(self, msg):
        info("\033[31m" + msg + "\033[m")

    def iteration_info(self, t, dt, iterations):
        return "at t = {:8.5f} after {:2} iteration(s) with dt = {:8.5f}.".format(
            t, iterations, dt
        )

    def success(self, t, dt, iterations):
        if self._show_bar:
            self._pbar.update(dt)
            self._pbar.set_description("dt = {:8.5f}".format(dt))
        else:
            self._green("Convergence " + self.iteration_info(t, dt, iterations))

    def error(self, t, dt, iterations):
        if not self._show_bar:
            self._red("No Convergence " + self.iteration_info(t, dt, iterations))

    def __del__(self):
        if self._show_bar:
            self._pbar.close()


class CheckPoints:
    def __init__(self, points, t_start, t_end, dt_min=1e-6):
        self.points = np.array(points)
        self.post_process_t0 = False

        if self.points.size == 0:
            return
        if self.points.max() > t_end or self.points.min() < t_start:
            raise RuntimeError("Checkpoints outside of integration range.")

        self.points = np.sort(np.unique(np.around(self.points, 6)))
        if np.any(self.points < t_start + dt_min):
            remaining = self.points >= t_start + dt_min
            self.points = self.points[remaining]
            self.post_process_t0 = True

    def timestep(self, t, dt):
        if self.points.size != 0 and self.points[0] < t + dt:
            dt = self.points[0] - t
        return dt

    def reached(self, t):
        if self.points.size != 0 and t == self.points[0]:
            self.points = np.delete(self.points, 0)


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

        checkpoints = CheckPoints(checkpoints, t_start, t_end, self.dt_min)
        if checkpoints.post_process_t0:
            self._post_process(t)

        while t < t_end:
            dt = checkpoints.timestep(t, dt)
            t += dt

            num_iter, converged = self._solve(t)
            if converged:
                progress.success(t, dt, num_iter)
                u_prev.assign(self._u)
                self._post_process(t)

                checkpoints.reached(t)

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
                progress.error(t, dt, num_iter)

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

        points_in_time = np.append(np.arange(t_start, t_end, dt), t_end)
        checkpoints = CheckPoints(checkpoints, t_start, t_end)
        points_in_time = np.append(points_in_time, checkpoints.points)

        for t in np.sort(points_in_time):
            num_iter, converged = self._solve(t)
            if converged:
                progress.success(t, dt, num_iter)
                self._post_process(t)
            else:
                progress.error(t, dt, num_iter)
                return False
        return True
