import numpy as np
from scipy.optimize import minimize


def cyclic_average(values, decay_rate=1.0, half_width=1):
    n = len(values)
    counts = np.zeros(n)
    totals = np.zeros(n)
    weights = decay_rate ** (half_width - np.append(np.arange(half_width + 1), np.arange(half_width)[::-1]))

    for i, x in enumerate(values):
        indices = np.arange(i - half_width, i + half_width + 1) % n
        totals[indices] += x * weights
        counts[indices] += weights

    return totals / counts


class UniformAverager:

    def __init__(self, n=24, decay_rate=1.0, half_width=1, init_value=None, init_count=1, max_value=float("inf"),
                 reset=False, reset_count=5):
        self.n = n
        self.decay_rate = decay_rate
        self.half_width = half_width
        self.init_value = init_value
        self.max_value = max_value
        self.reset = reset
        self.reset_count = reset_count

        if init_value is None:
            self.values = np.zeros(n, np.float32)
        elif isinstance(init_value, (float, int)):
            self.values = np.full(n, init_value, np.float32)
        else:
            self.values = init_value

        self.counts = np.full(n, init_count, np.float32)
        self.time_index = 0

    def update(self, x):

        if self.reset and (self.time_index > 0) and ((self.time_index % (24 * 30)) == 0):
            self.counts = np.full(len(self.counts), self.reset_count, dtype=np.int32)

        i = self.time_index % self.n

        if x < self.max_value:
            self.counts[i] += 1
            self.values[i] += (x - self.values[i]) / self.counts[i]

        self.time_index += 1

    def predict(self, indices):
        return cyclic_average(self.values, self.decay_rate, self.half_width)[indices % self.n]


class BlendedAverager:
    def __init__(self, decay_rate24=1.0, decay_rate168=1.0, half_width24=1, half_width168=1, init_count24=1,
                 init_count168=1, init_value24=None, init_value168=None, max_value=float("inf"), reset=False,
                 reset_count=5):

        self.tod_avg = UniformAverager(n=24, decay_rate=decay_rate24, half_width=half_width24, init_count=init_count24,
                                       init_value=init_value24, max_value=max_value, reset=reset, reset_count=reset_count)
        self.how_avg = UniformAverager(n=168, decay_rate=decay_rate168, half_width=half_width168, init_count=init_count168,
                                       init_value=init_value168, max_value=max_value, reset=reset, reset_count=reset_count)
        self.target_history = []
        self.tod_history = None
        self.how_history = None
        self.time_index = 0
        self.alpha = 1.0

    def update(self, x):
        self.tod_avg.update(x)
        self.how_avg.update(x)
        self.target_history.append(x)
        self.time_index += 1

        if (self.time_index > 168) and ((self.time_index % 168) == 0):
            target_values = np.array(self.target_history)[-168:]
            tod_forecasts = np.tile(self.tod_history, 7)
            how_forecasts = self.how_history

            def f(x):
                preds = (x * tod_forecasts) + (1 - x) * how_forecasts
                error = np.mean((target_values - preds) ** 2)
                return error

            self.alpha = minimize(f, x0=1.0, bounds=[(0.0, 1.0)]).x[0]

        if (self.time_index % 168) == 0:
            self.tod_history = self.tod_avg.values.copy()
            self.how_history = self.how_avg.values.copy()

    def predict(self, indices):
        tod_preds = self.tod_avg.predict(indices)
        how_preds = self.how_avg.predict(indices)
        return self.alpha * tod_preds + (1 - self.alpha) * how_preds


class MostRecentAverager:

    def __init__(self, alpha, decay_rate=0.1, half_width=0, init_count=1, init_value=None, max_value=float("inf")):

        self.averager = UniformAverager(
            decay_rate=decay_rate, half_width=half_width, init_count=init_count,
            init_value=init_value, max_value=max_value
        )

        self.average_history = []
        self.shift_history = []
        self.target_history = []
        self.time_index = 0
        self.alpha = alpha

    def update(self, x):
        self.averager.update(x)
        self.target_history.append(x)
        self.time_index += 1

        if self.time_index >= 24:

            # collect features and targets
            y = np.append(self.target_history, 48 * [np.nan])
            y = np.array([y[1 + i + np.arange(48)] for i in range(len(self.target_history) - 1)])
            averages = np.array(self.average_history)
            shifts = np.array(self.shift_history)[:, None]

            # optimize weights
            def f(x):
                w = x ** np.arange(48)[None]
                p = averages + w * shifts
                return np.nanmean((y - p) ** 2)

            self.alpha = minimize(f, 0.8, bounds=[(0, 1)]).x[0]

    def predict(self, indices):
        avg_values = self.averager.predict(indices)
        shift = (self.target_history[-1] - avg_values[0])
        self.average_history.append(avg_values)
        self.shift_history.append(shift)
        w = self.alpha ** np.arange(48)
        return avg_values + w * shift


class ConstantPredictor:

    def __init__(self, value):
        self.value = value

    def update(self, x):
        pass

    def predict(self, indices):
        return np.full(len(indices), self.value, dtype=np.float32).tolist()
