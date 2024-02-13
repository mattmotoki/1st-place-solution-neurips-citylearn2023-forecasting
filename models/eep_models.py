import numpy as np
from my_models.baselines4 import BlendedAverager, cyclic_average


DEFAULT_EEP = np.array([
    0.3231, 0.321, 0.3189, 0.324, 0.3455, 0.425, 0.497, 0.5698, 0.7067, 0.808, 0.8421, 0.802, 0.8156, 0.6794, 0.6632,
    0.6625, 0.6501, 0.6379, 0.6276, 0.6388, 0.6374, 0.6123, 0.4416, 0.3376
])


class GroupedUniformAverager:

    def __init__(self, n=24, n_groups=5, decay_rate=0.1, half_width=0, init_count=1, init_value=None,
                 max_value=float("inf")):
        self.n = n
        self.n_groups = n_groups
        self.decay_rate = decay_rate
        self.half_width = half_width
        self.max_value = max_value
        if init_value is None:
            self.values = np.zeros((n, n_groups), np.float32)
        else:
            self.values = np.repeat(init_value, n_groups).reshape(-1, n_groups)
        self.counts = 1e-3 + np.array(
            [[1, 29, 33, 27, 0], [0, 30, 32, 28, 0], [0, 30, 32, 28, 0], [0, 30, 32, 28, 0], [0, 33, 28, 29, 0],
             [0, 34, 30, 26, 0], [1, 43, 33, 13, 0], [6, 54, 21, 9, 0], [17, 50, 15, 8, 0], [23, 48, 16, 3, 0],
             [30, 41, 17, 2, 0], [35, 37, 18, 0, 0], [33, 37, 19, 1, 0], [35, 34, 19, 2, 0], [35, 32, 20, 3, 0],
             [28, 40, 19, 3, 0], [24, 41, 17, 8, 0], [20, 36, 22, 12, 0], [13, 32, 28, 17, 0], [6, 35, 28, 21, 0],
             [1, 32, 33, 24, 0], [0, 30, 34, 26, 0], [1, 29, 36, 24, 0], [1, 29, 34, 26, 0]], dtype=np.float32)
        self.counts = init_count * self.counts / self.counts.mean()
        self.time_index = 0

    def update(self, x, group_num):
        i = self.time_index % self.n
        if x < self.max_value:
            self.counts[i, group_num] += 1
            self.values[i, group_num] += (x - self.values[i, group_num]) / self.counts[i, group_num]
        self.time_index += 1

    def predict(self, indices):
        counts = self.counts[indices % self.n]
        preds = np.sum(self.values[indices % self.n] * counts / counts.sum(1, keepdims=True), 1)
        return cyclic_average(preds, self.decay_rate, self.half_width)


class EEPForecaster:

    def __init__(self, init_value):

        # occupancy averager
        self.alpha = 0.958
        self.occupancy_averager = GroupedUniformAverager(
            decay_rate=0.317, half_width=1, init_count=1.789,
            init_value=init_value * DEFAULT_EEP / DEFAULT_EEP.mean(), max_value=3.0
        )

        # baseline averager
        self.baseline_averager = BlendedAverager(
            decay_rate24=0.389, decay_rate168=0.0936, half_width24=1,
            half_width168=1, init_count24=1.761, init_count168=1.508,
            max_value=3.0, reset=False, reset_count=30,
            init_value24=DEFAULT_EEP / DEFAULT_EEP.mean() * init_value,
            init_value168=np.tile(DEFAULT_EEP, 7) / DEFAULT_EEP.mean() * init_value,
        )

    def update(self, eep, occupant_count):
        self.baseline_averager.update(eep)
        self.occupancy_averager.update(eep, occupant_count)

    def predict(self, indices):
        baseline = self.baseline_averager.predict(indices)[:, None]

        counts = self.occupancy_averager.counts[indices % 24]
        values = self.occupancy_averager.values[indices % 24]

        w = self.alpha ** counts
        values = w * baseline + (1 - w) * values

        preds = np.sum(values * counts / counts.sum(1, keepdims=True), 1)
        return cyclic_average(preds, self.decay_rate, self.half_width)
