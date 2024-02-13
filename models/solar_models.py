import numpy as np


DEFAULT_DIRECT_RADIATION = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 231.846, 403.6, 486.921, 570.687, 607.167, 607.134, 610.111, 600.205,
    588.15, 558.149, 519.062, 435.104, 257.739, 16.588, 0.0, 0.0, 0.0, 0.0
])

DEFAULT_DIFFUSE_RADIATION = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 62.261, 105.454, 148.35, 183.84, 219.174, 228.556, 230.291, 232.076,
    230.147, 202.12, 166.231, 117.399, 65.068, 4.739, 0.0, 0.0, 0.0, 0.0
])

DEFAULT_SOLAR_GENERATION = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 28.15, 113.1, 261.46, 409.2, 521.2, 574.3, 599.93, 593.4, 559.98, 477.89, 369.48,
    222.46, 70.18, 0.86, 0.0, 0.0, 0.0, 0.0
])


def solar_cyclic_average(values, decay_rate=1.0, half_width=1):
    n = len(values)
    counts = np.zeros(n)
    totals = np.zeros(n)
    weights = decay_rate ** (half_width - np.append(np.arange(half_width + 1), np.arange(half_width)[::-1]))

    for i, x in enumerate(values):
        indices = np.arange(i - half_width, i + half_width + 1) % n
        totals[indices] += weights * x
        counts[indices] += weights * float(x > 0)

    avgs = totals / np.where(counts == 0, 1, counts)
    avgs[counts != 2 * half_width + 1] = values[counts != 2 * half_width + 1]
    return avgs


class SolarAverager:

    def __init__(self, n=24, decay_rate=1.0, half_width=1, init_count=1):
        self.n = n
        self.decay_rate = decay_rate
        self.half_width = half_width
        self.values = DEFAULT_SOLAR_GENERATION.copy()
        self.counts = np.full(n, init_count, np.float32)
        self.time_index = 0

    def update(self, x):
        i = self.time_index % self.n
        self.counts[i] += 1
        self.values[i] += (x - self.values[i]) / self.counts[i]
        self.time_index += 1

    def predict(self, indices):
        return solar_cyclic_average(self.values, self.decay_rate, self.half_width)[indices % self.n]


class SolarForecaster:
    def __init__(self, b0_pv_capacity):

        params = {
            "averager": {"decay_rate": 0.192107815, "half_width": 17, "init_count": 248.333425},
            "direct_scale": 8.59058184, "diffuse_scale": 8.76028949, "shift": -6.27438633, "coef": 0.767685477,
            "alpha": 0.514582459, "direct_decay": 0.649011486, "diffuse_decay": 0.915372782, "avg_weight0": 0.69144079,
            "avg_weight1": 0.819944313
        }

        self.b0_pv_capacity = b0_pv_capacity
        self.averager = SolarAverager(**params["averager"])
        self.direct_scale = params["direct_scale"]
        self.diffuse_scale = params["diffuse_scale"]
        self.shift = params["shift"]
        self.coef = params["coef"]

        self.alpha = params["alpha"]

        self.direct_decay = params["direct_decay"]
        self.diffuse_decay = params["diffuse_decay"]

        self.avg_weight0 = params["avg_weight0"]
        self.avg_weight1 = params["avg_weight1"]

        self.direct_irradiance_forecast = DEFAULT_DIRECT_RADIATION.copy()
        self.direct_irradiance_forecast6 = DEFAULT_DIRECT_RADIATION.copy()
        self.direct_irradiance_forecast12 = DEFAULT_DIRECT_RADIATION.copy()
        self.direct_irradiance_forecast24 = DEFAULT_DIRECT_RADIATION.copy()

        self.diffuse_irradiance_forecast = DEFAULT_DIFFUSE_RADIATION.copy()
        self.diffuse_irradiance_forecast6 = DEFAULT_DIFFUSE_RADIATION.copy()
        self.diffuse_irradiance_forecast12 = DEFAULT_DIFFUSE_RADIATION.copy()
        self.diffuse_irradiance_forecast24 = DEFAULT_DIFFUSE_RADIATION.copy()

        self.time_index = 0

    def gumbel_l(self, x):
        return 1 - np.exp(-np.exp(x))

    def update(
            self, solar_generation,
            direct_irradiance, direct_irradiance6, direct_irradiance12, direct_irradiance24,
            diffuse_irradiance, diffuse_irradiance6, diffuse_irradiance12, diffuse_irradiance24
    ):

        self.averager.update(solar_generation / self.b0_pv_capacity * 1000)

        self.direct_irradiance_forecast[self.time_index % 24] = direct_irradiance
        self.direct_irradiance_forecast6[(self.time_index + 6) % 24] = direct_irradiance6
        self.direct_irradiance_forecast12[(self.time_index + 12) % 24] = direct_irradiance12
        self.direct_irradiance_forecast24[(self.time_index + 24) % 24] = direct_irradiance24

        self.diffuse_irradiance_forecast[self.time_index % 24] = diffuse_irradiance
        self.diffuse_irradiance_forecast6[(self.time_index + 6) % 24] = diffuse_irradiance6
        self.diffuse_irradiance_forecast12[(self.time_index + 12) % 24] = diffuse_irradiance12
        self.diffuse_irradiance_forecast24[(self.time_index + 24) % 24] = diffuse_irradiance24

        self.time_index += 1

    def predict(self, indices):
        direct_irradiance = solar_cyclic_average(self.direct_irradiance_forecast24)[indices[:24] % 24]
        direct_irradiance[:12] += self.direct_decay * (
                solar_cyclic_average(self.direct_irradiance_forecast12)[indices[:12] % 24] - direct_irradiance[:12])
        direct_irradiance[:6] += self.direct_decay * (
                solar_cyclic_average(self.direct_irradiance_forecast6)[indices[:6] % 24] - direct_irradiance[:6])
        direct_irradiance = solar_cyclic_average(direct_irradiance)

        diffuse_irradiance = solar_cyclic_average(self.diffuse_irradiance_forecast24)[indices[:24] % 24]
        diffuse_irradiance[:12] += self.diffuse_decay * (
                solar_cyclic_average(self.diffuse_irradiance_forecast12)[indices[:12] % 24] - diffuse_irradiance[
                                                                                              :12])
        diffuse_irradiance[:6] += self.diffuse_decay * (
                solar_cyclic_average(self.diffuse_irradiance_forecast6)[indices[:6] % 24] - diffuse_irradiance[:6])
        diffuse_irradiance = solar_cyclic_average(diffuse_irradiance)

        irradiance = self.direct_scale * direct_irradiance + self.diffuse_scale * diffuse_irradiance
        solar = 1000 * self.coef * self.gumbel_l(0.001 * irradiance + self.shift)

        w = self.alpha ** (1 + np.arange(24))
        next_solar = w * solar[-1] + (1 - w) * solar

        mask = np.tile((direct_irradiance > 0) * (diffuse_irradiance > 0), 2)

        forecast = solar_cyclic_average(np.append(solar, next_solar)) * mask
        baseline = self.averager.predict(indices) * mask

        return np.append(
            (1 - self.avg_weight0) * forecast[:24] + self.avg_weight0 * baseline[:24],
            (1 - self.avg_weight1) * forecast[24:] + self.avg_weight1 * baseline[24:],
        )
