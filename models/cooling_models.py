import numpy as np
from functools import partial
from scipy.optimize import minimize
from models.baselines import UniformAverager, BlendedAverager


DEFAULT_COOLING = np.array([
    1.749, 1.652, 1.275, 1.041, 0.922, 0.932, 1.208, 1.575, 1.98, 1.827, 2.059, 2.45,
    2.94, 3.229, 3.837, 4.252, 4.44, 4.155, 3.567, 3.063, 2.678, 2.464, 2.19, 1.987
])

DEFAULT_INDOOR_TEMPERATURES = np.array([
    23.858, 23.737, 23.882, 23.967, 23.98, 23.985, 24.023, 23.992, 24.071, 24.412, 24.475, 24.494, 24.504, 24.596,
    24.482, 24.45, 24.48, 24.734, 24.72, 24.588, 24.562, 24.429, 24.306, 24.11
])

DEFAULT_OUTDOOR_TEMPERATURES = np.array([
    25.457, 24.913, 24.519, 24.204, 23.942, 23.826, 24.881, 26.646, 28.752, 30.809, 32.654, 34.108, 35.196, 35.919,
    36.174, 36.025, 35.389, 34.287, 32.531, 30.096, 28.78, 27.853, 26.98, 26.174
])

DEFAULT_SETPOINTS = np.array([
    23.848, 23.749, 23.962, 24.036, 24.047, 24.048, 24.064, 24.001, 24.08, 24.43, 24.479, 24.496, 24.504, 24.598,
    24.465, 24.443, 24.48, 24.737, 24.721, 24.584, 24.561, 24.425, 24.3, 24.099
])


class IndoorTemperatureForecaster:

    def __init__(self, init_blend_rate=0.60, decay_rate=0.25, half_width=1, update_freq=24):
        self.averager = UniformAverager(n=24, decay_rate=decay_rate, half_width=half_width,
                                        init_value=DEFAULT_INDOOR_TEMPERATURES.copy())
        self.update_freq = update_freq
        self.feature_history = []
        self.target_history = []
        self.time_index = 0
        self.x = init_blend_rate

    def update(self, temperature):
        self.averager.update(temperature)
        self.target_history.append(temperature)
        self.time_index += 1

        if (self.time_index >= 24) and ((self.time_index % self.update_freq) == 0):
            # collect features and targets
            y = np.append(self.target_history, 48 * [np.nan])
            y = np.array([y[1 + i + np.arange(48)] for i in range(len(self.target_history) - 1)])
            X = np.array(self.feature_history)

            # optimize weights
            def f(x):
                w = x ** np.arange(48)
                w = np.array([w, 1 - w])
                return np.nanmean((y - np.sum(w * X, 1)) ** 2)

            self.x = minimize(f, 0.8, bounds=[(0, 1)]).x[0]

    def predict(self, indices):
        # calculate features
        most_recent = np.full(48, self.target_history[-1])
        avg_temperature = self.averager.predict(indices)
        features = np.vstack([most_recent, avg_temperature])
        self.feature_history.append(features)

        # combine features
        w = self.x ** np.arange(48)
        w = np.array([w, 1 - w])
        preds = (w * features).sum(0)
        return preds


class OutdoorTemperatureForecaster:

    def __init__(self, alpha=0.926025, beta=0.360831, decay_rate=0.1, half_width=0, init_count=0.58324):
        self.averager = UniformAverager(decay_rate=decay_rate, half_width=half_width,
                                        init_count=init_count, init_value=DEFAULT_OUTDOOR_TEMPERATURES.copy())
        self.feature_history = []
        self.target_history = []
        self.time_index = 0
        self.alpha = alpha
        self.beta = beta

        self.temperature6 = DEFAULT_OUTDOOR_TEMPERATURES.copy()
        self.temperature12 = DEFAULT_OUTDOOR_TEMPERATURES.copy()
        self.temperature24 = DEFAULT_OUTDOOR_TEMPERATURES.copy()

        self.x = np.array([1.0, 0.5])

    def update(self, temperature, temperature6, temperature12, temperature24):
        self.averager.update(temperature)
        self.temperature6[(self.time_index + 6) % 24] = temperature6
        self.temperature12[(self.time_index + 12) % 24] = temperature12
        self.temperature24[(self.time_index + 24) % 24] = temperature24
        self.target_history.append(temperature)
        self.time_index += 1

        if (self.time_index >= 24 * 2) and ((self.time_index % 1) == 0):
            # collect features and targets
            y = np.append(self.target_history, 48 * [np.nan])
            y = np.array([y[1 + i + np.arange(48)] for i in range(len(self.target_history) - 1)])
            X = np.array(self.feature_history)

            # optimize weights
            def f(x):
                w = np.append(np.full(24, x[0]), np.full(24, x[1]))
                w = np.array([w, 1 - w])
                return np.nanmean((y - np.sum(w * X, 1)) ** 2)

            self.x = minimize(f, 0.5 * np.ones(2), bounds=[(0, 1), (0, 1)]).x

    def predict(self, indices):
        weight_decay = self.beta * self.alpha ** np.arange(24)

        # calculate forecast
        forecast = self.temperature24[indices[:24] % 24]
        forecast[:12] += 0.75 * (self.temperature12[indices[:12] % 24] - forecast[:12])
        forecast[:6] += 0.75 * (self.temperature6[indices[:6] % 24] - forecast[:6])
        forecast = np.append(
            forecast + (self.target_history[-1] - forecast[0]) * weight_decay,
            forecast + (forecast[-1] - forecast[0]) * weight_decay,
        )

        # calculate recentered average
        avg_values = self.averager.predict(indices)[:24]
        avg_values = np.append(
            avg_values + (self.target_history[-1] - avg_values[0]) * weight_decay,
            avg_values + (forecast[-1] - avg_values[0]) * weight_decay,
        )

        # blend forecast and average
        features = np.vstack([forecast, avg_values])
        self.feature_history.append(features)

        w = np.append(np.full(24, self.x[0]), np.full(24, self.x[1]))
        w = np.array([w, 1 - w])
        preds = (w * features).sum(0)

        return preds


class BaseCoolingForcaster:
    def __init__(self, init_value, cooling_coef):

        params = {
            "temperature": {
                "init_count": 0.993
            },
            "baseline": {
                "decay_rate24": 0.258,
                "decay_rate168": 0.0665,
                "half_width24": 2,
                "half_width168": 5,
                "init_count24": 1.0,
                "init_count168": 0.0,
                "init_value24": -21.584,
                "init_value168": -21.584,
            },
            "cooling": {
                "scale": 0.732348987,
                "shift": -0.418839845,
                "coef": cooling_coef,
                "bias": 21.0775154
            }
        }

        self.params = params
        self.temperature_forecaster = OutdoorTemperatureForecaster(**params["temperature"])

        self.initial_averager = BlendedAverager(
            decay_rate24=0.0537522, decay_rate168=0.08, half_width24=15, half_width168=0,
            init_count24=1.3721732175469237, init_count168=1.4586715504373468e-05,
            init_value24=init_value * DEFAULT_COOLING / DEFAULT_COOLING.mean(),
            init_value168=np.tile(init_value * DEFAULT_COOLING / DEFAULT_COOLING.mean(), 7)
        )

        self.correction_averager = BlendedAverager(**params["baseline"])
        self.error_values = np.full((24, 48), error_init_value, dtype=np.float32)
        self.error_counts = np.full((24, 48), error_init_count, dtype=np.float32)

        self.forecast_history = []
        self.feature_history = []
        self.target_history = []
        self.time_index = 0

    def gumbel_r(self, x):
        return np.exp(-np.exp(-x))

    def get_cooling(self, temperature, scale, shift, coef, bias):
        temperature = (temperature - 29.588125) / 4.7843041
        return coef * self.gumbel_r(scale * temperature  + shift) + bias

    def update(self, temperature, temperature6, temperature12, temperature24, cooling_demand):

        self.temperature_forecaster.update(temperature, temperature6, temperature12, temperature24)
        self.initial_averager.update(cooling_demand)
        self.target_history.append(cooling_demand)

        cooling = self.get_cooling(temperature, **self.params["cooling"])
        self.correction_averager.update(cooling_demand - cooling)

        for horizon in range(48):
            if self.time_index > horizon:
                self.error_counts[(self.time_index - horizon - 1) % 24, horizon] += 1
                n = self.error_counts[(self.time_index - horizon - 1) % 24, horizon]

                error = cooling_demand - self.feature_history[-horizon - 1][horizon]
                value = self.error_values[(self.time_index - horizon - 1) % 24, horizon]

                self.error_values[(self.time_index - horizon - 1) % 24, horizon] += (error - value) / n

        self.time_index += 1

    def predict(self, indices):

        temperature = self.temperature_forecaster.predict(indices)
        cooling = self.get_cooling(temperature, **self.params["cooling"])

        correction = self.correction_averager.predict(indices)
        features = np.maximum(correction + cooling, 0)

        self.feature_history.append(features)

        error = self.error_values[self.time_index % 24]
        forecast = np.maximum(features + error, 0)

        self.forecast_history.append(forecast)
        return forecast

    def evaluate(self):
        y = np.append(self.target_history, 48 * [np.nan])
        y = np.array([y[1 + i + np.arange(48)] for i in range(len(self.target_history) - 1)])
        p = np.array(self.forecast_history)
        return np.nanmean((p - y)**2)


class CoolingForcaster:
    def __init__(self, init_value):
        self.forecaster_list = [
            BaseCoolingForcaster(init_value, cooling_coef=x*5.69146278)
            for x in np.logspace(-0.5, 0.5, 11, base=2)
        ]
        self.best_forecaster = 5
        self.time_index = 0

    def update(self, temperature, temperature6, temperature12, temperature24, cooling_demand):

        self.time_index += 1

        if self.time_index <= 30*24:
            for forecaster in self.forecaster_list:
                forecaster.update(temperature, temperature6, temperature12, temperature24, cooling_demand)
        else:
            self.forecaster_list[self.best_forecaster].update(temperature, temperature6, temperature12, temperature24, cooling_demand)

        if (7*24 <= self.time_index <= 30*24) & ((self.time_index % 24) == 0):
            errors = [forecaster.evaluate() for forecaster in self.forecaster_list]
            self.best_forecaster = np.argmin(errors)

    def predict(self, indices):
        if self.time_index <= 30*24:
            forecasts = [forecaster.predict(indices) for forecaster in self.forecaster_list]
            return forecasts[self.best_forecaster]
        else:
            return self.forecaster_list[self.best_forecaster].predict(indices)
