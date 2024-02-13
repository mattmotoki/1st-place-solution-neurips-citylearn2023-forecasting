import numpy as np
from scipy.optimize import minimize
from models.baselines import BlendedAverager


DEFAULT_EMISSIONS = np.array([
    0.436, 0.417, 0.401, 0.396, 0.4, 0.414, 0.429, 0.443, 0.454, 0.447, 0.46, 0.476, 0.485, 0.489, 0.489, 0.487,
    0.484, 0.481, 0.476, 0.473, 0.475, 0.474, 0.464, 0.451
])


class EmissionsForecaster:

    def __init__(self):

        params = {
            "averager": {
                "decay_rate24": 0.289, "decay_rate168": 0.580, "half_width24": 0,
                "half_width168": 0, "init_count24": 2.650, "init_count168": 8.793
            },
            "alpha": 0.93207663
        }

        self.averager = BlendedAverager(
            **params["averager"], init_value24=DEFAULT_EMISSIONS.copy(),
            init_value168=np.tile(DEFAULT_EMISSIONS.copy(), 7), reset=False
        )

        self.alpha = params["alpha"]
        self.feature_history = []
        self.target_history = []
        self.time_index = 0

    def update(self, emissions):
        self.averager.update(emissions)
        self.target_history.append(emissions)
        self.time_index += 1

        if self.time_index >= 24*10:

            # collect features and targets
            y = np.append(self.target_history, 48 * [np.nan])
            y = np.array([y[1 + i + np.arange(48)] for i in range(len(self.target_history) - 1)])
            X = np.array(self.feature_history)

            # optimize weights
            def f(x):
                w = x ** np.arange(48)
                w = np.array([w, 1 - w])
                return np.nanmean((y - np.sum(w * X, 1)) ** 2)

            self.alpha = minimize(f, 0.8, bounds=[(0, 1)]).x[0]

    def predict(self, indices):

        # calculate features
        avg_emissions = self.averager.predict(indices)
        most_recent = np.full(48, self.target_history[-1])
        most_recent = most_recent + (avg_emissions - avg_emissions[0])
        features = np.vstack([most_recent, avg_emissions])
        self.feature_history.append(features)

        # combine features
        w = self.alpha ** np.arange(48)
        w = np.array([w, 1 - w])
        preds = (w * features).sum(0)

        return preds

