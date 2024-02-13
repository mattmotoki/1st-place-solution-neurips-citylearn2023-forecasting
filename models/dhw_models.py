import numpy as np
from scipy.special import expit as sigmoid
from models.baselines import BlendedAverager


DEFAULT_DHW = np.array([
    0.0343, 0.0308, 0.0206, 0.0299, 0.039, 0.1489, 0.182, 0.1955, 0.2901, 0.2643, 0.1182, 0.1698,
    0.1751, 0.1038, 0.1088, 0.0805, 0.1427, 0.1435, 0.1768, 0.2331, 0.1993, 0.2054, 0.2109, 0.1261
])


class DHWForecaster:
    def __init__(
        self, init_value, max_value=3.0, decay_rate24=0.273, decay_rate168=1.0, 
        half_width24=1, half_width168=26, init_count24=103.75, init_count168=0.0, 
        scale=8.101e-01, alpha=4.811e-01, beta=26.556
    ):

        self.avg = BlendedAverager(
            decay_rate24=decay_rate24, decay_rate168=decay_rate168, half_width24=half_width24,
            half_width168=half_width168, init_count24=init_count24, init_count168=init_count168,
            max_value=max_value, reset=True, reset_count=5,
            init_value24=scale * DEFAULT_DHW / DEFAULT_DHW.mean() * init_value,
            init_value168=scale * np.tile(DEFAULT_DHW, 7) / DEFAULT_DHW.mean() * init_value
        )

        self.alpha = alpha
        self.beta = beta
        self.target_history = []
        self.time_index = 0

    def update(self, x):
        self.avg.update(x)
        self.target_history.append(x)
        self.time_index += 1

    def predict(self, indices):

        preds = self.avg.predict(indices)

        # adjust current day's predictions
        if (self.time_index >= 168) and ((self.time_index % 24) > 10):
            today = self.time_index // 24
            n_obs = self.time_index % 24
            today_indices = (24*today + np.arange(n_obs)) % 168
            today_demand = np.sum(self.target_history[-n_obs:])
            today_preds = self.avg.predict(today_indices).sum()
            preds[(indices // 24) == today] *= 1 + self.alpha*(sigmoid(self.beta*(today_preds - today_demand)) - 0.5)

        return preds
