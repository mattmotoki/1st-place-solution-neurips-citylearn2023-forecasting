import numpy as np
from models.base_predictor_model import BasePredictorModel
from models.emissions_models import EmissionsForecaster
from models.cooling_models import CoolingForcaster
from models.solar_models import SolarForecaster
from models.dhw_models import DHWForecaster
from models.eep_models import EEPForecaster


class SubmissionModel(BasePredictorModel):

    def __init__(self, env_data, tau):

        # parse input
        super().__init__(env_data)
        self.tau = tau
        self.history = []
        self.b0_pv_capacity = env_data["b0_pv_capacity"]
        self.predictors = {
            **{b_name: {
                "non_shiftable_load": EEPForecaster(init_value=init_eep),
                "dhw_demand": DHWForecaster(init_value=init_dhw),
                "cooling_demand": CoolingForcaster(init_value=init_cooling),
            } for b_name, init_eep, init_dhw, init_cooling in zip(
                self.building_names, self.eep_estimates, self.dhw_estimates, self.cooling_estimates)},
            "solar_generation": SolarForecaster(self.b0_pv_capacity),
            "carbon_intensity": EmissionsForecaster(),
        }

    def load(self):
        """No loading required for trivial example model."""
        pass

    def compute_forecast(self, observations):

        # parse observations
        observations = self.parse_observation(observations[0])
        pred_indices = 1 + len(self.history) + np.arange(48)

        # update predictors
        for b_name in self.building_names:

            self.predictors[b_name]["dhw_demand"].update(observations[b_name]["dhw_demand"])

            self.predictors[b_name]["non_shiftable_load"].update(
                eep=observations[b_name]["non_shiftable_load"],
                occupant_count=int(observations[b_name]["occupant_count"]),
            )

            self.predictors[b_name]["cooling_demand"].update(
                temperature=observations["outdoor_dry_bulb_temperature"],
                temperature6=observations["outdoor_dry_bulb_temperature_predicted_6h"],
                temperature12=observations["outdoor_dry_bulb_temperature_predicted_12h"],
                temperature24=observations["outdoor_dry_bulb_temperature_predicted_24h"],
                cooling_demand=observations[b_name]["cooling_demand"]
              )

        self.predictors["solar_generation"].update(
            solar_generation=observations["solar_generation"],
            direct_irradiance=observations["direct_solar_irradiance"],
            direct_irradiance6=observations["direct_solar_irradiance_predicted_6h"],
            direct_irradiance12=observations["direct_solar_irradiance_predicted_12h"],
            direct_irradiance24=observations["direct_solar_irradiance_predicted_24h"],
            diffuse_irradiance=observations["diffuse_solar_irradiance"],
            diffuse_irradiance6=observations["diffuse_solar_irradiance_predicted_6h"],
            diffuse_irradiance12=observations["diffuse_solar_irradiance_predicted_12h"],
            diffuse_irradiance24=observations["diffuse_solar_irradiance_predicted_24h"],
        )

        self.predictors["carbon_intensity"].update(observations["carbon_intensity"])

        # predict
        predictions_dict = {
            b_name: {x: self.predictors[b_name][x].predict(pred_indices) for x in
                     ["non_shiftable_load", "dhw_demand", "cooling_demand"]}
            for b_name in self.building_names
        }

        predictions_dict["solar_generation"] = self.predictors["solar_generation"].predict(pred_indices)
        predictions_dict["carbon_intensity"] = self.predictors["carbon_intensity"].predict(pred_indices)
        predictions_dict = self.parse_predictions(predictions_dict)

        # store observations
        self.history.append(observations)

        return predictions_dict