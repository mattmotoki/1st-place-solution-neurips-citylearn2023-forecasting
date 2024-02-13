import json
import numpy as np
from abc import ABC, abstractmethod


def jsonify(x, results):
    for k, v in x.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, (float, int)):
            v = float(v)
        elif isinstance(v, str):
            pass
        else:
            v = jsonify(v, {})
        results[k] = v
    return results

class BasePredictorModel(ABC):
    """Abstract class defining interfaces required by prediction
    model implementations."""

    def __init__(self, env_data):
        self.num_timesteps = env_data["time_steps"]
        self.num_buildings = env_data["num_buildings"]
        self.building_names = np.array(env_data["building_names"])
        self.observation_names = np.array(env_data["observation_names"][0])
        self.cooling_estimates = [x["annual_cooling_demand_estimate"] / self.num_timesteps for x in env_data["buildings_metadata"]]
        self.eep_estimates = [x["annual_non_shiftable_load_estimate"] / self.num_timesteps for x in env_data["buildings_metadata"]]
        self.dhw_estimates = [x["annual_dhw_demand_estimate"] / self.num_timesteps for x in env_data["buildings_metadata"]]

        # map observation names to indices
        self.obs_index_lookup = {}
        for obs_name in self.observation_names:
            indices = np.where(self.observation_names == obs_name)[0]
            if len(indices) == 1:
                self.obs_index_lookup[obs_name] = indices[0]
            elif obs_name == "solar_generation":
                self.obs_index_lookup[obs_name] = indices[0]
            elif obs_name == "carbon_intensity":
                self.obs_index_lookup[obs_name] = indices[0]
            else:
                assert len(self.building_names) == len(indices), (len(self.building_names), len(indices))
                for b_name, i in zip(self.building_names, indices):
                    if b_name not in self.obs_index_lookup:
                        self.obs_index_lookup[b_name] = {}
                    self.obs_index_lookup[b_name][obs_name] = i
         
    @abstractmethod
    def compute_forecast(self, observations):
        """Method to perform inference, generating predictions given
        current observations."""
        pass

    def parse_observation(self, observation):
        parse_obs = {}
        for key,value in self.obs_index_lookup.items():
            if isinstance(value, dict):
                parse_obs[key] = {k:observation[idx] for k,idx in value.items()}
            else:
                parse_obs[key] = observation[value]
        return parse_obs

    def parse_predictions(self, predictions_dict):
        for b_name in self.building_names:
            for pred_name, obs_name in zip(
                ["Equipment_Eletric_Power", "DHW_Heating", "Cooling_Load"],
                ["non_shiftable_load", "dhw_demand", "cooling_demand"]
            ):
                predictions_dict[b_name][pred_name] = predictions_dict[b_name][obs_name]
                del predictions_dict[b_name][obs_name]
        predictions_dict["Solar_Generation"] = predictions_dict["solar_generation"]
        predictions_dict["Carbon_Intensity"] = predictions_dict["carbon_intensity"]
        return predictions_dict

