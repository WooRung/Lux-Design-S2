from typing import Any, Dict

import gym
from numpy import typing as npt

from kits.rl.sb3.wrappers import SimpleUnitObservationWrapper as ISimpleUnitObservationWrapper


class SimpleUnitObservationWrapper(ISimpleUnitObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def observation(self, obs):
        return super().observation(obs)

    @staticmethod
    def convert_obs(obs: Dict[str, Any], env_cfg: Any) -> Dict[str, npt.NDArray]:
        return super().convert_obs(obs, env_cfg)


