# TODO: Env 공부
"""
1. SubProcVecEnv: env별로 프로세스 둠 (cpu core수 초과X)
2. DummyVecEnv
3. VecEnv
4. VecEnvWrapper
"""
import os
from dataclasses import asdict
from typing import Literal, Optional, Type, Union

import gym
from gym.wrappers import TimeLimit
from luxai_s2 import LuxAI_S2
from luxai_s2.config import EnvConfig
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from pettingzoo.utils.wrappers import CaptureStdoutWrapper, AssertOutOfBoundsWrapper, OrderEnforcingWrapper

# custom packages
from policies.factory_placement import random_factory_placement, place_near_random_ice, rl_factory_placement_policy
from wrappers.sb3_wrappers import SB3Wrapper
from wrappers.obs_wrappers import SimpleUnitObservationWrapper
from wrappers.controllers import SimpleUnitDiscreteController, RLFactoryController, RLRobotController
from wrappers.custom_env_wrappers import CustomEnvWrapper
from wrappers.replay_wrappers import ReplayWrapper

from pprint import pprint

FACTORY_PLACEMENT_POLICY = place_near_random_ice

# from kits.rl.sb3.train import make_env as old_make_env
# sample_make_env = old_make_env # 참고용


env_cfg = EnvConfig()
custom_env_cfg = {
    'verbose': 3
}


# TODO: 반드시 재정의가 필요합니다!
def make_env(env_id: str, rank: int, seed: int = 0, max_episode_steps=1000):
    # cfg_dict = asdict(env_config)
    # cfg_dict.update(custom_env_cfg)

    def _init() -> Union[gym.Env, LuxAI_S2]:
        # collect stats so we can create reward functions
        # env: OrderEnforcingWrapper(AssertOutOfBoundsWrapper(CaptureStdoutWrapper(LuxAI_S2))) = gym.make(env_id, collect_stats=True, **cfg_dict)
        env: Union[LuxAI_S2, CaptureStdoutWrapper, AssertOutOfBoundsWrapper, OrderEnforcingWrapper] = gym.make(
            env_id,
            collect_stats=True,
            **custom_env_cfg
            # **cfg_dict
        )

        env_cfg: EnvConfig = env.env_cfg

        # Add a SB3 wrapper to make it work with SB3 and simplify the action space with the controller
        # this will remove the bidding phase and factory placement phase. For factory placement we use
        # the provided place_near_random_ice function which will randomly select an ice tile and place a factory near it.

        env: Union[Type[env], SB3Wrapper] = SB3Wrapper(env,
                                                       factory_placement_policy=FACTORY_PLACEMENT_POLICY,
                                                       controller=SimpleUnitDiscreteController(env_cfg))

        env: Union[Type[env], SimpleUnitObservationWrapper] = SimpleUnitObservationWrapper(
            env
        )  # changes observation to include a few simple features
        env: Union[Type[env], CustomEnvWrapper] = CustomEnvWrapper(env)  # convert to single agent, add our reward
        env: Union[Type[env], TimeLimit] = TimeLimit(
            env, max_episode_steps=max_episode_steps
        )
        env: Union[Type[env], Monitor] = Monitor(env)  # for SB3 to allow it to record metrics

        if env.env_cfg.verbose == 3:
            print("******[EnvConfig]******")
            pprint(env.env_cfg)
            env: Union[Type[env], ReplayWrapper] = ReplayWrapper(env)

        env.reset(seed=seed + rank, rank=rank)

        set_random_seed(seed)

        return env

    return _init
