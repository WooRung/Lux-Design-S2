# filename
import logging
from pprint import pprint

import gym
from luxai_s2.config import EnvConfig
from luxai_s2.replay import generate_replay
import json

import os

REPLAY_DIR = 'logs/replays'
os.makedirs(REPLAY_DIR, exist_ok=True)

logger = logging.getLogger(__name__)


class ReplayWrapper(gym.Wrapper):
    RUN_COUNT = 1
    REPLAY_DICT = {}

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.state_list = []
        self._record = 0
        self.env_cfg: EnvConfig = self.env.env_cfg

        # generate_replay()

    def reset(self, **kwargs):
        if self._record: # 양수
            logger.info(f'Replay 기록 종료: {self._record}.json')
            replay = generate_replay(self.state_list)
            replay_json = json.dumps(replay)
            with open(os.path.join(REPLAY_DIR, f"{self._record}.json"), 'w') as f:
                f.write(replay_json)
            self.state_list = []
            self._record = 0

        rank = kwargs.get('rank')

        if rank not in ReplayWrapper.REPLAY_DICT:
            ReplayWrapper.REPLAY_DICT[rank] = 0
        else:
            ReplayWrapper.REPLAY_DICT[rank] += 1

        if ReplayWrapper.RUN_COUNT % 100 == 1:
            self._record = ReplayWrapper.RUN_COUNT # 양수
            logger.info(f'Replay 기록 시작: {self._record}.json')
            self.state_list = []

        ReplayWrapper.RUN_COUNT += 1

        # seed = kwargs.get('seed')

        # print(seed)

        # replay = generate_replay(self.state_list)
        # json.dumps(json.dumps())
        # self.state_list = []
        return super().reset(**kwargs)

    def step(self, action):
        # self.state_list.append(self.env.state)
        if self._record:
            self.state_list.append(self.env.state)

        return super().step(action)
