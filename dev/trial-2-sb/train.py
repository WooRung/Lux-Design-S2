# TODO: Env 공부
"""
1. SubProcVecEnv: env별로 프로세스 둠 (cpu core수 초과X)
2. DummyVecEnv
3. VecEnv
4. VecEnvWrapper
"""
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecEnv, VecEnvWrapper
from stable_baselines3.common.utils import set_random_seed

# custom packages
from tensorboard_callback import TensorboardCallback
from envs import make_env

# not important pacakges
from parse_args import parse_args

LOG_PATH = './logs'
# N_ENVS =  1 # default 8
N_ENVS =  os.cpu_count() # default 8
# N_ENVS = 8
# SEED = None  # default None
SEED = 12  # default None
MAX_EPISODE_STEPS = 1000  # DEFAULT:200 OR 100
# MAX_EPISODE_STEPS=200  # DEFAULT:200 OR 100
TOTAL_TIME_STEPS = 3_000_000
MODEL_PATH = None


def train(args, env_id, model: PPO):
    eval_env = SubprocVecEnv(
        [make_env(env_id, i, max_episode_steps=1000) for i in range(N_ENVS)]
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(LOG_PATH, "models"),
        log_path=os.path.join(LOG_PATH, "eval_logs"),
        eval_freq=24_000,
        deterministic=False,
        render=False,
        n_eval_episodes=5,
    )

    model.learn(
        TOTAL_TIME_STEPS,
        callback=[TensorboardCallback(tag="train_metrics"), eval_callback],
    )
    model.save(os.path.join(LOG_PATH, "models/latest_model"))


def main(args=None):
    print("Training with args", args)
    if SEED is not None:
        set_random_seed(SEED)

    env_id = "LuxAI_S2-v0"
    env = SubprocVecEnv(
        [
            make_env(env_id, i, max_episode_steps=MAX_EPISODE_STEPS)
            for i in range(N_ENVS)
        ]
    )
    env.reset()

    rollout_steps = 4000
    policy_kwargs = dict(net_arch=(128, 128))

    n_steps = rollout_steps // N_ENVS

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        # n_steps=rollout_steps // N_ENVS,
        batch_size=N_ENVS * n_steps,
        # batch_size=800,
        learning_rate=3e-4,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_epochs=2,
        target_kl=0.05,
        gamma=0.99,
        tensorboard_log=os.path.join(LOG_PATH),
    )
    train(args, env_id, model)


if __name__ == "__main__":
    main(parse_args())
