from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO,DQN
from tetris_env import TetrisEnv

def make_eval_env():
    # use exatamente os mesmos argumentos do treino (sem render)
    return Monitor(TetrisEnv(window_type="headless", memory_size=50))

eval_env = DummyVecEnv([make_eval_env])

model = DQN.load("dqn_tetris_v28")
model.exploration_rate = 0.0  # garante zero exploração
mean_reward, ep_len = evaluate_policy(
    model, eval_env, n_eval_episodes=100, deterministic=True, render=False,
    return_episode_rewards=True
)

import seaborn as sns 

sns.histplot(mean_reward, kde= True) 

sns.histplot(ep_len, kde= True) 