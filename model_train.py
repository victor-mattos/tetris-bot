

from stable_baselines3 import PPO, DQN
# from sb3_contrib import QRDQN

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from tetris_env import TetrisEnv

import os
import torch

log_dir = "./logs_csv"
os.makedirs(log_dir, exist_ok=True)

def make_env():
    return Monitor(TetrisEnv(window_type="headless"), filename=os.path.join(log_dir, "monitor.csv"))

root = os.getcwd()
model_save_path = os.path.join(root, "models")
os.makedirs(model_save_path, exist_ok=True)

# Wrap com DummyVecEnv
env = DummyVecEnv([make_env])
eval_env = DummyVecEnv([make_env])
eval_env = VecNormalize.load("vecnorm_v32.pkl", eval_env) if os.path.exists("vecnorm_v32.pkl") else VecNormalize(eval_env)
eval_env.training = False; eval_env.norm_reward = False

model = DQN(
    "MlpPolicy",
    # "CnnPolicy",
    env,
    learning_rate=1.5e-4,                   # Aprendizado mais agressivo
    buffer_size=500_000,                 # Reforça renovação do buffer
    learning_starts=50_000,               # Aprende quase imediatamente
    batch_size=256,
    tau=1.0,                             # Atualização direta (como original)
    gamma=0.995,
    train_freq=1,
    target_update_interval=10_000,
    exploration_fraction=0.3,
    exploration_final_eps=0.1,
    verbose=1,
    tensorboard_log="logs/",
    policy_kwargs=dict(net_arch=[256, 256]),  # Rede menor, menos overfitting
    device="cuda"
)

# model = DQN(
#     "MlpPolicy",
#     # "CnnPolicy",
#     env,
#     learning_rate=2.5e-4,                   # Aprendizado mais agressivo
#     buffer_size=100_000,                 # Reforça renovação do buffer
#     learning_starts=1_000,               # Aprende quase imediatamente
#     batch_size=256,
#     tau=1.0,                             # Atualização direta (como original)
#     gamma=0.99,
#     train_freq=1,
#     target_update_interval=3_000,
#     exploration_fraction=0.3,
#     exploration_final_eps=0.1,
#     verbose=1,
#     tensorboard_log="logs/",
#     policy_kwargs=dict(net_arch=[256, 256]),  # Rede menor, menos overfitting
#     device="cuda"
# )

eval_cb = EvalCallback(eval_env, best_model_save_path=os.path.join(model_save_path, "best"), eval_freq=50_000, n_eval_episodes=20, deterministic=True)
ckpt_cb = CheckpointCallback(save_freq=10_000, save_path=os.path.join(model_save_path, "ckpts"), name_prefix="dqn")

# Treinar
model.learn(total_timesteps=1_000_000)


file_name = "dqn_1m_area_next_piece_features.zip"
file_path = os.path.join(model_save_path, file_name)
# Salvar modelo e normalizador
model.save(file_path)