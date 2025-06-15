

from stable_baselines3 import PPO, DQN
# from sb3_contrib import QRDQN

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from tetris_env import TetrisEnv

import os
import torch
log_dir = "./logs_csv"
os.makedirs(log_dir, exist_ok=True)

def make_env():
    return Monitor(TetrisEnv(window_type="headless"), filename=os.path.join(log_dir, "monitor.csv"))

# Wrap com DummyVecEnv
env = DummyVecEnv([make_env])

# model = DQN(
#     "MlpPolicy",
#     env,
#     learning_rate=1e-4,
#     buffer_size=500_000,              # Tamanho do buffer de replay
#     learning_starts=5_000,            # Passos antes de começar a aprender
#     batch_size=512,
#     tau=1.0,                          # Fator de soft update (1.0 = update direto)
#     gamma=0.99,
#     train_freq=1,                     # Frequência de treino
#     target_update_interval=5_000,     # Frequência de atualização da rede alvo
#     exploration_fraction=0.5,         # Fração do treinamento com exploração
#     exploration_final_eps=0.1,       # Valor final de epsilon (ε-greedy)
#     verbose=1,
#     tensorboard_log="logs/",
#     policy_kwargs=dict(net_arch=[256, 256]),
#     device = "cuda"
# )

model = DQN(
    "MlpPolicy",
    env,
    learning_rate=3e-4,                   # Aprendizado mais agressivo
    buffer_size=100_000,                 # Reforça renovação do buffer
    learning_starts=1_000,               # Aprende quase imediatamente
    batch_size=512,
    tau=1.0,                             # Atualização direta (como original)
    gamma=0.99,
    train_freq=1,
    target_update_interval=3_000,
    exploration_fraction=0.4,
    exploration_final_eps=0.15,
    verbose=1,
    tensorboard_log="logs/",
    policy_kwargs=dict(net_arch=[128, 128]),  # Rede menor, menos overfitting
    device="cuda"
)

# Treinar
model.learn(total_timesteps=500_000)

# Salvar modelo e normalizador
model.save("dqn_tetris_v20")