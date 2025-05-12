

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from tetris_env import TetrisEnv

import os

log_dir = "./logs_csv"
os.makedirs(log_dir, exist_ok=True)

def make_env():
    return Monitor(TetrisEnv(window_type="headless"), filename=os.path.join(log_dir, "monitor.csv"))

# Wrap com DummyVecEnv
env = DummyVecEnv([make_env])

# Criar o modelo com o ambiente normalizado
# model = PPO(
#     "MlpPolicy",
#     env,
#     n_steps=2048,
#     batch_size=256,
#     n_epochs=10,
#     learning_rate=2.5e-4,
#     gamma=0.99,
#     gae_lambda=0.95,
#     clip_range=0.2,
#     ent_coef=0.03,
#     tensorboard_log="logs/",
#     verbose=1
# )


model = DQN(
    "MlpPolicy",
    env,
    learning_rate=2.5e-4,
    buffer_size=100_000,              # Tamanho do buffer de replay
    learning_starts=1_000,            # Passos antes de começar a aprender
    batch_size=256,
    tau=1.0,                          # Fator de soft update (1.0 = update direto)
    gamma=0.99,
    train_freq=1,                     # Frequência de treino
    target_update_interval=1_000,     # Frequência de atualização da rede alvo
    exploration_fraction=0.3,         # Fração do treinamento com exploração
    exploration_final_eps=0.05,       # Valor final de epsilon (ε-greedy)
    verbose=1,
    tensorboard_log="logs/"
)

# Treinar
model.learn(total_timesteps=100_000)

# Salvar modelo e normalizador
model.save("dqn_tetris")
# env.save("ppo_tetris_vecnormalize.pkl")
