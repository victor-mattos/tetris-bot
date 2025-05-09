
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from tetris_env import TetrisEnv

import os

log_dir = "./logs_csv"
os.makedirs(log_dir, exist_ok=True)

# def make_env():
#     return Monitor(TetrisEnv(window_type="headless"), filename=os.path.join(log_dir, "monitor.csv"))


# env = DummyVecEnv([make_env])  # Transforma em env vetorizado
def make_env():
    return TetrisEnv(window_type="headless")

# Wrap com DummyVecEnv
env = DummyVecEnv([make_env])


# Aplicar normalização de recompensa (e observação, opcional)
env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.0)

# Criar o modelo com o ambiente normalizado
model = PPO(
    "MlpPolicy",
    env,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    learning_rate=2.5e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    tensorboard_log="logs/",
    verbose=1
)

# Treinar
model.learn(total_timesteps=50_000)

# Salvar modelo e normalizador
model.save("ppo_tetris")
env.save("ppo_tetris_vecnormalize.pkl")
