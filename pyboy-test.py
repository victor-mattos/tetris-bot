
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from tetris_env import TetrisEnv

# Criar e checar o ambiente
env = TetrisEnv(window_type="headless")
# check_env(env, warn=True)  # verifica se está compatível com Gym

# Criar o modelo (usando MLP se obs for vetor/matriz)
model = PPO("MlpPolicy", env, verbose=1,tensorboard_log="logs/")

# Treinar
model.learn(total_timesteps=100_000)

# Salvar
model.save("ppo_tetris")

########################################################################################################################
########################################################################################################################
