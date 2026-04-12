import os
from pyboy.pyboy import *

from settings import ROM_PATH

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


from tetris_env import TetrisEnv
from stable_baselines3 import PPO,DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize # Importe aqui
from gymnasium.spaces import Discrete, Box
from utils import preprocess_game_area

done = False
step_count = 0



# env = TetrisEnv(window_type="SDL2", memory_size=50)
# print("\n[INFO] Reiniciando ambiente...")
# obs, _ = env.reset()


raw_env = TetrisEnv(window_type="SDL2", memory_size=50)

# 2. Envolva no DummyVecEnv (necessário para o VecNormalize)
env = DummyVecEnv([lambda: raw_env])

# 3. Carregue o normalizador salvo no final do treinamento
# Certifique-se de que o caminho aponta para o .pkl gerado junto com o modelo de 2M
norm_path = "models/vecnorm_2m.pkl" 
env = VecNormalize.load(norm_path, env)

# 4. CRÍTICO: Desligue o treinamento do normalizador na inferência
env.training = False 
env.norm_reward = False # Não precisamos normalizar a recompensa na hora de jogar/ver

obs = env.reset()

# Carrega o modelo treinado
# model = PPO.load("ppo_tetris")
model = DQN.load("models/ckpts/dqn_2m_2000000_steps")
reward_sum = 0
aux_list = []

while not done:

    # Get action from model
    action, _ = model.predict(obs, deterministic=True)
    print(f"\n[STEP {step_count}] Ação tomada: {action}")

    # Execute action
    # obs, reward, done, _, info = env.step(int(action))
    obs, reward, done, _, info = env.step(action)
    reward_sum += reward
    aux_list.append(reward_sum)
    
    print(f"[STEP {step_count}] Recompensa: {reward}")
    print(f"[STEP {step_count}] Done? {done}")
    print(f"[STEP {step_count}] Nova observação:")
    print(np.array(env.game_wrapper.game_area()).reshape(18, 10))

    # Small delay to make the game visible
    time.sleep(0.1)
    step_count += 1
print("\n[FIM] Episódio finalizado.")

done = False
df = pd.DataFrame(env.applied_rewards)
df['total'] = df.sum(axis = 1)
print(df)