import os
from pyboy.pyboy import *

from settings import ROM_PATH

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


from tetris_env import TetrisEnv
from stable_baselines3 import PPO,DQN
from gymnasium.spaces import Discrete, Box
from utils import preprocess_game_area

done = False
step_count = 0


env = TetrisEnv(window_type="SDL2")
print("\n[INFO] Reiniciando ambiente...")
obs, _ = env.reset()

# Carrega o modelo treinado
# model = PPO.load("ppo_tetris")
model = DQN.load("dqn_tetris")
reward_sum = 0
aux_list = []

while not done:

    # Get action from model
    action, _ = model.predict(obs, deterministic=True)
    print(f"\n[STEP {step_count}] Ação tomada: {action}")

    # Execute action
    obs, reward, done, _, info = env.step(int(action))
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
step_count = 0

df = pd.DataFrame(env.applied_rewards)
df['total'] = df.sum(axis = 1)
print(df)