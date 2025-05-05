import os
from pyboy.pyboy import *

from settings import ROM_PATH

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


from tetris_env import TetrisEnv
from stable_baselines3 import PPO
from gymnasium.spaces import Discrete, Box
from utils import preprocess_game_area

done = False
step_count = 0

env = TetrisEnv(window_type="SDL2")
print("\n[INFO] Reiniciando ambiente...")
obs,_ = env.reset()

# Carrega o modelo treinado
model = PPO.load("ppo_tetris")

reward_sum =0
aux_list = []
while not done:


    # print(f"[INFO] Observação inicial (shape: {obs.shape}):")
    # print(obs)

    # action = env.action_space.sample()  # por enquanto, usa ações aleatórias
    action, _ = model.predict(obs, deterministic=True)
    print(f"\n[STEP {step_count}] Ação tomada: {action}")

    obs, reward, done, _,info = env.step(int(action))
    reward_sum += reward
    aux_list.append(reward_sum)
    print(f"[STEP {step_count}] Recompensa: {reward}")
    print(f"[STEP {step_count}] Done? {done}")
    print(f"[STEP {step_count}] Nova observação:")
    # print(obs.reshape(18, 10))  # opcional: para ver como está o tabuleiro


    time.sleep(0.1)  # só para conseguir visualizar no terminal com calma
    step_count += 1

print("\n[FIM] Episódio finalizado.")

done = False
step_count = 0