
import os
from pyboy.pyboy import *

from settings import ROM_PATH

import time
import numpy as np
from tetris_env import TetrisEnv

from gym.spaces import Discrete, Box
from utils import preprocess_game_area

env = TetrisEnv(window_type="SDL2")

print("\n[INFO] Reiniciando ambiente...")
obs = env.reset()
print(f"[INFO] Observação inicial (shape: {obs.shape}):")
print(obs)

done = False
step_count = 0

while not done:
    action = env.action_space.sample()  # por enquanto, usa ações aleatórias
    print(f"\n[STEP {step_count}] Ação tomada: {action}")

    obs, reward, done, info = env.step(action)

    print(f"[STEP {step_count}] Recompensa: {reward}")
    print(f"[STEP {step_count}] Done? {done}")
    print(f"[STEP {step_count}] Nova observação:")
    print(obs.reshape(18, 10))  # opcional: para ver como está o tabuleiro


    time.sleep(0.1)  # só para conseguir visualizar no terminal com calma
    step_count += 1

print("\n[FIM] Episódio finalizado.")

done = False
step_count = 0

