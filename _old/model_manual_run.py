import os
import time
import numpy as np
from pyboy import PyBoy
from settings import ROM_PATH
from tetris_env import TetrisEnv

# Mapeia teclas para ações do ambiente
key_to_action = {
    'a': 0,  # Esquerda
    'd': 1,  # Direita
    'w': 2,  # Rotacionar
    's': 3,  # Baixo
    ' ': 4,  # Drop
    'q': 5,  # Sem ação (ou alguma neutra definida)
    'x': -1  # Sair
}

# Inicia o ambiente

done = False
step_count = 0
reward_sum = 0

with PyBoy(gamerom_file=ROM_PATH, game_wrapper = True, openai_gym = True) as pyboy:

    env = TetrisEnv(window_type="SDL2")
    
    obs, _ = env.reset()
    game_wrapper = pyboy.game_wrapper()
    game_wrapper.start_game()
    done = False
    while not done:
        action = input()

        try:
            action = key_to_action[action]
        except:
            action = 5

        # Executa a ação
        obs, reward, done, _, info = env.manual_step(action)


        # Lê nova observação
        area = env.game_wrapper.game_area()
        tiles = np.array(area)

        print(tiles)
        piece_in_play = env.analize_piece_in_play(area = tiles)
        
        if piece_in_play:
            env.calc_reward(piece_in_play=piece_in_play)

        print(env.applied_rewards[-1])
        
        step_count += 1

print("\n[FIM] Episódio encerrado.")
import pandas as pd
print(pd.DataFrame(env.applied_rewards))