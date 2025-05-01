import gym
from gym.spaces import Discrete, Box
import numpy as np
from pyboy import PyBoy, WindowEvent
from settings import ROM_PATH
from utils import preprocess_game_area
import time

from typing import Optional

class TetrisEnv(gym.Env):
    def __init__(self, window_type: Optional[str] = "headless"):
        super().__init__()

        self.action_space = Discrete(6)  # 6 ações possíveis
        self.observation_space = Box(low=0.0, high=1.0, shape=(180,), dtype=np.float32)

        self.pyboy = PyBoy(ROM_PATH, window_type=window_type, game_wrapper=True, openai_gym=True)
        
        self.game_wrapper = self.pyboy.game_wrapper()
        self.game_wrapper.start_game()

    # def reset(self):
    #     self.pyboy.send_input(WindowEvent.RESET)
    #     time.sleep(0.1)  # dá tempo do reset acontecer
    #     for _ in range(5):
    #         self.pyboy.tick()

    #     area = self.game_wrapper.game_area()
    #     tiles = np.array(area)
    #     obs = preprocess_game_area(tiles)
    #     return obs
    def reset(self):
        self.pyboy.stop()
        self.pyboy = PyBoy(ROM_PATH, window_type="headless", game_wrapper=True, openai_gym=True)
        self.game_wrapper = self.pyboy.game_wrapper()
        self.game_wrapper.start_game()

        for _ in range(5):
            self.pyboy.tick()

        area = self.game_wrapper.game_area()
        tiles = np.array(area)
        obs = preprocess_game_area(tiles)
        return obs

    def step(self, action):
        from pyboy import WindowEvent

        # Mapeamento da ação para teclas do Game Boy
        from settings import action_map, release_map

        # Envia o input correspondente
        press_event = action_map[action]
        self.pyboy.send_input(press_event)

        # Avança alguns frames para a ação surtir efeito
        for _ in range(5):
            self.pyboy.tick()

        # Libera a tecla pressionada (simula toque curto)
        release_event = release_map[press_event]
        self.pyboy.send_input(release_event)


        # Lê nova observação
        area = self.game_wrapper.game_area()
        tiles = np.array(area)
        obs = preprocess_game_area(tiles)

        # Recompensa simples: sobreviveu = 0.1, fim do jogo = -1.0
        if self.game_wrapper.game_over():
            reward = -1.0
            done = True
        else:
            reward = 0.1
            done = False

        info = {}

        return obs, reward, done, info