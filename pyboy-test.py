
import os
from pyboy.pyboy import *

from stable_baselines3 import PPO 
from stable_baselines3.common.callbacks import BaseCallback

from gym.spaces import Discrete

from settings import ROM_PATH
from utils import analize_tiles_on_screen



with PyBoy(gamerom_file=ROM_PATH, game_wrapper = True, openai_gym = True) as pyboy:

    game_wrapper = pyboy.game_wrapper()
    env = PyBoyGymEnv(pyboy = pyboy)
    model = PPO ('MlpPolicy', env, verbose=1, learning_rate=0.000001, n_steps = 512)
    game_wrapper.start_game()
    
    
    while not pyboy.tick():
        #print(pyboy.openai_gym())
        print(env.observation_space())
        #game_wrapper.start_game()
        area = game_wrapper.game_area()
        analize_tiles_on_screen(game_wrapper = area)


        pass

# # Stop PyBoy
# pyboy.stop()
# pil_image = pyboy.screen_image()
# pil_image.save('screenshot.png')