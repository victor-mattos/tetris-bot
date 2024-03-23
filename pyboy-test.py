
import os
from pyboy.pyboy import *

from settings import ROM_PATH
from utils import analize_tiles_on_screen
from tetris_model import TetrisModel
from tetris_agent import TetrisAgent



with PyBoy(gamerom_file=ROM_PATH, game_wrapper = True, openai_gym = True) as pyboy:


    input_shape = (180,)
    num_actions = 6

    game_wrapper = pyboy.game_wrapper()

    observation_type = 'tiles'
    action_type = 'press'

    env = PyBoyGymEnv(pyboy = pyboy, observation_type=observation_type)
    tetris_model = TetrisModel(input_shape = input_shape, num_actions = num_actions)
    agent = TetrisAgent(model = tetris_model)

    epochs = 40000

    #game_wrapper.start_game()

    for epoch in range(epochs):
        observation = env.reset()

        observation_new = env.step()

        area = game_wrapper.game_area()
        flat_area = analize_tiles_on_screen(game_wrapper = area)

        observation, reward 

        pass

# # Stop PyBoy
# pyboy.stop()
