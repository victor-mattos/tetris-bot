import os
from pyboy import WindowEvent
import numpy as np 


ROOT = os.getcwd()
ROM_NAME = 'tetris.gb'
ROM_PATH = os.path.join(ROOT,'game-file',ROM_NAME)

N_OF_ACTIONS = 6

action_map = {
    0: WindowEvent.PRESS_ARROW_LEFT,
    1: WindowEvent.PRESS_ARROW_RIGHT,
    2: WindowEvent.PRESS_ARROW_UP,
    3: WindowEvent.PRESS_ARROW_DOWN,
    4: WindowEvent.PRESS_BUTTON_A,
    5: WindowEvent.PRESS_BUTTON_B,
}

release_map = {
    WindowEvent.PRESS_ARROW_LEFT: WindowEvent.RELEASE_ARROW_LEFT,
    WindowEvent.PRESS_ARROW_RIGHT: WindowEvent.RELEASE_ARROW_RIGHT,
    WindowEvent.PRESS_ARROW_UP: WindowEvent.RELEASE_ARROW_UP,
    WindowEvent.PRESS_ARROW_DOWN: WindowEvent.RELEASE_ARROW_DOWN,
    WindowEvent.PRESS_BUTTON_A: WindowEvent.RELEASE_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B: WindowEvent.RELEASE_BUTTON_B,
}

piece_map = {'133':"T",
             "131":"Q",
             "129":"IL",
             "130":"H",
             "132":"L",
             "139":"I"}

PIECES_SHAPES = {
    # Peça I (linha)
    "I": np.array([
        [1, 1, 1, 1]
    ], dtype=np.uint8),

    # Peça O (quadrado)
    "O": np.array([
        [1, 1],
        [1, 1]
    ], dtype=np.uint8),

    # Peça T
    "T": np.array([
        [0, 1, 0],
        [1, 1, 1]
    ], dtype=np.uint8),

    # Peça S
    "S": np.array([
        [0, 1, 1],
        [1, 1, 0]
    ], dtype=np.uint8),

    # Peça Z
    "Z": np.array([
        [1, 1, 0],
        [0, 1, 1]
    ], dtype=np.uint8),

    # Peça J
    "J": np.array([
        [1, 0, 0],
        [1, 1, 1]
    ], dtype=np.uint8),

    # Peça L
    "L": np.array([
        [0, 0, 1],
        [1, 1, 1]
    ], dtype=np.uint8)
}
