import os
from pyboy import WindowEvent

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