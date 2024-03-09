from pyboy.pyboy import *

RUNNING_ON = 'windows'

if RUNNING_ON == "linux":
    rom_path = "/mnt/e/github/videogame-ai/game-file/tetris.gb"
else:
    rom_path = "E://github//videogame-ai//game-file/tetris.gb"

with PyBoy(rom_path) as pyboy:
    while not pyboy.tick():
        print(pyboy.game_wrapper())
        print(pyboy.game_wrapper())
        pass

# # Stop PyBoy
# pyboy.stop()
# pil_image = pyboy.screen_image()
# pil_image.save('screenshot.png')