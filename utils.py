from pyboy.pyboy import *
import numpy as np
#import matplotlib as plt


def analize_tiles_on_screen(game_wrapper):
    tiles = np.array(game_wrapper)

    flat_area = tiles.flatten()
    
    print(flat_area)
    return(flat_area)
    #print(tiles)
