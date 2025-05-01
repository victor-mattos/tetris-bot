from pyboy.pyboy import *
import numpy as np
#import matplotlib as plt


def analize_tiles_on_screen(game_wrapper):
    tiles = np.array(game_wrapper)

    flat_area = tiles.flatten()
    
    print(flat_area)
    return(flat_area)
    #print(tiles)

def preprocess_game_area(area: np.ndarray) -> np.ndarray:
    """Recebe a matriz da área do jogo e retorna vetor normalizado de entrada para o agente"""
    # Mapeamento manual
    processed = np.where(area == 47, 0, 1)  # 47 = vazio, qualquer outra coisa vira 1

    # Ou, se quiser diferenciar peças ativas (ex: 134)
    # processed = np.zeros_like(area)
    # processed[np.isin(area, [129, 131, 133])] = 1.0
    # processed[area == 134] = 0.5

    # Flatten e normaliza como float32
    processed = processed.flatten().astype(np.float32)
    
    return processed