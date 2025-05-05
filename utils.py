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

    # Flatten e normaliza como float32
    processed = processed.flatten().astype(np.float32)
    
    return processed


import numpy as np
import matplotlib.pyplot as plt
import math

def height_penalty(normalized_height):
    return ((-2 / (1 + math.exp(20 * (normalized_height - 0.4))) + 1) + 1)*10/2

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def plot_function(x_array:np.array,y_array:np.array):
    '''
    # Create x values (normalized heights from 0 to 1)
    x = np.linspace(0, 1, 1000)

    # Calculate y values (height penalties)
    y = [height_penalty(h) for h in x]
    '''

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_array, y_array, 'b-', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Normalized Height')
    plt.ylabel('Height Penalty')
    plt.title('Height Penalty vs Normalized Height')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0.4, color='r', linestyle='--', alpha=0.5, label='Threshold (0.4)')
    plt.legend()

    # Show the plot
    plt.show()