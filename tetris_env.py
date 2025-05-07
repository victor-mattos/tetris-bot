import gymnasium as gym
# from gym import Env

import time
import math
import numpy as np

from gymnasium.spaces import Discrete, Box
from pyboy import PyBoy, WindowEvent
from collections import deque


from settings import ROM_PATH
from typing import Optional

from utils import preprocess_game_area


class TetrisEnv(gym.Env):
    def __init__(self, window_type: Optional[str] = "headless"):
        super().__init__()

        self.action_space = Discrete(6)  # 6 ações possíveis
        self.observation_space = Box(low=0.0, high=1.0, shape=(180,), dtype=np.float32)

        self.window_type = window_type
        self.pyboy = PyBoy(ROM_PATH, window_type=self.window_type, game_wrapper=True, openai_gym=True)
        self.game_wrapper = self.pyboy.game_wrapper()
        self.game_wrapper.start_game()

        # Roda alguns ticks para a janela não travar
        for _ in range(5):
            self.pyboy.tick()


        self.score = 0
        self.last_score = 0


        self.memory_size = 3
        self.area_history = deque(maxlen=self.memory_size)
        self.fixed_gamearea_history = deque(maxlen=self.memory_size)

        self.last_piece_id = None
        self.current_piece_id = None
        self.piece_visible = False
        self.has_new_piece = False
        self.applied_rewards = []
    ###################################################################################
    ###################################################################################


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Call parent's reset to set the seed
        self.pyboy.stop()
        self.pyboy = PyBoy(ROM_PATH, window_type=self.window_type, game_wrapper=True, openai_gym=True)
        self.game_wrapper = self.pyboy.game_wrapper()
        self.game_wrapper.start_game()

        for _ in range(5):
            self.pyboy.tick()

        area = self.game_wrapper.game_area()
        tiles = np.array(area)
        obs = preprocess_game_area(tiles)
        return obs, {}  # Return observation and empty info dict

    
    ###################################################################################
    ###################################################################################


    def preprocess_game_area(self,area: np.ndarray) -> np.ndarray:
        """Recebe a matriz da área do jogo e retorna vetor normalizado de entrada para o agente"""
        # Mapeamento manual
        processed = np.where(area == 47, 0, 1)  # 47 = vazio, qualquer outra coisa vira 1

        # Flatten e normaliza como float32
        # processed = processed.flatten().astype(np.float32)
        
        return processed
    
    ###################################################################################
    ###################################################################################
    

    def analize_piece_in_play(self, area: np.ndarray) -> bool:
        ENTRY_SPACE_START_I = 0
        ENTRY_SPACE_END_I = 2
        ENTRY_SPACE_START_J = 4
        ENTRY_SPACE_END_J = 7

        spawn_center = area[ENTRY_SPACE_START_I:ENTRY_SPACE_END_I, ENTRY_SPACE_START_J:ENTRY_SPACE_END_J]

        diff_values = spawn_center[spawn_center != 47]
        if diff_values.size == 0:
            # Sem peça na área de spawn
            self.piece_visible = False
            return False

        # Há peça visível
        self.current_piece_id = np.unique(diff_values)[0]  # valor numérico único da peça (ex: 1, 2...)
        
        if not self.piece_visible:
            # A peça **acabou de aparecer**
            self.piece_visible = True
            if self.current_piece_id != self.last_piece_id:
                self.last_piece_id = self.current_piece_id
                return True  # Nova peça detectada!
        
        # Peça ainda visível, mas não é nova
        return False




    def step(self, action):

        # Mapeamento da ação para teclas do Game Boy
        from settings import action_map, release_map

        # Envia o input correspondente
        press_event = action_map[action]
        self.pyboy.send_input(press_event)

        # Avança alguns frames para a ação surtir efeito
        # for _ in range(10):
        self.pyboy.tick()

        # Libera a tecla pressionada (simula toque curto)
        release_event = release_map[press_event]
        self.pyboy.send_input(release_event)
        # self.pyboy.button(press_event)

        # Lê nova observação
        area = self.game_wrapper.game_area()
        tiles = np.array(area)

        if len(self.area_history) >= self.memory_size:
            piece_in_play = self.analize_piece_in_play(area = tiles)
        else:
            piece_in_play = False
        

        # Preprocessa a área do jogo
        obs = preprocess_game_area(tiles)
        obs = obs.flatten().astype(np.float32)

        # Recompensa simples: sobreviveu = 0.1, fim do jogo = -1.0
        if self.game_wrapper.game_over():
            reward = -100.0
            done = True
        else:
            reward = self.calc_reward(piece_in_play)
            done = False
 
        info = {}
        truncated = False
        return obs, reward, done, truncated,info
    
    ###################################################################################
    ###################################################################################

    def check_diff_between_tiles_frames(self, history_list: list[np.ndarray]) -> bool:
    
        area_different = np.any(history_list[0] != history_list[1])
        return area_different
    
    ###################################################################################
    ###################################################################################

    def check_fixed_gamearea_diff(self):
        fixed_gamearea_diff = self.check_diff_between_tiles_frames(self.fixed_gamearea_history)
        return fixed_gamearea_diff

    ###################################################################################
    ###################################################################################


    def update_current_score(self):
        self.score = self.game_wrapper.score 

    
    ###################################################################################
    ###################################################################################


    def update_current_gamearea(self):
        raw_game_area = self.game_wrapper.game_area()
        tiles = np.array(raw_game_area)
        self.game_area = self.preprocess_game_area(tiles)  # agora mantemos em 2D (18, 10)
        self.area_history.append(self.game_area)
    
    ###################################################################################
    ###################################################################################


    def update_current_fixed_gamearea(self):
        fixed_gamearea = self.get_fixed_gamearea()
        self.fixed_gamearea_history.append(fixed_gamearea)

    
    ###################################################################################
    ###################################################################################


    def get_fixed_gamearea(self) -> np.ndarray:
        current_game_area = self.area_history[-1]
        fixed_area = np.zeros_like(current_game_area, dtype=np.uint8)

        for i in reversed(range(current_game_area.shape[0])):  # De baixo para cima
            row = current_game_area[i]
            if np.any(row == 1):
                fixed_area[i] = row  # Copia a linha como parte da área fixa
            else:
                break  # Parou na primeira linha totalmente vazia
        return fixed_area.astype(np.uint8)


    ###################################################################################
    ###################################################################################

    def get_mobile_gamearea(self) -> np.ndarray:
        current_game_area = self.area_history[-1]  # shape (18, 10)
        fixed_game_area = self.get_fixed_gamearea()  # shape (18, 10)

        mobile = np.logical_and(current_game_area == 1, fixed_game_area == 0)
        return mobile.astype(np.uint8)  # shape (18, 10)

    ###################################################################################
    ###################################################################################

    def preprocess_game_area(self, area: np.ndarray) -> np.ndarray:
        """Recebe a matriz da área do jogo e retorna matriz binária 2D (18x10)"""
        processed = np.where(area == 47, 0, 1)  # 47 = vazio, qualquer outra coisa vira 1
        return processed.astype(np.uint8)




    def count_holes(self,area: np.ndarray) -> int:
        holes = 0
        rows, cols = area.shape

        for col in range(cols):
            column = area[:, col]
            seen_block = False
            for cell in column:
                if cell == 1:
                    seen_block = True
                elif seen_block and cell == 0:
                    holes += 1

        return holes

    ###################################################################################
    ###################################################################################

    def count_max_height(self, area: np.ndarray) -> int:
        """Retorna a altura máxima entre todas as colunas (quantas linhas até o primeiro bloco de cada coluna)."""
        
        rows, cols = area.shape
        max_height = 0

        for col in range(cols):
            column = area[:, col]
            for row_index, cell in enumerate(column):
                if cell == 1:
                    height = rows - row_index
                    max_height = max(max_height, height)
                    break  # achou o topo da pilha nesta coluna

        return max_height

    ###################################################################################
    ###################################################################################

    def height_penalty(self,normalized_height:np.array)->float:
        return ((-2 / (1 + math.exp(20 * (normalized_height - 0.4))) + 1) + 1)/2

    ###################################################################################
    ###################################################################################

    def calc_lower_pieces_reward(self, area: np.ndarray) -> float:

        blocks_in_rows = np.sum(area, axis=1)
        # multipliers = np.power(10, np.arange(18))/10e8
        multipliers = np.array([(2**n)/2**16 for n in range(18)])*10

        weighted_scores = blocks_in_rows * multipliers

        total_score = np.sum(weighted_scores)

        return total_score
    



    def calc_reward(self, piece_in_play: bool):

        # Atualiza a área de jogo
        self.update_current_gamearea()


        if piece_in_play:
            # Gera novo fixed_gamearea, mas ainda não adiciona ao histórico
            new_fixed_gamearea = self.get_fixed_gamearea()

            if len(self.fixed_gamearea_history) >= self.memory_size:
                old_fixed_gamearea = self.fixed_gamearea_history[-1]

            else:
                old_fixed_gamearea = new_fixed_gamearea.copy()

            # Adiciona ao histórico após a comparação
            self.fixed_gamearea_history.append(new_fixed_gamearea)

            # Buracos
            holes_before = self.count_holes(area = old_fixed_gamearea)
            holes_after = self.count_holes(area = new_fixed_gamearea)

            holes_diff = holes_after - holes_before
            hole_penalty = -holes_diff * 2 

            # Altura
            max_height_before = self.count_max_height(area = old_fixed_gamearea)
            max_height_after = self.count_max_height(area = new_fixed_gamearea)


            max_possible_height = new_fixed_gamearea.shape[0]
            normalized_height = max_height_after / max_possible_height  # entre 0 e 1

            height_penalty = - self.height_penalty(normalized_height)*10
 

            penalty = hole_penalty + height_penalty

            # lower_pieces_reward
            piece_position = new_fixed_gamearea - old_fixed_gamearea

            lower_pieces_reward = self.calc_lower_pieces_reward(area = piece_position)*2

            print("######## APLICANDO PENALIDADES ############")
            print(f"Penalidade: {penalty}")
            print(f"Altura: {max_height_after}")
            print(f"Buracos Antes: {holes_before}")
            print(f"Buracos Depois: {holes_after}")
            print(f"Buracos Diff {holes_diff}")
            print(f"Lower Pieces Reward: {lower_pieces_reward}")
            print("######## FIM DA PENALIDADE ############")

            

        else:
            penalty = 0
            hole_penalty=0
            height_penalty=0
            lower_pieces_reward = 0
        


        # Score
        last_score = self.score
        self.update_current_score()
        score_diff = self.score - last_score
        if score_diff == 0:
            score_reward = 0.0

        else:
            score_reward = score_diff * 100

        # Sobreviver
        survival_reward = 0.01

        # Penalidades
        reward = score_reward + survival_reward + lower_pieces_reward
        

        total_reward = reward + penalty


        reward_dict = {"hole_penalty":hole_penalty,
                       "height_penalty":height_penalty,
                       "lower_piece_reward":lower_pieces_reward,
                       "score_reward":score_reward,
                       "survival_reward":survival_reward,
                       }
        self.applied_rewards.append(reward_dict)

        return total_reward
        

    