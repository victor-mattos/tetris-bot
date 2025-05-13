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

        self.action_space = Discrete(10 * 4)  # 10 posições finais × 4 rotações


        self.score = 0
        self.last_score = 0
        self.step_count = 0

        self.memory_size = 3
        self.area_history = deque(maxlen=self.memory_size)
        self.fixed_gamearea_history = deque(maxlen=self.memory_size)

        self.last_piece_id = None
        self.current_piece_id = None
        self.piece_visible = False
        self.applied_rewards = []

    ###################################################################################
    ###################################################################################


    def reset(self, seed=None, options=None):
        import gc
        gc.collect()
        
        self.step_count = 0 
        self.last_score = 0
        self.score = 0
        self.area_history = deque(maxlen=self.memory_size)
        self.fixed_gamearea_history = deque(maxlen=self.memory_size)
        self.last_piece_id = None
        self.current_piece_id = None
        self.piece_visible = False
        self.applied_rewards = []
        
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

       # Atualiza observação
        area = self.game_wrapper.game_area()
        tiles = np.array(area)
        self.update_current_gamearea()

        piece_in_play = self.analize_piece_in_play(area = tiles)
        done = self.game_wrapper.game_over()
        # Define ações compostas: mover até a coluna X com rotação Y
        target_col = action // 4
        num_rotations = action % 4

        # Gira a peça
        for _ in range(num_rotations):
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
            self.pyboy.tick()

        # Obtem posição atual da peça (assuma central, ou pode tentar detectar)
        # Por simplicidade, vamos assumir que ela começa no centro (coluna 4)
        current_col = 4
        delta = target_col - current_col
        key = WindowEvent.PRESS_ARROW_RIGHT if delta > 0 else WindowEvent.PRESS_ARROW_LEFT

        for _ in range(abs(delta)):
            self.pyboy.send_input(key)
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT if delta > 0 else WindowEvent.RELEASE_ARROW_LEFT)
            self.pyboy.tick()

        # Derruba a peça
        # Avança alguns frames para a peça assentar
        # for _ in range(5):
        
        while not piece_in_play and not done:

            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN)
            

            game_area = self.game_wrapper.game_area()
            tiles = np.array(game_area)
            piece_in_play = self.analize_piece_in_play(area = tiles)
            done = self.game_wrapper.game_over()

        obs = preprocess_game_area(tiles).flatten().astype(np.float32)
        self.update_current_fixed_gamearea()
        done = self.game_wrapper.game_over()

        reward = -20.0 if done else self.calc_reward(piece_in_play = piece_in_play)
        
        self.step_count += 1

        if self.step_count > 1000:
            done = True
        return obs, reward, done, False, {}



    def manual_step(self, action):

        # Mapeamento da ação para teclas do Game Boy
        from settings import action_map, release_map

        self.update_current_gamearea()

        press_event = action_map[action]
        release_event = release_map[press_event]

        # Aperta a tecla
        self.pyboy.send_input(press_event)
        self.pyboy.tick()  # apenas 1 frame
        # Solta a tecla rapidamente
        self.pyboy.send_input(release_event)
        self.pyboy.tick()  # mais 1 frame para processar o release

        # Avança alguns frames apenas para o jogo "mostrar" a mudança, sem delay excessivo
        for _ in range(3):
            self.pyboy.tick()

        # Leitura do estado atual do jogo
        area = self.game_wrapper.game_area()
        tiles = np.array(area)

        if len(self.area_history) >= self.memory_size:
            piece_in_play = self.analize_piece_in_play(area = tiles)
        else:
            piece_in_play = False
        

        # Preprocessa a área do jogo
        obs = preprocess_game_area(tiles)
        obs = obs.flatten().astype(np.float32)

        self.update_current_fixed_gamearea()

        # Recompensa simples: sobreviveu = 0.1, fim do jogo = -1.0
        if self.game_wrapper.game_over():
            reward = -100.0
            done = True
        else:
            reward = self.calc_reward(piece_in_play)
            done = False
 
        info = {}
        truncated = False
        self.step_count += 1
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


    def calc_bumpiness(self, area: np.ndarray) -> float:
        # 1. Sanitiza: qualquer valor > 0 vira 1 (evita problemas com valores incorretos)
        area = np.where(area > 0, 1, 0)

        # 2. Calcula altura de cada coluna
        heights = [area.shape[0] - np.argmax(col[::-1]) if np.any(col) else 0
                for col in area.T]

        # 3. Diferença de altura entre colunas vizinhas
        diffs = np.abs(np.diff(heights))

        # 4. Penaliza apenas onde há diferença (bumpiness real)
        return np.count_nonzero(diffs)


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

    def calc_lower_pieces_reward(self) -> float:

               # Verifica se tem mais de uma peça em jogo
        if len(self.fixed_gamearea_history) >= 2:
            
            old_fixed_gamearea = self.fixed_gamearea_history[-2]
            new_fixed_gamearea = self.fixed_gamearea_history[-1] 
            
            area = new_fixed_gamearea - old_fixed_gamearea

            blocks_in_rows = np.sum(area, axis=1)
            # multipliers = np.power(10, np.arange(18))/10e8
            multipliers = np.array([(2**n)/2**17 for n in range(18)])

            weighted_scores = blocks_in_rows * multipliers

            reward = np.sum(weighted_scores)

        else:
            reward = 0
        return reward
    
    def calc_cleaning_line_progress_reward(self):
        # Verifica se tem histórico suficiente
        if len(self.fixed_gamearea_history) >= 2:
            old_fixed_gamearea = self.fixed_gamearea_history[-2]
            new_fixed_gamearea = self.fixed_gamearea_history[-1]

            # Soma o número de blocos (1s) por linha
            old_line_counts = np.count_nonzero(old_fixed_gamearea, axis=1)
            new_line_counts = np.count_nonzero(new_fixed_gamearea, axis=1)

            # Calcula o delta de preenchimento por linha
            line_diff = new_line_counts - old_line_counts

            # Considera apenas linhas já iniciadas
            active_lines = old_line_counts > 0
            relevant_diffs = line_diff * active_lines

            # Garante que só consideramos progresso positivo, e mitiga valores não binários
            positive_diffs = [diff for diff in relevant_diffs if diff > 0]

            # Conta o número de linhas que tiveram progresso
            num_lines_with_progress = np.count_nonzero(positive_diffs)*0.25

            # Calcula a recompensa total com peso progressivo
            weighted_reward = np.sum([np.log2(diff + 1) for diff in positive_diffs])

            # Combina os dois: número de linhas ativas e intensidade do progresso
            final_reward = num_lines_with_progress + weighted_reward

            return final_reward
        else:
            return 0.0


    def calc_bumpiness_penalty(self):
         # Verifica se tem mais de uma peça em jogo
        if len(self.fixed_gamearea_history) >= 2:
            
            old_fixed_gamearea = self.fixed_gamearea_history[-2]
            new_fixed_gamearea = self.fixed_gamearea_history[-1]

            old_bumpiness = self.calc_bumpiness(area = old_fixed_gamearea)
            new_bumpiness = self.calc_bumpiness(area = new_fixed_gamearea)

            delta_bumpiness = new_bumpiness - old_bumpiness 
            bumpiness_penalty = -delta_bumpiness * 0.25

        else:
            bumpiness_penalty = 0

        return bumpiness_penalty 


    def calc_height_penalty(self):
        
        # Verifica se tem mais de uma peça em jogo
        if len(self.fixed_gamearea_history) >= 2:
            
            old_fixed_gamearea = self.fixed_gamearea_history[-2]
            new_fixed_gamearea = self.fixed_gamearea_history[-1]   

            max_height_before = self.count_max_height(area = old_fixed_gamearea)
            max_height_after = self.count_max_height(area = new_fixed_gamearea)
            max_possible_height = new_fixed_gamearea.shape[0]

            if max_height_after > max_height_before:
                normalized_height = max_height_after / max_possible_height  # entre 0 e 1
                height_penalty = - self.height_penalty(normalized_height)*2

            else:
                height_penalty = 0
        
        else:
            height_penalty = 0
        
        return height_penalty

    def calc_hole_penalty(self):

        # Verifica se tem mais de uma peça em jogo
        if len(self.fixed_gamearea_history) >= 2:
            
            old_fixed_gamearea = self.fixed_gamearea_history[-2]
            new_fixed_gamearea = self.fixed_gamearea_history[-1]   

         
            holes_before = self.count_holes(area = old_fixed_gamearea)
            holes_after = self.count_holes(area = new_fixed_gamearea)

            holes_diff = holes_after - holes_before

            if holes_diff>0:
                hole_penalty = -holes_diff
            else:
                hole_penalty = 0
        else:
            hole_penalty = 0
        
        return hole_penalty 
            
            
    def calc_score_reward(self):
        # Score
        last_score = self.score
        self.update_current_score()
        score_diff = self.score - last_score

        if score_diff == 0:
            score_reward = 0.0

        else:
            score_reward = score_diff*10

        return score_reward
    
    def calc_reward(self, piece_in_play: bool):

        # Atualiza a área de jogo
        if piece_in_play:
            
            height_penalty = self.calc_height_penalty()
            hole_penalty = self.calc_hole_penalty()
            bumpiness_pentalty = self.calc_bumpiness_penalty()
            penalty = hole_penalty + height_penalty + bumpiness_pentalty
        else:
            penalty = 0
            hole_penalty=0
            height_penalty=0
            bumpiness_pentalty = 0
               
        score_reward = self.calc_score_reward()
        safe_reward = self.calc_lower_pieces_reward()

        if score_reward == 0 and self.count_max_height(self.fixed_gamearea_history[-1]) < 16:
            clean_line_progress_reward = self.calc_cleaning_line_progress_reward()

        else:
            clean_line_progress_reward = 0

        # Sobreviver
        survival_reward = 1
   
        # Penalidades
        reward = score_reward + survival_reward + safe_reward + clean_line_progress_reward  
        
        total_reward = reward + penalty

        total_reward = total_reward

        reward_dict = {"hole_penalty":hole_penalty,
                       "height_penalty":height_penalty,
                       "bumpiness_penalty":bumpiness_pentalty,
                       "clean_line_progress_reward":clean_line_progress_reward,
                       "score_reward":score_reward,
                       "survival_reward":survival_reward,
                       "safe_reward":safe_reward,
                       
                       }
        self.applied_rewards.append(reward_dict)
        # print(f"Score Reward: {total_reward}")
        return total_reward
        

    