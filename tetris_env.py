import gymnasium as gym
# from gym import Env

import time
import math
import numpy as np

from gymnasium.spaces import Discrete, Box
from pyboy import PyBoy,WindowEvent
# from pyboy.utils import WindowEvent
from collections import deque
from stable_baselines3 import PPO,DQN

from settings import ROM_PATH
from typing import Optional, Tuple,List


class TetrisEnv(gym.Env):

    def __init__(self, window_type: str = "headless", memory_size: int = 3):
        super().__init__()

        # self.observation_space = Box(low=0.0, high=1.0, shape=(180,), dtype=np.float32)

        self.action_space = Discrete(10 * 4)  # 10 posições finais × 4 rotações
        # self.observation_space = Box(low=0.0, high=1.0, shape=(40,18, 10), dtype=np.float32)

        # self.observation_space = Box(
        #     low=-np.inf,
        #     high=np.inf,
        #     shape=(40, 13),
        #     dtype=np.float32
        # )
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(40, 6),
            dtype=np.float32
        )

        self.window_type = window_type
        self.pyboy = PyBoy(ROM_PATH, window_type=self.window_type, game_wrapper=True, openai_gym=True)
        self.game_wrapper = self.pyboy.game_wrapper()
        self.game_wrapper.start_game()

        # Roda alguns ticks para a janela não travar
        for _ in range(5):
            self.pyboy.tick()



        self.score = 0
        self.last_score = 0
        self.step_count = 0

        self.memory_size = memory_size
        self.area_history = deque(maxlen=self.memory_size)
        self.fixed_gamearea_history = deque(maxlen=self.memory_size)
        self.possible_plays = deque(maxlen=self.memory_size)

        self.last_piece_id = None
        self.current_piece_id = None
        self.piece_visible = False
        self.applied_rewards = []

        from settings import PIECES_SHAPES, MEMORY_PIECE_MAP, PIECE_ORDER

        self.std_pieces_shapes = PIECES_SHAPES
        self.memory_piece_map = MEMORY_PIECE_MAP
        self.piece_order = PIECE_ORDER


    ###################################################################################
    ###################################################################################


    def reset(self, seed=None, options=None):
        import gc
        gc.collect()

        self.step_count = 0
        self.last_score = 0
        self.score = 0
        self.area_history.clear()
        self.fixed_gamearea_history.clear()
        self.possible_plays.clear()
        self.last_piece_id = None
        self.current_piece_id = None
        self.piece_visible = False
        self.applied_rewards = []

            
        self.game_wrapper.reset_game()

        for _ in range(5):
            self.pyboy.tick()

        obs_feature = self.get_all_obs_features()
        gc.collect()

        return obs_feature, {}

    
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
    
    def analize_piece_in_play(self) -> bool:
        """Simplificada: Retorna True se houver QUALQUER bloco na área de spawn."""
        ENTRY_SPACE_START_I = 0
        ENTRY_SPACE_END_I = 3
        ENTRY_SPACE_START_J = 4
        ENTRY_SPACE_END_J = 7

        area = self.get_current_gamearea()
        spawn_center = area[ENTRY_SPACE_START_I:ENTRY_SPACE_END_I, ENTRY_SPACE_START_J:ENTRY_SPACE_END_J]

        # Verifica simplesmente se existe pelo menos um valor não-zero na sub-área
        return np.any(spawn_center != 0)

    ###################################################################################
    ###################################################################################


    def rotate_piece_to_target_position(self,num_rotations:int):
        # Gira a peça
        for _ in range(num_rotations):
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
            self.pyboy.tick()

    ###################################################################################
    ###################################################################################

    def move_piece_to_target_column(self, target_col:int):
        ENTRY_SPACE_START_J = 3
        
        piece_shape = self.get_new_piece_shape()
        # Verifica se cada coluna tem pelo menos um '1'
        cols_occupied = np.any(piece_shape, axis=0)
        # Indexa a primeira coluna ocupada
        current_col = np.argmax(cols_occupied) + ENTRY_SPACE_START_J

        delta = target_col - current_col
        key = WindowEvent.PRESS_ARROW_RIGHT if delta > 0 else WindowEvent.PRESS_ARROW_LEFT

        for _ in range(abs(delta)):
            self.pyboy.send_input(key)
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT if delta > 0 else WindowEvent.RELEASE_ARROW_LEFT)
            self.pyboy.tick()

    ###################################################################################
    ###################################################################################

    def crop_piece(self,piece:np.ndarray)->np.ndarray:
        """Remove linhas e colunas que são todas zeros."""
        # Detect rows and columns that have any 1
        rows = np.any(piece, axis=1)
        cols = np.any(piece, axis=0)
        # Slice array to non-zero rows/cols
        cropped = piece[np.ix_(rows, cols)]
        return cropped

    ###################################################################################
    ###################################################################################

    def place_piece_in_standard_position(self):

        piece_shape = self.get_new_piece_shape()
        piece_shape = self.crop_piece(piece = piece_shape)

        done = self.game_wrapper.game_over()
        match_position = False
        
        while not match_position and not done:
            done = self.game_wrapper.game_over()

            for name, shape in self.std_pieces_shapes.items():

                if np.array_equal(piece_shape, shape):
                    
                    match_position = True
            
            if match_position == False:
                self.rotate_piece_to_target_position(num_rotations=1)
                piece_shape = self.get_new_piece_shape()
                piece_shape = self.crop_piece(piece = piece_shape)

    ###################################################################################
    ###################################################################################

    def move_piece_down(self,n_clicks:int=4):
        for _ in range(n_clicks):
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN)

    ###################################################################################
    ###################################################################################

    def hard_drop_piece(self):
        self.move_piece_down(n_clicks=6)
        piece_in_play = self.analize_piece_in_play()
        done = self.game_wrapper.game_over()
        
        while piece_in_play==False and done == False:
            done = self.game_wrapper.game_over()
            self.move_piece_down(n_clicks=2)          
            piece_in_play = self.analize_piece_in_play()

    ###################################################################################
    ###################################################################################

    def get_all_rotations(self,piece:np.ndarray)->list[np.ndarray]:
        """
        Gera todas as rotações válidas da peça (até 4),
        aparando linhas/colunas vazias e removendo duplicatas.
        Considera shape para diferenciar 1x4 de 4x1.
        """
        rotations = []
        seen = set()

        for k in range(4):
            rotated = np.rot90(piece, -k)

            # Remove linhas/colunas vazias
            rows = np.any(rotated, axis=1)
            cols = np.any(rotated, axis=0)
            if not rows.any() or not cols.any():
                continue

            trimmed = rotated[np.ix_(rows, cols)].astype(np.uint8, copy=False)
            # Inclui o shape na chave para diferenciar 1x4 de 4x1
            key = (trimmed.shape, trimmed.tobytes(order="C"))

            if key not in seen:
                rotations.append(trimmed)
                seen.add(key)

        return rotations 

    
    ###################################################################################
    ###################################################################################

    def get_new_piece_shape(self):
        ENTRY_SPACE_START_I = 0
        ENTRY_SPACE_END_I = 4
        ENTRY_SPACE_START_J = 3
        ENTRY_SPACE_END_J = 7


        current_area = self.get_current_gamearea() 
        board_height, board_width = current_area.shape

        piece_shape = current_area[ENTRY_SPACE_START_I:ENTRY_SPACE_END_I, ENTRY_SPACE_START_J:ENTRY_SPACE_END_J]

        return piece_shape

    ###################################################################################
    ###################################################################################

    def get_all_pieces_positions(self):


        piece_shape = self.get_new_piece_shape()
        piece_rotations = self.get_all_rotations(piece = piece_shape)
        possible_positions = []


        current_fixed_gamearea = self.get_fixed_gamearea().astype(int)
        possible_positions,moves = self.generate_all_final_positions( piece_rotations=piece_rotations, board=current_fixed_gamearea)

        # self.possible_plays.append([possible_positions,moves])

        return possible_positions,moves
    
    ###################################################################################
    ###################################################################################

    def clear_full_lines(self,board: np.ndarray) -> np.ndarray:

        """Remove linhas completas (todas as colunas == 1) e adiciona linhas vazias no topo."""
        board_h, board_w = board.shape
        # Mantém apenas as linhas que não estão completas
        not_full_rows = [row for row in board if not np.all(row == 1)]
        # Conta quantas linhas foram removidas
        num_cleared = board_h - len(not_full_rows)
        # Reconstroi o tabuleiro com linhas vazias no topo
        new_board = np.vstack([np.zeros((num_cleared, board_w), dtype=np.uint8),
                            np.array(not_full_rows, dtype=np.uint8)])
        return new_board
    
    ###################################################################################
    ###################################################################################

    def generate_all_final_positions(self,
        piece_rotations: list[np.ndarray], board: np.ndarray
        ) -> Tuple[list[np.ndarray], list[Tuple[int, int]]]:
        """
        Gera todas as posições finais possíveis para uma peça,
        considerando todas as rotações e colunas do tabuleiro.

        Args:
            piece_rotations: lista de np.ndarray (diferentes rotações da peça).
            board: estado atual do tabuleiro (np.ndarray 2D com 0 = vazio, 1 = ocupado).

        Returns:
            results: lista de tabuleiros após cada posição final válida.
            moves: lista de pares (rotacao, coluna) correspondentes a cada tabuleiro.
        """
        board_h, board_w = board.shape
        results = []
        moves = []

        for rot_idx, piece in enumerate(piece_rotations):
            ph, pw = piece.shape
            
            # percorre todas as colunas possíveis
            for col in range(board_w - pw + 1):
                row = 0

                # descer a peça até colidir
                while True:
                    if row + ph > board_h:  
                        break  # passou do fundo

                    region = board[row:row+ph, col:col+pw]

                    if np.any((region + piece) > 1):
                        break  # colisão

                    row += 1

                final_row = row - 1
                if final_row < 0:
                    continue  # não coube no tabuleiro

                new_board = board.copy()
                new_board[final_row:final_row+ph, col:col+pw] += piece

                new_board = self.clear_full_lines(new_board)
                results.append(new_board)
                moves.append((rot_idx, col))

        return results, moves

    ###################################################################################
    ###################################################################################

    def pad_results(self,
        results: List[np.ndarray], 
        moves: List[Tuple[int, int]], 
        max_len: int = 40, 
        board_shape: Tuple[int, int] = (18, 10)
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Ajusta os resultados para tamanho fixo (max_len).
        Repete o último estado/movimento até completar.
        """
        n = len(results)

        if n == 0:
            # Nenhum estado válido -> devolve todos zeros
            boards = np.zeros((max_len, *board_shape), dtype=np.uint8)
            moves_out = [(0, 0)] * max_len
            return boards, moves_out

        # Empilha resultados em (n, H, W)
        boards = np.stack(results, axis=0).astype(np.uint8)
        moves_out = list(moves)

        # Se precisar completar
        if n < max_len:
            last_board = boards[-1]
            last_move = moves_out[-1]
            pad_count = max_len - n

            boards = np.concatenate(
                [boards, np.repeat(last_board[None, :, :], pad_count, axis=0)],
                axis=0
            )
            moves_out.extend([last_move] * pad_count)

        elif n > max_len:
            # Corta o excesso (nunca deveria passar de 40 no Tetris clássico)
            boards = boards[:max_len]
            moves_out = moves_out[:max_len]

        return boards, moves_out

    ###################################################################################
    ###################################################################################

    def calc_area_features(self, area:np.array)->np.array:
        
        old_fixed_gamearea = self.get_fixed_gamearea()

        new_column_heights = self.get_column_heights(area = area)
        new_max_height = max(new_column_heights)/18
        new_mean_height = np.mean(new_column_heights)/18

        old_column_heights = self.get_column_heights(area = old_fixed_gamearea)
        old_max_height = max(old_column_heights)/18


        clear_line = max(old_max_height-new_max_height,0)*18
        
            
        bumpiness = self.calc_bumpiness(area = area)/180
        # holes = self.count_holes(area = area)/180
        holes = self.calc_hole_penalty(old_fixed_gamearea=old_fixed_gamearea, new_fixed_gamearea=area)/180
        

        clean_progress,_ = self.calc_cleaning_line_progress(old_fixed_gamearea=old_fixed_gamearea,
                                                            new_fixed_gamearea=area)
        clean_progress = clean_progress/10

        

        return np.array([new_max_height, new_mean_height, bumpiness, holes,clean_progress, clear_line])

    ###################################################################################
    ###################################################################################

    def get_next_piece_array(self)->np.array:
        next_piece_id =  self.pyboy.get_memory_value(0xC213) 
        piece = self.memory_piece_map.get(str(next_piece_id), None)

        one_hot = np.zeros(len(self.piece_order), dtype=np.float32)
        if piece is not None and piece in self.piece_order:
                idx = self.piece_order.index(piece)
                one_hot[idx] = 1.0

        return one_hot


    def get_all_obs_features(self)->np.array:

        obs_new, moves = self.get_all_pieces_positions()
        obs_new_padded, moves_padded = self.pad_results(obs_new, moves, max_len=40, board_shape=(18,10))
        
        next_piece_vec = self.get_next_piece_array()
        all_features = []

        for area in obs_new_padded:
            area_feature = self.calc_area_features(area=area)
            # feature_with_piece = np.concatenate([area_feature, next_piece_vec])
            all_features.append(area_feature)

        self.possible_plays.append([obs_new_padded,moves_padded])
    
        return np.array(all_features, dtype=np.float32)

    ###################################################################################
    ###################################################################################

    def step(self, action):
        '''
            self = TetrisEnv(window_type="headless", memory_size=50)
            obs_feature, _ = self.reset()
            model = DQN.load("dqn_tetris_v29")
            action, _ = model.predict(obs_feature, deterministic=True)
        '''


        # print(self.pyboy.get_memory_value(0xC213))
        # self.pyboy.screen_image()

        done = self.game_wrapper.game_over()

        possible_plays = self.possible_plays[-1]
        model_move = possible_plays[-1][action]



        # model_move = moves_padded[action]

        num_rotations = model_move[0]
        target_col = model_move[1]
        

        self.rotate_piece_to_target_position(num_rotations = num_rotations)
        self.move_piece_to_target_column(target_col= target_col)
        self.hard_drop_piece()      

        done = self.game_wrapper.game_over()


        if not done:

            self.place_piece_in_standard_position()
            self.update_current_gamearea()
            self.update_current_fixed_gamearea()
            
            # area = self.game_wrapper.game_area()
            # tiles = np.array(area)
            # obs = self.preprocess_game_area(tiles)

            obs_feature = self.get_all_obs_features()

            reward = self.calc_reward()
            
            self.step_count += 1

        else: 

            obs_feature = self.get_all_obs_features()
            reward = -100

        if self.step_count > 1000:
            done = True
        
        return obs_feature, reward, done, False, {}


    
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

    def get_current_gamearea(self)->np.ndarray:
        raw_game_area = self.game_wrapper.game_area()
        tiles = np.array(raw_game_area)
        # self.game_area = self.preprocess_game_area(tiles)  # agora mantemos em 2D (18, 10)
        game_area = np.where(tiles == 47, 0, 1)

        return game_area

    ###################################################################################
    ###################################################################################

    def update_current_gamearea(self):
        gamearea = self.get_current_gamearea()
        self.area_history.append(gamearea)
    
    ###################################################################################
    ###################################################################################


    def update_current_fixed_gamearea(self):
        fixed_gamearea = self.get_fixed_gamearea()
        self.fixed_gamearea_history.append(fixed_gamearea)

    
    ###################################################################################
    ###################################################################################


    def get_fixed_gamearea(self) -> np.ndarray:
        if len(self.area_history) == 0:
        # Retorna uma matriz vazia (sem peças posicionadas)
            return np.zeros((18, 10), dtype=np.uint8)

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

    def get_column_heights(self, area: np.ndarray) -> list[int]:
        """
        Retorna a altura de cada coluna do tabuleiro.
        A altura é calculada da primeira célula de cima (linha 0) até a primeira
        ocorrência de um número diferente de 1. Se todos forem 1, altura será 0.
        """
        rows, cols = area.shape
        heights = []

        for col_index in range(cols):
            col = area[:, col_index]
            height = 0
            for r in range(rows):  # de cima (0) para baixo (17)
                if col[r] == 1:
                    height = rows - r
                    break
            else:
                # Todos os valores são 1 na coluna
                height = 0
            heights.append(height)

        return heights

    ###################################################################################
    ###################################################################################

    def calc_bumpiness(self,area:np.ndarray)->int:
        """Calcula a bumpiness como a soma das diferenças absolutas de altura entre colunas adjacentes."""
        cols = area.shape[1]

        heights = self.get_column_heights(area = area)

        bumpiness = 0
        for i in range(cols - 1):
            bumpiness += abs(heights[i] - heights[i+1])

        return bumpiness

    ###################################################################################
    ###################################################################################

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

    def calc_lower_pieces_reward(self, old_fixed_gamearea:np.ndarray = None, new_fixed_gamearea:np.ndarray = None) -> float:

               # Verifica se tem mais de uma peça em jogo
        if len(self.fixed_gamearea_history) >= 2:
            if old_fixed_gamearea is None or new_fixed_gamearea is None:
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

        reward = min(reward,6)
        return reward

    ###################################################################################
    ###################################################################################

    def calc_cleaning_line_progress(self, old_fixed_gamearea:np.ndarray = None, new_fixed_gamearea:np.ndarray = None):

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
            num_lines_with_progress = np.count_nonzero(positive_diffs)

            return num_lines_with_progress, positive_diffs

    ###################################################################################
    ###################################################################################

    def calc_cleaning_line_progress_reward(self, old_fixed_gamearea:np.ndarray = None, new_fixed_gamearea:np.ndarray = None):
                # Verifica se tem histórico suficiente
        if len(self.fixed_gamearea_history) >= 2:

            if old_fixed_gamearea is None or new_fixed_gamearea is None:
                old_fixed_gamearea = self.fixed_gamearea_history[-2]
                new_fixed_gamearea = self.fixed_gamearea_history[-1]

            num_lines_with_progress, positive_diffs = self.calc_cleaning_line_progress(old_fixed_gamearea=old_fixed_gamearea, 
                                                                                       new_fixed_gamearea=new_fixed_gamearea)

            # Calcula a recompensa total com peso progressivo
            weighted_reward = np.sum([np.log2(diff + 1) for diff in positive_diffs])

            # Combina os dois: número de linhas ativas e intensidade do progresso
            final_reward = num_lines_with_progress + weighted_reward
        else:
            final_reward = 0

        return final_reward


    ###################################################################################
    ###################################################################################

    def calc_bumpiness_penalty(self, old_fixed_gamearea:np.ndarray = None, new_fixed_gamearea:np.ndarray = None):
         # Verifica se tem mais de uma peça em jogo
        if len(self.fixed_gamearea_history) >= 2:
            if old_fixed_gamearea is None or new_fixed_gamearea is None:
                old_fixed_gamearea = self.fixed_gamearea_history[-2]
                new_fixed_gamearea = self.fixed_gamearea_history[-1]

            old_bumpiness = self.calc_bumpiness(area = old_fixed_gamearea)
            new_bumpiness = self.calc_bumpiness(area = new_fixed_gamearea)

            delta_bumpiness = new_bumpiness - old_bumpiness 
            bumpiness_penalty = -delta_bumpiness

        else:
            bumpiness_penalty = 0

        return bumpiness_penalty 

    ###################################################################################
    ###################################################################################

    def calc_height_penalty(self, old_fixed_gamearea:np.ndarray = None, new_fixed_gamearea:np.ndarray = None):
        
        # Verifica se tem mais de uma peça em jogo
        if len(self.fixed_gamearea_history) >= 2:
            
            old_fixed_gamearea = self.fixed_gamearea_history[-2]
            new_fixed_gamearea = self.fixed_gamearea_history[-1]   

            max_height_before = self.count_max_height(area = old_fixed_gamearea)
            max_height_after = self.count_max_height(area = new_fixed_gamearea)
            max_possible_height = new_fixed_gamearea.shape[0]

            if max_height_after > max_height_before:
                normalized_height = max_height_after / max_possible_height  # entre 0 e 1
                height_penalty = - self.height_penalty(normalized_height)

            else:
                height_penalty = 0
        
        else:
            height_penalty = 0
        
        return height_penalty

    ###################################################################################
    ###################################################################################

    def calc_hole_penalty(self, old_fixed_gamearea:np.ndarray = None, new_fixed_gamearea:np.ndarray = None):

        # Verifica se tem mais de uma peça em jogo
        if len(self.fixed_gamearea_history) >= 2:
            
            if old_fixed_gamearea is None or new_fixed_gamearea is None:
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
            
    ###################################################################################
    ###################################################################################
            
    def calc_score_reward(self):
        # Score
        last_score = self.score
        self.update_current_score()
        score_diff = self.score - last_score

        if score_diff == 0:
            score_reward = 0.0

        else:
            score_reward = score_diff

        return score_reward

    ###################################################################################
    ###################################################################################

    def calc_reward(self):

        

        # Atualiza a área de jogo
        HEIGHT_MULT = 0.6
        HOLE_MULT = 0.36
        BUMP_MULT = 0.18

        ## Penalidades
        height_penalty = self.calc_height_penalty()*HEIGHT_MULT
        hole_penalty = self.calc_hole_penalty()*HOLE_MULT
        bumpiness_pentalty = -max(self.calc_bumpiness_penalty(),0)*BUMP_MULT
        penalty = hole_penalty + height_penalty + bumpiness_pentalty

        
        ## Rewards
        score_reward = self.calc_score_reward()
        safe_reward = self.calc_lower_pieces_reward()*0.5


        if score_reward == 0:
            # if hole_penalty == 0:
            clean_line_progress_reward = min(self.calc_cleaning_line_progress_reward()*0.6,3)

        else:
            clean_line_progress_reward = 0

        # Sobreviver
        survival_reward = 0.1
   
        # Penalidades
        reward = score_reward + survival_reward + safe_reward + clean_line_progress_reward  
        
        total_reward = reward + penalty

        reward_dict = {
                        "hole_penalty":hole_penalty,
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

    ###################################################################################
    ###################################################################################


    