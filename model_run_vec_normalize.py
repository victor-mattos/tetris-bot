import os
import time
import numpy as np
import pandas as pd 
from pyboy.pyboy import *
from settings import ROM_PATH
from tetris_env import TetrisEnv
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ---------------------------------------------------------
# 1. CONFIGURAÇÃO DO AMBIENTE VETORIZADO E NORMALIZAÇÃO
# ---------------------------------------------------------
print("\n[INFO] Inicializando ambiente base e vetorizado...")

# Instanciação do ambiente base (onde vivem as funções do jogo)
raw_env = TetrisEnv(window_type="SDL2", memory_size=50)

# Envelopamento para vetorização (necessário para o VecNormalize)
env = DummyVecEnv([lambda: raw_env])

# Carregamento do normalizador exato utilizado no treinamento de 2M steps
norm_path = "models/vecnorm_2m.pkl" 
env = VecNormalize.load(norm_path, env)

# CRÍTICO: Desligar o treinamento do normalizador na inferência para evitar Data Leakage
env.training = False 
env.norm_reward = False

# ---------------------------------------------------------
# 2. CARREGAMENTO DO MODELO
# ---------------------------------------------------------
print("\n[INFO] Carregando modelo DQN...")
model = DQN.load("models/best/best_model")

# ---------------------------------------------------------
# 3. LOOP DE INFERÊNCIA
# ---------------------------------------------------------
obs = env.reset()
reward_sum = 0
aux_list = []
step_count = 0
is_done = False # Uso de uma flag específica para controle do loop

while not is_done:

    # O modelo prevê a ação baseada na observação normalizada
    # action é um array (ex: [3]) devido ao VecEnv
    action, _ = model.predict(obs, deterministic=True)
    print(f"\n[STEP {step_count}] Ação tomada: {action[0]}")

    # Execução da ação: VecEnv retorna 4 ARRAYS (não 5 escalares)
    obs, rewards, dones, infos = env.step(action)
    # raw_env.render()
    # Extraímos os valores do array [0] já que temos apenas 1 ambiente rodando
    current_reward = rewards[0]
    is_done = dones[0]
    
    reward_sum += current_reward
    aux_list.append(reward_sum)
    
    print(f"[STEP {step_count}] Recompensa: {current_reward}")
    print(f"[STEP {step_count}] Done? {is_done}")
    
    # Para atributos customizados, acessamos SEMPRE o 'raw_env'
    print(f"[STEP {step_count}] Nova observação (Grid Real):")
    try:
        print(np.array(raw_env.game_wrapper.game_area()).reshape(18, 10))
    except Exception as e:
        print(f"[AVISO] Não foi possível renderizar a grid: {e}")

    # Pausa para visualização
    time.sleep(0.1)
    step_count += 1

print(f"\n[FIM] Episódio finalizado em {step_count} steps. Recompensa Total: {reward_sum}")

# ---------------------------------------------------------
# 4. ANÁLISE DE DADOS PÓS-EPISÓDIO
# ---------------------------------------------------------
try:
    # Novamente, acessamos as métricas customizadas através do 'raw_env'
    df = pd.DataFrame(raw_env.applied_rewards)
    df['total'] = df.sum(axis=1)
    print("\n[MÉTRICAS] Resumo das recompensas aplicadas:")
    print(df)
except AttributeError:
    print("\n[AVISO] Atributo 'applied_rewards' não encontrado no ambiente base.")