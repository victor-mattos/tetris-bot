import os
import torch
from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from typing import Callable
from tetris_env import TetrisEnv

# Certifique-se de que a função make_env e o TetrisEnv estejam no env_utils.py com window_type="null"
from env_utils import make_env 

datetime_now = datetime.now()
date_ref = datetime_now.strftime("%Y-%m-%d")

# 1. Função para Learning Rate Schedule (Decaimento Linear)
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

log_dir = "./logs_csv"
os.makedirs(log_dir, exist_ok=True)
model_save_path = os.path.join(os.getcwd(), f"models\\{date_ref}")
os.makedirs(model_save_path, exist_ok=True)


# =====================================================================
# BLOCO DE PROTEÇÃO OBRIGATÓRIO PARA MULTIPROCESSAMENTO
# Tudo que executa deve ficar daqui para baixo, indentado!
# =====================================================================
if __name__ == "__main__":
    
    NUM_CPUS = 6 
    print(f"\n[INFO] Inicializando com {NUM_CPUS} processos paralelos (start_method='spawn')...")

    # 1. Ambientes de TREINO
    env = SubprocVecEnv(
        [make_env(i, log_dir) for i in range(NUM_CPUS)], 
        start_method="spawn" 
    )
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.)
    print("[INFO] Ambientes de TREINO criados com sucesso!")

    # 2. Ambientes de AVALIAÇÃO (Limitado a 4 para não pesar)
    EVAL_CPUS = min(4, NUM_CPUS)
    eval_env = SubprocVecEnv(
        [make_env(i + NUM_CPUS, log_dir) for i in range(EVAL_CPUS)], 
        start_method="spawn"
    )
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)
    eval_env.training = False
    print("[INFO] Ambientes de AVALIAÇÃO criados com sucesso! Iniciando modelo...")

    # 3. Definição do Modelo Otimizado
    model = DQN(
        # "MlpPolicy",
        "MultiInputPolicy",
        env,
        learning_rate=linear_schedule(1.0e-4), 
        buffer_size=1_000_000,               
        learning_starts=50_000,              
        batch_size=512,
        tau=0.005,                           
        gamma=0.999,                         
        train_freq=max(1, 4 // NUM_CPUS),      
        # train_freq = 1,
        # gradient_steps=NUM_CPUS,
        gradient_steps=1,  
        target_update_interval=1,            
        exploration_fraction=0.3,            
        exploration_final_eps=0.01,          
        verbose=1,
        tensorboard_log="logs/",
        # policy_kwargs=dict(net_arch=[256, 256]), 
        policy_kwargs= dict(net_arch=[512, 512, 256]),
        device="cuda"
    )

    # 4. Callbacks e Treinamento
    eval_cb = EvalCallback(
        eval_env, 
        best_model_save_path=os.path.join(model_save_path, "best"), 
        log_path=log_dir,
        eval_freq=max(1, 50_000 // NUM_CPUS), # Evita floats e zeros 
        n_eval_episodes=50,                  
        deterministic=True
    )

    ckpt_cb = CheckpointCallback(
        save_freq=max(1, 100_000 // NUM_CPUS), # Ajuste para múltiplos ambientes
        save_path=os.path.join(model_save_path, "ckpts"), 
        name_prefix="dqn_5m"
    )

    print("\n[INFO] Iniciando treinamento para 2.000.000 steps...")
    model.learn(total_timesteps=5_000_000, callback=[eval_cb, ckpt_cb])

    # 5. Salvamento Seguro (Modelo + Normalizador) e Encerramento
    model.save(os.path.join(model_save_path, "dqn_5m_area_next_piece_features.zip"))
    env.save(os.path.join(model_save_path, "vecnorm_2m.pkl")) 
    
    env.close()
    eval_env.close()
    print("\n[INFO] Treinamento Finalizado com Sucesso.")