

# from stable_baselines3 import PPO, DQN
# # from sb3_contrib import QRDQN

# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
# from tetris_env import TetrisEnv

# import os
# import torch

# log_dir = "./logs_csv"
# os.makedirs(log_dir, exist_ok=True)

# def make_env():
#     return Monitor(TetrisEnv(window_type="headless"), filename=os.path.join(log_dir, "monitor.csv"))

# root = os.getcwd()
# model_save_path = os.path.join(root, "models")
# os.makedirs(model_save_path, exist_ok=True)

# # Wrap com DummyVecEnv
# env = DummyVecEnv([make_env])
# eval_env = DummyVecEnv([make_env])
# eval_env = VecNormalize.load("vecnorm_v32.pkl", eval_env) if os.path.exists("vecnorm_v32.pkl") else VecNormalize(eval_env)
# eval_env.training = False; eval_env.norm_reward = False

# model = DQN(
#     "MlpPolicy",
#     # "CnnPolicy",
#     env,
#     learning_rate=1.5e-4,                   # Aprendizado mais agressivo
#     buffer_size=500_000,                 # Reforça renovação do buffer
#     learning_starts=50_000,               # Aprende quase imediatamente
#     batch_size=256,
#     tau=1.0,                             # Atualização direta (como original)
#     gamma=0.995,
#     train_freq=1,
#     target_update_interval=10_000,
#     exploration_fraction=0.3,
#     exploration_final_eps=0.1,
#     verbose=1,
#     tensorboard_log="logs/",
#     policy_kwargs=dict(net_arch=[256, 256]),  # Rede menor, menos overfitting
#     device="cuda"
# )


# eval_cb = EvalCallback(eval_env, best_model_save_path=os.path.join(model_save_path, "best"), eval_freq=50_000, n_eval_episodes=20, deterministic=True)
# ckpt_cb = CheckpointCallback(save_freq=10_000, save_path=os.path.join(model_save_path, "ckpts"), name_prefix="dqn")

# # Treinar
# model.learn(total_timesteps=1_000_000)


# file_name = "dqn_2m_area_next_piece_features.zip"
# file_path = os.path.join(model_save_path, file_name)
# # Salvar modelo e normalizador
# model.save(file_path)

import os
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from typing import Callable
from tetris_env import TetrisEnv

# 1. Função para Learning Rate Schedule (Decaimento Linear)
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        # progress_remaining vai de 1.0 (início) até 0.0 (fim)
        return progress_remaining * initial_value
    return func

log_dir = "./logs_csv"
os.makedirs(log_dir, exist_ok=True)
model_save_path = os.path.join(os.getcwd(), "models")
os.makedirs(model_save_path, exist_ok=True)

def make_env():
    return Monitor(TetrisEnv(window_type="headless"), filename=os.path.join(log_dir, "monitor.csv"))

# 2. Correção Crítica: VecNormalize em AMBOS os ambientes
env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

eval_env = DummyVecEnv([make_env])
# O eval_env deve compartilhar as estatísticas exatas do env de treino
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)
eval_env.training = False # Desliga a atualização de médias na avaliação

# 3. Definição do Modelo Otimizado
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=linear_schedule(2.0e-4), # Começa em 2e-4 e cai até 0
    buffer_size=1_000_000,               # Aumentado para suportar os 2M steps
    learning_starts=50_000,              
    batch_size=256,
    tau=0.005,                           # Soft update para maior estabilidade
    gamma=0.999,                         # Horizonte estendido (Foco em sobrevivência)
    train_freq=4,                        # Atualiza a rede a cada 4 passos (Padrão DQN, mais estável)
    target_update_interval=1,            # Atualiza a rede alvo a cada step usando tau
    exploration_fraction=0.4,            # Explora durante 800k steps
    exploration_final_eps=0.02,          # Quase zero aleatoriedade no fim
    verbose=1,
    tensorboard_log="logs/",
    policy_kwargs=dict(net_arch=[256, 256]), 
    device="cuda"
)

# 4. Callbacks e Treinamento
eval_cb = EvalCallback(
    eval_env, 
    best_model_save_path=os.path.join(model_save_path, "best"), 
    log_path=log_dir,
    eval_freq=50_000, 
    n_eval_episodes=50,                  # Aumentado para ter maior confiança estatística
    deterministic=True
)

ckpt_cb = CheckpointCallback(save_freq=100_000, save_path=os.path.join(model_save_path, "ckpts"), name_prefix="dqn_2m")

print("\n[INFO] Iniciando treinamento para 2.000.000 steps...")
model.learn(total_timesteps=2_000_000, callback=[eval_cb, ckpt_cb])

# 5. Salvamento Seguro (Modelo + Normalizador)
model.save(os.path.join(model_save_path, "dqn_2m_area_next_piece_features.zip"))
env.save(os.path.join(model_save_path, "vecnorm_2m.pkl")) # Obrigatório salvar o env!