# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3 import PPO,DQN
# from tetris_env import TetrisEnv

# def make_eval_env():
#     # use exatamente os mesmos argumentos do treino (sem render)
#     return Monitor(TetrisEnv(window_type="headless", memory_size=50))

# eval_env = DummyVecEnv([make_eval_env])

# model = DQN.load("dqn_1m_area_features")
# model.exploration_rate = 0.0  # garante zero exploração
# mean_reward, ep_len = evaluate_policy(
#     model, eval_env, n_eval_episodes=100, deterministic=True, render=False,
#     return_episode_rewards=True
# )

# import seaborn as sns 

# sns.histplot(mean_reward, kde= True) 

# sns.histplot(ep_len, kde= True) 


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import DQN
from tetris_env import TetrisEnv
from tqdm.auto import tqdm
import numpy as np

# 1. Nova classe estruturada com __call__
class EvalProgressBar:
    def __init__(self, total_episodes: int):
        self.total_episodes = total_episodes
        self.pbar = None
        self.completed_episodes = 0

    def __call__(self, locals_, globals_):
        """
        Este método é chamado a cada step do evaluate_policy.
        locals_ contém o dicionário de variáveis internas da avaliação.
        """
        # Inicializa a barra de progresso no primeiro step
        if self.pbar is None:
            self.pbar = tqdm(total=self.total_episodes, desc="Avaliando Episódios")

        # Captura o array de booleanos que indica se o ambiente finalizou
        dones = locals_.get("dones")
        
        if dones is not None and np.any(dones):
            # np.sum(dones) é útil se você usar um VecEnv com múltiplos ambientes em paralelo
            increment = np.sum(dones)
            self.completed_episodes += increment
            self.pbar.update(increment)

        # Garante o fechamento limpo da barra
        if self.completed_episodes >= self.total_episodes:
            self.pbar.close()

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()

# 2. Configuração do Ambiente
def make_eval_env():
    return Monitor(TetrisEnv(window_type="headless", memory_size=50))

eval_env = DummyVecEnv([make_eval_env])

# 3. Carregamento do Modelo
model = DQN.load("dqn_1m_area_features")
model.exploration_rate = 0.0  # Exploração zero para avaliação pura

# 4. Execução da Avaliação com o Callback
n_episodes = 100
progress_callback = EvalProgressBar(total_episodes=n_episodes)

print(f"\n[INFO] Iniciando avaliação de {n_episodes} episódios...")

mean_reward, ep_len = evaluate_policy(
    model, 
    eval_env, 
    n_eval_episodes=n_episodes, 
    deterministic=True, 
    render=False,
    return_episode_rewards=True,
    callback=progress_callback  # <--- Passando o objeto "callable"
)

print(f"\n[RESULTADO] Recompensa Média: {np.mean(mean_reward):.2f} +/- {np.std(mean_reward):.2f}")

# 5. Visualização (Data Viz)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(mean_reward, kde=True, ax=axes[0], color="blue")
axes[0].set_title("Distribuição das Recompensas")
axes[0].set_xlabel("Recompensa Total")

sns.histplot(ep_len, kde=True, ax=axes[1], color="orange")
axes[1].set_title("Distribuição da Duração (Passos)")
axes[1].set_xlabel("Passos por Episódio")

plt.tight_layout()
plt.show()