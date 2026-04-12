import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize # Adicionado VecNormalize
from stable_baselines3 import DQN
from tetris_env import TetrisEnv

# 1. Classe estruturada com __call__ (Mantida - Excelente implementação!)
class EvalProgressBar:
    def __init__(self, total_episodes: int):
        self.total_episodes = total_episodes
        self.pbar = None
        self.completed_episodes = 0

    def __call__(self, locals_, globals_):
        if self.pbar is None:
            self.pbar = tqdm(total=self.total_episodes, desc="Avaliando Episódios")

        dones = locals_.get("dones")
        
        if dones is not None and np.any(dones):
            increment = np.sum(dones)
            self.completed_episodes += increment
            self.pbar.update(increment)

        if self.completed_episodes >= self.total_episodes:
            self.pbar.close()

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()

# 2. Configuração do Ambiente Base
print("\n[INFO] Configurando ambientes de avaliação...")
def make_eval_env():
    return Monitor(TetrisEnv(window_type="headless", memory_size=50))

# Envolve no DummyVecEnv
eval_env = DummyVecEnv([make_eval_env])

# 3. Correção Crítica: Aplicação do VecNormalize
# Carregamos as estatísticas salvas do seu modelo treinado
norm_path = "models/vecnorm_2m.pkl" 
if os.path.exists(norm_path):
    eval_env = VecNormalize.load(norm_path, eval_env)
    # Congela a atualização das médias/variâncias durante o teste
    eval_env.training = False 
    # CRÍTICO: Não normalizar a recompensa, para obtermos as métricas REAIS no Seaborn
    eval_env.norm_reward = False 
else:
    print(f"[AVISO] Arquivo {norm_path} não encontrado. Avaliação pode falhar.")

# 4. Carregamento do Modelo
# Ajustado para carregar o modelo correspondente ao VecNormalize (2M steps)
model_path = "models/best/best_model"
print(f"[INFO] Carregando modelo DQN de: {model_path}")
model = DQN.load(model_path)
model.exploration_rate = 0.0  # Exploração zero para avaliação pura (Exploitation)

# 5. Execução da Avaliação com o Callback
n_episodes = 100
progress_callback = EvalProgressBar(total_episodes=n_episodes)

print(f"\n[INFO] Iniciando avaliação de {n_episodes} episódios...")

mean_reward, ep_len = evaluate_policy(
    model, 
    eval_env, 
    n_eval_episodes=n_episodes, 
    deterministic=True, 
    render=False,
    return_episode_rewards=True, # Retorna listas de recompensas em vez de apenas um float
    callback=progress_callback
)

print(f"\n[RESULTADO FINAL]")
print(f"Recompensa Média: {np.mean(mean_reward):.2f} +/- {np.std(mean_reward):.2f}")
print(f"Duração Média: {np.mean(ep_len):.2f} passos")

# 6. Visualização Estatística (Data Viz)
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Recompensas
sns.histplot(mean_reward, kde=True, ax=axes[0], color="royalblue", bins=20)
axes[0].set_title("Distribuição das Recompensas (Performance)")
axes[0].set_xlabel("Recompensa Acumulada no Episódio")
axes[0].set_ylabel("Frequência")

# Gráfico 2: Sobrevivência
sns.histplot(ep_len, kde=True, ax=axes[1], color="darkorange", bins=20)
axes[1].set_title("Distribuição da Duração (Sobrevivência)")
axes[1].set_xlabel("Total de Passos por Episódio")
axes[1].set_ylabel("Frequência")

plt.tight_layout()
plt.show()