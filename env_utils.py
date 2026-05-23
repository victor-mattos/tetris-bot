import os
from stable_baselines3.common.monitor import Monitor
# Certifique-se de que o tetris_env.py não tem código executando fora das classes!
from tetris_env import TetrisEnv 

def make_env(rank: int, log_dir: str):
    def _init():
        # USO OBRIGATÓRIO DE "null" PARA MULTIPROCESSAMENTO
        env = TetrisEnv(window_type="null", memory_size=50) 
        log_file = os.path.join(log_dir, f"monitor_{rank}.csv")
        return Monitor(env, filename=log_file)
    return _init