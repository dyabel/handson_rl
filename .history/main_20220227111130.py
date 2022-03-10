from multiprocessing_env import SubprocVecEnv
import torch
import gym
def make_env(env_name):
    try:
        env = gym.make(env_name)
    except:
        raise ValueError(f"{env_name} not Valid")
    return 
class RLTrainer(object):
    def __init__(self,env,algo,num_updates,num_processes):
        self.env = env
    def collect_rollouts(self):
        pass

def main():
    pass

if __name__ == "__main__":
    main()