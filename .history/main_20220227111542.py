from multiprocessing_env import SubprocVecEnv
import torch
import gym

def make_env(env_name):
    try:
        env = gym.make(env_name)
    except:
        raise ValueError(f"{env_name} not Valid")
    return env

def get_env_info(env_name):
    env = gym.make(env_name)
    state_shape = env.observation_space.shape[0]
    action_shape = env.action_space.shape[0]

class RLTrainer(object):
    def __init__(self,env_name,algo,num_updates,num_processes):
        envs = [make_env(env_name) for _ in range(num_processes)]
        self.envs = SubprocVecEnv(envs)

    def collect_rollouts(self):
        pass

def main():
    pass

if __name__ == "__main__":
    main()