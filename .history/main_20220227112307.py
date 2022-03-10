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
    return state_shape, action_shape

class RLTrainer(object):
    def __init__(self,env_name,algo,num_env_steps,num_steps,args):
        envs = [make_env(env_name) for _ in range(args,num_processes)]
        self.envs = SubprocVecEnv(envs)
        state_shape, action_shape = get_env_info(env_name)
        if use_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
        self.algo = algo(state_shape, action_shape, device)
        num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    def collect_rollouts(self):
        pass

    def train(self):
        rollouts = self.collect_rollouts()
        self.algo.update(rollouts)

def main():
    pass

if __name__ == "__main__":
    main()