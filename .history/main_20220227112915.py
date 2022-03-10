from multiprocessing_env import SubprocVecEnv
import torch
import gym
import argparse

def get_args():
    # cuda_devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]

	# optimal cpu=2, device=cuda (rate 3.5)
	parser = argparse.ArgumentParser()
	parser.add_argument('-use_cuda', action='store_true', help="Which device to use")
    # parser.add_argument('-num_processes', type=int, default=256)

	# parser.add_argument('-device', type=str, choices=['auto', 'cpu', 'cuda'], default='cpu', help="Which device to use")
	cmd_args = parser.parse_args()

	return cmd_args

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
    def __init__(self,algo,args):
        envs = [make_env(args.env_name) for _ in range(args,args.num_processes)]
        self.envs = SubprocVecEnv(envs)
        state_shape, action_shape = get_env_info(args.env_name)
        if args.use_cuda:
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
    args = get_args()
    trainer = RLTrainer(args)
    pass

if __name__ == "__main__":
    main()