from multiprocessing_env import SubprocVecEnv
import torch
import gym
import argparse

def get_args():
    # cuda_devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]

	# optimal cpu=2, device=cuda (rate 3.5)
	parser = argparse.ArgumentParser()
	parser.add_argument('-device', type=str, choices=['auto', 'cpu', 'cuda'], default='cpu', help="Which device to use")
	parser.add_argument('-cpus', type=str, default='1', help="How many CPUs to use")
	parser.add_argument('-batch', type=int, default=256, help="Size of a batch / How many CPUs to use")
	parser.add_argument('-seed', type=int, default=None, help="Random seed") # seed in multiprocessing is not implemented
	parser.add_argument('-load_model', type=str, default=None, help="Load model from this file")
	parser.add_argument('-max_epochs', type=int, default=None, help="Terminate after this many epochs")
	parser.add_argument('-mp_iterations', type=int, default=10, help="Number of message passes")
	parser.add_argument('-epoch', type=int, default=1000, help="Epoch length")

	parser.add_argument('-subset', type=int, default=None, help="Use a subset of train set")
	parser.add_argument('--pos_feats', action='store_const', const=True, help="Enable positional features")
	parser.add_argument('--custom', type=str, default=None, help="Custom size (e.g. 10x10x4; else Boxoban)")

	parser.add_argument('-trace', action='store_const', const=True, help="Show trace of the agent")
	parser.add_argument('-eval', action='store_const', const=True, help="Evaluate the agent")

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