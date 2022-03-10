from multiprocessing_env import SubprocVecEnv
import torch
import gym
import argparse
from utils import RolloutStorage

def get_args():
    # cuda_devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]

	# optimal cpu=2, device=cuda (rate 3.5)
	parser = argparse.ArgumentParser()
	parser.add_argument('-use_cuda', action='store_true', help="Which device to use")
	parser.add_argument('-num_processes', type=int, default=32)
	parser.add_argument('-env_name', type=str, default="MountainCarContinuous-v0")
	parser.add_argument('-env_steps_per_update', type=int, default=1e3)
	parser.add_argument('-total_steps', type=int, default=1e6)

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
    state_shape = env.observation_space
    action_shape = env.action_space
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
        self.env_steps_per_update = args.env_steps_per_update
        self.num_updates = int(
        args.total_steps) // args.env_steps_per_update // args.num_processes
        self.rollouts = RolloutStorage(args.num_steps, args.num_processes,
                          state_shape.shape, action_shape,
                          algo.actor_critic.recurrent_hidden_state_size)

    def collect_rollouts(self,rollouts):
        for step in range(self.env_steps_per_update):
            action = self.algo.act(rollouts.obs[0])
            next_state, reward, done, _ = self.envs.step(action.cpu().numpy())
        self.rollouts.after_update()
        return rollouts

    def train(self):
        for i in range(self.num_updates):
            rollouts = self.collect_rollouts()
        self.algo.update(rollouts)

def main():
    args = get_args()
    trainer = RLTrainer(args)
    pass

if __name__ == "__main__":
    main()