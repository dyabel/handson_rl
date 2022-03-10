from multiprocessing_env import SubprocVecEnv
import torch
import gym
import argparse
from utils import RolloutStorage
from arguments import get_args
from model import ActorCritic
from ppo import PPO


def make_env(env_name):
    # try:
    def _thunk():
        env = gym.make(env_name)
        return env
    # env = gym.make(env_name)
    # except:
        # raise ValueError(f"{env_name} not Valid")
    return _thunk

def get_env_info(env_name):
    env = gym.make(env_name)
    state_shape = env.observation_space
    action_shape = env.action_space
    return state_shape, action_shape

class RLTrainer(object):
    def __init__(self,args,device):
        envs = [make_env(args.env_name) for _ in range(args.num_processes)]
        self.envs = SubprocVecEnv(envs)
        state_shape, action_shape = get_env_info(args.env_name)
        actor_critic = ActorCritic(state_shape.shape,action_shape,device)
        self.algo = PPO(actor_critic, args.clip_param, args.ppo_epoch, args.entropy_coef,
                            args.lr, args.eps, args.gamma, args.gae_lambda, args.num_mini_batch, 
                            True,args.value_loss_coef,args.max_grad_norm,
                            device)
        self.env_steps_per_update = args.env_steps_per_update
        self.num_updates = int(
        args.total_steps // args.env_steps_per_update // args.num_processes)
        self.rollouts = RolloutStorage(args.num_steps, args.num_processes,
                          state_shape.shape, action_shape,
                          self.algo.actor_critic.recurrent_hidden_state_size)
        obs = self.envs.reset()
        self.rollouts.obs[0].copy_(torch.from_numpy(obs))
        self.rollouts.to(device)

    def collect_rollouts(self):
        with torch.no_grad():
            for step in range(self.env_steps_per_update):
                value, action, action_log_prob, recurrent_hidden_states = self.algo.actor_critic.act(
                       self.rollouts.obs[step], self.rollouts.recurrent_hidden_states[step],
                        self.rollouts.masks[step])
                next_state, reward, done, _ = self.envs.step(action.cpu().numpy())
                mask = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                # bad_mask = torch.FloatTensor(
                #     [[0.0] if 'bad_transition' in info_.keys() else [1.0]
                #      for info_ in info])
                self.rollouts.insert(torch.from_numpy(next_state), recurrent_hidden_states, action, action_log_prob,
                  value, torch.from_numpy(reward)[:,None], mask)

    def train(self):
        for _ in range(self.num_updates):
            self.collect_rollouts()
            with torch.no_grad():
                next_value = self.algo.actor_critic.get_value(
                self.rollouts.obs[-1], self.rollouts.recurrent_hidden_states[-1],
                self.rollouts.masks[-1]).detach()

            self.algo.update(self.rollouts, next_value)
            self.rollouts.after_update()
            self.evaluate()

    def test_env(self, eval_envs, model, device, batch_size, deterministic=True):
        state = eval_envs.reset()
        obs = eval_envs.reset()
        eval_recurrent_hidden_states = torch.zeros(
            self.num_processes, self.actor_critic.recurrent_hidden_state_size, device=device)
        eval_masks = torch.zeros(self.num_processes, 1, device=device)
        done = False
        total_reward = 0
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            state = self.actor_critic.normalize_state(state, update=False)  # normalize state
            _, action, _, eval_recurrent_hidden_states = self.actor_critic.act(
            state, eval_recurrent_hidden_states, eval_masks,
            deterministic=True)
            next_state, reward, done, _ = eval_envs.step(action)
            state = next_state
            total_reward += reward
        return total_reward

    def evaluate(self):
        reward = self.test_env()
        pass

def main():
    torch.autograd.set_detect_anomaly(True)
    args = get_args()
    if args.use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    trainer = RLTrainer(args, device)
    trainer.train()

if __name__ == "__main__":
    main()