from multiprocessing_env import SubprocVecEnv
import torch
import gym
import argparse
from storage import RolloutStorage
# from utils import RolloutStorage
import utils
from arguments import get_args
from model import ActorCritic
from ppo import PPO
# from ppo_bak import PPO
import numpy as np
from copy import deepcopy
import os
from collections import deque
import time
from evaluation import evaluate


def make_env(env_name):
    # try:
    def _thunk():
        env = gym.make(env_name)
        return env
    # env = gym.make(env_name)
    # except:
        # raise ValueError(f"{env_name} not Valid")
    return _thunk
from envs import make_vec_envs
def get_env_info(env_name):
    env = gym.make(env_name)
    state_shape = env.observation_space
    action_shape = env.action_space
    return state_shape, action_shape

class RLTrainer(object):
    def __init__(self,args,device):
        envs = [make_env(args.env_name) for _ in range(args.num_processes)]
        self.envs = SubprocVecEnv(envs)
        # self.envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                        #  args.gamma, args.log_dir, device, False)
        self.eval_env = gym.make(args.env_name)
        state_shape, action_shape = get_env_info(args.env_name)
        #TODO
        actor_critic = ActorCritic(state_shape.shape,action_shape,device).to(device)
        #TODO
        # self.algo = PPO(actor_critic, args.clip_param, args.ppo_epochs, args.entropy_coef,
        #                     args.lr, args.eps, args.gamma, args.gae_lambda, args.num_mini_batch, 
        #                     True,args.value_loss_coef,args.max_grad_norm,
        #                     device)
        self.algo = PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epochs,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
       
        self.env_steps_per_update = args.env_steps_per_update
        self.num_test_episodes = args.num_test_episodes
        self.num_processes = args.num_processes
        self.use_linear_lr_decay = args.use_linear_lr_decay
        self.lr = args.lr
        self.args = args
        self.device = device  
        self.episode_reward = []  
        self.num_updates = int(
            args.total_steps // args.env_steps_per_update // args.num_processes)
        #TODO
        self.rollouts = RolloutStorage(args.env_steps_per_update, args.num_processes,
                          state_shape.shape, action_shape,
                          self.algo.actor_critic.recurrent_hidden_state_size)
        obs = self.envs.reset()
        # self.rollouts.obs[0].copy_(obs)
        self.rollouts.obs[0].copy_(torch.from_numpy(obs))
        self.rollouts.to(device)

    def collect_rollouts(self):
        with torch.no_grad():
            for step in range(self.env_steps_per_update):
                # self.rollouts.obs[step] = self.algo.actor_critic.normalize_state(self.rollouts.obs[step], update=False)  # normalize state
                value, action, action_log_prob, recurrent_hidden_states = self.algo.actor_critic.act(
                       self.rollouts.obs[step], self.rollouts.recurrent_hidden_states[step],
                        self.rollouts.masks[step])
                # next_state, reward, done, info = self.envs.step(action)
                next_state, reward, done, infos = self.envs.step(action.cpu().numpy())
                for info in infos:
                    if 'episode' in info.keys():
                        self.episode_rewards.append(info['episode']['r'])
                mask = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_mask = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info_.keys() else [1.0]
                     for info_ in infos])
                # self.rollouts.insert(next_state, recurrent_hidden_states, action, action_log_prob,
                #   value, reward, mask, bad_mask)
                self.rollouts.insert(torch.from_numpy(next_state).to(self.device), recurrent_hidden_states.to(self.device), action, action_log_prob,
                  value, torch.from_numpy(reward)[:,None].to(self.device), mask, bad_mask)

    def train(self):
        self.episode_rewards = deque(maxlen=10)
        for i in range(self.num_updates):
            start = time.time()
            if self.use_linear_lr_decay:
            # decrease learning rate linearly
                utils.update_linear_schedule(
                    self.algo.optimizer, i, self.num_updates,
                    self.lr)
            self.collect_rollouts()
            # self.episode_reward.append((self.rollouts.rewards*self.rollouts.masks[:-1]).sum())
            # print(self.episode_reward)
            # self.episode_reward = self.episode_reward[-10:]
            # print(sum(self.episode_reward)/10)

            with torch.no_grad():
                # self.rollouts.obs[-1] = self.algo.actor_critic.normalize_state(self.rollouts.obs[-1], update=False)  # normalize state
                next_value = self.algo.actor_critic.get_value(
                self.rollouts.obs[-1], self.rollouts.recurrent_hidden_states[-1],
                self.rollouts.masks[-1])
                
            self.rollouts.compute_returns(next_value, self.args.use_gae, self.args.gamma,
                                 self.args.gae_lambda, self.args.use_proper_time_limits)

            #TODO
            value_loss, action_loss, dist_entropy = self.algo.update(self.rollouts)
            # self.algo.update(self.rollouts, next_value)
            self.rollouts.after_update()
            print(f"update{i}")
            self.evaluate()
            # save for every interval-th episode or for the last epoch
            """
            if (i % self.args.save_interval == 0
                    or i == self.num_updates - 1) and self.args.save_dir != "":
                save_path = os.path.join(self.args.save_dir, self.args.algo)
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass

                torch.save([
                    self.algo.actor_critic,
                    getattr(utils.get_vec_normalize(self.envs), 'obs_rms', None)
                ], os.path.join(save_path, self.args.env_name + ".pt"))

            if i % self.args.log_interval == 0 and len(self.episode_rewards) > 1:
                total_num_steps = (i + 1) * self.args.num_processes * self.args.num_steps
                end = time.time()
                print(
                    "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(i, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(self.episode_rewards), np.mean(self.episode_rewards),
                            np.median(self.episode_rewards), np.min(self.episode_rewards),
                            np.max(self.episode_rewards), dist_entropy, value_loss,
                            action_loss))

            if (self.args.eval_interval is not None and len(self.episode_rewards) > 1
                    and i % self.args.eval_interval == 0):
                obs_rms = utils.get_vec_normalize(self.envs).obs_rms
                evaluate(self.algo.actor_critic, obs_rms, self.args.env_name, self.args.seed,
                         self.args.num_processes, self.eval_log_dir, self.device)
            """
        self.envs.close()
        torch.cuda.empty_cache()  # clear gpu memory if safe


    def test_env(self):
        state = self.eval_env.reset()
        eval_recurrent_hidden_states = torch.zeros(
            self.num_processes, self.algo.actor_critic.recurrent_hidden_state_size, device=self.device)
        eval_masks = torch.zeros(self.num_processes, 1, device=self.device)
        done = False
        total_reward = 0
        # cnt = 0
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                # state = self.algo.actor_critic.normalize_state(state, update=False)  # normalize state
                # value, action, action_log_prob, recurrent_hidden_states = self.algo.actor_critic.act(
                #        self.rollouts.obs[step], self.rollouts.recurrent_hidden_states[step],
                #         self.rollouts.masks[step])
                _, action, _, eval_recurrent_hidden_states = self.algo.actor_critic.act(
                state, eval_recurrent_hidden_states, eval_masks,
                deterministic=True)
                next_state, reward, done, _ = self.eval_env.step(action.cpu().numpy())
                state = next_state
                total_reward += reward
                # cnt += 1
        # print(cnt)
        return total_reward

    def evaluate(self):
        # if j % args.log_interval == 0 and len(episode_rewards) > 1:
        #         total_num_steps = (j + 1) * args.num_processes * args.num_steps
        # print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".format(j, total_num_steps,
        #                 int(total_num_steps / (end - start)),
        #                 len(episode_rewards), np.mean(episode_rewards),
        #                 np.median(episode_rewards), np.min(episode_rewards),
        #                 np.max(episode_rewards), dist_entropy, value_loss,
        #                 action_loss))
        test_reward_mean = np.mean([self.test_env() for _ in range(self.num_test_episodes)])
        print(test_reward_mean)

def main():
    # torch.autograd.set_detect_anomaly(True)
    args = get_args()
    if args.use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    trainer = RLTrainer(args, device)
    trainer.train()

if __name__ == "__main__":
    main()