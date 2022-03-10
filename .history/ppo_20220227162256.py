import torch
import numpy as np
class PPO(object):
    def __init__(self,actor_critic, clip_param, ppo_epoch, num_mini_batch, entropy_coef,
    lr, eps, gamma, gae_lambda, batch_size, device):
        self.actor_critic = actor_critic
        self.device = device 
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.ppo_epoch = ppo_epoch
        self.clip_param = clip_param
        self.num_mini_batch = num_mini_batch
        self.entropy_coef = entropy_coef
        self.lr = lr
        self.eps = eps

    def compute_gae(self, next_value, rewards, masks, values):
        gae = 0
        returns = []
        values += [next_value]
        for step in reversed(range(len(rewards)+1)):
            delta = rewards[step] + self.gamma * values[step+1] * masks[step] - values[step]
            gae = delta + self.gamma * self.gae_lambda * gae * masks[step]
            returns.insert(0,gae)
        return returns

    def normalize(self,x):
        x -= x.mean()
        x /= (x.std() + 1e-8)
        return x

    def ppo_iter(states, actions, log_probs, returns, advantage, minibatch_size):
        batch_size = states.size(0)
        # generates random mini-batches until we have covered the full batch
        #TODO
        for _ in range(batch_size // minibatch_size):
            rand_ids = np.random.randint(0, batch_size, minibatch_size)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[
                                                                                                       rand_ids, :]

    def ppo_update(self,rollouts,next_value):
        states, actions, rewards, values, action_log_probs, masks = rollouts.pop()
        returns = self.compute_gae(next_value,rewards,masks,values)
        advantages = returns - values
        advantages = self.normalize(advantages)
        for _ in range(self.ppo_epochs):
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(states, actions, action_log_probs,
                                                                     returns, advantages,
                                                                     minibatch_size=self.batch_size):
            

        
