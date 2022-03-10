import torch
import numpy as np
import torch.optim as optim
class PPO(object):
    def __init__(self,actor_critic, clip_param, ppo_epochs, entropy_coef,
                        lr, eps, gamma, gae_lambda, num_mini_batch, 
                        use_clipped_value_loss,value_loss_coef,max_grad_norm,
                        device):
        self.actor_critic = actor_critic
        self.device = device 
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.clip_param = clip_param
        #TODO
        self.num_mini_batch = num_mini_batch
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.lr = lr
        self.eps = eps
        self.use_clipped_value_loss = use_clipped_value_loss
        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    
    def compute_gae(self, next_value, rewards, masks, values):
        gae = 0
        returns = []
        values[-1] = next_value
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step+1] * masks[step] - values[step]
            gae = delta + self.gamma * self.gae_lambda * gae * masks[step]
            returns.insert(0,gae)
        return torch.stack(returns)

    def normalize(self,x):
        print(x)
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

    def update(self,rollouts,next_value):
        states, actions, rewards, values, action_log_probs, masks = rollouts.pop()
        returns = self.compute_gae(next_value,rewards,masks,values)
        advantages = returns - values[:-1]
        advantages = self.normalize(advantages)
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        for _ in range(self.ppo_epochs):
            # for state, action, old_log_probs, return_, advantage in self.ppo_iter(states, actions, action_log_probs,
            #                                                          returns, advantages,
            #                                                          minibatch_size=self.batch_size):
            if self.actor_critic.is_recurrent:
                    data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample
                obs_batch = self.actor_critic.normalize_state(obs_batch,False)
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                #TODO
                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
            

        
