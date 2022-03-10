
class PPO(object):
    def __init__(self,actor_critic,clip_param,ppo_epoch,num_mini_batch,entropy_coef,
    lr,eps,gamma,gae_lambda,device):
        self.actor_critic = actor_critic
        self.device = device 
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def compute_gae(self, next_value, rewards, masks, values):
        gae = []
        returns = []
        values += [next_values]
        for step in reversed(range(len(rewards)+1)):
            delta = rewards[step] + self.gamma * 

    def ppo_update(self,rollouts):
        returns, values, actons, action_log_probs = rollouts
        advantages = 
