
class PPO(object):
    def __init__(self,actor_critic,clip_param,ppo_epoch,num_mini_batch,entropy_coef,lr,eps,device):
        self.actor_critic = actor_critic
        self.device = device 

    def compute_gae(self):
        pass

    def ppo_update(self,rollouts):
        returns, values, actons, action_log_probs = rollouts
