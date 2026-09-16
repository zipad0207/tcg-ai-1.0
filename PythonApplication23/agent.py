import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

class TCGActorCritic(nn.Module):
    # 默认值已经同步为你升级后的 7 分制/7手牌 版本
    def __init__(self, obs_shape=(3, 13, 5), action_dim=29):
        super(TCGActorCritic, self).__init__()
        
        # 严格计算扁平化维度，确保是 195
        input_size = obs_shape[0] * obs_shape[1] * obs_shape[2] 
        
        # 共享特征提取层
        self.shared_net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.actor_head = nn.Linear(128, action_dim)
        self.critic_head = nn.Linear(128, 1)

    def forward(self, obs, action_mask=None):
        batch_size = obs.size(0)
        flat_obs = obs.view(batch_size, -1).float()
        
        features = self.shared_net(flat_obs)
        value = self.critic_head(features)
        logits = self.actor_head(features)
        
        if action_mask is not None:
            HUGE_NEG = -1e8
            logits = logits + (1.0 - action_mask) * HUGE_NEG
            
        dist = Categorical(logits=logits)
        return dist, value