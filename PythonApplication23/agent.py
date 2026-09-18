import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class CardNet(nn.Module):
    """
    TCG 智能体 Actor-Critic 策略与价值网络
    输入尺寸: 3 * 13 * 8 = 312
    动作空间: 29 (7张手牌 * 4种打出位置 + 1个结束回合动作)
    """
    def __init__(self, action_dim=29, obs_shape=(3, 13, 8)):
        super(CardNet, self).__init__()
        input_dim = obs_shape[0] * obs_shape[1] * obs_shape[2]
        self.shared_fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor = None):
        flattened = obs.view(obs.size(0), -1).float()
        feat = self.shared_fc(flattened)
        logits = self.actor(feat)
        value = self.critic(feat)
        if mask is not None:
            logits = torch.where(mask > 0.5, logits, torch.full_like(logits, -1e8))
        return logits, value

    def get_action_and_value(self, obs: torch.Tensor, mask: torch.Tensor = None, deterministic: bool = False):
        logits, value = self.forward(obs, mask)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

# 兼容旧命名别名
TCGActorCritic = CardNet