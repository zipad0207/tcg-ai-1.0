import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class CardNet(nn.Module):
    """TCG Actor-Critic network over (3, 13, 20) game observation tensors."""

    def __init__(self, action_dim: int = 29, obs_shape=(3, 13, 20)):
        super().__init__()
        self.obs_shape = tuple(obs_shape)
        input_dim = self.obs_shape[0] * self.obs_shape[1] * self.obs_shape[2]
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
        flattened = obs.reshape(obs.size(0), -1).float()
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


def load_cardnet_compatible(
    model: CardNet,
    ckpt_path: str,
    device: torch.device,
    legacy_cols: int = 8,
) -> str:
    """Load checkpoint into CardNet, expanding legacy (3, 13, 8) input weights when needed."""
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    target_state = model.state_dict()
    w_key = "shared_fc.0.weight"
    if w_key in state_dict and w_key in target_state:
        src_w = state_dict[w_key]
        dst_w = target_state[w_key]
        if src_w.shape == dst_w.shape:
            model.load_state_dict(state_dict)
            return "exact"

        ch, rows, new_cols = model.obs_shape
        expected_legacy_in = ch * rows * legacy_cols
        if (
            src_w.ndim == 2
            and dst_w.ndim == 2
            and src_w.shape[0] == dst_w.shape[0]
            and src_w.shape[1] == expected_legacy_in
            and new_cols > legacy_cols
        ):
            expanded = torch.zeros(dst_w.shape[0], ch, rows, new_cols, dtype=dst_w.dtype, device=dst_w.device)
            expanded[:, :, :, :legacy_cols] = src_w.to(device=dst_w.device, dtype=dst_w.dtype).view(
                dst_w.shape[0], ch, rows, legacy_cols
            )
            upgraded = dict(state_dict)
            upgraded[w_key] = expanded.view(dst_w.shape[0], -1)
            model.load_state_dict(upgraded)
            return "upgraded"

        raise RuntimeError(
            f"Incompatible checkpoint {ckpt_path}: weight {w_key} shape {tuple(src_w.shape)} "
            f"cannot adapt to {tuple(dst_w.shape)}"
        )

    model.load_state_dict(state_dict)
    return "exact"


def compute_zero_sum_gae(
    rewards,
    dones,
    values,
    acting_players,
    last_val: float = 0.0,
    last_player=None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
):
    """Compute alternating-turn zero-sum GAE advantages and lambda-returns."""
    n = len(rewards)
    v_ext = list(values) + [float(last_val)]
    advantages = []
    returns = []
    gae = 0.0

    for step in reversed(range(n)):
        curr_player = acting_players[step]
        if step + 1 < n:
            next_player = acting_players[step + 1]
        else:
            next_player = curr_player if last_player is None else last_player

        switched = curr_player != next_player
        next_val = -v_ext[step + 1] if switched else v_ext[step + 1]

        if dones[step]:
            delta = rewards[step] - v_ext[step]
            gae = delta
        else:
            delta = rewards[step] + gamma * next_val - v_ext[step]
            next_gae = -gae if switched else gae
            gae = delta + gamma * gae_lambda * next_gae

        advantages.append(gae)
        returns.append(gae + v_ext[step])

    advantages.reverse()
    returns.reverse()
    return advantages, returns


# 兼容旧命名别名
TCGActorCritic = CardNet