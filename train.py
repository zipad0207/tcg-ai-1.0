import os
import json
import signal
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 导入沙盒环境、阵营枚举与神经网络模型
from sandbox import DuelEnv, Faction
from agent import CardNet

# ==========================================
# 0. 实验阶段配置 (动态识别 baseline/tuned)
# ==========================================
parser = argparse.ArgumentParser(description="TCG PPO 自博弈训练流水线")
parser.add_argument("--stage", type=str, default="tuned", 
                    help="设置当前训练阶段: baseline, tuned, deepseek 等")
parser.add_argument("--brawl", action="store_true", 
                    help="开启三大阵营 (Red/Blue/Green) 多卡组随机自由混战模式")
parser.add_argument("--episodes", type=int, default=1000,
                    help="训练总对局轮数 (默认 1000)")
parser.add_argument("--cards", type=str, default=None,
                    help="自定义指定训练卡池配置文件路径 (默认根据 stage 自动匹配)")
args = parser.parse_args()
STAGE = args.stage

# ==========================================
# 1. 运算设备与训练超参数
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM = 0.5

ROLLOUT_STEPS = 1024     
BATCH_SIZE = 128         
UPDATE_EPOCHS = 4        
TOTAL_EPISODES = args.episodes    

# 动态保存路径：彻底防止数据覆盖
MODEL_SAVE_PATH = f"card_ppo_model_{STAGE}.pth"
METRICS_SAVE_PATH = f"training_metrics_{STAGE}.json"
FIGURE_SAVE_PATH = f"figure_{STAGE}.png"

# ==========================================
# 2. 经验回放缓冲区 (Buffer)
# ==========================================
class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states, self.actions, self.masks = [], [], []
        self.log_probs, self.rewards, self.dones, self.values = [], [], [], []
        self.acting_players = []

# ==========================================
# 3. PPO 优化与更新器
# ==========================================
class PPOTrainer:
    def __init__(self, action_dim=29):
        self.policy = CardNet(action_dim=action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.buffer = RolloutBuffer()

    def select_action(self, obs: np.ndarray, mask: np.ndarray):
        state_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        mask_t = torch.FloatTensor(mask).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits, value = self.policy(state_t, mask_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def get_value(self, obs: np.ndarray, mask: np.ndarray):
        state_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        mask_t = torch.FloatTensor(mask).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _, value = self.policy(state_t, mask_t)
        return value.item()

    def update(self, last_val: float = 0.0):
        states = torch.FloatTensor(np.array(self.buffer.states)).to(DEVICE)
        actions = torch.LongTensor(self.buffer.actions).to(DEVICE)
        masks = torch.FloatTensor(np.array(self.buffer.masks)).to(DEVICE)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(DEVICE)
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        values = self.buffer.values
        acting_players = self.buffer.acting_players

        advantages = []
        gae = 0.0
        values = values + [last_val]
        for t in reversed(range(len(rewards))):
            # 零和博弈交替回合价值反转判定: 若发生玩家换边，对手预估价值符号取反
            is_switch = (t + 1 < len(acting_players) and acting_players[t] != acting_players[t + 1])
            next_v = -values[t + 1] if is_switch else values[t + 1]

            delta = rewards[t] + GAMMA * next_v * (1.0 - dones[t]) - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * (1.0 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages).to(DEVICE)
        returns = advantages + torch.FloatTensor(values[:-1]).to(DEVICE)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss_accum = 0.0
        n_batches = 0
        dataset_size = len(rewards)

        for _ in range(UPDATE_EPOCHS):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_idx = indices[start:end]

                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_masks = masks[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]

                logits, new_values = self.policy(b_states, b_masks)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(new_values.squeeze(-1), b_returns)
                loss = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

                total_loss_accum += loss.item()
                n_batches += 1

        self.buffer.clear()
        return total_loss_accum / max(1, n_batches)

# ==========================================
# 5. 图表生成函数
# ==========================================
def auto_generate_plot():
    print("\n正在生成对局结果图表...")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    with open(METRICS_SAVE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    p0_winrate = data.get("p0_winrate", 0.0)
    p1_winrate = data.get("p1_winrate", 0.0)
    card_counts = data.get("card_play_count", {})

    top10 = sorted(card_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top10 = list(reversed(top10))
    card_names = [item[0] for item in top10]
    play_counts = [item[1] for item in top10]

    if STAGE == "tuned":
        fig_title = f"调优后对局胜率分布 (Tuned, PPO {TOTAL_EPISODES}局)"
        bar_title = "Top 10 对局卡牌出场频次分布 (调优后)"
    else:
        fig_title = f"基准环境对局胜率分布 (Baseline, PPO {TOTAL_EPISODES}局)"
        bar_title = "Top 10 对局卡牌出场频次分布 (基准组)"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # 左子图
    bars = ax1.bar(["红方 (快攻/突破)", "蓝方 (防守/控制)"], [p0_winrate, p1_winrate], color=["#e74c3c", "#3498db"], width=0.45)
    ax1.axhline(50.0, color="#7f8c8d", linestyle="--", linewidth=1.5, label="50% 平衡线")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("胜率 (%)", fontsize=11)
    ax1.set_title(fig_title, fontsize=12, pad=12, fontweight="bold")
    ax1.legend(loc="upper right")
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight="semibold")

    # 右子图
    ax2.barh(card_names, play_counts, color="#2ecc71", height=0.65)
    ax2.set_xlabel("对局出牌频次", fontsize=11)
    ax2.set_title(bar_title, fontsize=12, pad=12, fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIGURE_SAVE_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"图表已导出至: {FIGURE_SAVE_PATH}")

# ==========================================
# 6. 训练与指标统计主入口
# ==========================================
def main():
    if args.brawl:
        print("检测到 --brawl 标志，启动三大阵营自由混战训练模式。")
        import train_brawl
        train_brawl.TOTAL_EPISODES = args.episodes
        train_brawl.main()
        return

    print(f"[系统] 当前运行阶段: {STAGE.upper()} | 运算设备: {DEVICE}")
    cards_file = args.cards if args.cards else ("cards_config_baseline.json" if STAGE == "baseline" and os.path.exists("cards_config_baseline.json") else ("cards_config_tuned.json" if STAGE == "tuned" and os.path.exists("cards_config_tuned.json") else "cards_config.json"))
    print(f"[卡池加载] 阶段: {STAGE.upper()} | 卡池文件: {cards_file}")
    env = DuelEnv(p0_faction=Faction.RED, p1_faction=Faction.BLUE, cards_path=cards_file)
    trainer = PPOTrainer(action_dim=env.action_space_size)

    metrics = {
        "total_episodes": 0, "p0_wins": 0, "p1_wins": 0,
        "p0_winrate": 0.0, "p1_winrate": 0.0,
        "avg_steps_per_episode": 0.0, "card_play_count": {}
    }
    total_steps_history = []

    def save_metrics():
        if metrics["total_episodes"] > 0:
            metrics["p0_winrate"] = round((metrics["p0_wins"] / metrics["total_episodes"]) * 100, 2)
            metrics["p1_winrate"] = round((metrics["p1_wins"] / metrics["total_episodes"]) * 100, 2)
            metrics["avg_steps_per_episode"] = round(float(np.mean(total_steps_history)), 2)
        with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"[指标保存] 训练指标已写入: {METRICS_SAVE_PATH}")

    def handle_sigint(sig, frame):
        print("\n捕获中断信号，正在保存当前权重与数据...")
        torch.save(trainer.policy.state_dict(), MODEL_SAVE_PATH)
        save_metrics()
        auto_generate_plot()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)
    print(f"=== 开始 PPO 自博弈对抗训练 [{STAGE.upper()}] ===")
    
    step_accum = 0

    for episode in range(1, TOTAL_EPISODES + 1):
        obs = env.reset()
        done = False
        ep_len = 0

        while not done:
            acting_player = env.current_player
            mask = env.get_action_mask()
            action, log_prob, val = trainer.select_action(obs, mask)

            if action != env.action_space_size - 1:
                hand_idx = action // 4
                curr_player = env.players[acting_player]
                if hand_idx < len(curr_player.hand):
                    c_name = curr_player.hand[hand_idx].name
                    metrics["card_play_count"][c_name] = metrics["card_play_count"].get(c_name, 0) + 1

            next_obs, reward, done, info = env.step(action)

            trainer.buffer.states.append(obs)
            trainer.buffer.actions.append(action)
            trainer.buffer.masks.append(mask)
            trainer.buffer.log_probs.append(log_prob)
            trainer.buffer.rewards.append(reward)
            trainer.buffer.dones.append(done)
            trainer.buffer.values.append(val)
            trainer.buffer.acting_players.append(acting_player)

            obs = next_obs
            ep_len += 1
            step_accum += 1

            if step_accum >= ROLLOUT_STEPS:
                if done:
                    last_val = 0.0
                else:
                    next_mask = env.get_action_mask()
                    last_val = trainer.get_value(obs, next_mask)
                trainer.update(last_val=last_val)
                step_accum = 0

        metrics["total_episodes"] += 1
        total_steps_history.append(ep_len)
        p0_won = (env.winner == 0) if env.winner is not None else (env.players[0].score >= env.WIN_SCORE)
        if p0_won:
            metrics["p0_wins"] += 1
            winner_str = "玩家0 (红)"
        else:
            metrics["p1_wins"] += 1
            winner_str = "玩家1 (蓝)"

        if episode % 10 == 0 or episode == 1:
            win_rate_p0 = (metrics["p0_wins"] / metrics["total_episodes"]) * 100
            print(f"Episode {episode:04d} | 步数: {ep_len:03d} | 比分: [{env.players[0].score}:{env.players[1].score}] | 获胜方: {winner_str} | 红方胜率: {win_rate_p0:.1f}%")

        if episode % 50 == 0:
            torch.save(trainer.policy.state_dict(), MODEL_SAVE_PATH)
            save_metrics()

    # 正常训练结束归档并出图
    torch.save(trainer.policy.state_dict(), MODEL_SAVE_PATH)
    save_metrics()
    print(f"\n训练结束，模型权重已保存至: {MODEL_SAVE_PATH}")
    auto_generate_plot()

if __name__ == "__main__":
    main()