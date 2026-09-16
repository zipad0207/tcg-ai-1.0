"""
TCG-AI 多阵营自博弈强化学习训练脚本
支持红、蓝、绿三大阵营混战对局训练与胜率统计。
"""

import os
import sys
import json
import random
import signal
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from sandbox import DuelEnv, Faction
from agent import CardNet

# ==========================================
# 0. 参数解析与运行配置
# ==========================================
parser = argparse.ArgumentParser(description="TCG-AI 多卡组自由混战 PPO 训练流水线")
parser.add_argument("--episodes", type=int, default=2000,
                    help="训练总对局轮数 (默认 2000 局)")
parser.add_argument("--cards", type=str, default="cards_config.json",
                    help="卡池配置文件路径 (默认 cards_config.json)")
parser.add_argument("--decks", type=str, default="decks_config.json",
                    help="成熟卡组配置文件路径 (默认 decks_config.json)")
parser.add_argument("--save-model", type=str, default="card_ppo_model_brawl.pth",
                    help="模型保存路径")
parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
args = parser.parse_args()

TOTAL_EPISODES = args.episodes
CARDS_FILE = args.cards
DECKS_FILE = args.decks
MODEL_SAVE_PATH = args.save_model
TUNED_MODEL_PATH = "card_ppo_model_tuned.pth"
METRICS_SAVE_PATH = "training_metrics_brawl.json"
FIGURE_SAVE_PATH = "figure_brawl.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PPO 超参数
LR = args.lr
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.015
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM = 0.5

ROLLOUT_STEPS = 1024
BATCH_SIZE = 128
UPDATE_EPOCHS = 4

# ==========================================
# 1. 经验回放缓冲区 (Buffer)
# ==========================================
class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states, self.actions, self.masks = [], [], []
        self.log_probs, self.rewards, self.dones, self.values = [], [], [], []
        self.acting_players = []

# ==========================================
# 2. PPO 优化与更新器
# ==========================================
class PPOTrainer:
    def __init__(self, action_dim=29):
        self.policy = CardNet(action_dim=action_dim).to(DEVICE)
        # 若已有训练好的调优权重，优先热启载入继续增强
        if os.path.exists(TUNED_MODEL_PATH):
            try:
                state_dict = torch.load(TUNED_MODEL_PATH, map_location=DEVICE, weights_only=True)
                self.policy.load_state_dict(state_dict)
                print(f"[OK] 成功载入预训练权重: {TUNED_MODEL_PATH}，开启强化混战深造！")
            except Exception as e:
                print(f"[WARN] 载入预训练权重失败: {e}，将随机初始化开始训练")
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
        if len(self.buffer.states) == 0:
            return 0.0

        states = torch.FloatTensor(np.array(self.buffer.states)).to(DEVICE)
        actions = torch.LongTensor(self.buffer.actions).to(DEVICE)
        masks = torch.FloatTensor(np.array(self.buffer.masks)).to(DEVICE)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(DEVICE)

        # 计算 GAE 优势函数与回报目标 (针对回合交替自博弈)
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        values = self.buffer.values + [last_val]
        acting_players = self.buffer.acting_players

        returns = []
        advantages = []
        gae = 0.0

        for step in reversed(range(len(rewards))):
            curr_player = acting_players[step]
            next_val = values[step + 1]

            if step + 1 < len(acting_players):
                next_player = acting_players[step + 1]
                if curr_player != next_player:
                    next_val = -next_val

            delta = rewards[step] + (0.0 if dones[step] else GAMMA * next_val) - values[step]
            gae = delta + (0.0 if dones[step] else GAMMA * GAE_LAMBDA * gae)
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])

        returns = torch.FloatTensor(returns).to(DEVICE)
        advantages = torch.FloatTensor(advantages).to(DEVICE)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_samples = len(states)
        indices = np.arange(total_samples)
        total_loss_accum = 0.0
        n_batches = 0

        for _ in range(UPDATE_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, total_samples, BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_idx = indices[start:end]

                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_masks = masks[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]

                logits, val = self.policy(b_states, b_masks)
                val = val.squeeze(-1)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(val, b_returns)
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
# 3. 可视化图表生成 (三图看板)
# ==========================================
def generate_brawl_plots(metrics: dict):
    print("\n正在生成多阵营混战战绩分布看板...")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(18, 5.5), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.1, 1.2])

    # 1. 阵营总胜率柱状图
    ax1 = fig.add_subplot(gs[0])
    factions = ["Red", "Blue", "Green"]
    names = ["赤红 (Red)", "蔚蓝 (Blue)", "翠绿 (Green)"]
    colors = ["#ff4757", "#1e90ff", "#2ed573"]

    win_rates = [metrics["faction_stats"][f]["winrate"] for f in factions]
    play_counts = [metrics["faction_stats"][f]["matches"] for f in factions]

    bars = ax1.bar(names, win_rates, color=colors, width=0.5, edgecolor="#2f3640", linewidth=1.2)
    ax1.axhline(50.0, color="#7f8c8d", linestyle="--", linewidth=1.5, label="50% 平衡线")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("阵营综合胜率 (%)", fontsize=11, fontweight="bold")
    ax1.set_title(f"三大阵营混战综合胜率 (共 {metrics['total_episodes']} 局)", fontsize=12, pad=12, fontweight="bold")
    ax1.legend(loc="upper right")

    for bar, count in zip(bars, play_counts):
        height = bar.get_height()
        ax1.annotate(f"{height:.1f}%\n({count}局)", xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight="bold")

    # 2. 跨阵营对战克制矩阵热力图 (3x3 对称双向汇总)
    ax2 = fig.add_subplot(gs[1])
    matrix = np.zeros((3, 3))
    for i, fA in enumerate(factions):
        for j, fB in enumerate(factions):
            if i == j:
                # 内战（自己打自己）理论基准必然为 50.0%
                matrix[i, j] = 50.0
            else:
                k1 = f"{fA}_vs_{fB}"
                k2 = f"{fB}_vs_{fA}"
                r1 = metrics["matchups"].get(k1, {"total": 0})
                r2 = metrics["matchups"].get(k2, {"total": 0})
                total_games = r1.get("total", 0) + r2.get("total", 0)
                fA_wins = r1.get(f"{fA}_wins", 0) + r2.get(f"{fA}_wins", 0)
                matrix[i, j] = (fA_wins / total_games) * 100 if total_games > 0 else 50.0

    ax2.imshow(matrix, cmap="RdYlGn", vmin=35, vmax=65)
    ax2.set_xticks(range(3))
    ax2.set_yticks(range(3))
    ax2.set_xticklabels(["对手: 赤红", "对手: 蔚蓝", "对手: 翠绿"], fontsize=9.5)
    ax2.set_yticklabels(["本方: 赤红", "本方: 蔚蓝", "本方: 翠绿"], fontsize=9.5)
    ax2.set_title("跨阵营对弈胜率矩阵热力图 (%)", fontsize=12, pad=12, fontweight="bold")

    for i in range(3):
        for j in range(3):
            val = matrix[i, j]
            text_color = "black" if 42 <= val <= 58 else "white"
            label = "50.0%\n(内战)" if i == j else f"{val:.1f}%"
            ax2.text(j, i, label, ha="center", va="center", color=text_color, fontweight="bold", fontsize=10)

    # 3. 全局高频出牌热度 Top 10
    ax3 = fig.add_subplot(gs[2])
    card_counts = metrics.get("card_play_count", {})
    top10 = sorted(card_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top10 = list(reversed(top10))
    c_names = [item[0] for item in top10]
    c_plays = [item[1] for item in top10]

    ax3.barh(c_names, c_plays, color="#f39c12", height=0.6, edgecolor="#d35400", linewidth=1.1)
    ax3.set_xlabel("出战频次 (Play Count)", fontsize=10.5, fontweight="bold")
    ax3.set_title("多阵营混战核心出牌频次 Top 10", fontsize=12, pad=12, fontweight="bold")
    ax3.grid(axis="x", linestyle="--", alpha=0.5)

    for i, v in enumerate(c_plays):
        ax3.text(v + max(c_plays) * 0.01, i, str(v), va='center', fontsize=9, fontweight="bold", color="#2c3e50")

    plt.tight_layout()
    plt.savefig(FIGURE_SAVE_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"混战看板已导出至: {FIGURE_SAVE_PATH}")

# ==========================================
# 4. 主训练流程
# ==========================================
def main():
    print("=" * 70)
    print(f"TCG-AI 多阵营混战强化学习训练启动 (总对局: {TOTAL_EPISODES} 局)")
    print("=" * 70)
    print(f"[*] 运算设备: {DEVICE}")

    # 读取成熟卡组库 (如果存在)
    prebuilt_decks = {}
    if os.path.exists(DECKS_FILE):
        try:
            with open(DECKS_FILE, "r", encoding="utf-8") as f:
                decks_raw = json.load(f)
            for f_key in ["Red", "Blue", "Green"]:
                if f_key in decks_raw and "decklist" in decks_raw[f_key]:
                    prebuilt_decks[f_key] = decks_raw[f_key]["decklist"]
            print(f"[OK] 成功挂载成熟预建卡组库: {list(prebuilt_decks.keys())} 各包含 30 张实战卡牌")
        except Exception as e:
            print(f"[WARN] 读取卡组库失败: {e}，将全量使用动态卡池构筑")

    # 初始化沙盒环境
    env = DuelEnv(p0_faction=Faction.RED, p1_faction=Faction.BLUE, cards_path=CARDS_FILE)
    trainer = PPOTrainer(action_dim=env.action_space_size)

    faction_list = [Faction.RED, Faction.BLUE, Faction.GREEN]
    faction_names = {Faction.RED: "Red", Faction.BLUE: "Blue", Faction.GREEN: "Green"}

    metrics = {
        "total_episodes": 0,
        "faction_stats": {
            "Red": {"matches": 0, "wins": 0, "winrate": 0.0},
            "Blue": {"matches": 0, "wins": 0, "winrate": 0.0},
            "Green": {"matches": 0, "wins": 0, "winrate": 0.0}
        },
        "matchups": {},
        "card_play_count": {},
        "avg_steps": 0.0
    }

    step_accum = 0
    all_lengths = []

    def save_metrics():
        if metrics["total_episodes"] > 0:
            for f_k in ["Red", "Blue", "Green"]:
                m_cnt = metrics["faction_stats"][f_k]["matches"]
                w_cnt = metrics["faction_stats"][f_k]["wins"]
                metrics["faction_stats"][f_k]["winrate"] = round((w_cnt / max(1, m_cnt)) * 100, 2)
            metrics["avg_steps"] = round(float(np.mean(all_lengths)), 2)

        with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    def handle_sigint(sig, frame):
        print("\n捕获中断信号，正在保存混战模型与战绩...")
        torch.save(trainer.policy.state_dict(), MODEL_SAVE_PATH)
        torch.save(trainer.policy.state_dict(), TUNED_MODEL_PATH)
        save_metrics()
        generate_brawl_plots(metrics)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    for ep in range(1, TOTAL_EPISODES + 1):
        # 1. 双方随机独立抽取阵营
        f0 = random.choice(faction_list)
        f1 = random.choice(faction_list)

        name0 = faction_names[f0]
        name1 = faction_names[f1]

        # 2. 100% 选用该颜色卡组对应的 30 张专属构筑套牌（绝不随机掺杂杂牌）
        d0 = prebuilt_decks[name0]
        d1 = prebuilt_decks[name1]

        # 动态更新环境配置并重置
        env.p0_faction = f0
        env.p1_faction = f1
        env.p0_decklist = d0
        env.p1_decklist = d1

        obs = env.reset()
        done = False
        ep_len = 0

        while not done:
            acting_player = env.current_player
            mask = env.get_action_mask()
            action, log_prob, val = trainer.select_action(obs, mask)

            # 统计出牌记录
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

        # 战绩结算
        metrics["total_episodes"] += 1
        all_lengths.append(ep_len)

        p0_won = (env.players[0].score >= env.WIN_SCORE)
        winner_faction = name0 if p0_won else name1

        # 阵营战报
        metrics["faction_stats"][name0]["matches"] += 1
        metrics["faction_stats"][name1]["matches"] += 1
        if p0_won:
            metrics["faction_stats"][name0]["wins"] += 1
        else:
            metrics["faction_stats"][name1]["wins"] += 1

        # 对战矩阵统计
        m_key = f"{name0}_vs_{name1}"
        if name0 == name1:
            if m_key not in metrics["matchups"]:
                metrics["matchups"][m_key] = {
                    "total": 0,
                    "p0_first_wins": 0,
                    "p1_second_wins": 0,
                    "winrate": 50.0
                }
            metrics["matchups"][m_key]["total"] += 1
            if p0_won:
                metrics["matchups"][m_key]["p0_first_wins"] += 1
            else:
                metrics["matchups"][m_key]["p1_second_wins"] += 1
        else:
            if m_key not in metrics["matchups"]:
                metrics["matchups"][m_key] = {"total": 0, f"{name0}_wins": 0, f"{name1}_wins": 0}
            metrics["matchups"][m_key]["total"] += 1
            if p0_won:
                metrics["matchups"][m_key][f"{name0}_wins"] += 1
            else:
                metrics["matchups"][m_key][f"{name1}_wins"] += 1

        # 进度打印
        if ep % 50 == 0 or ep == 1:
            wr_r = (metrics["faction_stats"]["Red"]["wins"] / max(1, metrics["faction_stats"]["Red"]["matches"])) * 100
            wr_b = (metrics["faction_stats"]["Blue"]["wins"] / max(1, metrics["faction_stats"]["Blue"]["matches"])) * 100
            wr_g = (metrics["faction_stats"]["Green"]["wins"] / max(1, metrics["faction_stats"]["Green"]["matches"])) * 100

            print(f"Episode {ep:04d}/{TOTAL_EPISODES} | 对局: [{name0} vs {name1}] | 胜者: {winner_faction} | 胜率: [Red {wr_r:.1f}% | Blue {wr_b:.1f}% | Green {wr_g:.1f}%]")

        # 定期保存权重
        if ep % 200 == 0:
            torch.save(trainer.policy.state_dict(), MODEL_SAVE_PATH)
            torch.save(trainer.policy.state_dict(), TUNED_MODEL_PATH)
            save_metrics()

    # 训练完成终结归档
    torch.save(trainer.policy.state_dict(), MODEL_SAVE_PATH)
    torch.save(trainer.policy.state_dict(), TUNED_MODEL_PATH)
    save_metrics()
    print(f"\n混战训练完成，模型已保存至 {MODEL_SAVE_PATH} 与 {TUNED_MODEL_PATH}")
    generate_brawl_plots(metrics)

if __name__ == "__main__":
    main()
