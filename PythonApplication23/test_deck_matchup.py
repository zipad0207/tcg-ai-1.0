import os
import sys
import json
import argparse
import time
import torch
import numpy as np
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sandbox import DuelEnv, Faction
from agent import CardNet

def run_simulation(episodes: int = 500, decks_path: str = "decks_config.json", cards_path: str = "cards_config.json", model_path: str = "card_ppo_model_tuned.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p0_decklist, p1_decklist = None, None
    red_name, blue_name = "赤红AI卡组", "蔚蓝AI卡组"
    if os.path.exists(decks_path):
        with open(decks_path, "r", encoding="utf-8") as f:
            decks_cfg = json.load(f)
            if "Red" in decks_cfg:
                p0_decklist = decks_cfg["Red"].get("decklist")
                red_name = decks_cfg["Red"].get("deck_name", "赤红AI卡组")
            if "Blue" in decks_cfg:
                p1_decklist = decks_cfg["Blue"].get("decklist")
                blue_name = decks_cfg["Blue"].get("deck_name", "蔚蓝AI卡组")

    print("=" * 80)
    print(f"🔬 AI 构筑卡组竞技场对抗评测 (Monte Carlo Simulation)")
    print(f"🔴 红方卡组: 《{red_name}》 (30张)")
    print(f"🔵 蓝方卡组: 《{blue_name}》 (30张)")
    print(f"📊 评测总轮数: {episodes} 场 | 运算设备: {device}")
    print("=" * 80)

    env = DuelEnv(p0_faction=Faction.RED, p1_faction=Faction.BLUE, cards_path=cards_path,
                  p0_decklist=p0_decklist, p1_decklist=p1_decklist)

    model = CardNet(action_dim=env.action_space_size).to(device)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"✅ 已装载智能决策模型权重: {model_path}")
    else:
        print(f"⚠️ 未找到指定权重 {model_path}，将采用随机/策略推断")
    model.eval()

    p0_wins = 0
    p1_wins = 0
    draws = 0
    turns_history = []
    p0_scores = []
    p1_scores = []
    card_usage = Counter()

    start_time = time.time()

    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False
        step_count = 0

        while not done:
            curr_p = env.current_player
            mask = env.get_action_mask()
            state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            mask_t = torch.FloatTensor(mask).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, _ = model(state_t, mask_t)
                # 贪婪取最高概率合法动作
                action = torch.argmax(logits, dim=-1).item()

            if action != env.action_space_size - 1:
                hand_idx = action // 4
                curr_hand = env.players[curr_p].hand
                if hand_idx < len(curr_hand):
                    card_usage[curr_hand[hand_idx].name] += 1

            obs, reward, done, _ = env.step(action)
            step_count += 1

        s0 = env.players[0].score
        s1 = env.players[1].score
        p0_scores.append(s0)
        p1_scores.append(s1)
        turns_history.append(env.turn_count)

        if s0 >= env.WIN_SCORE and s1 < env.WIN_SCORE:
            p0_wins += 1
        elif s1 >= env.WIN_SCORE and s0 < env.WIN_SCORE:
            p1_wins += 1
        elif s0 >= env.WIN_SCORE and s1 >= env.WIN_SCORE:
            if s0 > s1:
                p0_wins += 1
            elif s1 > s0:
                p1_wins += 1
            else:
                draws += 1
        else:
            draws += 1

        if ep % (max(1, episodes // 5)) == 0 or ep == episodes:
            p0_rate = (p0_wins / ep) * 100
            p1_rate = (p1_wins / ep) * 100
            print(f"⏳ 进度: {ep:4d}/{episodes} 局 | 🔴 红方胜率: {p0_rate:5.1f}% | 🔵 蓝方胜率: {p1_rate:5.1f}% | 均轮: {np.mean(turns_history):.1f}")

    elapsed = time.time() - start_time
    p0_rate = (p0_wins / episodes) * 100
    p1_rate = (p1_wins / episodes) * 100
    draw_rate = (draws / episodes) * 100

    print("\n" + "═" * 80)
    print("🏆 【红蓝 AI 卡组对决测试报告】")
    print("═" * 80)
    print(f"⏱️ 测试总耗时: {elapsed:.2f} 秒 ({episodes / elapsed:.1f} 场/秒)")
    print(f"🔴 红方《{red_name}》: 胜 {p0_wins} 场 ({p0_rate:.2f}%) | 局均分: {np.mean(p0_scores):.2f}")
    print(f"🔵 蓝方《{blue_name}》: 胜 {p1_wins} 场 ({p1_rate:.2f}%) | 局均分: {np.mean(p1_scores):.2f}")
    if draws > 0:
        print(f"⚖️ 平局/超时: {draws} 场 ({draw_rate:.2f}%)")
    print(f"⌛ 平均对局轮数: {np.mean(turns_history):.1f} 回合 (中位数: {np.median(turns_history):.0f})")
    print(f"📈 轮数分布: 最短 {min(turns_history)} 回合 / 最长 {max(turns_history)} 回合")
    print("─" * 80)
    print("🌟 对战中使用频次最高的 TOP 8 核心单卡:")
    for rank, (cname, cnt) in enumerate(card_usage.most_common(8), 1):
        print(f"   {rank}. [{cname}]: 出牌 {cnt} 次 (局均 {cnt/episodes:.2f} 次)")
    print("═" * 80 + "\n")

    return {
        "episodes": episodes,
        "p0_winrate": p0_rate,
        "p1_winrate": p1_rate,
        "avg_turns": float(np.mean(turns_history)),
        "p0_avg_score": float(np.mean(p0_scores)),
        "p1_avg_score": float(np.mean(p1_scores))
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="红蓝 AI 卡组实机对抗批处理评测")
    parser.add_argument("--episodes", type=int, default=500, help="测试对战总轮数 (默认 500)")
    parser.add_argument("--decks", type=str, default="decks_config.json", help="卡组配置文件")
    parser.add_argument("--cards", type=str, default="cards_config.json", help="卡池配置文件")
    parser.add_argument("--model", type=str, default="card_ppo_model_tuned.pth", help="模型权重文件")
    args = parser.parse_args()

    run_simulation(episodes=args.episodes, decks_path=args.decks, cards_path=args.cards, model_path=args.model)
