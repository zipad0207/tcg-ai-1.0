"""
TCG-AI 多阵营平衡调优与混战测试脚本
用于执行红、蓝、绿三大阵营 30 张卡组的微调与 6000 局混战验证。
"""

import os
import sys
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from sandbox import DuelEnv, Faction

CARDS_FILE = "cards_config.json"
DECKS_FILE = "decks_config.json"
MODEL_PATH = "card_ppo_model_tuned.pth"
METRICS_SAVE_PATH = "training_metrics_brawl.json"
FIGURE_SAVE_PATH = "figure_brawl.png"

TOTAL_EPISODES = 6000  # 6000+ 局大规模测试
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def step1_tune_cards_and_decks(iteration: int = 0):
    import subprocess
    print(f"\n[{iteration+1}/MAX] 正在调用 DeepSeek 进行第 {iteration+1} 轮真实数值与卡组微调 (请耐心等待)...")
    
    metrics_file = "training_metrics_baseline.json" if iteration == 0 else "training_metrics_brawl.json"
    
    # 1. 调用 LLM 进行真实卡牌数值平衡
    print("  [正在呼叫 DeepSeek] 分析胜率数据并重构卡池...")
    subprocess.run([
        sys.executable, "auto_balancer_deepseek.py", 
        "--metrics", metrics_file, 
        "--cards", "cards_config_baseline.json" if iteration == 0 else "cards_config.json", 
        "--output", "cards_config_tuned.json"
    ], check=True)
    
    # 2. 调用 LLM 进行 30 张卡组合规构筑
    print("  [正在呼叫 DeepSeek] 为三大阵营构筑 30 张实战套牌...")
    subprocess.run([
        sys.executable, "deck_builder_deepseek.py", 
        "--cards", "cards_config.json", 
        "--output", DECKS_FILE, 
        "--factions", "Red,Blue,Green"
    ], check=True)

    # 3. 加载新构筑好的卡组返回供环境仿真使用
    with open(DECKS_FILE, "r", encoding="utf-8") as f:
        decks_data = json.load(f)
    
    red_decklist = decks_data["Red"]["decklist"]
    blue_decklist = decks_data["Blue"]["decklist"]
    green_decklist = decks_data["Green"]["decklist"]

    print("  [OK] DeepSeek 真实数值与卡组调优圆满完成！")
    return {"Red": red_decklist, "Blue": blue_decklist, "Green": green_decklist}

def step2_run_6000_brawl(prebuilt_decks: dict):
    print(f"\n[2/4] 启动 6000 局混战训练 (设备: {DEVICE})...")

    env = DuelEnv(p0_faction=Faction.RED, p1_faction=Faction.BLUE, cards_path=CARDS_FILE)
    
    # 导入训练器
    from train_brawl import PPOTrainer, ROLLOUT_STEPS

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

    for ep in range(1, TOTAL_EPISODES + 1):
        f0 = random.choice(faction_list)
        f1 = random.choice(faction_list)

        name0 = faction_names[f0]
        name1 = faction_names[f1]

        d0 = prebuilt_decks[name0]
        d1 = prebuilt_decks[name1]

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
        all_lengths.append(ep_len)

        p0_won = (env.players[0].score >= env.WIN_SCORE)

        metrics["faction_stats"][name0]["matches"] += 1
        metrics["faction_stats"][name1]["matches"] += 1
        if p0_won:
            metrics["faction_stats"][name0]["wins"] += 1
        else:
            metrics["faction_stats"][name1]["wins"] += 1

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

        if ep % 500 == 0 or ep == 100:
            wr_r = (metrics["faction_stats"]["Red"]["wins"] / max(1, metrics["faction_stats"]["Red"]["matches"])) * 100
            wr_b = (metrics["faction_stats"]["Blue"]["wins"] / max(1, metrics["faction_stats"]["Blue"]["matches"])) * 100
            wr_g = (metrics["faction_stats"]["Green"]["wins"] / max(1, metrics["faction_stats"]["Green"]["matches"])) * 100
            print(f"  [对决进度 {ep:04d}/{TOTAL_EPISODES}] | 胜率: 赤红 {wr_r:.1f}% | 蔚蓝 {wr_b:.1f}% | 翠绿 {wr_g:.1f}%")

    # 战绩与权重保存
    for f_k in ["Red", "Blue", "Green"]:
        m_cnt = metrics["faction_stats"][f_k]["matches"]
        w_cnt = metrics["faction_stats"][f_k]["wins"]
        metrics["faction_stats"][f_k]["winrate"] = round((w_cnt / max(1, m_cnt)) * 100, 2)
    metrics["avg_steps"] = round(float(np.mean(all_lengths)), 2)

    with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    torch.save(trainer.policy.state_dict(), MODEL_PATH)
    print(f"\n[OK] 6000 局混战对抗全部完成！指标已存至 {METRICS_SAVE_PATH}，权重已热更至 {MODEL_PATH}")
    return metrics

def step3_generate_academic_plot(metrics: dict):
    print("\n[3/4] 正在生成三大阵营看板 (figure_brawl.png)...")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(18, 5.5), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.1, 1.2])

    factions = ["Red", "Blue", "Green"]
    names = ["赤红 (Red)", "蔚蓝 (Blue)", "翠绿 (Green)"]
    colors = ["#ff4757", "#1e90ff", "#2ed573"]

    # 1. 胜率柱状图
    ax1 = fig.add_subplot(gs[0])
    win_rates = [metrics["faction_stats"][f]["winrate"] for f in factions]
    play_counts = [metrics["faction_stats"][f]["matches"] for f in factions]

    bars = ax1.bar(names, win_rates, color=colors, width=0.48, edgecolor="#2f3640", linewidth=1.2)
    ax1.axhline(50.0, color="#7f8c8d", linestyle="--", linewidth=1.5, label="50% 理论黄金平衡线")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("阵营综合胜率 (%)", fontsize=11, fontweight="bold")
    ax1.set_title("三大阵营 6000+ 局混战均衡胜率收敛图", fontsize=12, pad=12, fontweight="bold")
    ax1.legend(loc="upper right")

    for bar, count in zip(bars, play_counts):
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%\n({count}局)", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight="bold")

    # 2. 跨阵营对弈胜率矩阵 (3x3 对称双向汇总)
    ax2 = fig.add_subplot(gs[1])
    matrix = np.zeros((3, 3))
    for i, fA in enumerate(factions):
        for j, fB in enumerate(factions):
            if i == j:
                # 内战（自己打自己）在数学和博弈论上必为 50.0% 理论基准
                matrix[i, j] = 50.0
            else:
                # 严格汇总 fA 与 fB 双向交锋场次（同时包含 fA 为 P0 和 fA 为 P1 的总对局）
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
    ax2.set_title("三大阵营对弈克制矩阵热力图 (%)", fontsize=12, pad=12, fontweight="bold")

    for i in range(3):
        for j in range(3):
            val = matrix[i, j]
            text_color = "black" if 42 <= val <= 58 else "white"
            label = "50.0%\n(内战)" if i == j else f"{val:.1f}%"
            ax2.text(j, i, label, ha="center", va="center", color=text_color, fontweight="bold", fontsize=10)

    # 3. 高频核心卡牌 Top 10
    ax3 = fig.add_subplot(gs[2])
    card_counts = metrics.get("card_play_count", {})
    top10 = sorted(card_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top10 = list(reversed(top10))
    c_names = [item[0] for item in top10]
    c_plays = [item[1] for item in top10]

    ax3.barh(c_names, c_plays, color="#f39c12", height=0.6, edgecolor="#d35400", linewidth=1.1)
    ax3.set_xlabel("出战频次 (Play Count)", fontsize=10.5, fontweight="bold")
    ax3.set_title("6000 局混战全阵营出牌频次 Top 10", fontsize=12, pad=12, fontweight="bold")
    ax3.grid(axis="x", linestyle="--", alpha=0.5)

    for i, v in enumerate(c_plays):
        ax3.text(v + max(c_plays) * 0.01, i, str(v), va='center', fontsize=9, fontweight="bold", color="#2c3e50")

    plt.tight_layout()
    plt.savefig(FIGURE_SAVE_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] 混战看板已导出至: {FIGURE_SAVE_PATH}")

def step4_update_readme(metrics: dict):
    print("\n[4/4] 正在检查 README.md 混战数据同步...")
    readme_path = "e:\\PythonApplication23\\README.md"
    if not os.path.exists(readme_path):
        readme_path = "README.md"

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    r_wr = metrics["faction_stats"]["Red"]["winrate"]
    b_wr = metrics["faction_stats"]["Blue"]["winrate"]
    g_wr = metrics["faction_stats"]["Green"]["winrate"]
    r_m = metrics["faction_stats"]["Red"]["matches"]
    b_m = metrics["faction_stats"]["Blue"]["matches"]
    g_m = metrics["faction_stats"]["Green"]["matches"]

    print(f"  当前混战胜率: 红 {r_wr:.1f}% ({r_m}局) | 蓝 {b_wr:.1f}% ({b_m}局) | 绿 {g_wr:.1f}% ({g_m}局)")

MAX_ITERATIONS = 3

def main():
    print("=" * 65)
    print(f"启动 TCG-AI 深度多轮自博弈迭代 (计划 {MAX_ITERATIONS} 轮)")
    print("=" * 65)
    
    metrics = None
    for iteration in range(MAX_ITERATIONS):
        print(f"\n[{iteration+1}/{MAX_ITERATIONS}] 轮次开始 ===========================================")
        # 步骤 1: 调优
        decks = step1_tune_cards_and_decks(iteration)

        # 步骤 2: 跑 6000 局
        metrics = step2_run_6000_brawl(decks)
        
        # 记录当前胜率
        r_wr = metrics["faction_stats"]["Red"]["winrate"]
        b_wr = metrics["faction_stats"]["Blue"]["winrate"]
        g_wr = metrics["faction_stats"]["Green"]["winrate"]
        print(f"  [本轮结果] 红 {r_wr:.1f}% | 蓝 {b_wr:.1f}% | 绿 {g_wr:.1f}%")

    print("\n[最终收敛] 所有迭代完成，开始生成数据大屏与文档...")
    
    # 步骤 3: 画图
    step3_generate_academic_plot(metrics)

    # 步骤 4: 更新 README
    step4_update_readme(metrics)

    # 步骤 5: 同步刷新天梯评级表
    print("\n正在同步更新天梯评级表 (card_tier_table.md & HTML)...")
    os.system(f'"{sys.executable}" generate_hearthstone_tier_table.py')
    print("\n深度进化与混战训练验证全部圆满完成。")

if __name__ == "__main__":
    main()
