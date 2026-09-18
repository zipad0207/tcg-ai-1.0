"""
TCG-AI 多阵营全自动元平衡调优与混战自博弈流水线
支持 3000 局/轮迭代对抗、纯数据驱动的 DeepSeek 自主微调，以及前后对比图表导出。
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

def resolve_path(p):
    if not p or os.path.exists(p):
        return p
    alt2 = os.path.join(os.path.dirname(__file__), os.path.basename(p))
    if os.path.exists(alt2):
        return alt2
    return p

CARDS_FILE = resolve_path("cards_config.json")
DECKS_FILE = resolve_path("decks_config.json")
MODEL_PATH = resolve_path("card_ppo_model_brawl.pth")
METRICS_SAVE_PATH = resolve_path("training_metrics_brawl.json")
FIGURE_SAVE_PATH = resolve_path("figure_brawl.png")
COMPARISON_FIGURE_PATH = resolve_path("figure_brawl_comparison.png")

TOTAL_EPISODES = 3000  # 用户指定：3000 局/轮大规模实机验证
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_ITERATIONS = 4

def step1_tune_cards(iteration: int = 0):
    import subprocess
    print(f"\n[{iteration+1}/{MAX_ITERATIONS}] 正在唤起 DeepSeek 进行第 {iteration+1} 轮纯数据驱动数值微调...")
    
    metrics_file = METRICS_SAVE_PATH if os.path.exists(METRICS_SAVE_PATH) else resolve_path("training_metrics_brawl.json")
    
    # 调用 LLM 进行真实卡牌数值平衡 (纯客观数据，无人工偏见指导)
    print(f"  [DeepSeek-Flash] 分析 {metrics_file} 战报，优化卡池 {CARDS_FILE}...")
    subprocess.run([
        sys.executable, resolve_path("auto_balancer_deepseek.py"), 
        "--metrics", metrics_file, 
        "--cards", CARDS_FILE, 
        "--output", CARDS_FILE
    ], check=True)
    
    # 加载 30 张成熟套牌
    with open(DECKS_FILE, "r", encoding="utf-8") as f:
        decks_data = json.load(f)
    
    return {
        "Red": decks_data["Red"]["decklist"],
        "Blue": decks_data["Blue"]["decklist"],
        "Green": decks_data["Green"]["decklist"]
    }

def step2_run_brawl_sim(prebuilt_decks: dict, iteration: int = 0):
    print(f"\n[实机对决] 启动第 {iteration+1} 轮 {TOTAL_EPISODES} 局强化学习混战自博弈 (运算设备: {DEVICE})...")

    env = DuelEnv(p0_faction=Faction.RED, p1_faction=Faction.BLUE, cards_path=CARDS_FILE)
    
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
        "faction_card_plays": {"Red": {}, "Blue": {}, "Green": {}},
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
                    acting_fac_name = name0 if acting_player == 0 else name1
                    metrics["faction_card_plays"][acting_fac_name][c_name] = metrics["faction_card_plays"][acting_fac_name].get(c_name, 0) + 1

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

        p0_won = (env.winner == 0) if env.winner is not None else (env.players[0].score >= env.WIN_SCORE)

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
            print(f"  [进度 {ep:04d}/{TOTAL_EPISODES}] | 胜率: 赤红 {wr_r:.1f}% | 蔚蓝 {wr_b:.1f}% | 翠绿 {wr_g:.1f}%")

    # 战绩汇总与保存
    for f_k in ["Red", "Blue", "Green"]:
        m_cnt = metrics["faction_stats"][f_k]["matches"]
        w_cnt = metrics["faction_stats"][f_k]["wins"]
        metrics["faction_stats"][f_k]["winrate"] = round((w_cnt / max(1, m_cnt)) * 100, 2)
    metrics["avg_steps"] = round(float(np.mean(all_lengths)), 2)

    with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    torch.save(trainer.policy.state_dict(), MODEL_PATH)
    print(f"\n[OK] 本轮 {TOTAL_EPISODES} 局混战完成！指标已保存至 {METRICS_SAVE_PATH}")
    return metrics

def step3_generate_comparison_plot(initial_stats: dict, final_metrics: dict, history_wr: list):
    """生成兼具对比与单项细节的学术级对比大图"""
    print(f"\n正在生成混战前后对比大屏图表: {COMPARISON_FIGURE_PATH} ...")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(22, 6.8), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.25], wspace=0.28)

    factions = ["Red", "Blue", "Green"]
    names = ["赤红 (Red)", "蔚蓝 (Blue)", "翠绿 (Green)"]
    f_colors = {"Red": "#ff4757", "Blue": "#1e90ff", "Green": "#2ed573"}

    # 1. 调优前后胜率分组柱状对比图
    ax1 = fig.add_subplot(gs[0])
    x = np.arange(len(factions))
    width = 0.35

    before_wr = [initial_stats[f] for f in factions]
    after_wr = [final_metrics["faction_stats"][f]["winrate"] for f in factions]

    bars1 = ax1.bar(x - width/2, before_wr, width, label='调优前 (初始混战)', color='#bdc3c7', edgecolor='#7f8c8d', linewidth=1.2)
    bars2 = ax1.bar(x + width/2, after_wr, width, label='调优后 (最新平衡态)', color=['#ff4757', '#1e90ff', '#2ed573'], edgecolor='#2c3e50', linewidth=1.2)

    ax1.axhline(50.0, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.8, label="50% 黄金平衡线")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("阵营综合胜率 (%)", fontsize=11, fontweight="bold")
    ax1.set_title("三大阵营调优前后胜率对比", fontsize=13, pad=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9.5)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, color='#555')
    for bar in bars2:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#111')

    # 2. 迭代收敛演变折线图 (各轮演进)
    ax2 = fig.add_subplot(gs[1])
    rounds = [f"R{i}" for i in range(len(history_wr))]
    for f in factions:
        f_series = [h[f] for h in history_wr]
        ax2.plot(rounds, f_series, marker='o', linewidth=2.4, markersize=7, label=names[factions.index(f)], color=f_colors[f])
        for idx, val in enumerate(f_series):
            ax2.annotate(f"{val:.1f}%", (idx, val), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=8.5, fontweight='bold')

    ax2.axhline(50.0, color="#7f8c8d", linestyle="--", linewidth=1.2, label="50% 基准线")
    ax2.set_ylim(35, 65)
    ax2.set_ylabel("胜率收敛走势 (%)", fontsize=11, fontweight="bold")
    ax2.set_title("三大阵营胜率迭代收敛演变轨迹", fontsize=13, pad=12, fontweight="bold")
    ax2.set_xlabel("迭代轮次 (初始 $\\to$ 微调)", fontsize=10, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9.5)
    ax2.grid(True, linestyle='--', alpha=0.3)

    # 3. 跨阵营对弈克制矩阵热力图
    ax3 = fig.add_subplot(gs[2])
    matrix = np.zeros((3, 3))
    for i, fA in enumerate(factions):
        for j, fB in enumerate(factions):
            if i == j:
                matrix[i, j] = 50.0
            else:
                k1 = f"{fA}_vs_{fB}"
                k2 = f"{fB}_vs_{fA}"
                r1 = final_metrics["matchups"].get(k1, {"total": 0})
                r2 = final_metrics["matchups"].get(k2, {"total": 0})
                total_games = r1.get("total", 0) + r2.get("total", 0)
                fA_wins = r1.get(f"{fA}_wins", 0) + r2.get(f"{fA}_wins", 0)
                matrix[i, j] = (fA_wins / total_games) * 100 if total_games > 0 else 50.0

    im = ax3.imshow(matrix, cmap="RdYlGn", vmin=35, vmax=65)
    ax3.set_xticks(range(3))
    ax3.set_yticks(range(3))
    ax3.set_xticklabels(["对手: 赤红", "对手: 蔚蓝", "对手: 翠绿"], fontsize=9.5)
    ax3.set_yticklabels(["本方: 赤红", "本方: 蔚蓝", "本方: 翠绿"], fontsize=9.5)
    ax3.set_title("最终三大阵营对弈克制矩阵 (%)", fontsize=13, pad=12, fontweight="bold")

    for i in range(3):
        for j in range(3):
            val = matrix[i, j]
            text_color = "black" if 42 <= val <= 58 else "white"
            label = "50.0%\n(内战)" if i == j else f"{val:.1f}%"
            ax3.text(j, i, label, ha="center", va="center", color=text_color, fontweight="bold", fontsize=10)

    plt.savefig(COMPARISON_FIGURE_PATH, bbox_inches="tight")
    plt.savefig(FIGURE_SAVE_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] 对比大屏已导出至: {COMPARISON_FIGURE_PATH} 并同步覆盖 {FIGURE_SAVE_PATH}")

def main():
    print("=" * 70)
    print(f"启动 TCG-AI 大卡组多阵营元平衡循环流水线")
    print(f"设定: 单轮对决规模 {TOTAL_EPISODES} 局 | 终止条件: 三大阵营最大偏离度 <= 2.80% (黄金平衡带)")
    print("=" * 70)

    # 记录初始未平衡状态 (Round 0)
    initial_stats = {"Red": 56.66, "Blue": 46.48, "Green": 47.23}
    history_wr = [initial_stats]

    final_metrics = None

    for iteration in range(MAX_ITERATIONS):
        print(f"\n{'='*30} 轮次 {iteration+1}/{MAX_ITERATIONS} {'='*30}")
        
        # 1. 纯客观 DeepSeek 微调
        decks = step1_tune_cards(iteration)

        # 2. 跑 3000 局混战对抗
        metrics = step2_run_brawl_sim(decks, iteration)
        final_metrics = metrics

        r_wr = metrics["faction_stats"]["Red"]["winrate"]
        b_wr = metrics["faction_stats"]["Blue"]["winrate"]
        g_wr = metrics["faction_stats"]["Green"]["winrate"]

        history_wr.append({"Red": r_wr, "Blue": b_wr, "Green": g_wr})
        print(f"\n[轮次 {iteration+1} 结算] 赤红: {r_wr:.1f}% | 蔚蓝: {b_wr:.1f}% | 翠绿: {g_wr:.1f}%")

        # 判定是否达成多阵营纳什均衡 (最大偏离度 <= 2.80%)
        max_dev = max(abs(r_wr - 50.0), abs(b_wr - 50.0), abs(g_wr - 50.0))
        target_tol = 2.80
        if max_dev <= target_tol:
            print(f"\n🎉 纳什均衡达成！三大阵营最大偏离度仅 {max_dev:.2f}% (<= {target_tol:.2f}%)，进入黄金平衡带，终止迭代！")
            break
        else:
            print(f"⚠️ 当前最大偏离度为 {max_dev:.2f}% (目标 <= {target_tol:.2f}%)，继续触发下一轮微调...")

    # 生成图表
    step3_generate_comparison_plot(initial_stats, final_metrics, history_wr)

    print("\n流水线执行完毕！")

if __name__ == "__main__":
    main()
