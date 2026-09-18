import sys
import json
import os
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 配置中文字体，防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_path(p):
    if os.path.isabs(p) or os.path.exists(p):
        return p
    child_path = os.path.join(SCRIPT_DIR, p)
    if os.path.exists(child_path):
        return child_path
    return p

def resolve_output(p):
    if os.path.isabs(p):
        return p
    return os.path.join(SCRIPT_DIR, os.path.basename(p))

def plot_thesis_comparison(metrics_path, output_filename, is_tuned=False):
    """
    根据对战数据绘制对比图表
    :param metrics_path: 数据 JSON 路径
    :param output_filename: 保存图片的文件名
    :param is_tuned: 是否为调优后数据 (True: 调优后, False: 基准调优前)
    """
    metrics_path = resolve_path(metrics_path)
    output_filename = resolve_output(output_filename)
    if not os.path.exists(metrics_path):
        print(f"找不到文件: {metrics_path}")
        return

    with open(metrics_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    p0_winrate = data.get("p0_winrate", 0.0)
    p1_winrate = data.get("p1_winrate", 0.0)
    card_counts = data.get("card_play_count", {})
    total_episodes = data.get("total_episodes", 1000)

    # 提取 Top 10 卡牌并按频次升序排序（便于水平柱状图从下到上展示）
    top10 = sorted(card_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top10 = list(reversed(top10))
    card_names = [item[0] for item in top10]
    play_counts = [item[1] for item in top10]

    # 根据实验阶段设置图表标题
    if is_tuned:
        fig_title = f"调优后对局胜率分布 (Tuned, PPO {total_episodes}局)"
        bar_title = "Top 10 核心对局卡牌出场频次分布 (调优后)"
    else:
        fig_title = f"基准环境对局胜率分布 (Baseline, PPO {total_episodes}局)"
        bar_title = "Top 10 核心对局卡牌出场频次分布 (基准对照组)"

    # 初始化画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # ================= 1. 左子图：胜率对比 =================
    bars = ax1.bar(
        ["红方 (快攻/突破)", "蓝方 (防守/控制)"], 
        [p0_winrate, p1_winrate], 
        color=["#e74c3c", "#3498db"], 
        width=0.45
    )
    ax1.axhline(50.0, color="#7f8c8d", linestyle="--", linewidth=1.5, label="50% 理论基准线")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("胜率 (%)", fontsize=11)
    ax1.set_title(fig_title, fontsize=12, pad=12, fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9)

    # 柱状图顶部数值标注
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha='center', va='bottom', fontsize=10, fontweight="semibold"
        )

    # ================= 2. 右子图：卡牌频次分布 =================
    ax2.barh(card_names, play_counts, color="#2ecc71", height=0.65)
    ax2.set_xlabel("对局出牌频次", fontsize=11)
    ax2.set_title(bar_title, fontsize=12, pad=12, fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight")
    print(f"图表已导出至: {output_filename}")
    plt.close()

def plot_comprehensive_comparison(baseline_path="training_metrics_baseline.json", tuned_path="training_metrics_tuned.json", output_filename="figure_comparison.png"):
    """绘制调优前后对比图 (2x2 画布)"""
    baseline_path = resolve_path(baseline_path)
    tuned_path = resolve_path(tuned_path)
    output_filename = resolve_output(output_filename)
    if not os.path.exists(baseline_path) or not os.path.exists(tuned_path):
        print(f"数据文件缺失: {baseline_path} 或 {tuned_path}")
        return

    import numpy as np
    with open(baseline_path, 'r', encoding='utf-8') as f:
        b_data = json.load(f)
    with open(tuned_path, 'r', encoding='utf-8') as f:
        t_data = json.load(f)

    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    # 1. 左上：胜率直接对比
    ax1 = fig.add_subplot(gs[0, 0])
    stages = ['红方 (快攻/突破)', '蓝方 (防守/控制)']
    x = np.arange(len(stages))
    width = 0.32

    b_rates = [b_data['p0_winrate'], b_data['p1_winrate']]
    t_rates = [t_data['p0_winrate'], t_data['p1_winrate']]

    bars1 = ax1.bar(x - width/2, b_rates, width, label='调优前 (Baseline)', color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=0.8)
    bars2 = ax1.bar(x + width/2, t_rates, width, label='调优后 (Tuned)', color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=0.8)

    ax1.axhline(50.0, color='#7f8c8d', linestyle='--', linewidth=1.5, label='50% 平衡线')
    ax1.set_ylabel('胜率 (%)', fontsize=12, fontweight='bold')
    ax1.set_title('图 1: 调优前后对局胜率直接对比 (1000局 PPO 自博弈)', fontsize=13, pad=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax1.grid(axis='y', linestyle=':', alpha=0.6)

    for b in bars1:
        h = b.get_height()
        ax1.annotate(f'{h:.1f}%', xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for b in bars2:
        h = b.get_height()
        ax1.annotate(f'{h:.1f}%', xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 2. 右上：失衡偏离度收敛效果
    ax2 = fig.add_subplot(gs[0, 1])
    phases = ['基准对照组 (Baseline)', '闭环调优组 (Tuned)']
    deltas = [abs(b_data['p0_winrate'] - 50.0), abs(t_data['p0_winrate'] - 50.0)]
    colors = ['#e67e22', '#27ae60']
    bars_d = ax2.bar(phases, deltas, color=colors, width=0.45, edgecolor='black', linewidth=0.8)
    ax2.set_ylabel('偏离理论平衡线幅度 |WinRate - 50%| (%)', fontsize=11, fontweight='bold')
    ax2.set_title('图 2: 胜率失衡偏离度收敛效果 (|Δ - 50%|)', fontsize=13, pad=12, fontweight='bold')
    max_d = max(deltas) if deltas else 15
    ax2.set_ylim(0, max(25.0, max_d * 1.25))
    ax2.grid(axis='y', linestyle=':', alpha=0.6)
    for b in bars_d:
        h = b.get_height()
        ax2.annotate(f'±{h:.1f}%', xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 3. 左下：基准阶段核心卡牌 Top 10
    ax3 = fig.add_subplot(gs[1, 0])
    b_top = list(reversed(sorted(b_data.get('card_play_count', {}).items(), key=lambda x: x[1], reverse=True)[:10]))
    ax3.barh([x[0] for x in b_top], [x[1] for x in b_top], color='#3498db', alpha=0.85, edgecolor='black', linewidth=0.6)
    ax3.set_xlabel('打出频次 (局数累积)', fontsize=11, fontweight='bold')
    ax3.set_title('图 3: 基准阶段核心卡牌出场频次 Top 10', fontsize=13, pad=12, fontweight='bold')
    ax3.grid(axis='x', linestyle=':', alpha=0.6)

    # 4. 右下：调优阶段核心卡牌 Top 10
    ax4 = fig.add_subplot(gs[1, 1])
    t_top = list(reversed(sorted(t_data.get('card_play_count', {}).items(), key=lambda x: x[1], reverse=True)[:10]))
    ax4.barh([x[0] for x in t_top], [x[1] for x in t_top], color='#1abc9c', alpha=0.85, edgecolor='black', linewidth=0.6)
    ax4.set_xlabel('打出频次 (局数累积)', fontsize=11, fontweight='bold')
    ax4.set_title('图 4: 调优阶段核心卡牌出场频次 Top 10', fontsize=13, pad=12, fontweight='bold')
    ax4.grid(axis='x', linestyle=':', alpha=0.6)

    plt.suptitle('TCG 卡牌平衡系统：调优前后数据对比', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()
    print(f"对比图表已导出至: {output_filename}")

if __name__ == "__main__":
    # 1. 导出基准对照组图表
    plot_thesis_comparison(
        metrics_path="training_metrics_baseline.json", 
        output_filename="figure_baseline.png", 
        is_tuned=False
    )

    # 2. 导出实验组（调优后）图表
    plot_thesis_comparison(
        metrics_path="training_metrics_tuned.json", 
        output_filename="figure_tuned.png", 
        is_tuned=True
    )

    # 3. 导出对比图表
    plot_comprehensive_comparison(
        baseline_path="training_metrics_baseline.json",
        tuned_path="training_metrics_tuned.json",
        output_filename="figure_comparison.png"
    )