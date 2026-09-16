import json
import os
import matplotlib.pyplot as plt

# 配置中文字体，防止学术图表乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def plot_thesis_comparison(metrics_path, output_filename, is_tuned=False):
    """
    根据遥测数据绘制学术规范图表
    :param metrics_path: 遥测数据 JSON 路径
    :param output_filename: 保存图片的文件名
    :param is_tuned: 是否为调优后数据 (True: 调优后, False: 基准调优前)
    """
    if not os.path.exists(metrics_path):
        print(f"❌ 找不到文件: {metrics_path}")
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

    # 根据实验阶段设置规范学术标题
    if is_tuned:
        fig_title = f"LLM 闭环调优后对局胜率分布 (Tuned, PPO {total_episodes}局)"
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
    print(f"✅ 图表已导出至: {output_filename}")
    plt.show()

if __name__ == "__main__":
    # 1. 导出基准对照组（调优前）图表
    # 如果你的文件名不同，请替换为对应的 json 文件路径
    plot_thesis_comparison(
        metrics_path="training_metrics_baseline.json", 
        output_filename="figure_1_baseline.png", 
        is_tuned=False
    )

    # 2. 导出实验组（调优后）图表
    plot_thesis_comparison(
        metrics_path="training_metrics_tuned.json", 
        output_filename="figure_2_tuned.png", 
        is_tuned=True
    )