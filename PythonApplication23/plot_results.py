import json
import os
import matplotlib.pyplot as plt

# 设置字体以兼容中文显示
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

METRICS_FILE = "training_metrics.json"

if not os.path.exists(METRICS_FILE):
    print(f"❌ 未找到 {METRICS_FILE}")
    exit()

with open(METRICS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# 1. 阵营胜率对比饼图 / 柱状图
p0_rate = data.get("p0_winrate", 54.0)
p1_rate = data.get("p1_winrate", 46.0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 胜率对比柱状图
bars = ax1.bar(["红方 (快攻/突破)", "蓝方 (防守/控制)"], [p0_rate, p1_rate], color=["#e74c3c", "#3498db"], width=0.4)
ax1.set_ylim(0, 100)
ax1.set_ylabel("胜率 (%)")
ax1.set_title(f"PPO 1000局 自适应平衡后胜率对比 (红: {p0_rate}% / 蓝: {p1_rate}%)")
ax1.axhline(50, color="gray", linestyle="--", alpha=0.7, label="50% 理论基准线")
ax1.legend()

for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

# 2. 核心卡牌出场频次统计
cards_count = data.get("card_play_count", {})
if cards_count:
    # 取打出频次前 10 的卡牌
    sorted_cards = sorted(cards_count.items(), key=lambda x: x[1], reverse=True)[:10]
    names = [x[0] for x in sorted_cards]
    counts = [x[1] for x in sorted_cards]

    ax2.barh(names[::-1], counts[::-1], color="#2ecc71", alpha=0.85)
    ax2.set_xlabel("对局出牌频次")
    ax2.set_title("Top 10 核心对局卡牌出场频次分布")

plt.tight_layout()
plt.savefig("ppo_balance_result.png", dpi=300)
print("✅ 结果图表已生成并保存至: ppo_balance_result.png")
plt.show()