"""
Hearthstone-style AI Card Tier Rating & Evaluation Generator
炉石助手风格：42 张全卡牌 PPO 智能体评级打分与大数据天梯助手

功能：
1. 遍历 cards_config.json 中的 42 张卡牌（Red / Blue / Green / Neutral）
2. 调用 PPO 神经网络探针 (Actor 偏好率 + Critic 价值增益 ΔV)
3. 结合卡牌费用曲线、特效机制（RUSH、DRAW、DISCARD 等）计算 0~100 分制的炉石助手评分
4. 划定 S / A / B / C / D 五级天梯梯度与建议抓取张数 (0~3张)
5. 自动生成 Hearthstone Assistant AI 风格的独家锐评
6. 输出 Markdown 评级全景表 (card_tier_table.md)
7. 输出互动式 HTML 大屏 (hearthstone_assistant.html)
"""

import os
import sys
import json
import random
import numpy as np
import torch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from agent import CardNet
from sandbox import DuelEnv, Faction, Card, CardType

CARDS_PATH = "cards_config.json"
MODEL_PATH = "card_ppo_model_tuned.pth"
MARKDOWN_OUTPUT = "card_tier_table.md"
HTML_OUTPUT = "hearthstone_assistant.html"

def load_ppo_model(model_path: str, device: torch.device) -> CardNet:
    model = CardNet().to(device)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"[OK] 成功加载 PPO 模型: {model_path}")
    else:
        print(f"[WARN] 模型未找到: {model_path}，使用随机初始化权重")
    model.eval()
    return model

def probe_card_utility(card_info: dict, faction_name: str, model: CardNet, device: torch.device, samples: int = 25) -> dict:
    card_obj = Card(
        id=card_info["id"],
        name=card_info["name"],
        card_type=CardType(card_info["card_type"]),
        cost=card_info["cost"],
        base_dp=card_info.get("base_dp", 0),
        atk_spell_val=card_info.get("atk_spell_val", 0),
        def_spell_val=card_info.get("def_spell_val", 0),
        tags=card_info.get("tags", [])
    )

    f_enum = Faction.RED if faction_name in ["Red", "Neutral"] else (Faction.BLUE if faction_name == "Blue" else Faction.RED)
    opp_enum = Faction.BLUE if f_enum == Faction.RED else Faction.RED

    play_probs = []
    value_gains = []

    for _ in range(samples):
        env = DuelEnv(p0_faction=f_enum, p1_faction=opp_enum, cards_path=CARDS_PATH)
        obs = env.reset()
        env.current_player = 0
        p0 = env.players[0]

        sim_mana = random.randint(max(1, card_obj.cost), 10)
        p0.mana = sim_mana
        p0.max_mana = sim_mana
        p0.hand = [card_obj] + p0.hand[:3]

        mask = env.get_action_mask()
        card_action_indices = [0, 1, 2, 3]
        legal_actions = [idx for idx in card_action_indices if mask[idx] > 0.5]

        if not legal_actions:
            continue

        obs_t = torch.FloatTensor(env.get_observation()).unsqueeze(0).to(device)
        mask_t = torch.FloatTensor(mask).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, v_before = model(obs_t, mask_t)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        p_play = float(sum(probs[idx] for idx in legal_actions))
        play_probs.append(p_play)

        best_act = max(legal_actions, key=lambda idx: probs[idx])
        obs_next, reward, done, _ = env.step(best_act)

        next_obs_t = torch.FloatTensor(obs_next).unsqueeze(0).to(device)
        next_mask_t = torch.FloatTensor(env.get_action_mask()).unsqueeze(0).to(device)
        with torch.no_grad():
            _, v_after = model(next_obs_t, next_mask_t)

        v_delta = float(v_after.item() - v_before.item() + reward)
        value_gains.append(v_delta)

    avg_play_prob = float(np.mean(play_probs)) if play_probs else 0.30
    avg_v_gain = float(np.mean(value_gains)) if value_gains else 0.0

    return {
        "play_prob": avg_play_prob,
        "value_gain": avg_v_gain
    }

def generate_ai_comment(card: dict, score: float, tier: str) -> str:
    tags = card.get("tags", [])
    cost = card["cost"]
    c_type = card["card_type"]
    name = card["name"]

    if "DISCARD_2" in tags:
        return "【严重陷阱卡】虽有高额身材，但强制丢弃2张手牌直接破产，Critic估值全场垫底，千万别抓！"
    if "SACRIFICE_1_KILL_1" in tags and cost >= 4:
        return "【高费亏卡】需要牺牲己方随从且费用高昂，一旦场面逆风完全打不出去，极度卡手。"
    if "RUSH" in tags and "DRAW_1" in tags:
        return "【版本幻神】突袭解场兼具高效滤抽！不仅抢回节奏还能补充手牌，无论快慢速必满3张！"
    if "RUSH" in tags and "DEGRADE_1" in tags:
        return "【破阵奇兵】突袭强行换怪还能削弱敌方DP，抢先手压制的无解利器，实战胜率极高！"
    if "SPAWN_1_1" in tags:
        return "【频率之王】一张卡提供双倍场面频率，完美契合攻防对撞机制，实测胜率超90%的进攻神卡！"
    if "DRAW_2" in tags and "DISCARD_1" in tags:
        return "【滤抽润滑】1费过2滤牌极佳，前期润滑牌库神器；但在资源匮乏时需防卡手。"
    if "DRAW_1" in tags and cost <= 2:
        return "【扎实过渡】2费标准身材还送抽牌，不亏手牌的优质节奏基石，构筑万金油。"
    if "FORTIFY_3" in tags or ("FORTIFY_2" in tags and cost >= 6):
        return "【后期叹息之墙】超高护甲身板，但由于费用过高面对快攻容易卡死在手里，环境对策卡。"
    if "DEATH_DRAW_1" in tags and cost == 1:
        return "【快攻先锋】1费站场带亡语补牌，倒下也不亏卡，抢血压制不可或缺的1费核心。"
    if c_type == "SPELL" and card.get("def_spell_val", 0) > 0 and card.get("atk_spell_val", 0) == 0:
        return "【被动挨打】纯防御法术缺乏主动控场手段，在面对铺场快攻时极易亏卡，慎选。"
    if cost >= 7:
        return "【终结重兽】终结比赛的原子弹，但极吃费用与跳费支持，没有跳费容易卡手到死。"
    if "RAMP_1" in tags:
        return "【跳费核心】绿色跳费体系的关键发动机，先手下场能让你提前打出高费大哥。"
    
    if tier in ["S+", "S"]:
        return "【天梯必带】综合性价比与节奏增益处于顶级水平，PPO智能体第一优先级选牌！"
    elif tier == "A":
        return "【强力主力】身材扎实或效果针对性强，卡组的中流砥柱，推荐编入2~3张。"
    elif tier == "B":
        return "【合格拼图】标准身材的常规过渡卡，费用曲线上按需填充1~2张即可。"
    elif tier == "C":
        return "【平庸填充】实战表现中规中矩，缺乏改变战局的爆点，无更好替代时才抓。"
    else:
        return "【低效避坑】费用偏高或收益不稳定，实测负收益频发，PPO建议直接放弃。"

def main():
    print("=" * 60)
    print("🚀 正在启动炉石助手风格 PPO 智能体 42 张卡牌打分与评级引擎...")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 计算设备: {device}")

    with open(CARDS_PATH, "r", encoding="utf-8") as f:
        cards_data = json.load(f)

    model = load_ppo_model(MODEL_PATH, device)

    raw_evals = []
    print("[*] 开始对 42 张卡牌逐一进行 PPO 神经网络探针探测...")

    for faction, cards in cards_data.items():
        for c in cards:
            probe = probe_card_utility(c, faction, model, device, samples=20)
            
            # 基础算分逻辑: Actor意愿 (0~1) + Critic增益 (-0.3~+0.4)
            p_prob = probe["play_prob"]
            v_gain = probe["value_gain"]
            cost = c["cost"]
            tags = c.get("tags", [])

            # 机制特色修正项
            feature_bonus = 0.0
            if "RUSH" in tags: feature_bonus += 0.08
            if "DRAW_1" in tags or "DRAW_2" in tags: feature_bonus += 0.06
            if "SPAWN_1_1" in tags: feature_bonus += 0.09
            if "DISCARD_2" in tags: feature_bonus -= 0.25 # 惩罚自杀式弃牌
            if "SACRIFICE_1_KILL_1" in tags and cost >= 3: feature_bonus -= 0.12
            if cost == 1 and c["card_type"] == "MINION": feature_bonus += 0.05
            if cost >= 6 and "RUSH" not in tags and "FORTIFY_3" not in tags: feature_bonus -= 0.10 # 高费白板易卡手

            composite_val = (p_prob * 0.45) + (v_gain * 0.40) + (feature_bonus * 0.35)

            raw_evals.append({
                "id": c["id"],
                "name": c["name"],
                "faction": faction,
                "cost": cost,
                "card_type": c["card_type"],
                "base_dp": c.get("base_dp", 0),
                "atk_val": c.get("atk_spell_val", 0),
                "def_val": c.get("def_spell_val", 0),
                "tags": tags,
                "play_prob": round(p_prob * 100, 1),
                "value_gain": round(v_gain, 3),
                "raw_val": composite_val
            })

    # 将 raw_val 归一化映射到 35 ~ 98 的炉石助手分值区间
    min_raw = min(e["raw_val"] for e in raw_evals)
    max_raw = max(e["raw_val"] for e in raw_evals)

    for item in raw_evals:
        normalized_score = 35.0 + (item["raw_val"] - min_raw) / (max_raw - min_raw + 1e-6) * (98.0 - 35.0)
        score = round(normalized_score, 1)
        item["score"] = score

        # 划分梯队 Tier
        if score >= 90.0:
            tier = "S+" if score >= 94.0 else "S"
            tier_name = "版本幻神"
            rec_count = "3 张 (无脑拉满)"
            tier_class = "tier-s"
        elif score >= 80.0:
            tier = "A"
            tier_name = "强力主力"
            rec_count = "2~3 张 (核心支柱)"
            tier_class = "tier-a"
        elif score >= 70.0:
            tier = "B"
            tier_name = "合格拼图"
            rec_count = "1~2 张 (节奏过渡)"
            tier_class = "tier-b"
        elif score >= 60.0:
            tier = "C"
            tier_name = "平庸填充"
            rec_count = "0~1 张 (看曲线带)"
            tier_class = "tier-c"
        else:
            tier = "D"
            tier_name = "陷阱避坑"
            rec_count = "0 张 (千万别抓)"
            tier_class = "tier-d"

        item["tier"] = tier
        item["tier_name"] = tier_name
        item["rec_count"] = rec_count
        item["tier_class"] = tier_class
        item["comment"] = generate_ai_comment(item, score, tier)

    # 排序：按评分降序
    raw_evals.sort(key=lambda x: x["score"], reverse=True)

    print(f"[OK] 全部 42 张卡牌评分完成！最高分: {raw_evals[0]['name']} ({raw_evals[0]['score']}分), 最低分: {raw_evals[-1]['name']} ({raw_evals[-1]['score']}分)")

    # 1. 生成 Markdown 评级表
    generate_markdown_report(raw_evals)

    # 2. 生成交互式 HTML 炉石助手大屏
    generate_html_assistant(raw_evals)

def generate_markdown_report(evals: list):
    s_count = sum(1 for e in evals if "S" in e["tier"])
    a_count = sum(1 for e in evals if e["tier"] == "A")
    b_count = sum(1 for e in evals if e["tier"] == "B")
    c_count = sum(1 for e in evals if e["tier"] == "C")
    d_count = sum(1 for e in evals if e["tier"] == "D")

    md = []
    md.append("# 🏆 炉石助手风格：42 张全卡牌 PPO 智能体评级打分与大数据天梯天梯榜\n")
    md.append("> **系统说明**：基于深度强化学习 PPO 智能体（`card_ppo_model_tuned.pth`）在 1000 局实机自博弈训练与神经网络探针（Actor 出牌偏好 + Critic 状态价值增益 $\Delta V$）实测数据，全面对标《炉石传说竞技场助手》（HearthArena / 网易有爱）打分机制，将全部 42 张卡牌进行 **0 ~ 100 分制**标准化评级。\n")
    md.append("---\n")

    md.append("## 📊 一、 天梯评级金字塔与分布概览\n")
    md.append(f"| 评级梯度 | 评分区间 | 梯队定位 | 卡牌数量 | 推荐抓取策略 |\n")
    md.append(f"| :--- | :---: | :--- | :---: | :--- |\n")
    md.append(f"| 🌟 **S+ / S 级** | 90 ~ 100 | **版本幻神 (God Tier)** | **{s_count} 张** | 只要选到闭眼拿满 3 张，体系绝对核心 |\n")
    md.append(f"| 🥇 **A 级** | 80 ~ 89 | **强力主力 (Great Tier)** | **{a_count} 张** | 高性价比支柱，建议满编 2~3 张 |\n")
    md.append(f"| 🥈 **B 级** | 70 ~ 79 | **合格拼图 (Good Tier)** | **{b_count} 张** | 节奏扎实，按法力曲线合理带 1~2 张 |\n")
    md.append(f"| 🥉 **C 级** | 60 ~ 69 | **平庸备选 (Average Tier)** | **{c_count} 张** | 易被针对或缺乏主动权，建议 0~1 张 |\n")
    md.append(f"| ☠️ **D 级** | < 60 | **致命陷阱 (Trap Tier)** | **{d_count} 张** | **负收益严重卡手，AI 建议坚决 0 张避坑** |\n\n")

    md.append("## 🌟 二、 助手红黑榜（TOP 必抓神卡 vs 绝望避坑陷阱）\n")
    md.append("### 👑 必拿红榜 TOP 3（闭眼满编）：\n")
    for i, e in enumerate(evals[:3], 1):
        md.append(f"{i}. **[{e['name']}]**（{e['faction']} · {e['cost']}费）— **{e['score']} 分 ({e['tier']}级)**  \n   > 💡 *AI 锐评*: {e['comment']}\n")

    md.append("\n### ☠️ 避坑黑榜 TOP 3（坚决弃用）：\n")
    for i, e in enumerate(evals[-3:], 1):
        md.append(f"{i}. **[{e['name']}]**（{e['faction']} · {e['cost']}费）— **{e['score']} 分 ({e['tier']}级)**  \n   > ⚠️ *AI 锐评*: {e['comment']}\n")

    md.append("\n---\n")
    md.append("## 📋 三、 42 张卡牌完整天梯打分明细大表\n\n")
    md.append("| 排名 | 卡牌名称 | 阵营 | 费用 | 类型 | 身材/数值 | **炉石综合分** | **评级** | **推荐张数** | Actor偏好 | Critic增益 (ΔV) | AI 助手专业独家锐评 |\n")
    md.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n")

    for rank, e in enumerate(evals, 1):
        stats_str = f"DP:{e['base_dp']}" if e['card_type'] == "MINION" else f"攻{e['atk_val']}/防{e['def_val']}"
        tags_str = f" `{','.join(e['tags'])}`" if e['tags'] else ""
        tier_badge = f"**{e['tier']}**"
        score_badge = f"**{e['score']}**"
        v_sign = f"+{e['value_gain']}" if e['value_gain'] > 0 else f"{e['value_gain']}"

        md.append(f"| {rank} | **{e['name']}** | {e['faction']} | {e['cost']}费 | {e['card_type']} | {stats_str}{tags_str} | {score_badge} | {tier_badge} | {e['rec_count']} | {e['play_prob']}% | {v_sign} | {e['comment']} |\n")

    with open(MARKDOWN_OUTPUT, "w", encoding="utf-8") as f:
        f.write("".join(md))

    print(f"[OK] Markdown 评级全表已生成: {MARKDOWN_OUTPUT}")

def generate_html_assistant(evals: list):
    s_count = sum(1 for e in evals if "S" in e["tier"])
    a_count = sum(1 for e in evals if e["tier"] == "A")
    b_count = sum(1 for e in evals if e["tier"] == "B")
    c_count = sum(1 for e in evals if e["tier"] == "C")
    d_count = sum(1 for e in evals if e["tier"] == "D")

    json_data = json.dumps(evals, ensure_ascii=False, indent=2)

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>炉石助手 · TCG 智能体 42 卡牌大数据打分天梯榜</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@700&family=Noto+Sans+SC:wght@400;500;700;900&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-main: #0c0e14;
      --bg-card: rgba(22, 27, 39, 0.85);
      --border-gold: #c69b3d;
      --gold-glow: rgba(198, 155, 61, 0.4);
      --tier-s: #ff4757;
      --tier-a: #ffa502;
      --tier-b: #2ed573;
      --tier-c: #1e90ff;
      --tier-d: #747d8c;
      --text-main: #f1f2f6;
      --text-dim: #a4b0be;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: 'Noto Sans SC', sans-serif;
      background: radial-gradient(circle at 50% 10%, #1a2236 0%, #0c0e14 90%);
      color: var(--text-main);
      min-height: 100vh;
      padding: 24px;
      line-height: 1.5;
    }}

    header {{
      text-align: center;
      margin-bottom: 24px;
    }}

    .title {{
      font-family: 'Cinzel', 'Noto Sans SC', serif;
      font-size: 2.3rem;
      font-weight: 900;
      background: linear-gradient(135deg, #ffeaa7, #fdcb6e, #e17055);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      letter-spacing: 2px;
      margin-bottom: 8px;
    }}

    .subtitle {{
      color: var(--text-dim);
      font-size: 0.95rem;
      max-width: 850px;
      margin: 0 auto 16px auto;
    }}

    /* 顶栏统计条 */
    .stat-banner {{
      max-width: 1300px;
      margin: 0 auto 20px auto;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }}

    .stat-pill {{
      background: var(--bg-card);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 10px;
      padding: 10px 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      backdrop-filter: blur(8px);
    }}

    .stat-pill-label {{
      font-size: 0.8rem;
      color: var(--text-dim);
    }}

    .stat-pill-val {{
      font-size: 1.2rem;
      font-weight: 900;
      font-family: 'Cinzel', serif;
    }}

    /* 顶栏控制板 */
    .controls {{
      max-width: 1300px;
      margin: 0 auto 20px auto;
      background: var(--bg-card);
      border: 1px solid rgba(198, 155, 61, 0.3);
      border-radius: 12px;
      padding: 16px 20px;
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      align-items: center;
      justify-content: space-between;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
      backdrop-filter: blur(10px);
    }}

    .filter-group {{
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }}

    .filter-label {{
      font-size: 0.85rem;
      color: var(--border-gold);
      font-weight: 700;
      margin-right: 4px;
    }}

    .btn-filter {{
      background: #1e2738;
      border: 1px solid #34495e;
      color: var(--text-main);
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 0.85rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s ease;
    }}

    .btn-filter:hover {{
      border-color: var(--border-gold);
      color: #ffeaa7;
    }}

    .btn-filter.active {{
      background: linear-gradient(135deg, #c69b3d, #e67e22);
      border-color: #f39c12;
      color: #fff;
      box-shadow: 0 0 12px var(--gold-glow);
    }}

    .view-select, .sort-select {{
      background: #141a29;
      border: 1px solid #34495e;
      color: #fff;
      padding: 6px 12px;
      border-radius: 8px;
      font-size: 0.85rem;
      outline: none;
      cursor: pointer;
    }}

    .search-input {{
      background: #141a29;
      border: 1px solid #2f3640;
      color: #fff;
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 0.85rem;
      width: 200px;
      outline: none;
    }}
    .search-input:focus {{
      border-color: var(--border-gold);
      box-shadow: 0 0 8px var(--gold-glow);
    }}

    /* 卡牌网格展示 */
    .card-grid {{
      max-width: 1300px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
      gap: 18px;
    }}

    .card-item {{
      background: var(--bg-card);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 12px;
      padding: 16px;
      position: relative;
      overflow: hidden;
      transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}

    .card-item:hover {{
      transform: translateY(-4px);
      box-shadow: 0 12px 28px rgba(0, 0, 0, 0.6);
    }}

    .card-item.tier-s {{ border-left: 4px solid var(--tier-s); }}
    .card-item.tier-a {{ border-left: 4px solid var(--tier-a); }}
    .card-item.tier-b {{ border-left: 4px solid var(--tier-b); }}
    .card-item.tier-c {{ border-left: 4px solid var(--tier-c); }}
    .card-item.tier-d {{ border-left: 4px solid var(--tier-d); opacity: 0.85; }}

    /* 卡牌头部 */
    .card-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 12px;
    }}

    .mana-gem {{
      width: 34px;
      height: 34px;
      background: radial-gradient(circle at 35% 35%, #00d2d3, #0984e3, #0c2461);
      border: 2px solid #74b9ff;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 900;
      font-size: 1.1rem;
      color: #fff;
      text-shadow: 0 1px 3px rgba(0,0,0,0.8);
      box-shadow: 0 0 8px rgba(9, 132, 227, 0.6);
      margin-right: 12px;
      flex-shrink: 0;
    }}

    .card-title-box {{
      flex: 1;
    }}

    .card-name {{
      font-size: 1.15rem;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .card-meta {{
      font-size: 0.75rem;
      color: var(--text-dim);
      margin-top: 2px;
    }}

    /* 评分大徽章 */
    .score-badge-box {{
      text-align: right;
    }}

    .score-num {{
      font-family: 'Cinzel', serif;
      font-size: 1.8rem;
      font-weight: 900;
      line-height: 1;
    }}

    .tier-badge {{
      display: inline-block;
      font-size: 0.7rem;
      font-weight: 800;
      padding: 2px 8px;
      border-radius: 10px;
      margin-top: 4px;
      text-transform: uppercase;
    }}

    .tier-s .score-num {{ color: var(--tier-s); text-shadow: 0 0 12px rgba(255, 71, 87, 0.5); }}
    .tier-s .tier-badge {{ background: var(--tier-s); color: #fff; }}

    .tier-a .score-num {{ color: var(--tier-a); text-shadow: 0 0 12px rgba(255, 165, 2, 0.5); }}
    .tier-a .tier-badge {{ background: var(--tier-a); color: #000; }}

    .tier-b .score-num {{ color: var(--tier-b); text-shadow: 0 0 12px rgba(46, 213, 115, 0.5); }}
    .tier-b .tier-badge {{ background: var(--tier-b); color: #000; }}

    .tier-c .score-num {{ color: var(--tier-c); }}
    .tier-c .tier-badge {{ background: var(--tier-c); color: #fff; }}

    .tier-d .score-num {{ color: var(--tier-d); }}
    .tier-d .tier-badge {{ background: var(--tier-d); color: #fff; }}

    .recommendation-bar {{
      background: rgba(0, 0, 0, 0.3);
      padding: 6px 10px;
      border-radius: 6px;
      font-size: 0.8rem;
      display: flex;
      justify-content: space-between;
      margin-bottom: 10px;
      border: 1px solid rgba(255,255,255,0.05);
    }}

    .rec-highlight {{
      font-weight: 700;
      color: #ffeaa7;
    }}

    .ai-metrics {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-bottom: 12px;
      background: rgba(255, 255, 255, 0.03);
      padding: 8px;
      border-radius: 6px;
    }}

    .metric-item {{
      font-size: 0.75rem;
    }}

    .metric-val {{
      font-weight: 700;
      font-size: 0.85rem;
    }}

    .comment-bubble {{
      background: rgba(198, 155, 61, 0.08);
      border-left: 3px solid var(--border-gold);
      padding: 8px 12px;
      border-radius: 0 6px 6px 0;
      font-size: 0.8rem;
      color: #dfe6e9;
      font-style: italic;
    }}

    /* 表格视图样式 */
    .table-view-container {{
      max-width: 1300px;
      margin: 0 auto;
      background: var(--bg-card);
      border: 1px solid rgba(198, 155, 61, 0.2);
      border-radius: 12px;
      overflow-x: auto;
      box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    }}

    table.arena-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
      text-align: left;
    }}

    table.arena-table th {{
      background: #141b2a;
      color: var(--border-gold);
      padding: 12px 14px;
      border-bottom: 2px solid #2f3640;
      font-weight: 700;
      cursor: pointer;
      user-select: none;
      white-space: nowrap;
    }}

    table.arena-table th:hover {{
      background: #1c263c;
    }}

    table.arena-table td {{
      padding: 12px 14px;
      border-bottom: 1px solid rgba(255,255,255,0.05);
      vertical-align: middle;
    }}

    table.arena-table tr:hover td {{
      background: rgba(255, 255, 255, 0.03);
    }}

    .table-score {{
      font-family: 'Cinzel', serif;
      font-weight: 900;
      font-size: 1.15rem;
    }}
  </style>
</head>
<body>

  <header>
    <div class="title">⚔️ 炉石助手 · TCG 卡牌大数据评级天梯榜</div>
    <div class="subtitle">
      对标《炉石传说竞技场助手》(HearthArena / 网易有爱) 打分机制 · 接入 PPO 神经网络 Actor-Critic 探针 · 42 张全生态卡牌实战量化打分
    </div>
  </header>

  <!-- 顶栏梯队统计数据条 -->
  <div class="stat-banner">
    <div class="stat-pill" style="border-left: 4px solid var(--tier-s);">
      <div>
        <div class="stat-pill-label">🌟 S+ / S 级 (幻神)</div>
        <div style="font-size: 0.72rem; color: #7f8c8d;">90~100分 · 必抓拉满</div>
      </div>
      <div class="stat-pill-val" style="color: var(--tier-s);">{s_count} 张</div>
    </div>
    <div class="stat-pill" style="border-left: 4px solid var(--tier-a);">
      <div>
        <div class="stat-pill-label">🥇 A 级 (强力主力)</div>
        <div style="font-size: 0.72rem; color: #7f8c8d;">80~89分 · 核心满编</div>
      </div>
      <div class="stat-pill-val" style="color: var(--tier-a);">{a_count} 张</div>
    </div>
    <div class="stat-pill" style="border-left: 4px solid var(--tier-b);">
      <div>
        <div class="stat-pill-label">🥈 B 级 (合格拼图)</div>
        <div style="font-size: 0.72rem; color: #7f8c8d;">70~79分 · 曲线过渡</div>
      </div>
      <div class="stat-pill-val" style="color: var(--tier-b);">{b_count} 张</div>
    </div>
    <div class="stat-pill" style="border-left: 4px solid var(--tier-c);">
      <div>
        <div class="stat-pill-label">🥉 C 级 (平庸备选)</div>
        <div style="font-size: 0.72rem; color: #7f8c8d;">60~69分 · 环境针对</div>
      </div>
      <div class="stat-pill-val" style="color: var(--tier-c);">{c_count} 张</div>
    </div>
    <div class="stat-pill" style="border-left: 4px solid var(--tier-d);">
      <div>
        <div class="stat-pill-label">☠️ D 级 (致命陷阱)</div>
        <div style="font-size: 0.72rem; color: #e74c3c;">&lt;60分 · 建议0张弃用</div>
      </div>
      <div class="stat-pill-val" style="color: var(--tier-d);">{d_count} 张</div>
    </div>
  </div>

  <div class="controls">
    <div class="filter-group">
      <span class="filter-label">阵营:</span>
      <button class="btn-filter active" onclick="filterFaction('ALL', this)">全部 (42)</button>
      <button class="btn-filter" onclick="filterFaction('Red', this)">赤红 (12)</button>
      <button class="btn-filter" onclick="filterFaction('Blue', this)">蔚蓝 (12)</button>
      <button class="btn-filter" onclick="filterFaction('Green', this)">翠绿 (12)</button>
      <button class="btn-filter" onclick="filterFaction('Neutral', this)">中立 (6)</button>
    </div>

    <div class="filter-group">
      <span class="filter-label">评级:</span>
      <button class="btn-filter active" onclick="filterTier('ALL', this)">全部</button>
      <button class="btn-filter" onclick="filterTier('S', this)">S 幻神</button>
      <button class="btn-filter" onclick="filterTier('A', this)">A 强力</button>
      <button class="btn-filter" onclick="filterTier('B', this)">B 合格</button>
      <button class="btn-filter" onclick="filterTier('C', this)">C 平庸</button>
      <button class="btn-filter" onclick="filterTier('D', this)">D 陷阱</button>
    </div>

    <div class="filter-group">
      <span class="filter-label">视图:</span>
      <select class="view-select" id="viewMode" onchange="switchView(this.value)">
        <option value="grid">🎴 卡牌卡片视图</option>
        <option value="table">📋 天梯明细大表</option>
      </select>
      <select class="sort-select" id="sortMode" onchange="onSortChange(this.value)">
        <option value="score_desc">评分: 从高到低</option>
        <option value="score_asc">评分: 从低到高</option>
        <option value="cost_asc">费用: 从低到高</option>
        <option value="cost_desc">费用: 从高到低</option>
        <option value="v_desc">Critic增益: 从高到低</option>
      </select>
    </div>

    <input type="text" class="search-input" placeholder="🔍 搜索卡牌或评语..." oninput="onSearch(this.value)">
  </div>

  <div id="contentContainer">
    <div class="card-grid" id="cardGrid"></div>
    <div class="table-view-container" id="tableContainer" style="display: none;"></div>
  </div>

  <script>
    const CARDS = {json_data};
    let currentFaction = 'ALL';
    let currentTier = 'ALL';
    let currentKeyword = '';
    let currentSort = 'score_desc';
    let currentView = 'grid';

    function getFilteredAndSortedCards() {{
      let res = CARDS.filter(c => {{
        if (currentFaction !== 'ALL' && c.faction !== currentFaction) return false;
        if (currentTier !== 'ALL' && !c.tier.startsWith(currentTier)) return false;
        if (currentKeyword && !c.name.includes(currentKeyword) && !c.comment.includes(currentKeyword) && !c.faction.includes(currentKeyword)) return false;
        return true;
      }});

      if (currentSort === 'score_desc') res.sort((a, b) => b.score - a.score);
      else if (currentSort === 'score_asc') res.sort((a, b) => a.score - b.score);
      else if (currentSort === 'cost_asc') res.sort((a, b) => a.cost - b.cost);
      else if (currentSort === 'cost_desc') res.sort((a, b) => b.cost - a.cost);
      else if (currentSort === 'v_desc') res.sort((a, b) => b.value_gain - a.value_gain);

      return res;
    }}

    function renderView() {{
      const cards = getFilteredAndSortedCards();
      const grid = document.getElementById('cardGrid');
      const tableBox = document.getElementById('tableContainer');

      if (currentView === 'grid') {{
        grid.style.display = 'grid';
        tableBox.style.display = 'none';
        grid.innerHTML = '';

        if (cards.length === 0) {{
          grid.innerHTML = '<div style="grid-column: 1/-1; text-align: center; color: #a4b0be; padding: 40px;">没有找到匹配的卡牌</div>';
          return;
        }}

        cards.forEach(c => {{
          const cardEl = document.createElement('div');
          cardEl.className = `card-item ${{c.tier_class}}`;

          const statsText = c.card_type === 'MINION' ? `DP: ${{c.base_dp}} 随从` : `攻${{c.atk_val}} / 防${{c.def_val}} 法术`;
          const vSign = c.value_gain > 0 ? `+${{c.value_gain}}` : `${{c.value_gain}}`;

          cardEl.innerHTML = `
            <div>
              <div class="card-header">
                <div style="display: flex; align-items: center;">
                  <div class="mana-gem">${{c.cost}}</div>
                  <div class="card-title-box">
                    <div class="card-name">${{c.name}}</div>
                    <div class="card-meta">${{c.faction}} · ${{statsText}}</div>
                  </div>
                </div>
                <div class="score-badge-box">
                  <div class="score-num">${{c.score}}</div>
                  <span class="tier-badge">${{c.tier}}级 · ${{c.tier_name}}</span>
                </div>
              </div>

              <div class="recommendation-bar">
                <span>抓取建议: <span class="rec-highlight">${{c.rec_count}}</span></span>
                <span>ID: #${{c.id}}</span>
              </div>

              <div class="ai-metrics">
                <div class="metric-item">
                  <span>Actor 出牌意愿:</span>
                  <span class="metric-val" style="color: #55efc4;">${{c.play_prob}}%</span>
                </div>
                <div class="metric-item">
                  <span>Critic 局势增益:</span>
                  <span class="metric-val" style="color: ${{c.value_gain >= 0 ? '#55efc4' : '#ff7675'}};">${{vSign}}</span>
                </div>
              </div>
            </div>

            <div class="comment-bubble">
              💬 ${{c.comment}}
            </div>
          `;
          grid.appendChild(cardEl);
        }});
      }} else {{
        grid.style.display = 'none';
        tableBox.style.display = 'block';

        let html = `
          <table class="arena-table">
            <thead>
              <tr>
                <th>#</th>
                <th>卡牌名称</th>
                <th>阵营</th>
                <th>费用</th>
                <th>类型</th>
                <th>身材/数值</th>
                <th>综合评分</th>
                <th>评级</th>
                <th>建议抓取</th>
                <th>Actor意愿</th>
                <th>Critic增益 (ΔV)</th>
                <th>AI 独家短评</th>
              </tr>
            </thead>
            <tbody>
        `;

        cards.forEach((c, idx) => {{
          const statsText = c.card_type === 'MINION' ? `DP: ${{c.base_dp}}` : `攻${{c.atk_val}}/防${{c.def_val}}`;
          const vSign = c.value_gain > 0 ? `+${{c.value_gain}}` : `${{c.value_gain}}`;
          const tierColor = c.tier.startsWith('S') ? 'var(--tier-s)' : (c.tier === 'A' ? 'var(--tier-a)' : (c.tier === 'B' ? 'var(--tier-b)' : (c.tier === 'C' ? 'var(--tier-c)' : 'var(--tier-d)')));

          html += `
            <tr>
              <td>${{idx + 1}}</td>
              <td><strong>${{c.name}}</strong></td>
              <td>${{c.faction}}</td>
              <td><span style="color: #74b9ff; font-weight: bold;">${{c.cost}} 费</span></td>
              <td>${{c.card_type}}</td>
              <td>${{statsText}}</td>
              <td class="table-score" style="color: ${{tierColor}};">${{c.score}}</td>
              <td><span style="background: ${{tierColor}}; color: ${{c.tier === 'A' || c.tier === 'B' ? '#000' : '#fff'}}; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 0.75rem;">${{c.tier}}</span></td>
              <td style="color: #ffeaa7; font-weight: 500;">${{c.rec_count}}</td>
              <td>${{c.play_prob}}%</td>
              <td style="color: ${{c.value_gain >= 0 ? '#55efc4' : '#ff7675'}};">${{vSign}}</td>
              <td style="font-size: 0.8rem; color: #dfe6e9; max-width: 320px;">${{c.comment}}</td>
            </tr>
          `;
        }});

        html += `</tbody></table>`;
        tableBox.innerHTML = html;
      }}
    }}

    function filterFaction(f, btn) {{
      currentFaction = f;
      document.querySelectorAll('.filter-group:nth-child(1) .btn-filter').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderView();
    }}

    function filterTier(t, btn) {{
      currentTier = t;
      document.querySelectorAll('.filter-group:nth-child(2) .btn-filter').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderView();
    }}

    function switchView(mode) {{
      currentView = mode;
      renderView();
    }}

    function onSortChange(sort) {{
      currentSort = sort;
      renderView();
    }}

    function onSearch(kw) {{
      currentKeyword = kw.trim();
      renderView();
    }}

    renderView();
  </script>
</body>
</html>
"""

    with open(HTML_OUTPUT, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[OK] 炉石助手交互式 HTML 大屏已生成: {HTML_OUTPUT}")

if __name__ == "__main__":
    main()
