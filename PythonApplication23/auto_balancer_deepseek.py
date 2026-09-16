import os
import re
import sys
import json
from openai import OpenAI

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def get_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key and sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as env_key:
                key, _ = winreg.QueryValueEx(env_key, "DEEPSEEK_API_KEY")
        except Exception:
            pass
    return key

DEEPSEEK_API_KEY = get_api_key()

MODEL_NAME = "deepseek-flash"

CONFIG_FILE = "cards_config_baseline.json" if os.path.exists("cards_config_baseline.json") else "cards_config.json"
METRICS_FILE = "training_metrics_baseline.json" if os.path.exists("training_metrics_baseline.json") else "training_metrics.json"
OUTPUT_TUNED_FILE = "cards_config_tuned.json"
EXPANDED_FILE = "cards_config.json"  # 同步更新卡池

import argparse

def load_json(filepath: str) -> dict:
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_json_response(raw_text: str) -> str:
    """提取大模型返回文本中的合法 JSON 对象"""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_text.strip()

def run_deepseek_balance_and_expand(metrics_data: dict, cards_data: dict, client: OpenAI, history_metrics: dict = None) -> str:
    total = max(1, metrics_data.get("total_episodes", 1000))
    
    if "faction_stats" in metrics_data:
        # 三大阵营混战模式
        stats_lines = []
        for f, s in metrics_data["faction_stats"].items():
            stats_lines.append(f"- 【{f}】阵营: 胜率 {s['winrate']:.1f}% ({s['wins']} 胜 / {s['matches']} 场)")
        data_section = "\n".join(stats_lines)
        if "matchups" in metrics_data:
            data_section += f"\n- 两两交手战报: {json.dumps(metrics_data['matchups'], ensure_ascii=False)}"
        target_section = f"""当前三大阵营综合胜率为：
- 🔴 赤红 (Red): {metrics_data['faction_stats']['Red']['winrate']:.1f}%
- 🔵 蔚蓝 (Blue): {metrics_data['faction_stats']['Blue']['winrate']:.1f}%
- 🟢 翠绿 (Green): {metrics_data['faction_stats']['Green']['winrate']:.1f}%
请仔细结合对战数据与交手情况，挑选 2~3 张最关键的单卡进行精准数值微调（例如适度增强胜率略低阵营的前中期战力/解场，或微调胜率偏高阵营的身材），力求三大阵营综合胜率全部稳定在 48%~52% 黄金竞技平衡区间。"""
    else:
        # 传统双人模式
        p0_wins = metrics_data.get("p0_wins", 500)
        p1_wins = metrics_data.get("p1_wins", 500)
        p0_rate = (p0_wins / total) * 100
        p1_rate = (p1_wins / total) * 100
        data_section = f"- 红方胜率 (P0 - 快攻冲锋): {p0_rate:.1f}% ({p0_wins} 胜)\n- 蓝方胜率 (P1 - 控制防守): {p1_rate:.1f}% ({p1_wins} 胜)"
        target_section = f"当前红方胜率为 {p0_rate:.1f}%，蓝方胜率为 {p1_rate:.1f}%。请挑选 2~3 张最关键的卡牌进行微调，力求双方胜率接近 50% 的完美平衡线。"

    history_section = ""
    if history_metrics and history_metrics != metrics_data:
        history_section = f"""
### 历史对局参考：
{json.dumps(history_metrics.get("faction_stats", history_metrics), ensure_ascii=False, indent=2)}
"""

    prompt = f"""
你是一名 TCG 数值平衡工程师。
以下是通过强化学习（PPO 自博弈）收集的对战数据和当前卡池配置文件。

### 1. 训练对局数据:
- 总训练对局: {total} 局
{data_section}
- 卡牌使用频次统计: {json.dumps(metrics_data.get("card_play_count", {}), ensure_ascii=False)}
{history_section}
### 2. 核心机制特性与设计规则：
1. **蓄势规则**：随从怪兽打入进攻区后需要“蓄势一回合”（不能当回合冲锋），除非自带 "RUSH"（突袭）词条。
2. **阻挡机制**：防守怪兽提供站场阈值阻挡，进攻怪蓄势后同路合击突破，总战力 > 防守怪战力即可击穿并获得 1 分（先达 7 分获胜）。
3. **同名卡上限**：每套牌组同一张单卡严格最多携带 3 张。
4. **支持的合法属性字段**：
   - `id`: 卡牌唯一标识
   - `name`: 卡牌名称
   - `card_type`: "MINION" 或 "SPELL"
   - `cost`: 施法消耗法力
   - `base_dp`: 随从基础战力/阻挡阈值 (SPELL 则为 0)
   - `atk_spell_val`: 直伤法术削弱值 (无则为 0)
   - `def_spell_val`: 增益法术防御值 (无则为 0)
   - `tags`: 词条列表 (仅限已支持词条)
5. **已支持的核心词条系统**：
   - `RUSH`：突袭，当下回合可立刻发起冲锋。
   - `DEGRADE_X`：削弱，冲锋碰撞前永久扣除目标防守怪 X 点 DP 上限。
   - `FORTIFY_X`：坚守，打入防守区时立即增加自身 X 点 DP。
   - `SUPPORT_ATK_X`：光环，驻守在防守区时，为本路所有进攻怪冲锋提供 +X DP 战力支援。
   - `BONUS_SCORE_X`：得分强化，冲锋突破或打入空场时，额外增加 X 点获胜积分。
   - `SPAWN_X_Y`：召唤，战吼召唤 Y 只战力为 X 的衍生小兵。
   - `DEATH_DRAW_X`：亡语抽牌，阵亡后摸 X 张牌。
   - `DEATH_MANA_X`：亡语跳费，阵亡后永久上限 +X。
   - `SACRIFICE_1_KILL_1`：献祭己方一名单位，强制消灭敌方一名单位。
   - `ATTACK_ONLY`：限定只能打入进攻区。
   - `DRAW_X` / `RAMP_X` / `TEMP_MANA_1` / `DISCARD_X`：常规法术/战吼词条。

### 3. 待调整卡池配置:
{json.dumps(cards_data, indent=2, ensure_ascii=False)}

---
### 调整要求：
1. **自主数据分析与平衡目标**：
   请完全基于上述训练对局数据（阵营胜率、两两交手战报、单卡使用频次）以及核心游戏规则，自主深入分析导致失衡的根本原因，挑选 2~3 张最关键的单卡进行精准数值微调，力求各阵营胜率稳定在 48%~52% 黄金竞技平衡线。
2. **微调原则**：
   单卡微调应审慎克制（例如费用 ±1，随从战力 ±1~2 点，或适度调整词条参数），严禁大范围推倒重构成熟卡牌。
3. **数据完整性**：严格保持输入卡池中的所有阵营、卡牌名称、ID 与结构完全一致，仅修改所选 2~3 张卡牌的 cost/base_dp/atk_spell_val/def_spell_val/tags 属性数值，绝对不要新增、删除或重命名卡牌 ID。
4. **格式要求**：必须只输出合法且可直接解析的纯 JSON 字符串，不要包含任何解释文本或 Markdown 标记。
"""

    print(f"调用 [{MODEL_NAME}] 读取训练战报，执行数值微调...")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a professional TCG balance engineer. Output ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

def resolve_path(p):
    if not p or os.path.exists(p):
        return p
    alt1 = os.path.join("PythonApplication23", p)
    if os.path.exists(alt1):
        return alt1
    alt2 = os.path.join(os.path.dirname(__file__), os.path.basename(p))
    if os.path.exists(alt2):
        return alt2
    return p

def main():
    parser = argparse.ArgumentParser(description="TCG 卡牌数值自动平衡调优工具")
    parser.add_argument("--cards", type=str, default="cards_config.json", 
                        help="输入的卡池 JSON 文件")
    parser.add_argument("--metrics", type=str, default="training_metrics_brawl.json", 
                        help="输入的训练战报 JSON 文件")
    parser.add_argument("--output", type=str, default="cards_config_tuned.json", 
                        help="输出调优卡池文件路径")
    parser.add_argument("--history", type=str, default=None, 
                        help="基准/历史战报文件路径")
    args = parser.parse_args()

    if not DEEPSEEK_API_KEY:
        print("未找到有效 API Key，请配置 DEEPSEEK_API_KEY 环境变量。")
        return

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )

    cards_file = resolve_path(args.cards)
    metrics_file = resolve_path(args.metrics)
    output_file = resolve_path(args.output)
    history_file = resolve_path(args.history) if args.history else None

    if not os.path.exists(cards_file):
        print(f"找不到卡池配置文件: {cards_file}")
        return

    current_cards = load_json(cards_file)
    metrics_data = load_json(metrics_file)
    history_data = load_json(history_file) if history_file else None

    print(f"卡池输入: {cards_file} | 战报文件: {metrics_file}")
    if history_file:
        print(f"参考战报: {history_file}")

    # 提交请求
    raw_output = run_deepseek_balance_and_expand(metrics_data, current_cards, client, history_metrics=history_data)
    cleaned_json = clean_json_response(raw_output)

    try:
        new_card_pool = json.loads(cleaned_json)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(new_card_pool, f, indent=2, ensure_ascii=False)
            
        print(f"已完成卡池微调，已保存至: {output_file}")

        total_cards = sum(len(cards) for cards in new_card_pool.values())
        print(f"当前卡池规模: {total_cards} 张")
        for faction, cards in new_card_pool.items():
            print(f"  - {faction}: {len(cards)} 张")

    except json.JSONDecodeError as e:
        print(f"JSON 解析异常: {e}")
        print("模型原始输出内容如下：\n", raw_output)

if __name__ == "__main__":
    main()
