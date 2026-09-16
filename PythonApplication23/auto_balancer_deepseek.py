import os
import re
import sys
import json
import shutil
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
EXPANDED_FILE = "cards_config.json"  # 保持同步更新最新版本

import argparse

def load_json(filepath: str) -> dict:
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_json_response(raw_text: str) -> str:
    """提取大模型返回文本中的合法 JSON 对象，防御 Markdown 格式与思考文本干扰"""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_text.strip()

def run_deepseek_balance_and_expand(metrics_data: dict, cards_data: dict, client: OpenAI, history_metrics: dict = None) -> str:
    total = max(1, metrics_data.get("total_episodes", 1000))
    p0_wins = metrics_data.get("p0_wins", 500)
    p1_wins = metrics_data.get("p1_wins", 500)
    p0_rate = (p0_wins / total) * 100
    p1_rate = (p1_wins / total) * 100

    history_section = ""
    if history_metrics and history_metrics != metrics_data:
        h_total = max(1, history_metrics.get("total_episodes", 1000))
        h_p0 = (history_metrics.get("p0_wins", 500) / h_total) * 100
        h_p1 = (history_metrics.get("p1_wins", 500) / h_total) * 100
        history_section = f"""
### 历史演变轨迹参考：
- 【上一阶段 基准对照组】：红方胜率 {h_p0:.1f}% vs 蓝方胜率 {h_p1:.1f}% (当时防守流大幅占优)
- 【当前阶段 实际遥测数据】：红方胜率 {p0_rate:.1f}% vs 蓝方胜率 {p1_rate:.1f}%
- 【关键观察与反馈】：上一轮调整成功打破了蓝方的防守铁桶阵，但由于加费与数值强化幅度较大，导致红方胜率冲高到 {p0_rate:.1f}%，产生了典型的游戏数值“钟摆过调”效应。
"""

    prompt = f"""
你是一名资深 TCG（集换式卡牌游戏）数值策划与规则平衡工程师。
以下是我们通过强化学习（PPO 自博弈）在个人沙盒中真实跑出的对战数据和当前卡池配置文件。

### 1. 真实训练对局遥测 (当前输入):
- 总训练对局: {total} 局
- 红方胜率 (P0 - 快攻冲锋流): {p0_rate:.1f}% ({p0_wins} 胜)
- 蓝方胜率 (P1 - 控制防守流): {p1_rate:.1f}% ({p1_wins} 胜)
- 卡牌使用频次统计: {json.dumps(metrics_data.get("card_play_count", {}), ensure_ascii=False)}
{history_section}
### 2. 核心机制特性与设计规则：
1. **蓄势规则**：随从怪兽打入进攻区后需要“蓄势一回合”（不能当回合冲锋），除非自带 "RUSH"（突袭）词条。
2. **阻挡机制**：防守怪兽提供站场阈值阻挡，进攻怪蓄势后同路合击突破，总战力 > 防守怪战力即可击穿并获得 1 分（先达 7 分获胜）。
3. **同名卡上限**：每套牌组同一张单卡严格最多携带 3 张。
4. **支持的合法属性字段（严格限制，严禁使用 atk/hp 等未定义字段）**：
   - `id`: 卡牌唯一标识
   - `name`: 卡牌名称
   - `card_type`: "MINION" 或 "SPELL"
   - `cost`: 施法消耗法力
   - `base_dp`: 随从基础战力/阻挡阈值 (SPELL 则为 0)
   - `atk_spell_val`: 直伤法术削弱值 (无则为 0)
   - `def_spell_val`: 增益法术防御值 (无则为 0)
   - `tags`: 词条列表 (仅限已支持词条)
5. **已支持的核心词条系统（只能从以下词条中衍生，绝不能编造沙盒解析不了的词条）**：
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
### 你的核心任务：

任务一：【终极收敛手术级微调 (Final Convergence Micro-adjustments)】。当前红方胜率为 {p0_rate:.1f}%，蓝方胜率为 {p1_rate:.1f}%，离 50% 黄金平衡线只差最后一步！请仅挑选 2~3 张关键卡进行点对点精准微调：
  - 适度回调蓝方过度加费的卡牌（例如将 7 费石像鬼回调至 6 费，或将 4 费弓箭手回调至 3 费并平衡 DP，给蓝方适度节奏支撑）；
  - 或适当轻微削弱红方过强的主力卡（例如红色小队长 DP 由 4 微降至 3，或掠夺者费用微调）；
  - 严禁大改其他已有平衡卡，力求双方胜率收敛至 48% ~ 52% 纳什均衡黄金区间！
任务二：严格保持现有卡牌的名称与总数量完全一致，禁止新增卡牌或删除已有卡牌。
任务三：输出格式必须严格为合法 JSON。

---
### 格式输出硬性约束（至关重要）：
必须**严格只输出合法且可以直接解析的纯 JSON 字符串**（结构必须与输入一致，包含 Red, Blue, Green, Neutral 四个 Key），严禁夹带任何引言、中文解释、注释或额外 Markdown 标记。
"""

    print(f"🤖 正在调用 [{MODEL_NAME}] 读取训练战报，执行自适应数值微调...")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a professional TCG balance engineer. Output ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

def main():
    parser = argparse.ArgumentParser(description="DeepSeek TCG 自适应数值平衡器")
    parser.add_argument("--cards", type=str, default=None, 
                        help="输入的卡池 JSON 文件 (默认自动检测最新调优文件)")
    parser.add_argument("--metrics", type=str, default=None, 
                        help="输入的训练战报 JSON 文件 (默认自动检测最新战报)")
    parser.add_argument("--output", type=str, default="cards_config_tuned.json", 
                        help="输出调优卡池文件路径")
    parser.add_argument("--history", type=str, default="training_metrics_baseline.json", 
                        help="基准/历史战报文件路径 (提供给大模型作为对比参考)")
    args = parser.parse_args()

    if not DEEPSEEK_API_KEY:
        print("❌ 未找到有效 API Key，请检查 DEEPSEEK_API_KEY 配置！")
        return

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )

    # 智能推断输入卡池与战报：
    # 如果已存在 tuned 卡池和 tuned 战报，说明是第 2+ 轮迭代，以 tuned 为输入基准；
    # 否则以 baseline 为输入。
    if args.cards:
        cards_file = args.cards
    elif os.path.exists("cards_config_tuned.json") and os.path.exists("training_metrics_tuned.json"):
        cards_file = "cards_config_tuned.json"
    else:
        cards_file = "cards_config_baseline.json" if os.path.exists("cards_config_baseline.json") else "cards_config.json"

    if args.metrics:
        metrics_file = args.metrics
    elif os.path.exists("training_metrics_tuned.json"):
        metrics_file = "training_metrics_tuned.json"
    else:
        metrics_file = "training_metrics_baseline.json" if os.path.exists("training_metrics_baseline.json") else "training_metrics.json"

    history_file = args.history if (args.history and os.path.exists(args.history) and args.history != metrics_file) else None

    if not os.path.exists(cards_file):
        print(f"❌ 找不到卡池配置文件: {cards_file}")
        return

    current_cards = load_json(cards_file)
    metrics_data = load_json(metrics_file)
    history_data = load_json(history_file) if history_file else None

    print(f"📦 [调优输入] 卡池源文件: {cards_file} | 战报文件: {metrics_file}")
    if history_file:
        print(f"📜 [历史参考] 历史基准战报: {history_file}")

    # 提交 API 请求
    raw_output = run_deepseek_balance_and_expand(metrics_data, current_cards, client, history_metrics=history_data)
    cleaned_json = clean_json_response(raw_output)

    try:
        new_card_pool = json.loads(cleaned_json)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(new_card_pool, f, indent=2, ensure_ascii=False)
        with open("cards_config.json", "w", encoding="utf-8") as f:
            json.dump(new_card_pool, f, indent=2, ensure_ascii=False)
            
        print(f"🚀 成功自适应微调卡池！已保存至: {args.output}")
        if os.path.exists("cards_config_baseline.json"):
            print(f"🔒 用户基准卡池 cards_config_baseline.json 始终处于受保护只读状态。")

        total_cards = sum(len(cards) for cards in new_card_pool.values())
        print(f"📊 当前卡池总规模: {total_cards} 张卡")
        for faction, cards in new_card_pool.items():
            print(f"   └─ {faction}: {len(cards)} 张")

    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析异常: {e}")
        print("模型原始输出内容如下：\n", raw_output)

if __name__ == "__main__":
    main()
