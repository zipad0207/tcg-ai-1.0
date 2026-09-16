import os
import re
import sys
import json
import shutil
from openai import OpenAI

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

MODEL_NAME = "deepseek-chat"

CONFIG_FILE = "cards_config.json"
METRICS_FILE = "training_metrics.json"
BACKUP_FILE = "cards_config_backup.json"
EXPANDED_FILE = "cards_config.json"  # 直接更新卡池

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

def run_deepseek_balance_and_expand(metrics_data: dict, cards_data: dict, client: OpenAI) -> str:
    total = max(1, metrics_data.get("total_episodes", 250))
    p0_wins = metrics_data.get("p0_wins", 100)
    p1_wins = metrics_data.get("p1_wins", 150)
    p0_rate = (p0_wins / total) * 100
    p1_rate = (p1_wins / total) * 100

    prompt = f"""
你是一名资深 TCG（集换式卡牌游戏）数值策划与规则平衡工程师。
以下是我们通过强化学习（PPO 自博弈）在个人沙盒中真实跑出的对战数据和当前卡池配置文件。

### 1. 真实训练数据 (training_metrics.json):
- 总训练对局: {total} 局
- 红方胜率 (P0 - 快攻冲锋流): {p0_rate:.1f}% ({p0_wins} 胜)
- 蓝方胜率 (P1 - 控制防守流): {p1_rate:.1f}% ({p1_wins} 胜)
- 卡牌使用频次统计: {json.dumps(metrics_data.get("card_play_count", {}), ensure_ascii=False)}

### 2. 核心机制特性与设计规则：
1. **蓄势规则**：随从怪兽打入进攻区后需要“蓄势一回合”（不能当回合冲锋），除非自带 "RUSH"（突袭）词条。
2. **阻挡机制**：防守怪兽提供站场阈值阻挡，进攻怪蓄势后同路合击突破，总战力 > 防守怪战力即可击穿并获得 1 分（先达 7 分获胜）。
3. **支持的合法属性字段（严格限制，严禁使用 atk/hp 等未定义字段）**：
   - `id`: 卡牌唯一标识
   - `name`: 卡牌名称
   - `card_type`: "MINION" 或 "SPELL"
   - `cost`: 施法消耗法力
   - `base_dp`: 随从基础战力/阻挡阈值 (SPELL 则为 0)
   - `atk_spell_val`: 直伤法术削弱值 (无则为 0)
   - `def_spell_val`: 增益法术防御值 (无则为 0)
   - `tags`: 词条列表 (仅限已支持词条)
4. **已支持的核心词条系统（只能从以下词条中衍生，绝不能编造沙盒解析不了的词条）**：
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

### 3. 当前卡池完整配置 (cards_config.json):
{json.dumps(cards_data, indent=2, ensure_ascii=False)}

---
### 你的核心任务：

任务一：根据对局遥测数据（胜率偏离度与高频卡牌分布），仅对既有卡牌进行精准的数值微调（包括 cost, base_dp, atk_spell_val, def_spell_val, tags 数值）。
任务二：严格保持现有卡牌的名称与总数量不变，禁止新增卡牌或删除已有卡牌。
任务三：削弱出场率畸高且胜率贡献过大的强势阵营核心卡，增强弱势阵营核心单卡，使双方理论胜率收敛至 45%~55% 纳什均衡区间。

---
### 格式输出硬性约束（至关重要）：
必须**严格只输出合法且可以直接解析的纯 JSON 字符串**（结构必须与原 cards_config.json 完全一致，包含 Red, Blue, Green, Neutral 四个 Key），严禁夹带任何引言、中文解释、注释或额外文本。
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
    if not DEEPSEEK_API_KEY:
        print("⚠️ [安全提示] 未检测到 DEEPSEEK_API_KEY 环境变量！")
        print("为保证密钥安全，已移除源码中的硬编码 Key。请先配置环境变量：")
        print("  Windows PowerShell: $env:DEEPSEEK_API_KEY=\"你的API_KEY\"")
        print("  Linux / macOS:     export DEEPSEEK_API_KEY=\"你的API_KEY\"")
        return

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )

    if not os.path.exists(CONFIG_FILE):
        print(f"❌ 找不到基础卡池配置文件: {CONFIG_FILE}")
        return

    # 1. 读取原卡池与训练数据
    current_cards = load_json(CONFIG_FILE)
    metrics_data = load_json(METRICS_FILE)
    
    # 容错：如果刚清空环境还没产生 metrics，给一套默认值
    if not metrics_data:
        metrics_data = {
            "total_episodes": 1000,
            "p0_wins": 401,
            "p1_wins": 599,
            "card_play_count": {"石像鬼": 182, "大块头": 140, "藤甲兵": 160}
        }

    # 2. 备份原卡池
    shutil.copyfile(CONFIG_FILE, BACKUP_FILE)
    print(f"📦 已备份当前卡池至: {BACKUP_FILE}")

    # 3. 提交 API 请求
    raw_output = run_deepseek_balance_and_expand(metrics_data, current_cards, client)
    cleaned_json = clean_json_response(raw_output)

    # 4. 解析写入
    try:
        new_card_pool = json.loads(cleaned_json)
        with open(EXPANDED_FILE, "w", encoding="utf-8") as f:
            json.dump(new_card_pool, f, indent=2, ensure_ascii=False)
        print(f"🚀 成功自适应微调卡池！已保存至 {EXPANDED_FILE}")

        total_cards = sum(len(cards) for cards in new_card_pool.values())
        print(f"📊 当前卡池总规模: {total_cards} 张卡")
        for faction, cards in new_card_pool.items():
            print(f"   └─ {faction}: {len(cards)} 张")

    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析异常: {e}")
        print("模型原始输出内容如下：\n", raw_output)

if __name__ == "__main__":
    main()