import os
import re
import sys
import json
import argparse
from typing import Dict, List, Any
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

# 沙盒已完整支持的核心词条规则定义
LEGAL_TAG_PATTERNS = [
    r"^RUSH$",
    r"^FORTIFY_\d+$",
    r"^DEGRADE_\d+$",
    r"^SUPPORT_ATK_\d+$",
    r"^BONUS_SCORE_\d+$",
    r"^SPAWN_\d+_\d+$",
    r"^DEATH_DRAW_\d+$",
    r"^DEATH_MANA_\d+$",
    r"^SACRIFICE_1_KILL_1$",
    r"^ATTACK_ONLY$",
    r"^DRAW_\d+$",
    r"^RAMP_\d+$",
    r"^TEMP_MANA_\d+$",
    r"^DISCARD_\d+$",
]

def is_legal_tag(tag: str) -> bool:
    return any(re.match(pattern, tag) for pattern in LEGAL_TAG_PATTERNS)

def load_json(filepath: str) -> dict:
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_json_response(raw_text: str) -> str:
    """提取 Markdown 代码块中的纯 JSON 内容"""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_text.strip()

def get_next_id(existing_cards: List[dict], default_start: int) -> int:
    """自动计算不冲突的安全卡牌 ID"""
    if not existing_cards:
        return default_start
    return max(c.get("id", default_start) for c in existing_cards) + 1

def build_prompt(current_pool: dict, faction: str, count: int, theme: str) -> str:
    faction_desc = {
        "Red": "红方（赤红军团）：特色为快攻冲锋、突袭（RUSH）、召唤炮灰小怪（SPAWN_X_Y）、牺牲自爆以小换大（SACRIFICE_1_KILL_1）、破甲削弱（DEGRADE）与空场高伤突破。必须有小怪生成机制来配合自爆与血祭牺牲！",
        "Blue": "蓝方（蔚蓝守卫）：特色为高额阻挡（高DP/坚守FORTIFY）、阵地驻防光环（SUPPORT_ATK）、控制护盾与防线延阻。",
        "Green": "绿方（翡翠林野）：特色为极致跳费成长（RAMP/DEATH_MANA）、单体超模远古巨兽/巨龙（高费高DP大怪，坚守FORTIFY）。核心打法是前期跳费、后期拍巨型大怪正面碾压，绝不生杂毛小怪！",
        "Neutral": "中立（雇佣酒馆）：提供过牌抽卡（DRAW）、通用阻挡身材与战术润滑单卡。"
    }.get(faction, f"{faction} 阵营")

    return f"""
你是一名资深 TCG（集换式卡牌游戏）首席卡牌架构师与创新数值设计师。
你的目标是为我们的【双路对撞战术卡牌游戏（DuelEnv）】设计并印刷全新单卡（印卡系统）。

### 1. 本次印卡需求：
- 目标阵营: 【{faction}】 ({faction_desc})
- 印刷新卡数量: 【{count}】张
- 设计主题/风格诉求: 【{theme if theme else '符合阵营特色，富有构筑深度与博弈互动的全新机制卡'}】

### 2. 核心物理沙盒机制与规则边界：
1. **双路对撞体系**：左右两路独立攻防，随从打入进攻区默认需【蓄势一回合】方可冲锋（除非拥有 `RUSH` 突袭词条）。
2. **同名卡构筑上限**：单卡上限 3 张，牌库 30 张。单卡设计需具备合理曲线与构筑价值，切忌不可替代的绝对单卡。
3. **支持的属性字段（严禁使用 atk/hp 等非沙盒字段）**：
   - `id`: 卡牌整数标识
   - `name`: 中文名称（具有鲜明奇幻风味）
   - `card_type`: "MINION" 或 "SPELL"
   - `cost`: 施法消耗 (0~10)
   - `base_dp`: 随从基础战力/阻挡阈值 (SPELL 必须为 0)
   - `atk_spell_val`: 直伤削弱数值 (无则为 0)
   - `def_spell_val`: 增益护盾数值 (无则为 0)
   - `tags`: 词条列表 (必须严格从下方已支持词条中挑选组装)
4. **沙盒已完整支持的词条系统（严禁虚构沙盒无法解析的词条）**：
   - `RUSH`：突袭，打出当回合立刻就绪冲锋。
   - `FORTIFY_X`：坚守，打入防守区时自身立即增加 X 点 DP。
   - `DEGRADE_X`：削弱，冲锋碰撞前永久扣除目标防守怪 X 点 DP 上限。
   - `SUPPORT_ATK_X`：光环，驻守防守区时，为本路所有己方冲锋怪提供 +X DP 冲锋战力支援。
   - `BONUS_SCORE_X`：得分强化，冲锋突破或打入空场时，额外增加 X 点获胜积分。
   - `SPAWN_X_Y`：召唤，战吼召唤 Y 只战力为 X 的衍生小兵。
   - `DEATH_DRAW_X`：亡语抽牌，阵亡后摸 X 张牌。
   - `DEATH_MANA_X`：亡语跳费，阵亡后法力上限永久 +X。
   - `SACRIFICE_1_KILL_1`：法术专属，献祭己方一名单位，强制消灭敌方一名单位。
   - `ATTACK_ONLY`：限定只能打入进攻区。
   - `DRAW_X`：战吼/施法抽 X 张牌。
   - `RAMP_X`：战吼/施法永久增加 X 点法力上限。
   - `TEMP_MANA_X`：施法当回合获得 X 点临时法力。
   - `DISCARD_X`：负面补偿，使用时从手牌弃掉 X 张牌。

### 3. 当前参考卡池现状 (现有卡牌):
{json.dumps(current_pool, indent=2, ensure_ascii=False)}

---
### 输出格式硬性要求：
必须严格输出纯合法 JSON，结构如下：
{{
  "new_cards": {{
    "{faction}": [
      {{
        "id": 0,
        "name": "卡牌名",
        "card_type": "MINION",
        "cost": 3,
        "base_dp": 3,
        "atk_spell_val": 0,
        "def_spell_val": 0,
        "tags": ["FORTIFY_1"]
      }}
    ]
  }},
  "design_notes": [
    {{
      "card_name": "卡牌名",
      "flavor_and_strategy": "设计意图、策略定位及博弈价值简述"
    }}
  ]
}}
严禁夹带任何额外解释、注释或 Markdown 外壳！
"""

def print_card_table(cards_dict: Dict[str, List[dict]], design_notes: List[dict]):
    """在终端渲染清晰美观的印卡战报"""
    notes_map = {n.get("card_name"): n.get("flavor_and_strategy", "") for n in design_notes}
    print("\n" + "═" * 95)
    print("🖨️  【TCG-AI 印卡工坊】全新生成卡牌一览")
    print("═" * 95)
    print(f"{'ID':<6}{'阵营':<8}{'名称':<14}{'类型':<8}{'费用':<6}{'DP/数值':<10}{'词条 (Tags)':<26}{'设计意图'}")
    print("─" * 95)
    
    for f, cards in cards_dict.items():
        for c in cards:
            cid = c.get("id", 0)
            name = c.get("name", "未知")
            ctype = c.get("card_type", "MINION")
            cost = c.get("cost", 0)
            dp_str = f"DP:{c.get('base_dp', 0)}" if ctype == "MINION" else f"攻{c.get('atk_spell_val',0)}/防{c.get('def_spell_val',0)}"
            tags_str = ",".join(c.get("tags", [])) if c.get("tags") else "无"
            note = notes_map.get(name, "")[:22]
            print(f"{cid:<6}{f:<8}{name:<14}{ctype:<8}{cost:<6}{dp_str:<10}{tags_str:<26}{note}")
    print("═" * 95 + "\n")

def main():
    parser = argparse.ArgumentParser(description="TCG-AI 独立印卡工坊 (Card Designer & Expander)")
    parser.add_argument("--base", type=str, default="cards_config_baseline.json",
                        help="参考基础卡池文件 (默认 cards_config_baseline.json)")
    parser.add_argument("--output", type=str, default="cards_config_expanded.json",
                        help="扩写后导出的目标卡池文件 (默认 cards_config_expanded.json)")
    parser.add_argument("--faction", type=str, default="all",
                        choices=["Red", "Blue", "Green", "Neutral", "all"],
                        help="目标扩充阵营 (默认 all 全阵营)")
    parser.add_argument("--count", type=int, default=5,
                        help="各阵营印刷新卡数量 (默认 5 张)")
    parser.add_argument("--theme", type=str, default=None,
                        help="自定义设计主题或特色需求描述")
    parser.add_argument("--merge-into", type=str, default=None,
                        help="可选：将新卡直接合并写入到指定卡池文件 (如 cards_config.json)")
    args = parser.parse_args()

    if not DEEPSEEK_API_KEY:
        print("⚠️ [安全提示] 未检测到 DEEPSEEK_API_KEY 环境变量！")
        print("请先配置环境变量：")
        print("  Windows PowerShell: $env:DEEPSEEK_API_KEY=\"你的API_KEY\"")
        print("  Linux / macOS:     export DEEPSEEK_API_KEY=\"你的API_KEY\"")
        return

    base_path = args.base if os.path.exists(args.base) else "cards_config.json"
    print(f"📖 读取基准参考卡池: {base_path}")
    current_pool = load_json(base_path)

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )

    faction_id_starts = {
        "Red": 100,
        "Blue": 200,
        "Green": 300,
        "Neutral": 900
    }

    if args.faction == "all":
        factions_to_process = [
            ("Red", args.count, "红方：强化快攻冲锋（RUSH）、破甲削弱（DEGRADE）与牺牲直伤斩杀"),
            ("Blue", args.count, "蓝方：强化阵地驻防（FORTIFY）、光环增益（SUPPORT_ATK）与护盾控制"),
            ("Green", args.count, "绿方：强化跳费成长（RAMP）、远古巨兽巨龙与衍生物召唤（SPAWN）"),
            ("Neutral", max(2, args.count - 2), "中立：强化战术润滑、通用过牌抽卡（DRAW）与身材博弈")
        ]
    else:
        theme = args.theme if args.theme else f"{args.faction} 阵营核心机制扩充"
        factions_to_process = [(args.faction, args.count, theme)]

    expanded_pool = json.loads(json.dumps(current_pool))
    all_printed_cards = {}
    all_design_notes = []

    for f_name, f_count, f_theme in factions_to_process:
        print(f"\n" + "─" * 70)
        print(f"🤖 正在调用 [{MODEL_NAME}] 印刷阵营: 【{f_name}】 (目标: {f_count} 张)...")
        print(f"🎯 设计特色主题: {f_theme}")

        prompt = build_prompt(expanded_pool, f_name, f_count, f_theme)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a professional TCG game designer. Output ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        raw_output = response.choices[0].message.content
        cleaned = clean_json_response(raw_output)

        try:
            res_data = json.loads(cleaned)
            new_cards_dict = res_data.get("new_cards", {})
            notes = res_data.get("design_notes", [])
            all_design_notes.extend(notes)

            if f_name not in expanded_pool:
                expanded_pool[f_name] = []
            if f_name not in all_printed_cards:
                all_printed_cards[f_name] = []

            cards_list = new_cards_dict.get(f_name, [])
            if not cards_list and len(new_cards_dict) > 0:
                cards_list = list(new_cards_dict.values())[0]

            for c in cards_list:
                allocated_id = get_next_id(expanded_pool[f_name], faction_id_starts.get(f_name, 500))
                c["id"] = allocated_id
                c["tags"] = [t for t in c.get("tags", []) if is_legal_tag(t)]
                c["cost"] = max(0, int(c.get("cost", 1)))
                c["base_dp"] = max(0, int(c.get("base_dp", 0)))
                c["atk_spell_val"] = max(0, int(c.get("atk_spell_val", 0)))
                c["def_spell_val"] = max(0, int(c.get("def_spell_val", 0)))

                expanded_pool[f_name].append(c)
                all_printed_cards[f_name].append(c)

            print(f"✅ 【{f_name}】阵营成功印制 {len(cards_list)} 张新卡！")

        except json.JSONDecodeError as e:
            print(f"❌ 解析【{f_name}】大模型返回 JSON 异常: {e}")
            print("原始返回内容：\n", raw_output)

    # 打印全局可视化印卡总表
    print_card_table(all_printed_cards, all_design_notes)

    # 写入独立扩充卡池文件
    output_file = args.output
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(expanded_pool, f, indent=2, ensure_ascii=False)
    print(f"💾 全阵营扩充后卡池已保存至专属文件: {output_file}")
    print(f"🔒 原基准卡池 {base_path} 处于只读保护，未受影响。")

    # 统计卡池总数
    print(f"\n📊 扩充后卡池总规模统计:")
    total_cards = sum(len(cards) for cards in expanded_pool.values())
    print(f"   总卡牌数: {total_cards} 张")
    for f, c_list in expanded_pool.items():
        print(f"   └─ {f}: {len(c_list)} 张 (新增 {len(all_printed_cards.get(f, []))} 张)")

    # 若指定了合并写入
    if args.merge_into:
        with open(args.merge_into, "w", encoding="utf-8") as f:
            json.dump(expanded_pool, f, indent=2, ensure_ascii=False)
        print(f"🔗 已同步合并更新至生产卡池: {args.merge_into}")

if __name__ == "__main__":
    main()
