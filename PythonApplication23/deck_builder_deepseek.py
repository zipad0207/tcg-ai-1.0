import os
import re
import sys
import json
import argparse
from typing import Dict, List, Any, Tuple
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
MODEL_NAME = "deepseek-chat"

MAX_COPIES_PER_CARD = 3
DECK_SIZE = 30

def load_json(filepath: str) -> dict:
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_json_response(raw_text: str) -> str:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_text.strip()

def get_faction_pool(card_database: dict, faction: str) -> List[dict]:
    f_cards = card_database.get(faction, [])
    n_cards = card_database.get("Neutral", [])
    return f_cards + n_cards

def heuristic_deck_build(faction: str, pool: List[dict]) -> dict:
    """当 LLM 不可用或超额时，由高级启发式策略算法兜底构建战术卡组"""
    pool_dict = {c["id"]: c for c in pool}
    counts: Dict[int, int] = {c["id"]: 0 for c in pool}

    if faction == "Red":
        # 红方快攻/爆燃流偏好：低费随从、衍生小怪、自爆、突袭
        name = "赤红·裂甲爆燃突袭流"
        archetype = "Swarm Aggro & Sacrifice"
        concept = "前期依靠突击手、号手快速铺场并生成衍生兵，配合射线抢占先手；中期利用自爆与血祭以小换大摧毁防线，后手利用突袭迅速抢分斩杀。"
        combos = [
            "集结号手 (2费生1/1) + 自爆 (2费)：以小兵牺牲瞬间清除敌方重装大怪",
            "裂甲掷斧手 (2费突袭破甲) + 射线 (1费直伤)：快速击穿敌方防线造成突破伤害",
            "赤红突击手 (1费亡语抽卡) + 酒馆密账 (1费抽2弃1)：维持快攻手牌续航"
        ]
        priority = [
            (100, 3), # 赤红突击手
            (107, 3), # 集结号手
            (108, 3), # 裂甲掷斧手
            (103, 3), # 射线
            (101, 3), # 红色小队长
            (102, 2), # 自爆
            (104, 2), # 切割者
            (109, 2), # 破阵狂徒
            (110, 2), # 血祭爆燃
            (111, 2), # 赤红掠袭者
            (903, 2), # 酒馆密账
            (905, 2), # 佣兵斥候
            (105, 1), # 掠夺者
        ]
    elif faction == "Blue":
        # 蓝方壁垒控制流偏好：高坚守、支援光环、法术延阻、厚实大怪
        name = "蔚蓝·晶壁坚垒反制流"
        archetype = "Fortress Control & Support"
        concept = "依托坚守属性将防守区打造为坚不可摧的高DP壁垒；配合支援光环为进攻提供协同战力，中后期出动石像鬼与蔚蓝要塞进行全面战力碾压。"
        combos = [
            "盾兵/霜盾见习官 + 壁垒工匠：叠加强力坚守与支援光环，单路DP迅速突破8点",
            "防御！+ 寒晶护壁：兼具强力防御提升与抽牌润滑，拖延敌方进攻节奏",
            "蔚蓝要塞 (6费坚守3+支援1) + 石像鬼 (8DP)：终结对局的终极攻防一体阵列"
        ]
        priority = [
            (201, 3), # 盾兵
            (200, 3), # 蔚蓝卫士
            (207, 3), # 霜盾见习官
            (208, 3), # 壁垒工匠
            (203, 3), # 防御！
            (209, 3), # 寒晶护壁
            (202, 2), # 弓箭手
            (205, 2), # 藤甲兵
            (206, 2), # 火铳手
            (210, 2), # 冰封禁制
            (204, 2), # 石像鬼
            (211, 2), # 蔚蓝要塞
        ]
    else: # Green
        name = "翡翠·古树巨龙跳费流"
        archetype = "Pure Ramp & Giant Colossus"
        concept = "专注前期跳费提升法力上限，中后期连续召唤远古巨树、翡翠巨熊与灭世巨龙，利用庞大DP与突袭直接压垮对手。"
        combos = [
            "翠绿萌芽 + 芽苗祭司：连续跳费，4回合即可进入7-8费大怪爆发期",
            "狂暴生长 + 翡翠幼龙：冲锋试探，死后返还法力，衔接后期巨龙",
            "世界树恩泽 + 灭世翡翠巨龙 (11DP突袭)：终极清屏突破"
        ]
        priority = [
            (300, 3), # 翠绿萌芽
            (307, 3), # 芽苗祭司
            (311, 3), # 世界树恩泽
            (308, 3), # 翡翠巨熊
            (301, 2), # 树人
            (302, 2), # 剧毒花
            (303, 2), # 狂暴生长
            (304, 2), # 森林之狼
            (305, 2), # 远古巨树
            (306, 2), # 荆棘缠绕
            (309, 2), # 翡翠幼龙
            (310, 2), # 灭世翡翠巨龙
            (900, 2), # 商人
        ]

    for cid, count in priority:
        if cid in pool_dict:
            counts[cid] = min(MAX_COPIES_PER_CARD, count)

    # 严格校验总数至 30
    counts = normalize_counts(counts, pool)
    return {
        "deck_name": name,
        "archetype": archetype,
        "tactical_concept": concept,
        "key_combos": combos,
        "card_allocation": counts
    }

def normalize_counts(counts: Dict[int, int], pool: List[dict]) -> Dict[int, int]:
    """严格规范化卡牌数量为 30 张，且每张卡 0~3 张"""
    pool_ids = [c["id"] for c in pool]
    # 清除不在卡池中的非法 ID
    clean_counts = {cid: min(MAX_COPIES_PER_CARD, max(0, counts.get(cid, 0))) for cid in pool_ids}
    
    total = sum(clean_counts.values())
    
    # 若总数不足 30，按卡牌优先级依次补充至 3 张
    if total < DECK_SIZE:
        for cid in pool_ids:
            while clean_counts[cid] < MAX_COPIES_PER_CARD and total < DECK_SIZE:
                clean_counts[cid] += 1
                total += 1
            if total == DECK_SIZE:
                break
                
    # 若总数超过 30，按高费/高数量卡牌依次削减
    elif total > DECK_SIZE:
        # 逆序削减
        for cid in reversed(pool_ids):
            while clean_counts[cid] > 0 and total > DECK_SIZE:
                clean_counts[cid] -= 1
                total -= 1
            if total == DECK_SIZE:
                break

    return clean_counts

def build_prompt_for_deck(faction: str, pool: List[dict]) -> str:
    pool_desc = []
    for c in pool:
        ctype = c["card_type"]
        dp_str = f"DP:{c.get('base_dp', 0)}" if ctype == "MINION" else f"攻{c.get('atk_spell_val', 0)}/防{c.get('def_spell_val', 0)}"
        tags_str = ",".join(c.get("tags", [])) if c.get("tags") else "无词条"
        pool_desc.append(f"- ID:{c['id']} | [{c['name']}] | 类型:{ctype} | 费用:{c['cost']} | 数值:{dp_str} | 词条:{tags_str}")

    cards_text = "\n".join(pool_desc)

    faction_styles = {
        "Red": "赤红快攻爆发流：核心战术为低费铺场、利用 SPAWN 召唤小兵打炮灰，利用 SACRIFICE_1_KILL_1 (自爆/血祭) 以小换大摧毁高防随从，配合 RUSH 冲锋快速获取胜点 (WIN_SCORE=7)。费用曲线应偏低 (1~3 费为主)。",
        "Blue": "蔚蓝阵地控制流：核心战术为利用 FORTIFY 坚守和 SUPPORT_ATK 支援光环，构筑高 DP 防守壁垒，配合法术延阻敌方突袭，中后期凭借石像鬼/蔚蓝要塞等高战力重随从稳健取胜。费用曲线偏平稳 (2~4 费为主，配高费斩杀)。",
        "Green": "翠绿古树跳费流：核心战术为前期快速跳费 (RAMP)，中期控场，中后期拍下高额 DP 巨兽 (远古巨树、灭世巨龙) 实施全面压制。绝不使用小型杂兵铺场。"
    }.get(faction, "均衡战术卡组")

    return f"""你是一名世界顶尖的 TCG（集换式卡牌游戏）职业选手兼卡组构筑架构师。
我们正在进行【双路对撞 TCG 物理沙盒】的 AI 卡组构筑。

### 规则硬性约束：
1. **卡组总张数**：必须【恰好 30 张卡】。
2. **单卡携带上限**：任何同名卡牌的携带张数【必须在 0 到 3 张之间】（即 0, 1, 2, 3）。
3. **可用卡池**：只能从下方提供的【{faction}】候选卡池（包含阵营专属卡与中立单卡）中选择，不得虚构不存在的 ID。
4. **游戏胜负机制**：先达 7 分者获胜；每回合增长 1 点最大法力（上限 10 点）；双路各自最多容纳 3 个随从。

### 本次构筑阵营：【{faction}】
阵营战术定位：{faction_styles}

### 可选卡池全清单：
{cards_text}

---
### 输出格式规范：
请直接返回合法的纯 JSON 格式（不要输出冗余 markdown 说明），JSON 结构如下：
{{
  "deck_name": "卡组中文响亮名称",
  "archetype": "流派类型（如 Aggro/Sacrifice/Control/Ramp）",
  "tactical_concept": "卡组核心战术思路与获胜逻辑概述（100字以内）",
  "key_combos": [
    "核心联动连招 1 说明",
    "核心联动连招 2 说明"
  ],
  "card_allocation": {{
    "卡牌ID(如100)": 张数(0~3),
    ...
  }}
}}
注意：card_allocation 中所有选定卡牌的张数总和必须严格等于 30！"""

def call_deepseek_deckbuild(faction: str, pool: List[dict]) -> dict:
    if not DEEPSEEK_API_KEY:
        print(f"⚠️ [提示] 未检测到 DEEPSEEK_API_KEY，将启用内置职业级启发式构筑算法构建 {faction} 卡组...")
        return heuristic_deck_build(faction, pool)

    print(f"🤖 正在连接 DeepSeek AI 大模型，为【{faction}】量身构筑 30 张竞技卡组...")
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )
    prompt = build_prompt_for_deck(faction, pool)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一名严谨的 TCG 首席构筑大师与数学分析专家，精通法力曲线平衡与单卡张数分配，严格遵守输出 JSON 约束与 30 张牌库上限。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        content = response.choices[0].message.content
        cleaned = clean_json_response(content)
        parsed = json.loads(cleaned)

        raw_alloc = parsed.get("card_allocation", {})
        int_alloc = {}
        for k, v in raw_alloc.items():
            try:
                int_alloc[int(k)] = int(v)
            except Exception:
                pass

        # 归一化保障绝对等于 30 张
        normalized_alloc = normalize_counts(int_alloc, pool)
        parsed["card_allocation"] = normalized_alloc
        return parsed

    except Exception as e:
        print(f"⚠️ DeepSeek 接口交互或解析异常: {e}，自动切换至启发式大师构筑引擎...")
        return heuristic_deck_build(faction, pool)

def generate_deck_details(deck_data: dict, pool: List[dict]) -> dict:
    pool_dict = {c["id"]: c for c in pool}
    alloc: Dict[int, int] = deck_data["card_allocation"]
    
    decklist = []
    mana_curve = {i: 0 for i in range(10)}
    card_details = []
    minion_count = 0
    spell_count = 0
    total_mana = 0

    for cid, count in sorted(alloc.items(), key=lambda x: (pool_dict.get(x[0], {}).get("cost", 0), x[0])):
        if count <= 0 or cid not in pool_dict:
            continue
        c = pool_dict[cid]
        for _ in range(count):
            decklist.append(cid)
            cost_clamp = min(c["cost"], 9)
            mana_curve[cost_clamp] += 1
            total_mana += c["cost"]
            if c["card_type"] == "MINION":
                minion_count += 1
            else:
                spell_count += 1

        card_details.append({
            "id": cid,
            "name": c["name"],
            "cost": c["cost"],
            "card_type": c["card_type"],
            "base_dp": c.get("base_dp", 0),
            "atk_spell_val": c.get("atk_spell_val", 0),
            "def_spell_val": c.get("def_spell_val", 0),
            "tags": c.get("tags", []),
            "count": count
        })

    avg_cost = round(total_mana / max(1, len(decklist)), 2)

    return {
        "deck_name": deck_data.get("deck_name", "AI 自选卡组"),
        "archetype": deck_data.get("archetype", "Custom"),
        "tactical_concept": deck_data.get("tactical_concept", ""),
        "key_combos": deck_data.get("key_combos", []),
        "total_cards": len(decklist),
        "minion_count": minion_count,
        "spell_count": spell_count,
        "avg_cost": avg_cost,
        "mana_curve": mana_curve,
        "card_allocation": {str(k): v for k, v in alloc.items() if v > 0},
        "card_details": card_details,
        "decklist": decklist
    }

def print_deck_profile(faction: str, deck_info: dict):
    print("\n" + "═" * 80)
    print(f"🃏 【{faction}】AI 竞技卡组发布: 《{deck_info['deck_name']}》")
    print(f"📌 流派定位: {deck_info['archetype']} | 平均费用: {deck_info['avg_cost']} 费 | 构成: 随从 {deck_info['minion_count']} 张 / 法术 {deck_info['spell_count']} 张")
    print("─" * 80)
    print(f"💡 构筑战术设计理念:")
    print(f"   {deck_info['tactical_concept']}")
    if deck_info.get("key_combos"):
        print("🔗 核心战术配合 (Key Combos):")
        for idx, cb in enumerate(deck_info["key_combos"], 1):
            print(f"   {idx}. {cb}")
    print("─" * 80)
    print(f"{'ID':<6}{'费用':<6}{'名称':<12}{'类型':<8}{'数值/DP':<10}{'张数':<6}{'关键词条'}")
    print("─" * 80)
    for c in deck_info["card_details"]:
        dp_str = f"DP:{c['base_dp']}" if c["card_type"] == "MINION" else f"攻{c['atk_spell_val']}/防{c['def_spell_val']}"
        tags_str = ",".join(c["tags"]) if c["tags"] else "-"
        print(f"{c['id']:<6}{c['cost']:<6}{c['name']:<12}{c['card_type']:<8}{dp_str:<10}{c['count']:<6}{tags_str}")
    
    print("─" * 80)
    print("📊 法力曲线分布 (Mana Curve):")
    max_count = max(deck_info["mana_curve"].values()) if deck_info["mana_curve"].values() else 1
    for cost in range(1, 8):
        cnt = deck_info["mana_curve"].get(cost, 0)
        bar = "█" * (cnt * 2)
        print(f"   {cost} 费: {bar:<20} ({cnt} 张)")
    cnt_high = sum(deck_info["mana_curve"].get(c, 0) for c in range(8, 10))
    if cnt_high > 0:
        bar = "█" * (cnt_high * 2)
        print(f" 8+ 费: {bar:<20} ({cnt_high} 张)")
    print(f"   总卡牌数: {deck_info['total_cards']} / 30 张 [合规校验通过]")
    print("═" * 80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="TCG-AI 智能选卡构筑大师 (AI Deckbuilder)")
    parser.add_argument("--cards", type=str, default="cards_config.json",
                        help="卡池配置文件路径 (默认 cards_config.json)")
    parser.add_argument("--output", type=str, default="decks_config.json",
                        help="输出卡组保存文件 (默认 decks_config.json)")
    parser.add_argument("--factions", type=str, default="Red,Blue",
                        help="目标构建阵营 (默认 Red,Blue，支持 Red,Blue,Green)")
    parser.add_argument("--heuristic", action="store_true",
                        help="直接使用纯启发式专家算法，不调用 LLM")
    args = parser.parse_args()

    cards_db = load_json(args.cards)
    if not cards_db:
        print(f"❌ 无法读取卡池数据: {args.cards}")
        return

    factions_to_build = [f.strip() for f in args.factions.split(",") if f.strip()]
    decks_result = {}

    # 若已有现存卡组文件，先加载保留其他阵营
    if os.path.exists(args.output):
        decks_result = load_json(args.output)

    print("\n" + "⚔️ " * 20)
    print("🏛️  TCG AI 大模型卡组构筑系统启动 (Deckbuilder Engine)")
    print("⚔️ " * 20)

    for faction in factions_to_build:
        pool = get_faction_pool(cards_db, faction)
        if not pool:
            print(f"⚠️ 未找到阵营 {faction} 的可用卡池！")
            continue

        if args.heuristic:
            raw_deck = heuristic_deck_build(faction, pool)
        else:
            raw_deck = call_deepseek_deckbuild(faction, pool)

        deck_info = generate_deck_details(raw_deck, pool)
        print_deck_profile(faction, deck_info)
        decks_result[faction] = deck_info

    # 导出卡组配置
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(decks_result, f, indent=2, ensure_ascii=False)

    abs_out = os.path.abspath(args.output)
    print(f"💾 AI 构筑竞技卡组已成功保存至: {abs_out}")

if __name__ == "__main__":
    main()
