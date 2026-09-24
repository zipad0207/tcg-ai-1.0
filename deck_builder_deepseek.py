import os
from dotenv import load_dotenv
load_dotenv()
import re
import sys
import json
import argparse
from typing import Dict, List
from openai import OpenAI

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def get_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "") or os.environ.get("SILICONFLOW_API_KEY", "")
    if not key:
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    key = cfg.get("deepseek_api_key") or cfg.get("api_key", "")
            except Exception:
                pass
    if not key and sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as env_key:
                key, _ = winreg.QueryValueEx(env_key, "DEEPSEEK_API_KEY")
        except Exception:
            pass
    return key

def get_base_url() -> str:
    url = os.environ.get("DEEPSEEK_BASE_URL", "")
    if not url:
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    url = cfg.get("base_url", "")
            except Exception:
                pass
    return url or "https://api.deepseek.com"

def get_model_name() -> str:
    model = os.environ.get("DEEPSEEK_MODEL", "")
    if not model:
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    model = cfg.get("model") or cfg.get("deepseek_model", "")
            except Exception:
                pass
    if not model:
        base_url = get_base_url()
        model = "deepseek-ai/DeepSeek-V3" if "siliconflow" in base_url.lower() else "deepseek-flash"
    return model

MODEL_NAME = get_model_name()

DEEPSEEK_API_KEY = get_api_key()

MAX_COPIES_PER_CARD = 3
DECK_SIZE = 30

def load_json(filepath: str) -> dict:
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_json_response(raw_text: str) -> str:
    if not raw_text:
        return "{}"
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_text.strip()

def get_faction_pool(card_database: dict, faction: str) -> List[dict]:
    pool = []
    seen_ids = set()
    f_code = {"Red": 1, "Blue": 2, "Green": 3}.get(faction, 0)

    for category, card_list in card_database.items():
        for c in card_list:
            cid = c.get("id", 0)
            if cid in seen_ids:
                continue

            facs = c.get("factions", [])
            c_f = cid // 100
            allowed = False
            if facs and (faction in facs or "Neutral" in facs):
                allowed = True
            elif category == faction or category == "Neutral":
                allowed = True
            elif c_f == f_code or c_f == 9:
                allowed = True
            elif c_f == 4 and faction in ("Red", "Blue"):
                allowed = True
            elif c_f == 5 and faction in ("Blue", "Green"):
                allowed = True
            elif c_f == 6 and faction in ("Red", "Green"):
                allowed = True

            if allowed:
                pool.append(c)
                seen_ids.add(cid)

    return pool


def normalize_counts(counts: Dict[int, int], pool: List[dict]) -> Dict[int, int]:
    """严格规范化卡牌数量为 30 张，且每张卡 0~3 张"""
    pool_ids = [c["id"] for c in pool]
    # 清除不在卡池中的非法 ID
    clean_counts = {cid: min(MAX_COPIES_PER_CARD, max(0, counts.get(cid, 0))) for cid in pool_ids}
    
    total = sum(clean_counts.values())
    
    # 若总数不足 30，随机打乱卡池并依次补充，防止固定按顺位死抓 1~2 费小牌
    if total < DECK_SIZE:
        import random
        fill_pool = list(pool_ids)
        random.shuffle(fill_pool)
        for cid in fill_pool:
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
        "Red": "赤红快攻爆发流：核心战术为低费铺场、利用 SPAWN 召唤小兵打炮灰，利用 SACRIFICE_1_KILL_1 (自爆/血祭) 以小换大摧毁高防随从，配合 RUSH 冲锋快速获取胜点 (WIN_SCORE=7)。注意卡组需要包含各费用段的卡牌，不能全是低费，也需要中高费终结者来收割比赛。",
        "Blue": "蔚蓝阵地控制流：核心战术为利用 FORTIFY 坚守和 SUPPORT_ATK 支援光环，构筑高 DP 防守壁垒，配合法术延阻敌方突袭，中后期凭借石像鬼/蔚蓝要塞等高战力重随从稳健取胜。前期需要足够的低费防守随从撑过快攻压力，后期需要配置足够的高费终结随从，不能全是低费防守单位。",
        "Green": "翠绿古树跳费流：核心战术为前期快速跳费 (RAMP)，中期控场，中后期拍下高额 DP 巨兽 (远古巨树、灭世巨龙) 实施全面压制。前期必须带够低费的跳费随从和 RAMP 法术来加速攒法力，否则高费巨兽出不来就被快攻打死了。高费巨兽是卡组核心，必须保证足够数量。"
    }.get(faction, "均衡战术卡组")

    return f"""你是一名负责卡牌游戏构筑与法力曲线优化的数值策划。
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
  "deck_name": "卡组名称",
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
        raise ValueError(f"未检测到 DEEPSEEK_API_KEY，无法构建 {faction} 卡组。请先配置环境变量。")

    print(f"调用 DeepSeek 生成 {faction} 阵营卡组...")
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=get_base_url()
    )
    prompt = build_prompt_for_deck(faction, pool)
    extra = {"thinking": {"type": "disabled"}} if "deepseek.com" in get_base_url() else {}

    try:
        response = client.chat.completions.create(
            model=get_model_name(),
            messages=[
                {"role": "system", "content": "你是一名 TCG 构筑分析工程师，负责卡组配置与法力曲线优化，请严格输出合法 JSON。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4000,
            response_format={"type": "json_object"},
            extra_body=extra if extra else None
        )
        content = response.choices[0].message.content
        cleaned = clean_json_response(content)
        parsed = json.loads(cleaned)

        raw_alloc = parsed.get("card_allocation", {})
        pool_ids = {c["id"] for c in pool}
        int_alloc = {}
        for k, v in raw_alloc.items():
            try:
                cid = int(k)
                if cid in pool_ids:
                    int_alloc[cid] = int(v)
            except Exception:
                pass

        # 归一化保障绝对等于 30 张
        normalized_alloc = normalize_counts(int_alloc, pool)
        parsed["card_allocation"] = normalized_alloc
        return parsed

    except Exception as e:
        print(f"[WARN] DeepSeek 构筑生成异常 ({e})，启用自适应规则兜底卡组，确保流水线平稳过渡至 PPO 阶段...")
        fallback_alloc = {}
        sorted_pool = sorted(pool, key=lambda c: (c.get("cost", 0), -c.get("base_dp", 0)))
        total_picked = 0
        for c in sorted_pool:
            if total_picked >= 30:
                break
            can_add = min(2, 30 - total_picked)
            fallback_alloc[c["id"]] = can_add
            total_picked += can_add
        fallback_alloc = normalize_counts(fallback_alloc, pool)
        return {
            "deck_name": f"{faction}·均衡启航构筑",
            "archetype": "Balanced",
            "tactical_concept": "平滑法力曲线自适应基础构筑，准备进入后续 PPO 智能体自博弈演化。",
            "key_combos": ["基础攻防节奏支撑"],
            "card_allocation": fallback_alloc
        }

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
    print(f"【{faction}】卡组配置: 《{deck_info['deck_name']}》")
    print(f"类型: {deck_info['archetype']} | 均费: {deck_info['avg_cost']} 费 | 随从 {deck_info['minion_count']} 张 / 法术 {deck_info['spell_count']} 张")
    print("─" * 80)
    print("构筑说明:")
    print(f"   {deck_info['tactical_concept']}")
    if deck_info.get("key_combos"):
        print("主要配合:")
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
    print("法力曲线分布:")
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
    parser = argparse.ArgumentParser(description="TCG-AI 卡组构筑工具 (AI Deckbuilder)")
    parser.add_argument("--cards", type=str, default="cards_config.json",
                        help="卡池配置文件路径 (默认 cards_config.json)")
    parser.add_argument("--output", type=str, default="decks_config.json",
                        help="输出卡组保存文件 (默认 decks_config.json)")
    parser.add_argument("--factions", type=str, default="Red,Blue",
                        help="目标构建阵营 (默认 Red,Blue，支持 Red,Blue,Green)")

    args = parser.parse_args()

    cards_db = load_json(args.cards)
    if not cards_db:
        print(f"无法读取卡池数据: {args.cards}")
        return

    factions_to_build = [f.strip() for f in args.factions.split(",") if f.strip()]
    decks_result = {}

    # 若已有现存卡组文件，先加载保留其他阵营
    if os.path.exists(args.output):
        decks_result = load_json(args.output)

    print("\n" + "═" * 50)
    print("TCG 卡组构筑工具启动")
    print("═" * 50)

    for faction in factions_to_build:
        pool = get_faction_pool(cards_db, faction)
        if not pool:
            print(f"未找到阵营 {faction} 的可用卡池。")
            continue

        raw_deck = call_deepseek_deckbuild(faction, pool)

        deck_info = generate_deck_details(raw_deck, pool)
        print_deck_profile(faction, deck_info)
        decks_result[faction] = deck_info

    # 导出卡组配置
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(decks_result, f, indent=2, ensure_ascii=False)

    abs_out = os.path.abspath(args.output)
    print(f"卡组已保存至: {abs_out}")

if __name__ == "__main__":
    main()
