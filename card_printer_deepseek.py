import os
import re
import sys
import json
import time
import argparse
from typing import Dict, List
from collections import Counter
import torch
import numpy as np
from openai import OpenAI
from sandbox import DuelEnv, Faction
from agent import CardNet

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
    if not isinstance(tag, str):
        return False
    return any(re.match(pattern, tag.strip().upper()) for pattern in LEGAL_TAG_PATTERNS)

def extract_valid_tags(c: dict) -> List[str]:
    raw_tags = c.get("tags")
    if raw_tags is None:
        raw_tags = c.get("keywords") or c.get("tag") or []
    if isinstance(raw_tags, str):
        raw_tags = [raw_tags]
    elif not isinstance(raw_tags, list):
        raw_tags = []

    kw_values = c.get("keyword_values", {})
    valid_tags = []
    default_sub_map = {
        "FORTIFY_X": "FORTIFY_2",
        "DEGRADE_X": "DEGRADE_1",
        "SUPPORT_ATK_X": "SUPPORT_ATK_1",
        "BONUS_SCORE_X": "BONUS_SCORE_1",
        "SPAWN_X_Y": "SPAWN_1_1",
        "DEATH_DRAW_X": "DEATH_DRAW_1",
        "DEATH_MANA_X": "DEATH_MANA_1",
        "DRAW_X": "DRAW_1",
        "RAMP_X": "RAMP_1",
        "TEMP_MANA_X": "TEMP_MANA_1",
        "DISCARD_X": "DISCARD_1",
    }

    for t in raw_tags:
        if not isinstance(t, str):
            continue
        t_clean = t.strip().upper()
        if is_legal_tag(t_clean):
            if t_clean not in valid_tags:
                valid_tags.append(t_clean)
            continue
        if "_X" in t_clean or "_Y" in t_clean:
            val = None
            if isinstance(kw_values, dict):
                for k, v in kw_values.items():
                    if k.strip().upper() in t_clean or t_clean.startswith(k.strip().upper()):
                        val = v
                        break
            if val is not None:
                sub_tag = re.sub(r"_[XY]", f"_{val}", t_clean)
                if is_legal_tag(sub_tag) and sub_tag not in valid_tags:
                    valid_tags.append(sub_tag)
                    continue
            if t_clean in default_sub_map and is_legal_tag(default_sub_map[t_clean]):
                cand = default_sub_map[t_clean]
                if cand not in valid_tags:
                    valid_tags.append(cand)

    return valid_tags

def load_json(filepath: str) -> dict:
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_json_response(raw_text: str) -> str:
    """提取 Markdown 代码块中的纯 JSON 内容"""
    if not raw_text:
        return "{}"
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_text.strip()

SYSTEM_RESERVED_IDS = {996, 997, 998, 999}

def get_next_id(existing_cards: List[dict], default_start: int) -> int:
    """自动计算不冲突的安全卡牌 ID (严格规避系统衍生牌 990~999 保留区)"""
    if not existing_cards:
        return default_start
    curr_id = max(c.get("id", default_start) for c in existing_cards) + 1
    while curr_id in SYSTEM_RESERVED_IDS or (990 <= curr_id <= 999):
        curr_id += 1
    return curr_id

def build_prompt(current_pool: dict, faction: str, count: int, theme: str) -> str:
    faction_desc = {
        "Red": "红方：快攻冲锋、突袭（RUSH）、召唤衍生单位（SPAWN）、破甲削弱（DEGRADE）",
        "Blue": "蓝方：高额防御（FORTIFY）、光环增益（SUPPORT_ATK）、控制护盾",
        "Green": "绿方：跳费成长（RAMP/DEATH_MANA）、高费随从（FORTIFY）",
        "Neutral": "中立：过牌（DRAW）、通用防御与辅助"
    }.get(faction, f"{faction} 阵营")

    return f"""
你是一名 TCG 卡牌设计与数值平衡工程师。
请为双路集换式卡牌对战环境（DuelEnv）设计新卡。

### 1. 本次设计需求：
- 目标阵营: 【{faction}】 ({faction_desc})
- 数量: 【{count}】张
- 设计主题: 【{theme if theme else '符合阵营特色，具备一定战术搭配价值的机制卡'}】

### 2. 核心机制与规则边界：
1. **双路对撞**：左右两路独立攻防，随从打入进攻区默认需蓄势一回合（除非拥有 `RUSH` 突袭词条）。
2. **构筑上限**：单卡上限 3 张，牌库 30 张。
3. **支持的属性字段**：
   - `id`: 卡牌整数标识
   - `name`: 中文名称
   - `card_type`: "MINION" 或 "SPELL"
   - `cost`: 施法消耗 (0~10)
   - `base_dp`: 随从基础战力/阻挡阈值 (SPELL 必须为 0)
   - `atk_spell_val`: 直伤削弱数值 (无则为 0)
   - `def_spell_val`: 增益护盾数值 (无则为 0)
   - `tags`: 词条列表 (仅限已支持词条)
4. **支持的词条**：
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
5. **严禁恶性完爆（Anti-Strict-Outclassing Rule，绝对红线）**：
   - 严禁出现同费用、同阵营下的绝对完爆关系！
   - 绝不允许新卡在费用相同的情况下，身材、词条以及机制全方位碾压现有同阵营卡牌（例如同样 3 费，一张 3 DP 生 1/1 能攻能守，另一张 1 DP 生 1/1 且仅能进攻）；
   - 同费卡牌之间必须有明确的属性权衡或功能差异（如高攻脆皮 vs 稳健肉盾、即时冲锋 vs 亡语后劲、单点突破 vs 横向多体），确保每张卡具有独特的战术生态位。
6. **严禁极度亏模废牌（Anti-Deficit Rule，全费用通用绝对红线）**：
   - 规则不限于高费，严禁在任何费用区间（无论是 1~4 费低中费，还是 5~8+ 费高费）设计出「身材/数值极度亏模且缺乏强力词条机制补偿」的垃圾废牌！
   - 随从身材模型基准：
     a. 纯白板随从（无任何词条）：必须严格遵守超模身材补偿基准线，底线为 DP >= 费用 + 1（例如 1费白板 >= 2 DP，2费白板 >= 3 DP，3费白板 >= 4 DP，4费白板 >= 5 DP，5费白板 >= 6 DP，6费白板 >= 7~8 DP，7费白板 >= 8~9 DP，8费白板 >= 9~10 DP）。绝对严禁设计出 1费1DP、2费2DP、3费3DP、4费4DP 甚至 7~8费5DP 这类没有任何词条还严重亏模的废卡！
     b. 亏模随从（DP <= 费用）：必须携带足够强力或关键的战术词条（如 RUSH、FORTIFY、DEGRADE、BONUS_SCORE、DEATH_DRAW、SPAWN 等）作为亏模补偿。若 DP < 费用 - 1，其机制价值必须极其扎实，严禁低数值白板！
   - 法术数值模型基准：
     纯数值法术的总点数必须合理匹配费用（基础标准通常为 点数 >= 费用 * 1.5 到 2，如 1费直伤 2、2费直伤 3~4、3费护盾 5~6）。严禁出现 3 费仅提供 3 点护盾且无任何词条/跳费效果的极度亏模法术！

### 3. 当前参考卡池现状:
{json.dumps(current_pool, indent=2, ensure_ascii=False)}

---
### 输出格式：
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
      "flavor_and_strategy": "设计说明"
    }}
  ]
}}
不要包含额外解释或 Markdown 标记。
"""

def print_card_table(cards_dict: Dict[str, List[dict]], design_notes: List[dict]):
    """在终端打印新生成卡牌信息"""
    notes_map = {n.get("card_name"): n.get("flavor_and_strategy", "") for n in design_notes}
    print("\n" + "═" * 95)
    print("【TCG-AI】新生成卡牌一览")
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

def find_model_path() -> str:
    candidates = [
        "card_ppo_model_tuned.pth",
        "PythonApplication23/card_ppo_model_tuned.pth",
        "card_ppo_model.pth",
        "PythonApplication23/card_ppo_model.pth",
        "card_ppo_model_baseline.pth",
        "PythonApplication23/card_ppo_model_baseline.pth"
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def run_post_print_benchmark(cards_path: str, all_printed_cards: Dict[str, List[dict]],
                             episodes: int = 1000, auto_balance: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_file = find_model_path()

    print("\n" + "═" * 85)
    print("启动印后数值实测 (1000 局)")
    print(f"实测卡池: {cards_path} | 对战规模: {episodes} 局 | 设备: {device}")
    if model_file:
        print(f"模型权重: {model_file}")
    print("═" * 85)

    new_card_info = {}
    for f_name, cards in all_printed_cards.items():
        for c in cards:
            cid = c["id"]
            new_card_info[cid] = {
                "name": c["name"],
                "faction": f_name,
                "cost": c["cost"],
                "type": c["card_type"],
                "dp": c.get("base_dp", 0)
            }

    has_green = "Green" in all_printed_cards and len(all_printed_cards["Green"]) > 0
    if has_green:
        matchups = [
            (Faction.RED, Faction.BLUE, episodes // 2),
            (Faction.GREEN, Faction.BLUE, episodes // 2)
        ]
    else:
        matchups = [(Faction.RED, Faction.BLUE, episodes)]

    card_played_total = Counter()
    card_played_win = Counter()
    total_turns = []
    red_wins = 0
    blue_wins = 0
    green_wins = 0
    total_played_games = 0

    start_time = time.time()

    for p0_f, p1_f, sub_eps in matchups:
        env = DuelEnv(p0_faction=p0_f, p1_faction=p1_f, cards_path=cards_path)
        model = CardNet(action_dim=env.action_space_size).to(device)
        if model_file and os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
        model.eval()

        for ep in range(1, sub_eps + 1):
            total_played_games += 1
            obs = env.reset()
            done = False
            p0_played = set()
            p1_played = set()

            while not done:
                curr_p = env.current_player
                mask = env.get_action_mask()
                state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                mask_t = torch.FloatTensor(mask).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits, _ = model(state_t, mask_t)
                    act = torch.argmax(logits, dim=-1).item()

                if act != env.action_space_size - 1:
                    h_idx = act // 4
                    hand = env.players[curr_p].hand
                    if h_idx < len(hand):
                        card = hand[h_idx]
                        card_played_total[card.name] += 1
                        if curr_p == 0:
                            p0_played.add(card.name)
                        else:
                            p1_played.add(card.name)

                obs, _, done, _ = env.step(act)

            total_turns.append(env.turn_count)
            s0 = env.players[0].score
            s1 = env.players[1].score

            if s0 > s1 and s0 >= env.WIN_SCORE:
                if p0_f == Faction.RED: red_wins += 1
                elif p0_f == Faction.GREEN: green_wins += 1
                for cname in p0_played: card_played_win[cname] += 1
            elif s1 > s0 and s1 >= env.WIN_SCORE:
                if p1_f == Faction.BLUE: blue_wins += 1
                for cname in p1_played: card_played_win[cname] += 1

            if total_played_games % 250 == 0 or total_played_games == episodes:
                print(f"   ⏳ 评测进度: {total_played_games:4d}/{episodes} 局...")

    elapsed = time.time() - start_time
    avg_turns = float(np.mean(total_turns)) if total_turns else 0.0

    p0_rate = (red_wins / max(1, total_played_games)) * 100
    p1_rate = (blue_wins / max(1, total_played_games)) * 100

    print("\n" + "═" * 85)
    print("【印后数值平衡实测结果】")
    print("═" * 85)
    print(f"⏱️ 测试总耗时: {elapsed:.2f} 秒 ({total_played_games/max(0.01, elapsed):.1f} 局/秒)")
    print(f"对局统计: 红方胜 {red_wins} 场 ({p0_rate:.1f}%) | 蓝方胜 {blue_wins} 场 ({p1_rate:.1f}%)" + (f" | 🟢 绿方胜 {green_wins} 场" if has_green else ""))
    print(f"⌛ 平均对局回合: {avg_turns:.1f} 轮")
    print("─" * 85)
    print("新卡实战表现与胜率统计:")
    print(f"{'ID':<6}{'阵营':<8}{'名称':<14}{'费用':<6}{'出牌频次':<20}{'实战胜率':<12}{'状态评定'}")
    print("─" * 85)

    has_imbalance = False
    for cid, info in sorted(new_card_info.items(), key=lambda x: x[0]):
        cname = info["name"]
        played = card_played_total.get(cname, 0)
        wins = card_played_win.get(cname, 0)
        win_pct = (wins / played * 100) if played > 0 else 0.0

        if played == 0:
            health = "未出场"
        elif win_pct >= 65.0 and played >= 10:
            health = "偏强 (胜率>65%)"
            has_imbalance = True
        elif win_pct <= 35.0 and played >= 10:
            health = "偏弱 (胜率<35%)"
        else:
            health = "正常"

        played_str = f"{played} 次 (局均{played/max(1, total_played_games):.2f})"
        win_str = f"{win_pct:.1f}%" if played > 0 else "-"
        print(f"{cid:<6}{info['faction']:<8}{cname:<14}{info['cost']:<6}{played_str:<20}{win_str:<12}{health}")

    print("─" * 85)
    if not has_imbalance:
        print("总体评估: 新卡表现平稳，环境处于正常区间。")
    else:
        print("总体评估: 部分新卡实测胜率偏高，建议后续微调。")
        print("提示: 可运行 python auto_balancer_deepseek.py 进行数值微调。")

    print("═" * 85)

    metrics_export = {
        "total_episodes": total_played_games,
        "p0_wins": red_wins,
        "p1_wins": blue_wins,
        "p0_winrate": round((red_wins / max(1, total_played_games)) * 100, 2),
        "p1_winrate": round((blue_wins / max(1, total_played_games)) * 100, 2),
        "avg_steps_per_episode": round(avg_turns, 2),
        "card_play_count": dict(card_played_total)
    }
    with open("training_metrics_post_print.json", "w", encoding="utf-8") as mf:
        json.dump(metrics_export, mf, indent=2, ensure_ascii=False)
    print("对战数据已保存至: training_metrics_post_print.json\n")


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
    parser.add_argument("--test-episodes", type=int, default=1000,
                        help="印卡后自动实机运行对局评估轮数 (默认 1000 局，设为 0 跳过)")
    parser.add_argument("--auto-balance", action="store_true",
                        help="若检测到新卡导致严重失衡，提示或启动闭环调优")
    args = parser.parse_args()

    if not DEEPSEEK_API_KEY:
        print("未检测到 DEEPSEEK_API_KEY 环境变量，请先配置：")
        print("  Windows PowerShell: $env:DEEPSEEK_API_KEY=\"你的API_KEY\"")
        print("  Linux / macOS:     export DEEPSEEK_API_KEY=\"你的API_KEY\"")
        return

    base_path = args.base if os.path.exists(args.base) else "cards_config.json"
    print(f"读取基准卡池: {base_path}")
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
            ("Red", args.count, "红方：快攻冲锋（RUSH）、破甲削弱（DEGRADE）与牺牲直伤"),
            ("Blue", args.count, "蓝方：阵地驻防（FORTIFY）、光环增益（SUPPORT_ATK）与护盾控制"),
            ("Green", args.count, "绿方：跳费成长（RAMP）、高费随从与召唤（SPAWN）"),
            ("Neutral", max(2, args.count - 2), "中立：过牌（DRAW）与通用辅助")
        ]
    else:
        theme = args.theme if args.theme else f"{args.faction} 阵营机制扩充"
        factions_to_process = [(args.faction, args.count, theme)]

    expanded_pool = json.loads(json.dumps(current_pool))
    all_printed_cards = {}
    all_design_notes = []

    for f_name, f_count, f_theme in factions_to_process:
        print("\n" + "─" * 70)
        print(f"正在调用 [{MODEL_NAME}] 生成阵营卡牌: {f_name} ({f_count} 张)...")
        print(f"设计主题: {f_theme}")

        prompt = build_prompt(expanded_pool, f_name, f_count, f_theme)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a professional TCG game designer. Output ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}}
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
                c["tags"] = extract_valid_tags(c)
                c["cost"] = max(0, int(c.get("cost", 1)))
                c["base_dp"] = max(0, int(c.get("base_dp", 0)))
                c["atk_spell_val"] = max(0, int(c.get("atk_spell_val", 0)))
                c["def_spell_val"] = max(0, int(c.get("def_spell_val", 0)))

                expanded_pool[f_name].append(c)
                all_printed_cards[f_name].append(c)

            print(f"{f_name} 阵营成功生成 {len(cards_list)} 张新卡。")

        except json.JSONDecodeError as e:
            print(f"解析大模型返回 JSON 异常: {e}")
            print("原始返回内容：\n", raw_output)

    # 打印全局卡牌表
    print_card_table(all_printed_cards, all_design_notes)

    # 写入文件
    output_file = args.output
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(expanded_pool, f, indent=2, ensure_ascii=False)
    print(f"扩充后卡池已保存至: {output_file}")

    # 统计卡池总数
    print("\n扩充后卡池总规模:")
    total_cards = sum(len(cards) for cards in expanded_pool.values())
    print(f"   总卡牌数: {total_cards} 张")
    for f, c_list in expanded_pool.items():
        print(f"   - {f}: {len(c_list)} 张 (新增 {len(all_printed_cards.get(f, []))} 张)")

    # 若指定了合并写入
    if args.merge_into:
        with open(args.merge_into, "w", encoding="utf-8") as f:
            json.dump(expanded_pool, f, indent=2, ensure_ascii=False)
        print(f"已同步更新至卡池: {args.merge_into}")

    # 印完卡后立即先跑 1000 局对抗压力测试，严密监控平衡性
    if args.test_episodes > 0 and any(len(cards) > 0 for cards in all_printed_cards.values()):
        target_test_pool = args.merge_into if args.merge_into else output_file
        run_post_print_benchmark(
            cards_path=target_test_pool,
            all_printed_cards=all_printed_cards,
            episodes=args.test_episodes,
            auto_balance=args.auto_balance
        )

if __name__ == "__main__":
    main()
