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
    return any(re.match(p, tag) for p in LEGAL_TAG_PATTERNS)

def run_deepseek_balance_and_expand(metrics_data: dict, cards_data: dict, client: OpenAI, history_metrics: dict = None) -> str:
    total = max(1, metrics_data.get("total_episodes", 1000))
    
    max_dev = 0.0
    if "faction_stats" in metrics_data:
        stats_lines = []
        deviations = {}
        for f in ["Red", "Blue", "Green"]:
            s = metrics_data["faction_stats"].get(f, {"winrate": 50.0, "wins": 0, "matches": 0})
            wr = s.get("winrate", 50.0)
            dev = abs(wr - 50.0)
            deviations[f] = dev
            stats_lines.append(f"- 【{f}】阵营: 胜率 {wr:.1f}% ({s.get('wins', 0)} 胜 / {s.get('matches', 0)} 场) | 偏离 50% 黄金基准: {dev:.1f}%")
        data_section = "\n".join(stats_lines)
        if "matchups" in metrics_data:
            data_section += f"\n- 两两交手战报: {json.dumps(metrics_data['matchups'], ensure_ascii=False)}"
        max_dev = max(deviations.values()) if deviations else 0.0
    else:
        p0_wins = metrics_data.get("p0_wins", 500)
        p1_wins = metrics_data.get("p1_wins", 500)
        p0_rate = (p0_wins / total) * 100
        p1_rate = (p1_wins / total) * 100
        data_section = f"- 红方胜率 (P0 - 快攻冲锋): {p0_rate:.1f}% ({p0_wins} 胜)\n- 蓝方胜率 (P1 - 控制防守): {p1_rate:.1f}% ({p1_wins} 胜)"
        max_dev = abs(p0_rate - 50.0)

    history_section = ""
    if history_metrics and history_metrics != metrics_data:
        history_section = f"""
### 历史对局参考：
{json.dumps(history_metrics.get("faction_stats", history_metrics), ensure_ascii=False, indent=2)}
"""

    # 动态分析各阵营真实强弱格局
    overperforming = []
    underperforming = []
    balanced = []
    
    if "faction_stats" in metrics_data:
        for f, s in metrics_data["faction_stats"].items():
            wr = s.get("winrate", 50.0)
            if wr > 52.8:
                overperforming.append((f, wr, wr - 50.0))
            elif wr < 47.2:
                underperforming.append((f, wr, 50.0 - wr))
            else:
                balanced.append((f, wr))

    over_instructions = []
    for f, wr, diff in overperforming:
        over_instructions.append(f"""
    * 【削弱过强阵营 —— {f} (当前胜率 {wr:.1f}%，高出基准 +{diff:.1f}%)】：
      - 阶梯式平衡逻辑（避免死砍身材）：
        1. 常规调整：优先通过适度提高费用 (+1 费) 或小幅下调身材 (-1~2 DP) 进行平滑抑制；
        2. 转向词条平衡：若发现削弱身材收效甚微（例如带有 RUSH 突袭或 BONUS_SCORE 破阵得分的终结怪，即便面板被砍依然能凭借机制强行偷鸡斩杀），或者身材已达到该费用的合理底线（6费 >= 4 DP, 7~8费 >= 5 DP, 9费 >= 6 DP），说明问题根源在于机制而非白值，此时应果断转向削弱或剥离过于强势的词条（如剥离 BONUS_SCORE_1、将 RUSH 突袭改为蓄势、将双抽/双跳下调为单抽/单跳）；
        3. 坚决杜绝畸形面板：绝对禁止在保留霸道词条的同时，盲目把高费随从一路砍到 1~2 DP！""")

    under_instructions = []
    for f, wr, diff in underperforming:
        under_instructions.append(f"""
   * 【补强落后阵营 —— {f} (当前胜率 {wr:.1f}%，落后基准 -{diff:.1f}%)】：
     - 若关键牌先前被过度调整：适度回调费用 (-1 费) 或增强身材 (+1~+2 DP)；
     - 前期直伤与解场手段：提高低费直伤 atk_spell_val (+1~+2 点) 或降低单解消耗 (-1 费)；
     - 防守与阻挡能力：提升低费防守怪 base_dp 或赋予 FORTIFY_2/FORTIFY_3 坚守词条；
     - 中后期制胜核心：适度降低费用 (-1 费) 提升出场率。""")

    over_text = "\n".join(over_instructions) if over_instructions else "   * 暂无超标阵营（各阵营均在安全线以下）。"
    under_text = "\n".join(under_instructions) if under_instructions else "   * 暂无垫底阵营（各阵营均在安全线以上）。"

    if max_dev >= 8.0:
        mode_banner = f"【大幅数值调整模式 (Major Overhaul Mode) | 最大偏离度: {max_dev:.1f}%】"
        balance_instructions = f"""
### 【大幅数值调整说明 (Major Overhaul)】
当前阵营间胜率差距较为显著（最大偏离度 {max_dev:.1f}%）。
需针对关键卡牌进行多维度属性调整：

1. **调整规模**：
   - 对 4 ~ 8 张核心卡牌进行针对性属性调整；
   - 过强阵营削弱 2~4 张，弱势阵营补强 2~4 张；
   - 必须输出明确的修改。

2. **阵营针对性调整**：
{over_text}
{under_text}

3. **词条微调**：
   - 可对过强卡牌精简过于强势的词条（如取消 RUSH 或减少数值）；
   - 可为弱势卡牌补充急需词条（如赋予 FORTIFY、DEGRADE 等合法词条）。
"""
    else:
        mode_banner = f"【精细微调模式 (Fine-Tuning Mode) | 最大偏离度: {max_dev:.1f}%】"
        balance_instructions = f"""
### 【局部精细微调说明 (Fine-Tuning)】
当前各阵营胜率已接近平衡区间（最大偏离度 {max_dev:.1f}% < 8%）。
需基于对战统计进行小幅修剪：
1. **调整规模**：挑选 2 ~ 4 张关键卡牌进行微调。
2. **微调原则**：
{over_text}
{under_text}
   - 费用调整 ±1，随从身材 ±1~2 点，法术数值 ±1~2 点；
   - 保持卡牌体系稳定性，平滑引导各阵营胜率趋近 50%。
"""

    prompt = f"""
你是一名 TCG 卡牌总监兼数值平衡科学家。
以下是通过强化学习（PPO 自博弈）收集的实战遥测对战数据和当前卡池配置文件。

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
5. **已支持的核心合法词条系统**：
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
6. **严禁恶性完爆（Anti-Strict-Outclassing Rule，绝对红线）**：
   - 严禁调出同费用、同阵营下的绝对完爆关系！
   - 严禁出现一张卡在同费用下，身材、词条以及机制全方位被另一张卡绝对碾压（如同样 3 费，一张 3 DP 生 1/1 能攻能守，另一张 1 DP 生 1/1 且仅能进攻）；
   - 若发现同阵营卡牌功能重叠或存在完爆隐患，必须通过调整费用（如将低阶卡降费以拉开费用阶梯）、重塑差异化词条或调整战术定位来化解，确保每张卡具有不可替代的构筑价值。

### 3. 待调整卡池配置:
{json.dumps(cards_data, indent=2, ensure_ascii=False)}

---
### 调优工作指示:
{mode_banner}
{balance_instructions}

### 输出格式与极致精简要求（非常重要）：
1. **只输出实际被修改调整的卡牌列表**（严禁将未修改的几十张卡牌原样复制输出，必须精炼快速）！
2. 必须输出合法纯 JSON 对象，格式严格如下：
{{
  "modifications": [
    {{
      "id": 103,
      "cost": 2,
      "base_dp": 0,
      "atk_spell_val": 3,
      "def_spell_val": 0,
      "tags": ["ATTACK_ONLY"],
      "reason": "简述调整理由"
    }}
  ]
}}
3. 严禁改动卡牌的 id 与 card_type；仅针对性调整选中的 2~8 张卡牌的 cost, base_dp, atk_spell_val, def_spell_val, tags。
4. 严禁输出任何解释性废话。
"""

    print(f"\n调用 [{MODEL_NAME}] 读取训练战报，执行卡池数值调优...")
    print(f"模式: {mode_banner}")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a professional TCG balance engineer. Output ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=8192,
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}}
    )
    return response.choices[0].message.content

def extract_card_list(data) -> list:
    """提取任何嵌套格式下的卡牌对象列表"""
    cards = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "id" in item:
                cards.append(item)
    elif isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict) and "id" in item:
                        cards.append(item)
            elif isinstance(v, dict) and "id" in v:
                cards.append(v)
    return cards

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

    new_card_pool = None
    try:
        new_card_pool = json.loads(cleaned_json)
    except json.JSONDecodeError as e:
        print(f"[*] 遇到 JSON 局部格式异常 ({e})，启动自适应单卡特征正则修复提取...")
        # 正则兜底提取：即使外层缺少括号或逗号，只要单卡对象结构存在即可挽救
        cand_matches = re.findall(r'\{[^{}]*"id"\s*:\s*\d+[^}]*\}', raw_output)
        if cand_matches:
            recovered_cards = []
            for m in cand_matches:
                try:
                    c_obj = json.loads(m)
                    if "id" in c_obj:
                        recovered_cards.append(c_obj)
                except Exception:
                    pass
            if recovered_cards:
                print(f"[*] 正则修复成功提取到 {len(recovered_cards)} 张微调卡牌数据！")
                new_card_pool = {"modifications": recovered_cards}

    if not new_card_pool:
        preview = raw_output[:200].replace('\n', ' ') + "..." if len(raw_output) > 200 else raw_output
        print(f"[警告] 本轮未能解析出有效的数值微调指令 (片段: {preview})，保持现有卡池进入下阶段。")
        return

    # 稳健合并逻辑：以原卡池为基准进行靶向覆盖，确保卡牌 ID、阵营与名称永不遗失
    updated_pool = {}
    for f, card_list in current_cards.items():
        updated_pool[f] = [dict(c) for c in card_list]

    # 建立 id -> (faction, index) 索引
    id_index = {}
    for f, card_list in updated_pool.items():
        for idx, c in enumerate(card_list):
            id_index[c["id"]] = (f, idx)

    # 进行靶向更新与合规校验
    updated_count = 0
    cand_cards = extract_card_list(new_card_pool)
    print(f"[*] 成功提取 {len(cand_cards)} 张调整卡牌，正在校验合规性...")
    for new_c in cand_cards:
        cid = new_c.get("id")
        if cid in id_index:
            orig_f, orig_idx = id_index[cid]
            orig_card = updated_pool[orig_f][orig_idx]
            # 仅更新允许修改的数值与词条字段
            for field in ["cost", "base_dp", "atk_spell_val", "def_spell_val", "tags"]:
                if field in new_c:
                    new_val = new_c[field]
                    if field == "cost":
                        try:
                            new_val = min(10, max(1, int(new_val)))
                        except Exception:
                            continue
                    elif field == "base_dp":
                        try:
                            val = int(new_val)
                            cost = orig_card.get("cost", 1)
                            if orig_card.get("card_type") == "MINION":
                                # 防御机制：高费随从身材保底，防止大模型将高费怪削弱为畸形 1/1
                                if cost >= 8:
                                    val = max(5, val)
                                elif cost >= 6:
                                    val = max(3, val)
                                elif cost >= 4:
                                    val = max(2, val)
                                else:
                                    val = max(1, val)
                            new_val = min(15, val)
                        except Exception:
                            continue
                    elif field in ["atk_spell_val", "def_spell_val"]:
                        try:
                            new_val = max(0, int(new_val))
                        except Exception:
                            continue
                    elif field == "tags":
                        if isinstance(new_val, list):
                            new_val = [t for t in new_val if isinstance(t, str) and is_legal_tag(t)]
                        else:
                            continue

                    if new_val != orig_card.get(field):
                        print(f"  [数值调整] ID {cid} {orig_card['name']}: {field} 从 {orig_card.get(field)} -> {new_val}")
                        orig_card[field] = new_val
                        updated_count += 1

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(updated_pool, f, indent=2, ensure_ascii=False)
        
    print(f"已完成卡池数值调优 (共更新 {updated_count} 处属性/词条)，已保存至: {output_file}")

if __name__ == "__main__":
    main()
