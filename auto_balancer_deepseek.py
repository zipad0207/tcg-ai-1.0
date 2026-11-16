import os
import re
import sys
import json
import copy
import argparse
from openai import OpenAI

from tcg_common import (
    ROOT_DIR as SCRIPT_DIR,
    LEGAL_TAG_PATTERNS,
    OUTCLASSING_RULE_BODY,
    DEFICIT_RULE_BODY,
    atomic_write_json,
    clean_json_response,
    clean_secret as _clean_secret,
    compute_pairwise_matchups,
    enforce_temp_mana_cost,
    get_api_key,
    get_base_url,
    get_model_name,
    is_legal_tag,
    load_json,
    load_llm_config as _load_llm_config,
    resolve_path,
    to_float as _to_float,
    to_int as _to_int,
    to_int_default as _to_int_default,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_NAME = get_model_name()
DEEPSEEK_API_KEY = get_api_key()


def _compute_balance_metrics(metrics_data: dict, target_pairwise_tolerance: float) -> dict:
    summary = compute_pairwise_matchups(metrics_data)
    summary["effective_max_dev"] = (
        max(summary["max_dev"], summary["max_pairwise_dev"])
        if target_pairwise_tolerance > 0
        else summary["max_dev"]
    )
    return summary


def _strip_cards_for_prompt(cards_data: dict, card_in_decks: dict = None) -> dict:
    keep_fields = ("id", "name", "card_type", "cost", "base_dp", "atk_spell_val", "def_spell_val", "tags", "factions")
    stripped = {}
    for k, v in cards_data.items():
        if isinstance(v, list):
            items = []
            for c in v:
                if not isinstance(c, dict):
                    continue
                entry = {fld: c[fld] for fld in keep_fields if fld in c}
                if card_in_decks is not None:
                    cid = _to_int(c.get("id"))
                    if cid is not None and cid in card_in_decks:
                        entry["in_decks"] = card_in_decks[cid]
                items.append(entry)
            stripped[k] = items
        else:
            stripped[k] = v
    return stripped


def _build_active_deck_context(decks_data: dict, cards_data: dict) -> tuple:
    """Summarize each faction's active 30-card roster and map card_id -> {faction: count}."""
    if not isinstance(decks_data, dict) or not decks_data:
        return {}, ""

    by_id = {}
    for _, clist in cards_data.items():
        if isinstance(clist, list):
            for c in clist:
                if isinstance(c, dict) and _to_int(c.get("id")) is not None:
                    by_id[_to_int(c.get("id"))] = c

    card_in_decks = {}
    roster_lines = []
    for f in ("Red", "Blue", "Green"):
        pkg = decks_data.get(f)
        if not isinstance(pkg, dict):
            continue
        decklist = pkg.get("decklist") or []
        if not decklist:
            continue
        counts = {}
        for raw_id in decklist:
            cid = _to_int(raw_id)
            if cid is None:
                continue
            counts[cid] = counts.get(cid, 0) + 1
            card_in_decks.setdefault(cid, {})[f] = counts[cid]

        parts = []
        for cid, cnt in sorted(counts.items(), key=lambda x: (-x[1], by_id.get(x[0], {}).get("cost", 0), x[0])):
            c = by_id.get(cid)
            if not c:
                parts.append(f"ID{cid}x{cnt}")
                continue
            stat = f"{c.get('base_dp', 0)}DP" if c.get("card_type") == "MINION" else f"攻{c.get('atk_spell_val', 0)}/防{c.get('def_spell_val', 0)}"
            parts.append(f"ID{cid} {c.get('name', '')}x{cnt}({c.get('cost', 0)}费/{stat})")
        avg_c = pkg.get("avg_cost", "?")
        roster_lines.append(f"  - 【{f}】现役 30 张 (均费 {avg_c}): " + ", ".join(parts))

    if not roster_lines:
        return card_in_decks, ""

    section = (
        "\n### 各阵营当前 30 张现役卡组携带实况 (已携带卡牌在卡池含 in_decks 字段):\n"
        + "\n".join(roster_lines)
        + "\n  * 重要调优准则: 削弱强势阵营时必须优先从其现役携带 (>=1张) 的卡中挑选；加强弱势阵营时若调整 0 携带冷门卡，必须给予显著加强（如 -1费 或 +2数值/关键强力词条）以确保 Phase 3 构筑器能将其选入现役卡组。\n"
    )
    return card_in_decks, section


def _build_card_role_maps(cards_data: dict) -> tuple:
    excluded_names = {"幸运币", "法力过载"}
    dual_names = set()
    for category, card_list in cards_data.items():
        if not isinstance(card_list, list):
            continue
        for c in card_list:
            if not isinstance(c, dict):
                continue
            name = c.get("name", "")
            f_list = c.get("factions") or []
            if c.get("is_token") or category == "Neutral" or f_list == ["Neutral"]:
                if name:
                    excluded_names.add(name)
            elif category == "Dual" or len(f_list) > 1:
                if name:
                    dual_names.add(name)
    return excluded_names, dual_names

def run_deepseek_balance_and_expand(metrics_data: dict, cards_data: dict, client: OpenAI, history_metrics: dict = None, target_tolerance: float = 5.0, target_pairwise_tolerance: float = 8.0, target_pairwise_balance: float = None, severe_threshold: float = 10.0, tabu_list: dict = None, weak_factions: list = None, decks_data: dict = None) -> str:
    if target_pairwise_balance is not None:
        target_pairwise_tolerance = target_pairwise_balance

    summary = _compute_balance_metrics(metrics_data, target_pairwise_tolerance)
    total = summary["total"]
    factions = summary["factions"]
    deviations = summary["deviations"]
    pairwise_stats = summary["pairwise_stats"]
    max_dev = summary["max_dev"]
    max_pairwise_dev = summary["max_pairwise_dev"]
    effective_max_dev = summary["effective_max_dev"]

    data_section = ""
    pairwise_imbalanced = []

    f_stats_raw = metrics_data.get("faction_stats")
    if isinstance(f_stats_raw, dict):
        stats_lines = []
        for f in factions:
            s = f_stats_raw.get(f) or {}
            wr = _to_float(s.get("winrate"), 50.0)
            wins_cnt = _to_int_default(s.get("wins"), 0)
            matches_cnt = _to_int_default(s.get("matches"), 0)
            dev = deviations.get(f, abs(wr - 50.0))
            stats_lines.append(f"- 【{f}】阵营: 胜率 {wr:.1f}% ({wins_cnt} 胜 / {matches_cnt} 场) | 偏离 50% 基准: {dev:.1f}%")
        data_section = "\n".join(stats_lines)

        if pairwise_stats:
            pairwise_lines = []
            for (f1, f2), ps in pairwise_stats.items():
                tot = ps["total"]
                f1_wins = ps["f1_wins"]
                f2_wins = ps["f2_wins"]
                wr1 = ps["wr1"]
                wr2 = ps["wr2"]
                p_dev = ps["dev"]
                is_p_ok = (target_pairwise_tolerance <= 0) or (p_dev <= target_pairwise_tolerance)
                flag = "均势" if is_p_ok else f"偏离 (偏离 50% 达 ±{p_dev:.1f}%)"
                pairwise_lines.append(
                    f"  * 【{f1} vs {f2}】: {f1} 胜率 {wr1:.1f}% ({f1_wins} 胜) vs {f2} 胜率 {wr2:.1f}% ({f2_wins} 胜) / 共 {tot} 局 -> 偏离度: {p_dev:.1f}% [{flag}]"
                )
                if target_pairwise_tolerance > 0 and p_dev > target_pairwise_tolerance:
                    dom = f1 if wr1 > 50 else f2
                    udg = f2 if wr1 > 50 else f1
                    d_w = wr1 if wr1 > 50 else wr2
                    u_w = wr2 if wr1 > 50 else wr1
                    pairwise_imbalanced.append((dom, udg, d_w, u_w, p_dev))

            data_section += "\n- 跨阵营两两对抗战报 (实机双向合并，攻守兼备，核心考核指标):\n" + "\n".join(pairwise_lines)
    else:
        p0_wins = _to_int_default(metrics_data.get("p0_wins"), 500)
        p1_wins = _to_int_default(metrics_data.get("p1_wins"), 500)
        p0_rate = (p0_wins / total) * 100
        p1_rate = (p1_wins / total) * 100
        data_section = f"- 红方胜率 (P0 - 快攻冲锋): {p0_rate:.1f}% ({p0_wins} 胜)\n- 蓝方胜率 (P1 - 控制防守): {p1_rate:.1f}% ({p1_wins} 胜)"

    history_section = ""
    if history_metrics and history_metrics != metrics_data:
        history_section = f"""
### 历史对局参考：
{json.dumps(history_metrics.get("faction_stats", history_metrics), ensure_ascii=False, indent=2)}
"""

    overperforming = []
    underperforming = []

    if isinstance(f_stats_raw, dict):
        for f, s in f_stats_raw.items():
            wr = _to_float((s or {}).get("winrate"), 50.0)
            if wr > (50.0 + target_tolerance):
                overperforming.append((f, wr, wr - 50.0))
            elif wr < (50.0 - target_tolerance):
                underperforming.append((f, wr, 50.0 - wr))

    over_instructions = []
    for f, wr, diff in overperforming:
        over_instructions.append(f"""
    * 【适度削弱大盘超标阵营 —— {f} (当前胜率 {wr:.1f}%，超出 50% 达 +{diff:.1f}%)】：
      - 阶梯式平衡逻辑（避免死砍身材）：
        1. 常规调整：优先通过适度提高费用 (+1 费) 或小幅下调身材 (-1~2 DP) 进行平滑抑制；
        2. 转向词条平衡：若发现削弱身材收效甚微，或者身材已达到该费用的合理底线（6费 >= 4 DP, 7~8费 >= 5 DP, 9费 >= 6 DP），说明问题根源在于机制而非白值，此时应果断转向削弱或剥离过于强势的词条（如剥离 BONUS_SCORE_1、将 RUSH 突袭改为蓄势、将双抽/双跳下调为单抽/单跳）；
        3. 坚决杜绝畸形面板：绝对禁止在保留霸道词条的同时，盲目把高费随从一路砍到 1~2 DP！""")

    under_instructions = []
    for f, wr, diff in underperforming:
        under_instructions.append(f"""
    * 【适度补强大盘落后阵营 —— {f} (当前胜率 {wr:.1f}%，低于 50% 达 -{diff:.1f}%)】：
      - 若关键牌先前被过度削弱：适度回调费用 (-1 费) 或增强身材 (+1~+2 DP)；
      - 前期直伤与解场手段：提高低费直伤 atk_spell_val (+1~+2 点) 或降低单解消耗 (-1 费)；
      - 防守与阻挡能力：提升低费防守怪 base_dp 或赋予 FORTIFY_1/FORTIFY_2 坚守词条；
      - 中后期制胜核心：适度降低费用 (-1 费) 提升出场率。""")

    over_text = "\n".join(over_instructions) if over_instructions else "   * 暂无大盘总胜率超标阵营。"
    under_text = "\n".join(under_instructions) if under_instructions else "   * 暂无大盘总胜率垫底阵营。"

    pairwise_instructions = []
    for dom, udg, d_w, u_w, diff in pairwise_imbalanced:
        pairwise_instructions.append(f"""
    * 【两两对抗克制失衡专项修正 —— {dom} 压制 {udg} (胜率 {d_w:.1f}% vs {u_w:.1f}%，偏离 ±{diff:.1f}% > 目标容差 ±{target_pairwise_tolerance:.1f}%)】：
      - 诊断方向：
        1. 【{dom}】是否存在令【{udg}】无力应对的过度压制机制（例如：低费高爆发突袭 RUSH、强力破甲 DEGRADE、或过快跳费 RAMP）？
        2. 【{udg}】在面对【{dom}】特定战术时是否存在防守/对策真空期（例如：面对快攻缺乏 1~2 费坚守 FORTIFY 单位或低费护盾法术；面对大身材随从缺乏单解）？
      - 调整指示：
        1. 针对性下调【{dom}】压制【{udg}】最关键的单卡收益（+1 费、-1~2 DP 或调整过激词条）；
        2. 补强【{udg}】的抗压/阻挡组件（降低抗压牌费用或提高驻守战力）；
        3. 协同注意：调整时须顾及对第三阵营的平衡联动，切忌矫枉过正引发连锁失衡。""")
    pairwise_text = "\n".join(pairwise_instructions) if pairwise_instructions else "   * 暂无两两对抗失衡（各组两两对战均在容差安全线以内）。"

    is_overall_imbalanced = max_dev > target_tolerance
    is_pairwise_imbalanced = (target_pairwise_tolerance > 0) and (max_pairwise_dev > target_pairwise_tolerance)
    is_maintenance = False

    if effective_max_dev >= severe_threshold:
        mode_banner = f"【大幅数值调整模式 (含两两克制严重失衡) | 综合最大偏离度: {effective_max_dev:.1f}% >= {severe_threshold:.1f}% (总偏离: {max_dev:.1f}%, 两两最大偏离: {max_pairwise_dev:.1f}%)】"
        balance_instructions = f"""
### 【大幅数值调整说明 (Major Overhaul)】
当前阵营间存在严重失衡或极端对局克制（综合最大偏离度 {effective_max_dev:.1f}% >= {severe_threshold:.1f}%）。
必须重点针对【两两对抗压制对局】及失衡阵营的关键卡牌进行实质性数值微调：
1. **调整规模（强制硬性要求）**：必须针对性调整 4 ~ 6 张核心卡牌（严禁仅调整 1~2 张！必须包含至少 2 张强势阵营核心削弱牌与至少 2 张弱势阵营补强牌）。
2. **两两克制失衡修正专项（核心重点）**：
{pairwise_text}
3. **大盘阵营总胜率调整**：
{over_text}
{under_text}
"""
    elif is_overall_imbalanced or is_pairwise_imbalanced:
        mode_banner = f"【轻量两两克制微调模式 | 综合最大偏离度: {effective_max_dev:.1f}% (总偏离: {max_dev:.1f}%, 两两偏离: {max_pairwise_dev:.1f}%)】"
        balance_instructions = f"""
### 【轻量微调说明】
当前部分阵营总胜率或两两交手胜率轻微浮出平衡容差（偏离度在目标容差至严重失衡线 ±{severe_threshold:.1f}% 之间）。
仅需对 1 ~ 3 张关键对策卡牌进行 ±1 费或 ±1 DP 的极小幅修剪，重点解决两两对抗中的轻度克制倾向：
{pairwise_text}
{over_text}
{under_text}
"""
    else:
        is_maintenance = True
        mode_banner = f"【全面平衡维持模式 | 阵营总偏离 {max_dev:.1f}% <= {target_tolerance}%, 两两对抗最大偏离 {max_pairwise_dev:.1f}% <= {target_pairwise_tolerance}%】"
        balance_instructions = f"""
### 【平衡维持说明】
当前各阵营综合胜率与两两实战对抗胜率全部进入了目标平衡带。
各阵营对局胜率均在安全容差以内，属于健康的竞技互动与阵营风格差异。
**本轮原则上无需对数值进行大规模修改，保护当前稳定的全量牌池生态！**
"""

    excluded_names, dual_names = _build_card_role_maps(cards_data)
    f_card_plays = metrics_data.get("faction_card_plays", {})
    card_plays_section = ""
    if f_card_plays:
        play_lines = []
        for f, c_dict in f_card_plays.items():
            filtered = [
                (name, cnt) for name, cnt in c_dict.items()
                if name not in excluded_names and not name.endswith("的小兵")
            ]
            top_cards = sorted(filtered, key=lambda x: x[1], reverse=True)[:5]
            if top_cards:
                top_str = ", ".join([
                    f"{name}{'[双色]' if name in dual_names else ''}(出场{cnt}次)"
                    for name, cnt in top_cards
                ])
                play_lines.append(f"  - 【{f}】本阵营实战高频出牌 Top 5: {top_str}")
        if play_lines:
            card_plays_section = "\n### 各阵营实战高频出牌遥测 (已剔除中立与衍生物，Top 5 核心支柱):\n" + "\n".join(play_lines) + "\n"

    constraints = []
    tabu_ids = list(tabu_list.keys()) if tabu_list else []
    if tabu_ids:
        constraints.append(f"**Tabu记忆冻结（严禁选中）**：以下卡牌刚在近期被调整过，本轮强制冷却，绝对禁止任何改动: {tabu_ids}。系统将硬拦截这些卡牌，请务必从其它卡牌中选取调整目标！")
    if weak_factions:
        constraints.append(f"**弱势阵营绝对保护**：禁止削弱综合胜率最低的弱势阵营（{weak_factions}）。对这些阵营的卡牌，强制只能加数值或降费，严禁增加 cost、降低 base_dp/atk_spell_val/def_spell_val 或剥离正面词条！")
    if overperforming and not is_maintenance:
        over_names = [x[0] for x in overperforming]
        constraints.append(f"**定向削弱指引**：强势胜率超标阵营（{over_names}），请优先从其上述【本阵营实战高频出牌 Top 5】的非双色单阵营卡中挑选削弱对象（除 Tabu 冻结卡牌外），削弱其费用(+1费)或身材(-1~2 DP)，切断其过度压制对手的节奏点！")

    if constraints:
        numbered = "\n".join(f"{idx}. {txt}" for idx, txt in enumerate(constraints, 1))
        balance_instructions += f"\n### 特殊硬性约束：\n{numbered}\n"

    mod_count_instruction = (
        "若判定为【全面平衡维持模式】无需修改任何卡牌，请直接输出 `{\"modifications\": []}`；如需微调，请仅调整 1~2 张。"
        if is_maintenance else
        "严禁改动卡牌的 id 与 card_type；仅针对性调整选中的 2~8 张卡牌的 cost, base_dp, atk_spell_val, def_spell_val, tags。"
    )

    card_in_decks, active_deck_section = _build_active_deck_context(decks_data, cards_data)
    prompt_cards = _strip_cards_for_prompt(cards_data, card_in_decks=card_in_decks or None)

    prompt = f"""
你是一名经验丰富的 TCG 卡牌设计师兼数值策划。
以下是通过强化学习（PPO 自博弈）收集的实战遥测对战数据和当前卡池配置文件。

### 1. 训练对局数据:
- 总训练对局: {total} 局
{data_section}
{card_plays_section}{active_deck_section}- 卡牌全局使用频次统计: {json.dumps(metrics_data.get("card_play_count", {}), ensure_ascii=False)}
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
   - `DRAW_X` / `RAMP_X` / `TEMP_MANA_X` / `DISCARD_X`：常规法术/战吼词条。
6. **严禁恶性完爆（Anti-Strict-Outclassing Rule，绝对红线）**：
{OUTCLASSING_RULE_BODY}
7. **严禁极度亏模废牌（Anti-Deficit Rule，全费用通用绝对红线）**：
{DEFICIT_RULE_BODY}
   - 平衡调整核心原则：
     在削弱胜率过高的卡牌时，若剥离了其突袭（RUSH）或核心词条，绝不能把数值留在残缺低位使其沦为亏模白板废卡！若去掉核心词条，必须同步补足基础身材；若压低身材，必须保留功能性机制或降费。

### 3. 待调整卡池配置:
{json.dumps(prompt_cards, indent=2, ensure_ascii=False)}

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
3. {mod_count_instruction}
4. 严禁输出任何解释性废话。
"""

    print(f"\n调用 [{get_model_name()}] 读取训练战报，执行卡池数值调优...")
    print(f"模式: {mode_banner}")
    extra = {"thinking": {"type": "disabled"}} if "deepseek.com" in get_base_url().lower() else {}
    response = client.chat.completions.create(
        model=get_model_name(),
        messages=[
            {"role": "system", "content": "You are a professional TCG balance engineer. Output ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=8192,
        response_format={"type": "json_object"},
        extra_body=extra if extra else None
    )
    return response.choices[0].message.content or ""


def extract_card_list(data) -> list:
    cards = []

    def _collect(node):
        if isinstance(node, dict):
            if "id" in node:
                cards.append(node)
            else:
                for v in node.values():
                    _collect(v)
        elif isinstance(node, list):
            for item in node:
                _collect(item)

    _collect(data)
    return cards

_POS_BOOL_TAGS = {"RUSH", "SACRIFICE_1_KILL_1"}
_POS_PREFIX_TAGS = (
    "FORTIFY_", "DEGRADE_", "SUPPORT_ATK_", "BONUS_SCORE_",
    "DEATH_DRAW_", "DEATH_MANA_", "DRAW_", "RAMP_", "TEMP_MANA_",
)


def _tag_profile(tags: list) -> dict:
    bool_pos = set()
    val_pos = {}
    spawn_dp = 0
    attack_only = False
    discard = 0
    for raw_t in tags or []:
        if not isinstance(raw_t, str):
            continue
        t = raw_t.strip().upper()
        if not is_legal_tag(t):
            continue
        if t in _POS_BOOL_TAGS:
            bool_pos.add(t)
        elif t == "ATTACK_ONLY":
            attack_only = True
        elif t.startswith("DISCARD_"):
            val = _to_int(t.split("_", 1)[1])
            if val:
                discard += val
        elif t.startswith("SPAWN_"):
            parts = t.split("_")
            if len(parts) == 3:
                x_val, y_val = _to_int(parts[1]), _to_int(parts[2])
                if x_val and y_val:
                    spawn_dp += x_val * y_val
        else:
            for pfx in _POS_PREFIX_TAGS:
                if t.startswith(pfx):
                    val = _to_int(t[len(pfx):])
                    if val:
                        val_pos[pfx] = val_pos.get(pfx, 0) + val
                    break
    return {
        "bool_pos": bool_pos,
        "val_pos": val_pos,
        "spawn_dp": spawn_dp,
        "attack_only": attack_only,
        "discard": discard,
    }


def _is_protected_weak_card(orig_f: str, orig_card: dict, weak_factions: list) -> bool:
    if not weak_factions:
        return False
    if orig_f in weak_factions:
        return True
    f_list = orig_card.get("factions") or []
    if orig_f == "Dual" and f_list:
        return all(f in weak_factions for f in f_list)
    return False


def _is_nerf_for_weak_faction(orig_card: dict, new_c: dict) -> bool:
    if "cost" in new_c:
        nc = _to_int(new_c["cost"])
        oc = _to_int(orig_card.get("cost"))
        if nc is not None and oc is not None and nc > oc:
            return True
    for fld in ("base_dp", "atk_spell_val", "def_spell_val"):
        if fld in new_c:
            nv = _to_int(new_c[fld])
            ov = _to_int(orig_card.get(fld)) or 0
            if nv is not None and nv < ov:
                return True
    if "tags" in new_c and isinstance(new_c["tags"], list):
        orig_p = _tag_profile(orig_card.get("tags") or [])
        new_p = _tag_profile(new_c["tags"])
        if not orig_p["bool_pos"].issubset(new_p["bool_pos"]):
            return True
        for k, v in orig_p["val_pos"].items():
            if new_p["val_pos"].get(k, 0) < v:
                return True
        if new_p["spawn_dp"] < orig_p["spawn_dp"]:
            return True
        if new_p["attack_only"] and not orig_p["attack_only"]:
            return True
        if new_p["discard"] > orig_p["discard"]:
            return True
    return False


def _apply_candidates(candidates: list, updated_pool: dict, id_index: dict, weak_factions: list, tabu_list: dict) -> tuple:
    applied_cards = 0
    updated_fields = 0

    for cid, new_c in candidates:
        orig_f, orig_idx = id_index[cid]
        orig_card = updated_pool[orig_f][orig_idx]

        if _is_protected_weak_card(orig_f, orig_card, weak_factions) and _is_nerf_for_weak_faction(orig_card, new_c):
            print(f"  [保护] ID {cid} 属于弱势阵营 {orig_f}，拦截任何形式的削弱（数值、费用或词条）。")
            continue

        before_card = copy.deepcopy(orig_card)
        is_spell = (orig_card.get("card_type") == "SPELL")
        card_name = orig_card.get("name", f"ID_{cid}")
        orig_cost = _to_int_default(before_card.get("cost"), 1)

        for field in ["tags", "cost", "base_dp", "atk_spell_val", "def_spell_val"]:
            if field not in new_c:
                continue
            new_val = new_c[field]

            if field == "tags":
                if not isinstance(new_val, list):
                    continue
                norm_tags = []
                for t in new_val:
                    if isinstance(t, str) and is_legal_tag(t):
                        nt = t.strip().upper()
                        if nt not in norm_tags:
                            norm_tags.append(nt)
                new_val = norm_tags
            elif field == "cost":
                parsed = _to_int(new_val)
                if parsed is None:
                    continue
                has_temp_mana = any(
                    isinstance(t, str) and t.startswith("TEMP_MANA_")
                    for t in (orig_card.get("tags") or [])
                )
                min_cost = 0 if (is_spell and (has_temp_mana or orig_cost == 0)) else 1
                new_val = min(10, max(min_cost, parsed))
            elif field == "base_dp":
                parsed = _to_int(new_val)
                if parsed is None:
                    continue
                if is_spell:
                    new_val = 0
                else:
                    o_dp = _to_int_default(orig_card.get("base_dp"), 0)
                    if parsed == o_dp:
                        new_val = o_dp
                    else:
                        cost = _to_int_default(orig_card.get("cost"), 1)
                        floor = 5 if cost >= 8 else (3 if cost >= 6 else (2 if cost >= 4 else 1))
                        if parsed < o_dp and o_dp <= floor:
                            parsed = o_dp
                        elif parsed < floor:
                            parsed = floor
                        new_val = min(15, parsed)
            elif field in ("atk_spell_val", "def_spell_val"):
                parsed = _to_int(new_val)
                if parsed is None:
                    continue
                new_val = max(0, parsed)

            if field != "tags":
                o_val = _to_int_default(orig_card.get(field), 0)
                if abs(new_val - o_val) > 1:
                    new_val = o_val + (1 if new_val > o_val else -1)

            orig_card[field] = new_val

        # 临时法力倒亏安全拦截与修复
        if is_spell:
            curr_cost = _to_int_default(orig_card.get("cost"), 0)
            fixed_cost, corrected = enforce_temp_mana_cost(
                "SPELL",
                curr_cost,
                _to_int_default(orig_card.get("atk_spell_val"), 0),
                _to_int_default(orig_card.get("def_spell_val"), 0),
                orig_card.get("tags") or [],
            )
            if corrected:
                print(f"  [安全兜底] 拦截到纯临时法力倒亏法术 ID {cid} {card_name} (cost={curr_cost})，自动修正为 0 费！")
                orig_card["cost"] = fixed_cost

        card_changed = False
        for field in ["cost", "base_dp", "atk_spell_val", "def_spell_val", "tags"]:
            if field == "tags":
                old_v = before_card.get("tags") or []
                new_v = orig_card.get("tags") or []
                old_norm = [t.strip().upper() for t in old_v if isinstance(t, str)]
                if len(new_v) == len(old_norm) and set(new_v) == set(old_norm):
                    orig_card["tags"] = old_v
                    continue
                if new_v != old_v:
                    print(f"  [词条调整] ID {cid} {card_name}: {field} 从 {old_v} -> {new_v}")
                    card_changed = True
                    updated_fields += 1
            else:
                old_v = _to_int_default(before_card.get(field), 0)
                new_v = _to_int_default(orig_card.get(field), 0)
                if new_v != old_v:
                    print(f"  [数值调整] ID {cid} {card_name}: {field} 从 {old_v} -> {new_v}")
                    card_changed = True
                    updated_fields += 1

        if card_changed:
            applied_cards += 1
            tabu_list[str(cid)] = 2

    return applied_cards, updated_fields


def main():
    parser = argparse.ArgumentParser(description="TCG 卡牌数值自动平衡调优工具")
    parser.add_argument("--cards", type=str, default="cards_config.json", help="输入的卡池 JSON 文件")
    parser.add_argument("--metrics", type=str, default="training_metrics_brawl.json", help="输入的训练战报 JSON 文件")
    parser.add_argument("--decks", type=str, default=None, help="现役卡组配置 JSON 文件 (可选，默认自动查找同目录 decks_config.json)")
    parser.add_argument("--output", type=str, default="cards_config_tuned.json", help="输出调优卡池文件路径")
    parser.add_argument("--history", type=str, default=None, help="基准/历史战报文件路径")
    parser.add_argument("--target-balance", type=float, default=5.0, help="阵营总胜率平衡容差 (默认 5.0%%)")
    parser.add_argument("--target-pairwise-balance", type=float, default=8.0, help="两两对抗平衡容差 (默认 8.0%%)")
    parser.add_argument("--severe-threshold", "--severe-imbalance-threshold", type=float, default=10.0, help="严重失衡触发大改的偏离阈值 (默认 10.0%%)")
    args = parser.parse_args()

    api_key = DEEPSEEK_API_KEY or get_api_key()
    if not api_key:
        print("\n" + "!" * 70)
        print("[致命错误] 未找到有效 DEEPSEEK_API_KEY，无法调用大模型执行数值调优！")
        print("!" * 70)
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=get_base_url())

    cards_file = resolve_path(args.cards)
    metrics_file = resolve_path(args.metrics)
    output_file = resolve_path(args.output)
    history_file = resolve_path(args.history) if args.history else None

    if not os.path.exists(cards_file):
        print(f"找不到卡池配置文件: {cards_file}")
        return
    if not os.path.exists(metrics_file):
        print(f"找不到训练战报文件: {metrics_file}")
        return

    current_cards = load_json(cards_file)
    if not isinstance(current_cards, dict) or not current_cards:
        print(f"卡池配置为空或格式非法: {cards_file}")
        return

    metrics_data = load_json(metrics_file)
    history_data = load_json(history_file) if history_file else None
    decks_file = resolve_path(args.decks) if args.decks else os.path.join(os.path.dirname(os.path.abspath(cards_file)), "decks_config.json")
    decks_data = load_json(decks_file) if decks_file and os.path.exists(decks_file) else None

    print(f"卡池输入: {cards_file} | 战报文件: {metrics_file}")
    if history_file:
        print(f"参考战报: {history_file}")

    tabu_file = os.path.join(os.path.dirname(os.path.abspath(output_file)), "balancer_tabu_list.json")
    raw_tabu = {}
    if os.path.exists(tabu_file):
        try:
            raw_tabu = load_json(tabu_file)
        except Exception:
            raw_tabu = {}
    tabu_list = {}
    if isinstance(raw_tabu, dict):
        for cid, wait in raw_tabu.items():
            w = _to_int(wait)
            if w is not None and w > 1:
                tabu_list[str(cid)] = w - 1

    weak_threshold = 50.0 - args.target_balance
    weak_factions = []
    f_stats = metrics_data.get("faction_stats")
    if isinstance(f_stats, dict):
        for f, s in f_stats.items():
            if _to_float((s or {}).get("winrate"), 50.0) < weak_threshold:
                weak_factions.append(f)

    try:
        raw_output = run_deepseek_balance_and_expand(
            metrics_data, current_cards, client,
            history_metrics=history_data,
            target_tolerance=args.target_balance,
            target_pairwise_balance=args.target_pairwise_balance,
            severe_threshold=args.severe_threshold,
            tabu_list=tabu_list,
            weak_factions=weak_factions,
            decks_data=decks_data,
        )
    except Exception as e:
        print(f"\n[警告] DeepSeek 模型调用异常 ({e})！保持当前卡池不变，平滑进入下一阶段。")
        return

    raw_output = raw_output or ""
    cleaned_json = clean_json_response(raw_output)

    new_card_pool = None
    try:
        new_card_pool = json.loads(cleaned_json)
    except json.JSONDecodeError as e:
        print(f"[*] 遇到 JSON 局部格式异常 ({e})，启动自适应单卡特征正则修复提取...")
        cand_matches = re.findall(r'\{[^{}]*"id"\s*:\s*"?\d+"?[^{}]*\}', raw_output)
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

    updated_pool = {}
    for f, card_list in current_cards.items():
        if isinstance(card_list, list):
            updated_pool[f] = [copy.deepcopy(c) for c in card_list if isinstance(c, dict)]
        else:
            updated_pool[f] = copy.deepcopy(card_list)

    id_index = {}
    for f, card_list in updated_pool.items():
        if not isinstance(card_list, list):
            continue
        for idx, c in enumerate(card_list):
            cid_key = _to_int(c.get("id"))
            if cid_key is not None:
                id_index[cid_key] = (f, idx)

    summary = _compute_balance_metrics(metrics_data, args.target_pairwise_balance)
    effective_max_dev = summary["effective_max_dev"]

    cand_cards = extract_card_list(new_card_pool)
    print(f"[*] 成功提取 {len(cand_cards)} 张调整卡牌，正在校验合规性...")

    dedup_candidates = []
    seen_cids = set()
    for new_c in cand_cards:
        if not isinstance(new_c, dict):
            continue
        cid = _to_int(new_c.get("id"))
        if cid is None or cid not in id_index or cid in seen_cids:
            continue
        seen_cids.add(cid)
        dedup_candidates.append((cid, new_c))

    non_tabu = []
    tabu_skipped = []
    for cid, new_c in dedup_candidates:
        if str(cid) in tabu_list:
            tabu_skipped.append((cid, new_c))
        else:
            non_tabu.append((cid, new_c))

    applied_cards_count, updated_count = _apply_candidates(
        non_tabu, updated_pool, id_index, weak_factions, tabu_list
    )

    # Aspiration Criterion: unlock Tabu cards if non-Tabu candidates yield 0 changes or <2 during severe imbalance.
    if tabu_skipped and (
        applied_cards_count == 0
        or (applied_cards_count < 2 and effective_max_dev >= args.severe_threshold)
    ):
        print(f"  [特赦准则激活 (Aspiration Criterion)] 触发 Tabu 特赦解除保护，强制采纳 {len(tabu_skipped)} 张冷却中卡牌的必要数值调整，防止系统死锁！")
        extra_cards, extra_fields = _apply_candidates(
            tabu_skipped, updated_pool, id_index, weak_factions, tabu_list
        )
        applied_cards_count += extra_cards
        updated_count += extra_fields
    elif tabu_skipped:
        for cid, _ in tabu_skipped:
            print(f"  [保护] ID {cid} 处于 Tabu 冷却期，跳过调整。")

    if updated_count == 0:
        print("  [死锁防范] 本轮有效卡牌调整数为 0，自动清空 Tabu 禁忌列表，解除下一轮潜在死锁。")
        tabu_list.clear()

    try:
        atomic_write_json(tabu_file, tabu_list, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"  [警告] 保存 Tabu 列表失败 ({tabu_file}): {e}")

    atomic_write_json(output_file, updated_pool, indent=2, ensure_ascii=False)

    print(f"已完成卡池数值调优 (共更新 {updated_count} 处属性/词条)，已保存至: {output_file}")


if __name__ == "__main__":
    main()
