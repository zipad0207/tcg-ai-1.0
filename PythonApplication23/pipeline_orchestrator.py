"""
TCG-AI 全自动端到端扩展包与自平衡流水线总控系统 (Pipeline Orchestrator)

功能闭环：
1. [阶段一] DeepSeek 环境诊断与“缺啥补啥” 6 张新卡智能印制 (4单位 + 2法术，含1张双色卡，词条自由组合)
2. [阶段二] DeepSeek 初始 30 张卡组理论预构筑
3. [阶段三~五] 双环自适应平衡状态机：
   - 内环：PPO 智能体自博弈演化构筑 (“谁打的谁构筑”)
   - 审计：实机 3,000 局高并发多阵营混战遥测
   - 判定：各阵营胜率是否落入黄金平衡带 (|WR - 50%| <= 阈值)
   - 外环：若失衡，唤起 DeepSeek 定向数值微调，并自动通知内环重新调构筑
4. [阶段六] 自动成果归档：生成学术图表、单卡梯队榜及全景 Markdown 战报
"""

import os
import sys
import json
import re
import time
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from openai import OpenAI

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_path(p: str) -> str:
    if not p:
        return p
    if os.path.isabs(p) and os.path.exists(p):
        return p
    candidates = [
        p,
        os.path.join(SCRIPT_DIR, p),
        os.path.join(SCRIPT_DIR, "PythonApplication23", p),
        os.path.join(os.path.dirname(SCRIPT_DIR), p),
        os.path.join(os.path.dirname(SCRIPT_DIR), "PythonApplication23", p),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.abspath(os.path.join(SCRIPT_DIR, p))

def get_subprocess_env() -> dict:
    env = os.environ.copy()
    py_paths = [
        SCRIPT_DIR,
        os.path.join(SCRIPT_DIR, "PythonApplication23"),
        os.path.dirname(SCRIPT_DIR),
        os.path.join(os.path.dirname(SCRIPT_DIR), "PythonApplication23"),
    ]
    cur_pypath = env.get("PYTHONPATH", "")
    all_paths = [p for p in py_paths if os.path.exists(p)]
    if cur_pypath:
        all_paths.append(cur_pypath)
    env["PYTHONPATH"] = os.pathsep.join(all_paths)
    return env

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

# 支持的核心合法词条
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

# ==============================================================================
# Phase 1: DeepSeek 环境诊断与 6 卡智能扩展包印制 (缺啥补啥 + 双色协同)
# ==============================================================================
def step1_print_expansion_pack(cards_file: str, metrics_file: str, pack_name: str, theme: str) -> Tuple[dict, List[dict]]:
    print("\n" + "═" * 80)
    print(f"【阶段一】DeepSeek 观察实战遥测，执行【缺啥补啥】6 卡智能扩展包印制")
    print(f"扩展包名称: 《{pack_name}》 | 设计主题: {theme}")
    print("═" * 80)

    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("未检测到 DEEPSEEK_API_KEY 环境变量，请配置 API 密钥后再运行。")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    with open(cards_file, "r", encoding="utf-8") as f:
        current_pool = json.load(f)

    telemetry_summary = "暂无混战遥测数据，采用阵营通用设计。"
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                m_data = json.load(f)
            f_stats = m_data.get("faction_stats", {})
            r_wr = f_stats.get("Red", {}).get("winrate", 50.0)
            b_wr = f_stats.get("Blue", {}).get("winrate", 50.0)
            g_wr = f_stats.get("Green", {}).get("winrate", 50.0)
            sorted_factions = sorted([("赤红 (Red)", r_wr), ("蔚蓝 (Blue)", b_wr), ("翠绿 (Green)", g_wr)], key=lambda x: x[1])
            weakest_name, weakest_wr = sorted_factions[0]
            mid_name, mid_wr = sorted_factions[1]
            strongest_name, strongest_wr = sorted_factions[-1]

            telemetry_summary = f"""
- 历史总局数: {m_data.get('total_episodes', 3000)} 局
- 各阵营综合胜率: 赤红(Red) {r_wr:.1f}% | 蔚蓝(Blue) {b_wr:.1f}% | 翠绿(Green) {g_wr:.1f}%
- 关键诊断与补强目标:
  * 胜率相对偏低的阵营为 {weakest_name} ({weakest_wr:.1f}%)，需重点强化其针对快攻冲击与高身材随从的防御、解场与资源润滑手段；
  * 居中阵营 {mid_name} ({mid_wr:.1f}%) 表现相对平稳，需丰富其中期战术选择与反击手段；
  * 胜率相对偏高的阵营 {strongest_name} ({strongest_wr:.1f}%) 体系成熟，需提供非极端的多元打法与中后期变奏；
  * 双色卡作为桥梁，促进三大阵营攻防克制闭环。
"""
        except Exception as e:
            telemetry_summary = f"遥测读取解析异常 ({e})，将使用默认均衡诊断。"

    prompt = f"""
你是一名资深的 TCG 卡牌总设计师与数值平衡科学家。
我们当前有一个基于强化学习对抗的卡牌游戏对战环境（DuelEnv），包含三大阵营：赤红 (Red)、蔚蓝 (Blue)、翠绿 (Green)。
请你根据以下【当前环境遥测战报】进行【缺啥补啥】的针对性补强设计，印制一套完整的版本扩展包。

### 1. 当前环境遥测诊断战报：
{telemetry_summary}

### 2. 扩展包设计规格硬性约束：
本次扩展包针对【赤红 (Red)】、【蔚蓝 (Blue)】、【翠绿 (Green)】三个颜色分别印制，**每个颜色各印制 6 张卡，全套扩展包总计 18 张新卡**！

每个颜色内部的 6 张新卡必须严格遵守以下规范：
1. **类型配比严格锁定**：每个颜色内部必须恰好为 **4 张随从单位 (MINION) + 2 张法术 (SPELL)**。
2. **双色卡分配**：
   - **每个颜色内部必须恰好包含 1 张【双色卡】**（如赤红配 1 张红蓝或红绿双色卡，蔚蓝配 1 张蓝绿双色卡，翠绿配 1 张红绿双色卡）；
   - **其余 5 张为纯色本阵营卡**（赤红配 5 张红卡，蔚蓝配 5 张蓝卡，翠绿配 5 张绿卡）；
   - 三大阵营的双色卡形成红蓝(4xx)、蓝绿(5xx)、红绿(6xx)的互补闭环。
3. **践行【缺啥补啥】原则**：
   - 针对当前胜率相对弱势的阵营，重点补充中前期护盾反击、有效控场随从与解场手段；
   - 针对攻势单一或依赖极端打法的阵营，提供多元化战术、优质配合与中期突破点；
   - 针对跳费或成长阵营，补充平滑过渡的驻防随从或法力/抽牌润滑组件；
   - 双色卡（红蓝 4xx、蓝绿 5xx、红绿 6xx）互相融合两色机制，形成互补闭环。
4. **合法词条自由组合**：
   可以从下列底层引擎已支持的 14 个词条中【自由组合、混合搭配】（每张卡 0~2 个词条）：
   `RUSH` (突袭), `FORTIFY_X` (坚守), `DEGRADE_X` (削弱), `SUPPORT_ATK_X` (光环), `BONUS_SCORE_X` (击穿得分),
   `SPAWN_X_Y` (衍生随从), `DEATH_DRAW_X` (亡语抽牌), `DEATH_MANA_X` (亡语临时法力), `SACRIFICE_1_KILL_1` (献祭强解),
   `ATTACK_ONLY` (限进攻区), `DRAW_X` (抽牌), `RAMP_X` (跳费水晶上限), `TEMP_MANA_X` (临时法力), `DISCARD_X` (弃牌)。
5. **数值模型合理**：
   - 随从费用 (cost) 1~8 费，基础战力 (base_dp) 1~8 点；法术 base_dp 必须为 0；
   - 法术若为直接伤害填入 atk_spell_val，若为护盾填入 def_spell_val；
   - 绝不允许捏造未列入上述清单的非法词条。

### 3. 输出格式要求：
必须严格输出纯 JSON 对象，格式如下：
{{
  "diagnostic_rationale": "整体环境诊断说明（三大阵营各缺啥，如何通过这 18 张新卡实现补强）",
  "factions": {{
    "Red": [
      {{
        "name": "卡牌名",
        "card_type": "MINION",
        "cost": 3,
        "base_dp": 3,
        "atk_spell_val": 0,
        "def_spell_val": 0,
        "tags": ["RUSH"],
        "is_dual": false,
        "factions": ["Red"]
      }},
      {{
        "name": "赤蓝交织者",
        "card_type": "MINION",
        "cost": 4,
        "base_dp": 4,
        "atk_spell_val": 0,
        "def_spell_val": 0,
        "tags": ["SUPPORT_ATK_1"],
        "is_dual": true,
        "factions": ["Red", "Blue"]
      }}
    ],
    "Blue": [
      ... (共 6 张: 4 单位 + 2 法术，含 1 张双色卡)
    ],
    "Green": [
      ... (共 6 张: 4 单位 + 2 法术，含 1 张双色卡)
    ]
  }}
}}
不要输出任何 Markdown 外壳以外的废话。
"""

    print("  [DeepSeek-Flash] 正在进行全阵营环境分析与 18 卡完整扩展包印制...")
    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=8192,
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}}
    )

    raw_text = response.choices[0].message.content or ""
    if not raw_text.strip():
        reasoning = getattr(response.choices[0].message, "reasoning_content", "")
        finish_reason = response.choices[0].finish_reason
        raise RuntimeError(
            f"DeepSeek 模型返回内容为空 (finish_reason='{finish_reason}')！\n"
            f"原因分析: 模型启用了思考模式并在思考阶段耗尽了 token 配额 ({len(reasoning)} 字思考)，未能输出正式内容。"
        )

    cleaned = clean_json_response(raw_text)
    data = json.loads(cleaned)

    rationale = data.get("diagnostic_rationale", "")
    factions_dict = data.get("factions", {})

    print(f"\n[DeepSeek 环境诊断结论]:\n  {rationale}\n")

    # 分配安全唯一的卡牌 ID (严格避开系统保留卡牌 ID: 996幸运币, 997法力过载, 998壁垒, 999衍生小兵)
    SYSTEM_RESERVED_IDS = {996, 997, 998, 999}
    existing_all_ids = set(SYSTEM_RESERVED_IDS)
    for cat, c_list in current_pool.items():
        for c in c_list:
            existing_all_ids.add(c["id"])

    def alloc_id(prefix: int) -> int:
        upper_limit = 989 if prefix == 9 else (prefix + 1) * 100
        for idx in range(prefix * 100, upper_limit):
            if idx not in existing_all_ids and idx not in SYSTEM_RESERVED_IDS:
                existing_all_ids.add(idx)
                return idx
        # 若本段耗尽，平滑升档至 4 位数扩展区间 (如 1000+, 2000+)
        for idx in range(prefix * 1000, (prefix + 1) * 1000):
            if idx not in existing_all_ids and idx not in SYSTEM_RESERVED_IDS:
                existing_all_ids.add(idx)
                return idx
        raise RuntimeError(f"阵营编号前缀 {prefix} 可用 ID 空间完全耗尽！")

    final_pack = []
    if "Dual" not in current_pool:
        current_pool["Dual"] = []

    for f_name in ["Red", "Blue", "Green"]:
        cards_list = factions_dict.get(f_name, [])
        if not cards_list:
            continue

        minion_c = sum(1 for c in cards_list if c.get("card_type") == "MINION")
        spell_c = sum(1 for c in cards_list if c.get("card_type") == "SPELL")
        dual_c = sum(1 for c in cards_list if c.get("is_dual", False))
        print(f"[*] 【{f_name}】生成卡牌统计: 单位 {minion_c} 张 | 法术 {spell_c} 张 | 双色卡 {dual_c} 张 (共 {len(cards_list)} 张)")

        for c in cards_list:
            c_type = "MINION" if c.get("card_type") == "MINION" else "SPELL"
            is_dual = c.get("is_dual", False)
            fac_list = c.get("factions", [f_name])
            if not is_dual and f_name not in fac_list:
                fac_list = [f_name]

            # 确定 ID 前缀
            if is_dual:
                fac_set = set(fac_list)
                if fac_set == {"Red", "Blue"}:
                    pfx = 4
                elif fac_set == {"Blue", "Green"}:
                    pfx = 5
                elif fac_set == {"Red", "Green"}:
                    pfx = 6
                else:
                    pfx = 4 if f_name == "Red" else (5 if f_name == "Blue" else 6)
            else:
                pfx = {"Red": 1, "Blue": 2, "Green": 3}.get(f_name, 9)

            cid = alloc_id(pfx)
            valid_tags = [t for t in c.get("tags", []) if is_legal_tag(t)]

            card_dict = {
                "id": cid,
                "name": c.get("name", f"{f_name}新卡_{cid}"),
                "card_type": c_type,
                "cost": max(1, min(8, int(c.get("cost", 2)))),
                "base_dp": max(0, min(8, int(c.get("base_dp", 0)))) if c_type == "MINION" else 0,
                "atk_spell_val": max(0, int(c.get("atk_spell_val", 0))) if c_type == "SPELL" else 0,
                "def_spell_val": max(0, int(c.get("def_spell_val", 0))) if c_type == "SPELL" else 0,
                "tags": valid_tags,
                "factions": fac_list
            }

            if is_dual:
                current_pool["Dual"].append(card_dict)
            else:
                if f_name not in current_pool:
                    current_pool[f_name] = []
                current_pool[f_name].append(card_dict)

            final_pack.append(card_dict)

    # 保存扩充后卡池
    with open(cards_file, "w", encoding="utf-8") as f:
        json.dump(current_pool, f, indent=2, ensure_ascii=False)

    # 终端打印表格
    print("\n" + "═" * 105)
    print(f"【{pack_name}】全套共 {len(final_pack)} 张新卡详细属性一览 (每阵营 6 卡: 4单位+2法术，含1张双色卡)")
    print("═" * 105)
    print(f"{'ID':<6}{'阵营':<14}{'卡牌名称':<16}{'类型':<8}{'费用':<6}{'身材/数值':<12}{'词条 (Tags)':<28}")
    print("─" * 105)
    for c in final_pack:
        fac_str = "/".join(c.get("factions", [])) if "factions" in c else ("双色" if c["id"]//100 in (4,5,6) else "常规")
        val_str = f"DP:{c['base_dp']}" if c["card_type"] == "MINION" else f"攻{c['atk_spell_val']}/防{c['def_spell_val']}"
        tags_str = ",".join(c["tags"]) if c["tags"] else "无"
        print(f"{c['id']:<6}{fac_str:<14}{c['name']:<16}{c['card_type']:<8}{c['cost']:<6}{val_str:<12}{tags_str:<28}")
    print("═" * 105 + "\n")

    return current_pool, final_pack

# ==============================================================================
# Phase 2: DeepSeek 初始 30 张理论预构筑
# ==============================================================================
def step2_generate_prebuild_decks(cards_file: str, decks_file: str):
    print("\n" + "═" * 80)
    print("【阶段二】DeepSeek 针对新卡池快速生成各阵营 30 张推荐初始预构筑")
    print("═" * 80)

    deck_builder_script = resolve_path("deck_builder_deepseek.py")
    cmd = [
        sys.executable, deck_builder_script,
        "--cards", cards_file,
        "--output", decks_file,
        "--factions", "Red,Blue,Green"
    ]
    print(f"  [执行指令] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=get_subprocess_env())
    print("  [OK] DeepSeek 30 张初始预构筑已生成并更新至 decks_config.json")

# ==============================================================================
# Phase 3: PPO 智能体自主选卡进化 ("谁打的谁构筑")
# ==============================================================================
def step3_ppo_deck_evolution(cards_file: str, decks_file: str, generations: int = 5, games_per_gen: int = 50):
    print("\n" + "═" * 80)
    print(f"【阶段三】PPO 智能体自主构筑演化 ('谁打的谁构筑') | 代数: {generations} 代")
    print("═" * 80)

    ppo_builder_script = resolve_path("deck_builder_ppo.py")
    cmd = [
        sys.executable, ppo_builder_script,
        "--cards", cards_file,
        "--output", decks_file,
        "--factions", "Red,Blue,Green",
        "--generations", str(generations),
        "--games-per-gen", str(games_per_gen)
    ]
    print(f"  [执行指令] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=get_subprocess_env())
    print("  [OK] PPO 智能体已根据对局胜率与 Critic 价值评估完成 30 张实战构筑自主更新！")

# ==============================================================================
# Phase 4: 3,000 局高并发多阵营混战实机对抗遥测
# ==============================================================================
def step4_run_brawl_audit(cards_file: str, decks_file: str, metrics_file: str, episodes: int = 3000) -> dict:
    print("\n" + "═" * 80)
    print(f"【阶段四】3,000 局多阵营实机混战遥测审计 (自博弈规模: {episodes} 局)")
    print("═" * 80)

    brawl_script = resolve_path("train_brawl.py")
    cmd = [
        sys.executable, brawl_script,
        "--cards", cards_file,
        "--decks", decks_file,
        "--episodes", str(episodes)
    ]
    print(f"  [执行指令] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=get_subprocess_env())

    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    f_stats = metrics["faction_stats"]
    print("\n" + "─" * 60)
    print("【实机遥测战报结算】")
    for f_name in ["Red", "Blue", "Green"]:
        s = f_stats[f_name]
        print(f"  * {f_name:<6}: 对局 {s['matches']:<5} 胜场 {s['wins']:<5} 胜率: {s['winrate']:.2f}%")
    print("─" * 60)
    return metrics

# ==============================================================================
# Phase 5: 自平衡状态审计与 DeepSeek 外环数值微调
# ==============================================================================
def check_balance_status(metrics: dict, target_tolerance: float = 2.8) -> Tuple[bool, float]:
    f_stats = metrics.get("faction_stats", {})
    deviations = {f: abs(f_stats.get(f, {}).get("winrate", 50.0) - 50.0) for f in ["Red", "Blue", "Green"]}
    max_dev = max(deviations.values())

    print(f"\n[平衡偏离度审计] 最大偏离: {max_dev:.2f}% | 允许目标: <= {target_tolerance:.2f}%")
    for f in ["Red", "Blue", "Green"]:
        wr = f_stats.get(f, {}).get("winrate", 50.0)
        dev = deviations[f]
        flag = "✅" if dev <= target_tolerance else "❌"
        print(f"  {flag} 【{f:<5}】当前胜率: {wr:5.1f}% (偏离 50% 达 {dev:4.1f}%)")

    if max_dev <= target_tolerance:
        print("🎉 恭喜！三大阵营综合胜率均落入黄金平衡带，达成纳什自平衡！")
        return True, max_dev

    return False, max_dev

def step5_deepseek_rebalance_cards(cards_file: str, metrics_file: str):
    print("\n" + "═" * 80)
    print("【阶段五·外环触发】构筑优化已达瓶颈，遥测战报扔回 DeepSeek 执行卡牌数值靶向微调")
    print("═" * 80)
    
    balancer_script = resolve_path("auto_balancer_deepseek.py")
    cmd = [
        sys.executable, balancer_script,
        "--cards", cards_file,
        "--metrics", metrics_file,
        "--output", cards_file
    ]
    print(f"  [执行指令] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=get_subprocess_env())
    print("  [OK] DeepSeek 已根据最新 3,000 局对战遥测数据完成卡牌属性与费用靶向微调！")

# ==============================================================================
# Phase 6: 自动可视化导出与全景战报打包
# ==============================================================================
def step6_export_reports_and_charts(cards_file: str, metrics_file: str, history_records: list):
    print("\n" + "═" * 80)
    print("【阶段六】自动导出学术看板、单卡梯队天梯榜与全景 Markdown 战报")
    print("═" * 80)

    # 1. 导出单卡胜率贡献榜 (Hearthstone Tier Table)
    tier_script = resolve_path("generate_hearthstone_tier_table.py")
    if os.path.exists(tier_script):
        try:
            print("  [1/3] 正在生成单卡贡献天梯榜 (card_tier_table.md)...")
            subprocess.run([sys.executable, tier_script], check=True, env=get_subprocess_env())
        except Exception as e:
            print(f"  [警告] 梯队榜生成异常: {e}")

    # 2. 生成前后对比大屏图表
    try:
        print("  [2/3] 正在生成对比大屏 (figure_brawl_comparison.png)...")
        with open(metrics_file, "r", encoding="utf-8") as f:
            final_metrics = json.load(f)

        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        fig = plt.figure(figsize=(22, 6.8), dpi=300)
        gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.25], wspace=0.28)
        factions = ["Red", "Blue", "Green"]
        names = ["赤红 (Red)", "蔚蓝 (Blue)", "翠绿 (Green)"]

        # 左图：胜率对比
        ax1 = fig.add_subplot(gs[0])
        x = np.arange(len(factions))
        init_wr = [history_records[0][f] for f in factions]
        curr_wr = [final_metrics["faction_stats"][f]["winrate"] for f in factions]
        width = 0.35
        ax1.bar(x - width/2, init_wr, width, label='初始阶段', color='#bdc3c7', edgecolor='#7f8c8d')
        ax1.bar(x + width/2, curr_wr, width, label='当前收敛态', color=['#ff4757', '#1e90ff', '#2ed573'], edgecolor='#2c3e50')
        ax1.axhline(50.0, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.8, label="50% 平衡线")
        ax1.set_ylim(0, 100)
        ax1.set_ylabel("胜率 (%)", fontsize=11, fontweight="bold")
        ax1.set_title("三大阵营调优前后胜率对比", fontsize=13, pad=12, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, fontsize=10, fontweight="bold")
        ax1.legend(loc="upper right")
        ax1.grid(axis='y', linestyle='--', alpha=0.3)

        # 中图：演进轨迹
        ax2 = fig.add_subplot(gs[1])
        rounds = [f"R{i}" for i in range(len(history_records))]
        f_colors = {"Red": "#ff4757", "Blue": "#1e90ff", "Green": "#2ed573"}
        for f in factions:
            series = [h[f] for h in history_records]
            ax2.plot(rounds, series, marker='o', linewidth=2.4, label=names[factions.index(f)], color=f_colors[f])
            for idx, val in enumerate(series):
                ax2.annotate(f"{val:.1f}%", (idx, val), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=8.5, fontweight='bold')
        ax2.axhline(50.0, color="#7f8c8d", linestyle="--", linewidth=1.2)
        ax2.set_ylim(30, 70)
        ax2.set_ylabel("胜率收敛走势 (%)", fontsize=11, fontweight="bold")
        ax2.set_title("迭代收敛演变轨迹", fontsize=13, pad=12, fontweight="bold")
        ax2.legend(loc="upper right")
        ax2.grid(True, linestyle='--', alpha=0.3)

        # 右图：克制矩阵
        ax3 = fig.add_subplot(gs[2])
        matrix = np.zeros((3, 3))
        for i, fA in enumerate(factions):
            for j, fB in enumerate(factions):
                if i == j:
                    matrix[i, j] = 50.0
                else:
                    k1 = f"{fA}_vs_{fB}"
                    k2 = f"{fB}_vs_{fA}"
                    r1 = final_metrics["matchups"].get(k1, {"total": 0})
                    r2 = final_metrics["matchups"].get(k2, {"total": 0})
                    total_games = r1.get("total", 0) + r2.get("total", 0)
                    fA_wins = r1.get(f"{fA}_wins", 0) + r2.get(f"{fA}_wins", 0)
                    matrix[i, j] = (fA_wins / total_games) * 100 if total_games > 0 else 50.0
        ax3.imshow(matrix, cmap="RdYlGn", vmin=35, vmax=65)
        ax3.set_xticks(range(3))
        ax3.set_yticks(range(3))
        ax3.set_xticklabels(["对手: 赤红", "对手: 蔚蓝", "对手: 翠绿"], fontsize=9.5)
        ax3.set_yticklabels(["本方: 赤红", "本方: 蔚蓝", "本方: 翠绿"], fontsize=9.5)
        ax3.set_title("最终三大阵营对弈克制矩阵 (%)", fontsize=13, pad=12, fontweight="bold")
        for i in range(3):
            for j in range(3):
                val = matrix[i, j]
                label = "50.0%\n(内战)" if i == j else f"{val:.1f}%"
                ax3.text(j, i, label, ha="center", va="center", color="black" if 42 <= val <= 58 else "white", fontweight="bold", fontsize=10)

        cmp_out = resolve_path("figure_brawl_comparison.png")
        plt.savefig(cmp_out, bbox_inches="tight")
        plt.close(fig)
        print(f"  [OK] 对比大屏已更新: {cmp_out}")
    except Exception as e:
        print(f"  [警告] 对比大屏图表生成异常: {e}")

    # 3. 根目录与子目录双向资产同步
    try:
        import shutil
        root_dir = os.path.dirname(SCRIPT_DIR)
        sync_files = [
            "figure_brawl.png", "figure_brawl_comparison.png", "training_metrics_brawl.json",
            "cards_config.json", "decks_config.json", "card_tier_table.md", "ppo_introspection_report.md",
            "pipeline_orchestrator.py", "deck_builder_ppo.py", "auto_balancer_deepseek.py"
        ]
        for fname in sync_files:
            src = resolve_path(fname)
            dst = os.path.join(root_dir, fname)
            if os.path.exists(src) and src != dst:
                shutil.copy(src, dst)
        print("  [3/3] 核心资产与战报已自动同步至工作区根目录与子目录！")
    except Exception as e:
        print(f"  [警告] 资产同步异常: {e}")

# ==============================================================================
# Main 流水线总控入口 (双环嵌套闭环系统)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="TCG-AI 全自动扩展包印制与双环自平衡协同流水线 (Pipeline Orchestrator)")
    parser.add_argument("--pack-name", type=str, default="破晓对决补充包", help="扩展包名称")
    parser.add_argument("--theme", type=str, default="环境数据驱动缺啥补啥与双色协同", help="设计主题")
    parser.add_argument("--episodes", type=int, default=3000, help="每轮自博弈混战对局规模 (默认 3000)")
    parser.add_argument("--target-balance", type=float, default=2.8, help="目标平衡偏离容差 (默认 2.8 百分点)")
    parser.add_argument("--max-deck-attempts", type=int, default=3, help="同一卡池下 PPO 自主微调构筑的尝试次数 (默认 3 次，即尝试调 2~3 次构筑)")
    parser.add_argument("--max-outer-iterations", type=int, default=10, help="最大 DeepSeek 外环数值微调迭代轮次 (默认 10 轮)")
    parser.add_argument("--max-iterations", type=int, default=None, help="(兼容旧参数) 等价于 --max-outer-iterations")
    parser.add_argument("--skip-print", action="store_true", help="跳过印卡阶段，直接基于现有卡池开始闭环调优")
    parser.add_argument("--dry-run", action="store_true", help="快速演练模式 (小规模局数快速验证端到端状态机)")
    args = parser.parse_args()

    max_outer = args.max_iterations if args.max_iterations is not None else args.max_outer_iterations
    cards_file = resolve_path("cards_config.json")
    decks_file = resolve_path("decks_config.json")
    metrics_file = resolve_path("training_metrics_brawl.json")

    brawl_episodes = 60 if args.dry_run else args.episodes
    ppo_gens = 2 if args.dry_run else 5
    ppo_games = 15 if args.dry_run else 50
    deck_attempts = 2 if args.dry_run else args.max_deck_attempts

    print("=" * 85)
    print(" 🚀 TCG-AI 端到端全自动扩展包演化流水线 (Pipeline Orchestrator) 启动")
    print(f" 运作模式: {'[DRY-RUN 快速演练]' if args.dry_run else '[FULL PRODUCTION 实机全量]'}")
    print(f" 混战规模: {brawl_episodes} 局/轮 | 目标平衡容差: ±{args.target_balance:.1f}%")
    print(f" 嵌套架构: PPO 内环连续演化构筑 {deck_attempts} 次 -> 若失衡则触发 DeepSeek 外环改卡 (上限 {max_outer} 轮)")
    print("=" * 85)

    # 1. 阶段一：DeepSeek 诊断与印卡
    if not args.skip_print:
        step1_print_expansion_pack(cards_file, metrics_file, args.pack_name, args.theme)
    else:
        print("\n[*] 跳过印卡阶段，沿用当前卡池进行闭环平衡。")

    # 2. 阶段二：DeepSeek 初始预构筑 (仅在 decks_config 不存在时创建)
    if not os.path.exists(decks_file):
        step2_generate_prebuild_decks(cards_file, decks_file)
    else:
        print(f"\n[*] 检测到既有卡组配置 ({decks_file})，保留作为 PPO 初始种群基底。")

    # 3. 记录初始胜率历史与基准偏离度
    history_records = []
    latest_max_dev = 25.0
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                old_m = json.load(f)
            f_stats = old_m.get("faction_stats", {})
            if f_stats:
                latest_max_dev = max(abs(s.get("winrate", 50.0) - 50.0) for s in f_stats.values())
            history_records.append({
                "Red": old_m["faction_stats"]["Red"]["winrate"],
                "Blue": old_m["faction_stats"]["Blue"]["winrate"],
                "Green": old_m["faction_stats"]["Green"]["winrate"]
            })
        except Exception:
            history_records.append({"Red": 50.0, "Blue": 50.0, "Green": 50.0})
    else:
        history_records.append({"Red": 50.0, "Blue": 50.0, "Green": 50.0})

    # 4. 阶段三~五：双环嵌套自适应平衡状态机
    # 外环：DeepSeek 卡牌数值版本迭代 (Patch Era)
    # 内环：PPO 智能体自主构筑演化微调 (2~3 次尝试，自适应局数实测)
    outer_round = 1
    is_balanced = False
    latest_metrics = None

    while not is_balanced and outer_round <= max_outer:
        print("\n" + "█" * 85)
        print(f"【外环·版本周期】第 {outer_round} / {max_outer} 轮卡牌数值补丁启动")
        print("█" * 85)

        for attempt in range(1, deck_attempts + 1):
            print("\n" + "─" * 70)
            print(f"  ▶【内环·构筑演进】第 {attempt} / {deck_attempts} 次 PPO 智能体自主微调构筑 ('谁打的谁构筑')")
            print("─" * 70)

            # 3A: PPO 智能体自主调构筑 (增量演化)
            step3_ppo_deck_evolution(cards_file, decks_file, generations=ppo_gens, games_per_gen=ppo_games)

            # 自适应混战局数设定：失衡越严重，用越少的局数快速捕获方向，成倍节省时间
            if args.dry_run:
                current_brawl_episodes = 60
            elif latest_max_dev >= 12.0:
                current_brawl_episodes = 500
                print(f"  ⚡ [自适应加速] 检测到严重失衡 (偏离度 {latest_max_dev:.1f}%)，采用高速遥测模式 ({current_brawl_episodes} 局/轮)...")
            elif latest_max_dev >= 6.0:
                current_brawl_episodes = 1200
                print(f"  ⚡ [自适应平衡] 检测到中度失衡 (偏离度 {latest_max_dev:.1f}%)，采用中度遥测模式 ({current_brawl_episodes} 局/轮)...")
            else:
                current_brawl_episodes = args.episodes
                print(f"  🎯 [黄金带精准验证] 生态已接近平衡 (偏离度 {latest_max_dev:.1f}%)，采用高精度混战 ({current_brawl_episodes} 局/轮)...")

            # 3B: 实机混战对抗遥测
            metrics = step4_run_brawl_audit(cards_file, decks_file, metrics_file, episodes=current_brawl_episodes)
            latest_metrics = metrics

            curr_stats = {
                "Red": metrics["faction_stats"]["Red"]["winrate"],
                "Blue": metrics["faction_stats"]["Blue"]["winrate"],
                "Green": metrics["faction_stats"]["Green"]["winrate"]
            }
            history_records.append(curr_stats)

            # 3C: 平衡带审计判定
            is_balanced, max_dev = check_balance_status(metrics, target_tolerance=args.target_balance)
            latest_max_dev = max_dev

            if is_balanced:
                print("\n" + "★" * 70)
                print(f"🎉 完美达成纳什自平衡！(第 {attempt} 次构筑微调收敛)")
                print(f"   三大阵营最大偏离度仅 {max_dev:.2f}% <= 目标容差 {args.target_balance:.2f}%！")
                print("   全生态自平衡达成，立即退出双环迭代！")
                print("★" * 70)
                break
            else:
                # 严重失衡断崖判定：如果偏离度 >= 16.0%，即使在第 1 次构筑尝试，也无需徒劳重跑构筑，直接呼叫 DeepSeek 大刀阔斧改卡
                if max_dev >= 16.0 and attempt == 1 and deck_attempts > 1:
                    print(f"\n🚨【底层数值严重失衡预警 (偏离度 {max_dev:.2f}% >= 16.0%)】")
                    print(f"   胜率存在断层级鸿沟，卡组构筑已无法跨越此结构性差距！")
                    print(f"   ⚡ 立即跳过剩余构筑尝试，直接提前呼叫 DeepSeek 启动【重拳大刀阔斧调优模式】！")
                    break

                if attempt < deck_attempts:
                    print(f"\n⚠️ 当前构筑下胜率偏离度 {max_dev:.2f}% > 目标 {args.target_balance:.2f}%，生态尚未平衡。")
                    print(f"   💡 让 PPO 智能体继续在现有卡池中自博弈演化构筑 (进行第 {attempt + 1} / {deck_attempts} 次尝试)...")
                else:
                    print(f"\n⚠️ PPO 智能体已在当前卡池下连续微调构筑 {deck_attempts} 次，胜率偏离度仍为 {max_dev:.2f}%！")
                    print("   🚨【判定：发现构筑不行】纯靠卡组构筑已达调优瓶颈，确认存在底层单卡数值失衡！")
                    print("   📦 准备将遥测数据扔回 DeepSeek 进行底层数值与费用大刀阔斧调优...")

        if is_balanced:
            break

        # 若内环尝试后依然失衡，扔回 DeepSeek 调数据
        if outer_round < max_outer:
            step5_deepseek_rebalance_cards(cards_file, metrics_file)
            outer_round += 1
        else:
            print(f"\n⚠️ 已达到外环最大迭代轮次 ({max_outer})，结束微调循环。")
            break

    # 5. 阶段六：自动导出成果与报表
    step6_export_reports_and_charts(cards_file, metrics_file, history_records)

    print("\n" + "═" * 85)
    print(" ✅ TCG-AI 端到端全自动扩展包演化流水线执行圆满完毕！")
    print("═" * 85)

if __name__ == "__main__":
    main()
