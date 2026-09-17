"""
TCG-AI 多阵营扩展包生成与自平衡调度流水线

执行流程：
1. 阶段一：阵营对局遥测分析与扩展卡牌生成
2. 阶段二：生成各阵营 30 张初始套牌构筑
3. 阶段三：PPO 智能体自博弈对抗与卡组自适应微调
4. 阶段四：多阵营实机对战遥测与胜率偏离度统计
5. 阶段五：阵营胜率未收敛时执行卡牌数值微调
6. 阶段六：导出分析数据与可视化图表
"""

import os
import sys
import json
import re
import time
import argparse
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
# Phase 1: 卡牌扩展设计
# ==============================================================================
def step1_print_expansion_pack(cards_file: str, metrics_file: str, pack_name: str, theme: str) -> Tuple[dict, List[dict]]:
    print("\n" + "═" * 80)
    print(f"【阶段一】卡牌扩展设计: 《{pack_name}》 (主题: {theme})")
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
本次扩展包针对三大阵营与中立牌池全量印制，**包含 18 张职业阵营卡 + 3 张中立通用卡，全套扩展包总计 21 张新卡**！

详细规格配比：
1. **三大阵营卡牌 (各 6 张，共 18 张)**：
   - 【赤红 (Red)】：6 张 (4 随从 + 2 法术，包含 1 张双色卡)
   - 【蔚蓝 (Blue)】：6 张 (4 随从 + 2 法术，包含 1 张双色卡)
   - 【翠绿 (Green)】：6 张 (4 随从 + 2 法术，包含 1 张双色卡)
   - 双色卡（红蓝 4xx、蓝绿 5xx、红绿 6xx）互相融合两色机制，形成互补闭环。
2. **中立通用卡牌 (共 3 张)**：
   - 【中立 (Neutral)】：**恰好 3 张新卡**（建议为 2 张随从 + 1 张法术，或 3 张随从）；
   - 定位：全阵营通用功能组件，重点承担过牌（DRAW）、通用阻挡（FORTIFY）、临时润滑或针对性环境对策；
   - `factions` 字段必须为 `["Neutral"]`，`is_dual` 为 `false`。
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
  "diagnostic_rationale": "整体环境诊断说明（三大阵营与中立各缺啥，如何通过这 21 张新卡实现补强）",
  "factions": {{
    "Red": [
      ... (共 6 张: 4 单位 + 2 法术，含 1 张双色卡)
    ],
    "Blue": [
      ... (共 6 张: 4 单位 + 2 法术，含 1 张双色卡)
    ],
    "Green": [
      ... (共 6 张: 4 单位 + 2 法术，含 1 张双色卡)
    ],
    "Neutral": [
      ... (共 3 张中立通用卡: factions 为 [\"Neutral\"])
    ]
  }}
}}
不要输出任何 Markdown 外壳以外的废话。
"""

    print("  [模型调用] 正在生成扩展包卡牌数据...")
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
            f"模型返回内容为空 (finish_reason='{finish_reason}')！"
        )

    cleaned = clean_json_response(raw_text)
    data = json.loads(cleaned)

    rationale = data.get("diagnostic_rationale", "")
    factions_dict = data.get("factions", {})

    print(f"\n[环境诊断]:\n  {rationale}\n")

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

    for f_name in ["Red", "Blue", "Green", "Neutral"]:
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
# Phase 2: 初始 30 张预构筑
# ==============================================================================
def step2_generate_prebuild_decks(cards_file: str, decks_file: str):
    print("\n" + "═" * 80)
    print("【阶段二】生成各阵营 30 张初始套牌构筑")
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
    print("  [完成] 初始套牌构筑已生成并更新至 decks_config.json")

# ==============================================================================
# Phase 3: PPO 智能体卡组微调
# ==============================================================================
def step3_ppo_deck_evolution(cards_file: str, decks_file: str, generations: int = 5, games_per_gen: int = 50, samples: int = 15):
    print("\n" + "═" * 80)
    print(f"【阶段三】PPO 策略卡组自适应微调 | 代数: {generations} 代 | 采样: {samples} 次")
    print("═" * 80)

    ppo_builder_script = resolve_path("deck_builder_ppo.py")
    cmd = [
        sys.executable, ppo_builder_script,
        "--cards", cards_file,
        "--output", decks_file,
        "--factions", "Red,Blue,Green",
        "--generations", str(generations),
        "--games-per-gen", str(games_per_gen),
        "--samples", str(samples)
    ]
    print(f"  [执行指令] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=get_subprocess_env())
    print("  [完成] 卡组自适应微调已更新至 decks_config.json")

# ==============================================================================
# Phase 4: 多阵营对战遥测审计 (PPO 真强化学习演化)
# ==============================================================================
def step4_run_brawl_audit(cards_file: str, decks_file: str, metrics_file: str, episodes: int = 3000, eval_only: bool = False) -> dict:
    print("\n" + "═" * 80)
    print(f"【阶段四】多阵营实机对战遥测 (对局规模: {episodes} 局 | PPO 强化学习: {'关 (纯评估)' if eval_only else '开 (梯度反向传播与模型权重更新)'})")
    print("═" * 80)

    brawl_script = resolve_path("train_brawl.py")
    cmd = [
        sys.executable, brawl_script,
        "--cards", cards_file,
        "--decks", decks_file,
        "--episodes", str(episodes)
    ]
    if eval_only:
        cmd.append("--eval-only")
    print(f"  [执行指令] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=get_subprocess_env())

    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    f_stats = metrics["faction_stats"]
    print("\n" + "─" * 60)
    print("【对战遥测统计结果】")
    for f_name in ["Red", "Blue", "Green"]:
        s = f_stats[f_name]
        print(f"  * {f_name:<6}: 对局 {s['matches']:<5} 胜场 {s['wins']:<5} 胜率: {s['winrate']:.2f}%")
    print("─" * 60)
    return metrics

# ==============================================================================
# Phase 5: 胜率偏离度判定与数值微调
# ==============================================================================
def check_balance_status(metrics: dict, target_tolerance: float = 2.8) -> Tuple[bool, float, float]:
    f_stats = metrics.get("faction_stats", {})
    deviations = {f: abs(f_stats.get(f, {}).get("winrate", 50.0) - 50.0) for f in ["Red", "Blue", "Green"]}
    winrates = [f_stats.get(f, {}).get("winrate", 50.0) for f in ["Red", "Blue", "Green"]]
    max_dev = max(deviations.values()) if deviations else 0.0
    spread = (max(winrates) - min(winrates)) if winrates else 0.0

    print(f"\n[胜率偏离检测] 最大偏离: {max_dev:.2f}% | 阵营极差: {spread:.2f}% | 目标容差: <= {target_tolerance:.2f}%")
    for f in ["Red", "Blue", "Green"]:
        wr = f_stats.get(f, {}).get("winrate", 50.0)
        dev = deviations[f]
        flag = "✅" if dev <= target_tolerance else "❌"
        print(f"  {flag} 【{f:<5}】当前胜率: {wr:5.1f}% (偏离 50% 达 {dev:4.1f}%)")

    if max_dev <= target_tolerance:
        print(f"[判定通过] 三大阵营综合胜率均落入目标平衡区间 (最大偏离 <= {target_tolerance:.2f}%)")
        return True, max_dev, spread

    return False, max_dev, spread

def step5_deepseek_rebalance_cards(cards_file: str, metrics_file: str):
    print("\n" + "═" * 80)
    print("【阶段五】根据遥测战报调整卡牌基础数值与费用")
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
    print("  [完成] 卡牌数值与费用调整完毕并保存至 cards_config.json")

# ==============================================================================
# Phase 6: 导出分析看板与对比图表
# ==============================================================================
def step6_export_reports_and_charts(cards_file: str, metrics_file: str, history_records: list):
    print("\n" + "═" * 80)
    print("【阶段六】导出分析数据与可视化对比图表")
    print("═" * 80)

    # 1. 导出单卡胜率贡献榜
    tier_script = resolve_path("generate_hearthstone_tier_table.py")
    if os.path.exists(tier_script):
        try:
            print("  [1/3] 正在导出单卡胜率与评估数据 (card_tier_table.md)...")
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
        init_wr = [history_records[0].get(f, 50.0) if history_records else 50.0 for f in factions]
        curr_wr = [final_metrics.get("faction_stats", {}).get(f, {}).get("winrate", 50.0) for f in factions]
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
            series = [h.get(f, 50.0) for h in history_records]
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
            "hearthstone_assistant.html", "generate_hearthstone_tier_table.py",
            "pipeline_orchestrator.py", "deck_builder_ppo.py", "auto_balancer_deepseek.py", "sandbox.py",
            "agent.py", "eval_play.py", "train.py", "visualizer.py", "card_printer_deepseek.py",
            "deck_builder_deepseek.py", "plot_experiments.py", "test_deck_matchup.py",
            "cards_config_baseline.json", "cards_config_tuned.json"
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
    parser.add_argument("--episodes", type=int, default=3000, help="每轮自博弈混战对局规模 (默认 3000 局，保证学术统计置信度)")
    parser.add_argument("--target-balance", type=float, default=2.8, help="目标平衡偏离容差 (默认 2.8 百分点)")
    parser.add_argument("--max-deck-attempts", type=int, default=2, help="同一卡池下 PPO 自主微调构筑的尝试次数 (默认 2 次，兼顾智能体构筑优化与流水线效率)")
    parser.add_argument("--severe-imbalance-threshold", type=float, default=6.0, help="阵营胜率严重失衡偏离度阈值 (当最大偏离度 >= 该值时，判定为单卡数值硬伤，卡组微调无法弥补，跳过剩余卡组微调直接进入 DeepSeek 数值调整，默认 6.0%%)")
    parser.add_argument("--severe-spread-threshold", type=float, default=10.0, help="阵营胜率极差严重失衡阈值 (当最高与最低胜率阵营之差 >= 该值时，直接触发 DeepSeek 数值微调，默认 10.0%%)")
    parser.add_argument("--max-outer-iterations", type=int, default=10, help="最大 DeepSeek 外环数值微调迭代轮次 (默认 10 轮)")
    parser.add_argument("--max-iterations", type=int, default=None, help="(兼容旧参数) 等价于 --max-outer-iterations")
    parser.add_argument("--skip-print", action="store_true", help="跳过印卡阶段，直接基于现有卡池开始闭环调优")
    parser.add_argument("--eval-only", action="store_true", help="纯评估模式 (跳过 PPO 模型梯度更新，默认关闭以执行真强化学习)")
    parser.add_argument("--dry-run", action="store_true", help="快速演练模式 (小规模局数快速验证端到端状态机)")
    args = parser.parse_args()

    max_outer = args.max_iterations if args.max_iterations is not None else args.max_outer_iterations
    cards_file = resolve_path("cards_config.json")
    decks_file = resolve_path("decks_config.json")
    metrics_file = resolve_path("training_metrics_brawl.json")

    brawl_episodes = 60 if args.dry_run else args.episodes
    ppo_gens = 2 if args.dry_run else 5
    ppo_games = 15 if args.dry_run else 50
    ppo_samples = 4 if args.dry_run else 15
    deck_attempts = 1 if args.dry_run else args.max_deck_attempts

    print("=" * 85)
    print(" TCG-AI 多阵营扩展包生成与平衡调度流水线启动")
    print(f" 模式: {'[快速演练]' if args.dry_run else '[全量执行]'}")
    print(f" 混战规模: {brawl_episodes} 局/轮 (PPO 强化学习: {'纯评估' if args.eval_only else '真自博弈训练'}) | 目标平衡偏离度: <= {args.target_balance:.1f}%")
    print(f" 迭代配置: 卡组微调尝试 {deck_attempts} 次 | 数值微调上限 {max_outer} 轮")
    print(f" 熔断机制: 最大偏离度 >= {args.severe_imbalance_threshold:.1f}% 或 胜率极差 >= {args.severe_spread_threshold:.1f}% 自动快转至 DeepSeek 数值微调")
    print("=" * 85)

    # 1. 阶段一：卡牌扩展设计
    if not args.skip_print:
        step1_print_expansion_pack(cards_file, metrics_file, args.pack_name, args.theme)
    else:
        print("\n[*] 跳过印卡阶段，沿用当前卡池。")

    # 2. 阶段二：初始预构筑 (若印制了新扩展包，必须重新生成包含新卡的各阵营初始推荐套牌)
    if not args.skip_print:
        step2_generate_prebuild_decks(cards_file, decks_file)
    else:
        if not os.path.exists(decks_file):
            step2_generate_prebuild_decks(cards_file, decks_file)
        else:
            print(f"\n[*] 跳过印卡阶段，保留既有卡组配置 ({decks_file}) 作为初始种群。")

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
                "Red": f_stats.get("Red", {}).get("winrate", 50.0),
                "Blue": f_stats.get("Blue", {}).get("winrate", 50.0),
                "Green": f_stats.get("Green", {}).get("winrate", 50.0)
            })
        except Exception:
            history_records.append({"Red": 50.0, "Blue": 50.0, "Green": 50.0})
    else:
        history_records.append({"Red": 50.0, "Blue": 50.0, "Green": 50.0})

    # 4. 阶段三~五：双环自适应平衡状态机
    outer_round = 1
    is_balanced = False
    latest_metrics = None

    while not is_balanced and outer_round <= max_outer:
        print("\n" + "█" * 85)
        print(f"【轮次 {outer_round} / {max_outer}】数值微调与评估周期")
        print("█" * 85)

        for attempt in range(1, deck_attempts + 1):
            print("\n" + "─" * 70)
            print(f"  ▶ [卡组自适应] 第 {attempt} / {deck_attempts} 次微调")
            print("─" * 70)

            # 3A: PPO 智能体调构筑 (增量演化)
            step3_ppo_deck_evolution(cards_file, decks_file, generations=ppo_gens, games_per_gen=ppo_games, samples=ppo_samples)

            # 3B: 实机混战对抗遥测兼 PPO 强化学习演化 (按设定规模全量采样，保证统计置信度)
            current_brawl_episodes = 60 if args.dry_run else args.episodes
            metrics = step4_run_brawl_audit(cards_file, decks_file, metrics_file, episodes=current_brawl_episodes, eval_only=args.eval_only)
            latest_metrics = metrics

            curr_stats = {
                "Red": metrics.get("faction_stats", {}).get("Red", {}).get("winrate", 50.0),
                "Blue": metrics.get("faction_stats", {}).get("Blue", {}).get("winrate", 50.0),
                "Green": metrics.get("faction_stats", {}).get("Green", {}).get("winrate", 50.0)
            }
            history_records.append(curr_stats)

            # 3C: 平衡带审计判定
            is_balanced, max_dev, spread = check_balance_status(metrics, target_tolerance=args.target_balance)
            latest_max_dev = max_dev

            if is_balanced:
                print("\n" + "★" * 70)
                print(f"[判定] 胜率达成平衡收敛条件 (第 {outer_round} 轮数值调整，第 {attempt} 次卡组微调)")
                print(f"   三大阵营最大偏离度: {max_dev:.2f}% <= 目标阈值: {args.target_balance:.2f}% (胜率极差: {spread:.2f}%)")
                if not args.dry_run and current_brawl_episodes < 3000:
                    print("   [验证] 执行 3,000 局全量对战验收遥测...")
                    metrics = step4_run_brawl_audit(cards_file, decks_file, metrics_file, episodes=3000, eval_only=args.eval_only)
                    latest_metrics = metrics
                print("   已达成收敛，退出迭代循环。")
                print("★" * 70)
                break
            else:
                is_severe = (max_dev >= args.severe_imbalance_threshold) or (spread >= args.severe_spread_threshold)
                if is_severe:
                    print("\n" + "!" * 70)
                    print(f"[严重失衡熔断] 检测到阵营胜率严重失衡:")
                    print(f"   最大偏离度: {max_dev:.2f}% (熔断阈值: >= {args.severe_imbalance_threshold:.1f}%) | 阵营极差: {spread:.2f}% (熔断阈值: >= {args.severe_spread_threshold:.1f}%)")
                    print(f"   原因诊断: 存在显著的单卡数值/费用硬伤（非卡组构筑微调所能弥补）。")
                    print(f"   执行动作: 提前终止当前内环构筑探索 (当前第 {attempt}/{deck_attempts} 次)，直接快转至外环 DeepSeek 调卡牌数值！")
                    print("!" * 70)
                    break
                elif attempt < deck_attempts:
                    print(f"\n[检测] 当前偏离度 {max_dev:.2f}% > 目标 {args.target_balance:.2f}% (极差 {spread:.2f}%)，处于温和偏离区间，继续由 PPO 进行卡组微调 ({attempt + 1} / {deck_attempts})...")
                else:
                    print(f"\n[检测] PPO 已完成全部 {deck_attempts} 次卡组构筑探索，偏离度仍为 {max_dev:.2f}% (极差 {spread:.2f}%)，触发外环卡牌数值微调。")

        if is_balanced:
            break

        # 若内环尝试后依然失衡，执行数值调整
        if outer_round < max_outer:
            step5_deepseek_rebalance_cards(cards_file, metrics_file)
            outer_round += 1
        else:
            print(f"\n[提示] 已达到最大迭代轮次 ({max_outer})，结束调优循环。")
            break

    # 5. 阶段六：导出成果与报表
    step6_export_reports_and_charts(cards_file, metrics_file, history_records)

    print("\n" + "═" * 85)
    print(" [完成] 流水线执行结束，所有数据与图表已生成。")
    print("═" * 85)

if __name__ == "__main__":
    main()
