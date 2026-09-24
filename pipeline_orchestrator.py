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
from dotenv import load_dotenv
load_dotenv()
import re
import time
import argparse
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from openai import OpenAI

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if sys.platform == "win32":
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.SetPriorityClass.restype = wintypes.BOOL
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_path(p: str) -> str:
    if not p:
        return p
    if os.path.isabs(p) and os.path.exists(p):
        return p
    candidates = [
        p,
        os.path.join(SCRIPT_DIR, p),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.abspath(os.path.join(SCRIPT_DIR, p))

def get_subprocess_env() -> dict:
    env = os.environ.copy()
    py_paths = [
        SCRIPT_DIR,
    ]
    cur_pypath = env.get("PYTHONPATH", "")
    all_paths = [p for p in py_paths if os.path.exists(p)]
    if cur_pypath:
        all_paths.append(cur_pypath)
    env["PYTHONPATH"] = os.pathsep.join(all_paths)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
    env["VECLIB_MAXIMUM_THREADS"] = "2"
    env["NUMEXPR_NUM_THREADS"] = "2"
    env["TORCH_NUM_THREADS"] = "2"
    env["OMP_WAIT_POLICY"] = "PASSIVE"
    env["KMP_BLOCKTIME"] = "0"
    return env

def run_command_safe(cmd: list) -> subprocess.CompletedProcess:
    cflags = 0x00004000 if sys.platform == "win32" else 0  # BELOW_NORMAL_PRIORITY_CLASS
    return subprocess.run(cmd, check=True, env=get_subprocess_env(), creationflags=cflags)

def get_api_key() -> str:
    key = (os.environ.get("DEEPSEEK_API_KEY", "") or os.environ.get("SILICONFLOW_API_KEY", "")).strip()
    if not key or "••••" in key or key.startswith("sk-•••"):
        key = ""
        cfg_path = os.path.join(SCRIPT_DIR, "llm_config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    raw_k = (cfg.get("deepseek_api_key") or cfg.get("api_key", "")).strip()
                    if raw_k and "••••" not in raw_k and not raw_k.startswith("sk-•••"):
                        key = raw_k
            except Exception:
                pass
    if not key and sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as env_key:
                raw_k, _ = winreg.QueryValueEx(env_key, "DEEPSEEK_API_KEY")
                raw_k = (raw_k or "").strip()
                if raw_k and "••••" not in raw_k and not raw_k.startswith("sk-•••"):
                    key = raw_k
        except Exception:
            pass
    return key

def get_base_url() -> str:
    url = os.environ.get("DEEPSEEK_BASE_URL", "")
    if not url:
        cfg_path = os.path.join(SCRIPT_DIR, "llm_config.json")
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
        cfg_path = os.path.join(SCRIPT_DIR, "llm_config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    model = cfg.get("model", "")
            except Exception:
                pass
    return model or "deepseek-flash"

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
    if not isinstance(tag, str):
        return False
    return any(re.match(p, tag.strip().upper()) for p in LEGAL_TAG_PATTERNS)

def extract_valid_tags(c: dict) -> List[str]:
    """从大模型生成的卡牌对象中健壮提取合法词条，兼容多种键名、大小写及占位符"""
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
        # 占位符兼容处理
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
def step1_print_expansion_pack(cards_file: str, metrics_file: str, pack_name: str, theme: str, auto_art: bool = True) -> Tuple[dict, List[dict]]:
    print("\n" + "═" * 80)
    print(f"【阶段一】卡牌扩展设计: 《{pack_name}》 (主题: {theme})")
    print("═" * 80)

    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("未检测到 DEEPSEEK_API_KEY 环境变量，请配置 API 密钥后再运行。")

    client = OpenAI(api_key=api_key, base_url=get_base_url())

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
4. **机制牌与白板牌的黄金配比（重要核心法则）**：
   - **不能张张都是词条怪，可以有白板傻大个，但绝不能全是白板！**
   - **机制牌（带 1~2 个合法词条，填入 tags 数组）**：应占卡牌总数的大多数（约 60%~75%，全套 21 张中约 13~16 张），负责承载阵营特色、解场、突袭、过牌、亡语、跳费等核心战术；
   - **纯数值/白板牌（无词条，tags 填空数组 []）**：应保留合理比例（约 25%~35%，全套 21 张中约 5~8 张），允许身材扎实的「白板傻大个随从」以及纯直伤/纯护盾法术；
   - **白板随从身材补偿线**：白板随从没有任何词条机制，必须给足扎实身材（DP >= 费用 + 1，如 1费2DP、2费3DP、3费4DP、4费5DP、5费6DP 等）；带有强力机制的随从身材可适当让步（DP <= 费用）。
5. **合法词条规则（仅限 14 个底层已支持词条，参数必须为具体整数，严禁带 X/Y 占位符）**：
   - `RUSH` (突袭)
   - `FORTIFY_1` ~ `FORTIFY_3` (坚守)
   - `DEGRADE_1` ~ `DEGRADE_3` (削弱)
   - `SUPPORT_ATK_1` ~ `SUPPORT_ATK_2` (友方攻击光环)
   - `BONUS_SCORE_1` (击穿得分)
   - `SPAWN_1_1` ~ `SPAWN_2_1` (召唤衍生随从)
   - `DEATH_DRAW_1` (亡语抽牌)
   - `DEATH_MANA_1` (亡语跳费)
   - `SACRIFICE_1_KILL_1` (献祭强解)
   - `ATTACK_ONLY` (限进攻区)
   - `DRAW_1` ~ `DRAW_2` (抽牌)
   - `RAMP_1` (永久法力水晶+1)
   - `TEMP_MANA_1` ~ `TEMP_MANA_2` (临时法力)
   - `DISCARD_1` (弃牌)
   卡牌词条必须填入 `tags` 数组（例如 `["RUSH", "DEGRADE_1"]` 或纯白板 `[]`），严禁使用 `keywords` 键名！
6. **数值模型合理**：
   - 随从费用 (cost) 1~8 费，基础战力 (base_dp) 1~8 点；法术 base_dp 必须为 0；
   - 法术若为直接伤害填入 atk_spell_val，若为护盾填入 def_spell_val；
   - 绝不允许捏造未列入上述清单的非法词条。
7. **严禁恶性完爆（Anti-Strict-Outclassing Rule，绝对红线）**：
   - 严禁出现同阵营、同费用区间的绝对完爆（Strictly Worse / Outclassed）！
   - 绝不允许设计出一张新牌在费用相同的情况下，身材、词条以及机制全维度完全碾压现有卡牌（例如同为 3 费，一张 3 DP 生 1/1 且攻守兼备，另一张 1 DP 生 1/1 且仅能进攻，这属于绝对恶性完爆）；
   - 同费卡牌之间必须有明确的属性权衡或功能差异（如高攻脆皮 vs 稳健肉盾、即时冲锋 vs 亡语后劲、单点突破 vs 横向多体），确保每张卡具有不可替代的战术生态位。
8. **严禁极度亏模废牌（Anti-Deficit Rule，全费用通用绝对红线）**：
   - 规则不限于高费，严禁在任何费用区间（无论是 1~4 费低中费，还是 5~8+ 费高费）设计或修改出「身材/数值极度亏模且缺乏强力词条机制补偿」的垃圾废牌！
   - 随从身材模型基准：
     a. 纯白板随从（无任何词条）：必须严格遵守超模身材补偿基准线，底线为 DP >= 费用 + 1（例如 1费白板 >= 2 DP，2费白板 >= 3 DP，3费白板 >= 4 DP，4费白板 >= 5 DP，5费白板 >= 6 DP，6费白板 >= 7~8 DP，7费白板 >= 8~9 DP，8费白板 >= 9~10 DP）。绝对严禁出现 1费1DP、2费2DP、3费3DP、4费4DP 甚至 7~8费5DP 这类没有任何词条还严重亏模的废卡！
     b. 亏模随从（DP <= 费用）：必须携带足够强力或关键的战术词条（如 RUSH、FORTIFY、DEGRADE、BONUS_SCORE、DEATH_DRAW、SPAWN 等）作为亏模补偿。若 DP < 费用 - 1，其机制价值必须极其扎实，严禁低数值白板！
   - 法术数值模型基准：
     纯数值法术的总点数必须合理匹配费用（基础标准通常为 点数 >= 费用 * 1.5 到 2，如 1费直伤 2、2费直伤 3~4、3费护盾 5~6）。严禁出现 3 费仅提供 3 点护盾且无任何词条/跳费效果的极度亏模法术！
   - **严禁“倒亏法力”与逻辑错乱的法力法术（绝对核心红线，违者直接判废）：**
     - `TEMP_MANA_X` 为当前回合临时获得的法力水晶（打出当回合生效，如同炉石传说幸运币或激活）。
     - **纯临时法力法术**（仅包含 `TEMP_MANA_X`，无直伤、无护盾、无抽牌）：其法力消耗 (cost) **必须严格小于获得的临时法力数值**！标准设计只能是 **0 费消耗获得 1~2 点临时法力**（用于前期抢节奏、打连携爆发）；
     - **绝对严禁出现 `cost >= TEMP_MANA` 的反智倒亏设计**（例如：绝对严禁设计出「5 费法术仅提供 TEMP_MANA_2」、「2 费法术仅提供 TEMP_MANA_1」这种花更多费用换取更少临时法力、净亏费用还白白浪费手牌的荒谬智商税废卡）！
     - 若一张法术的费用较高（如 2~5 费）且带有 `TEMP_MANA`，它必须作为高价值复合收益的返费润滑手段（例如：带有直伤打怪 `atk_spell_val`、强力护盾 `def_spell_val`，或者配合抽牌 `DRAW_2`），**绝不允许高费单挂一个 `TEMP_MANA` 却毫无其他任何正面效果**！
   - 平衡调整核心准则：
     在削弱胜率过高的卡牌时，若剥离了其突袭（RUSH）或核心词条，绝不能把数值留在残缺低位使其沦为亏模白板废卡！若去掉核心词条，必须同步补足基础身材；若压低身材，必须保留功能性机制或降费。

### 3. 输出格式要求：
必须严格输出纯 JSON 对象，字段名称必须统一使用 "tags"（数组），严禁使用 keywords！
格式如下：
{{
  "diagnostic_rationale": "整体环境诊断说明（三大阵营与中立各缺啥，如何通过这 21 张新卡实现补强，白板与机制牌配比规划）",
  "factions": {{
    "Red": [
      {{
        "name": "赤焰突袭兵",
        "card_type": "MINION",
        "cost": 2,
        "base_dp": 2,
        "atk_spell_val": 0,
        "def_spell_val": 0,
        "tags": ["RUSH", "DEGRADE_1"],
        "is_dual": false
      }},
      {{
        "name": "赤红重装巨像",
        "card_type": "MINION",
        "cost": 4,
        "base_dp": 5,
        "atk_spell_val": 0,
        "def_spell_val": 0,
        "tags": [],
        "is_dual": false
      }},
      {{
        "name": "红莲爆裂",
        "card_type": "SPELL",
        "cost": 2,
        "base_dp": 0,
        "atk_spell_val": 4,
        "def_spell_val": 0,
        "tags": [],
        "is_dual": false
      }},
      ... (共 6 张: 4 随从 + 2 法术，包含 1 张双色卡)
    ],
    "Blue": [
      ... (共 6 张: 4 随从 + 2 法术，包含 1 张双色卡)
    ],
    "Green": [
      ... (共 6 张: 4 随从 + 2 法术，包含 1 张双色卡)
    ],
    "Neutral": [
      ... (共 3 张中立通用卡: factions 为 [\"Neutral\"])
    ]
  }}
}}
不要输出任何 Markdown 外壳以外的废话。
"""

    print("  [模型调用] 正在生成扩展包卡牌数据...")
    extra = {"thinking": {"type": "disabled"}} if "deepseek.com" in get_base_url() else {}
    response = client.chat.completions.create(
        model=get_model_name(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=8192,
        response_format={"type": "json_object"},
        extra_body=extra if extra else None
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
            raw_type = str(c.get("card_type", c.get("type", "MINION"))).upper()
            raw_dp = int(c.get("base_dp", 0))
            # 智能判断随从还是法术：若注明 MINION/UNIT 或提供了 base_dp > 0，则判定为随从
            if "MINION" in raw_type or "UNIT" in raw_type or raw_dp > 0:
                c_type = "MINION"
            else:
                c_type = "SPELL"

            is_dual = c.get("is_dual", False)
            fac_list = c.get("factions", [f_name])
            if not is_dual and f_name not in fac_list:
                fac_list = [f_name]

            # 确定 ID 前缀与双色阵营归属
            if is_dual:
                fac_set = set(fac_list)
                if fac_set == {"Red", "Blue"}:
                    pfx = 4
                    fac_list = ["Red", "Blue"]
                elif fac_set == {"Blue", "Green"}:
                    pfx = 5
                    fac_list = ["Blue", "Green"]
                elif fac_set == {"Red", "Green"}:
                    pfx = 6
                    fac_list = ["Red", "Green"]
                else:
                    if f_name == "Red":
                        pfx = 4
                        fac_list = ["Red", "Blue"]
                    elif f_name == "Blue":
                        pfx = 5
                        fac_list = ["Blue", "Green"]
                    else:
                        pfx = 6
                        fac_list = ["Red", "Green"]
            else:
                pfx = {"Red": 1, "Blue": 2, "Green": 3}.get(f_name, 9)

            cid = alloc_id(pfx)
            valid_tags = extract_valid_tags(c)
            raw_cost = int(c.get("cost", 2))

            if c_type == "MINION":
                cost_val = max(1, min(8, raw_cost))
                dp_val = max(1, min(8, raw_dp))
                if dp_val == 0:
                    dp_val = cost_val + 1 if not valid_tags else max(1, cost_val - 1)
                atk_val = 0
                def_val = 0
            else:
                cost_val = max(0, min(8, raw_cost))
                dp_val = 0
                atk_val = max(0, int(c.get("atk_spell_val", 0)))
                def_val = max(0, int(c.get("def_spell_val", 0)))
                # 兜底：若纯法术完全没有任何攻防点数和词条，按费用自动补足合理点数，绝不产生 0/0 空壳
                if atk_val == 0 and def_val == 0 and not valid_tags:
                    cost_val = max(1, cost_val)
                    if f_name == "Red":
                        atk_val = cost_val * 2
                    elif f_name == "Blue":
                        def_val = cost_val * 2
                    else:
                        def_val = cost_val + 2

                # 倒亏法力防护：拦截纯临时法力法术（如花5费只给2点临时法力）
                temp_mana_val = sum(int(t.split("_")[2]) for t in valid_tags if t.startswith("TEMP_MANA_"))
                has_comp = (atk_val > 0 or def_val > 0 or any(t.startswith("DRAW_") or t.startswith("SPAWN_") or t.startswith("RAMP_") or t == "SACRIFICE_1_KILL_1" for t in valid_tags))
                if temp_mana_val > 0 and not has_comp:
                    if cost_val >= temp_mana_val:
                        print(f"  [规则拦截] 发现纯临时法力倒亏卡牌 '{c.get('name')}' (费用 {cost_val} >= 临时法力 {temp_mana_val})，自动修正为 0 费爆发牌！")
                        cost_val = 0

            card_dict = {
                "id": cid,
                "name": c.get("name", f"{f_name}新卡_{cid}"),
                "card_type": c_type,
                "cost": cost_val,
                "base_dp": dp_val,
                "atk_spell_val": atk_val,
                "def_spell_val": def_val,
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

    # 保存扩充后卡池 (原子写入防止并发读取异常)
    tmp_cards = cards_file + ".tmp"
    try:
        with open(tmp_cards, "w", encoding="utf-8") as f:
            json.dump(current_pool, f, indent=2, ensure_ascii=False)
        os.replace(tmp_cards, cards_file)
    except Exception:
        with open(cards_file, "w", encoding="utf-8") as f:
            json.dump(current_pool, f, indent=2, ensure_ascii=False)
        if os.path.exists(tmp_cards):
            try:
                os.remove(tmp_cards)
            except OSError:
                pass

    # 终端打印表格与配比统计
    mechanic_cards = [c for c in final_pack if c.get("tags")]
    vanilla_cards = [c for c in final_pack if not c.get("tags")]
    print("\n" + "═" * 105)
    print(f"【{pack_name}】全套共 {len(final_pack)} 张新卡详细属性一览 (每阵营 6 卡: 4单位+2法术，含1张双色卡)")
    print(f"[*] 配比统计: 机制卡牌 {len(mechanic_cards)} 张 ({len(mechanic_cards)/max(1,len(final_pack))*100:.1f}%) | 纯数值/白板卡牌 {len(vanilla_cards)} 张 ({len(vanilla_cards)/max(1,len(final_pack))*100:.1f}%)")
    print("═" * 105)
    print(f"{'ID':<6}{'阵营':<14}{'卡牌名称':<16}{'类型':<8}{'费用':<6}{'身材/数值':<12}{'词条 (Tags)':<28}")
    print("─" * 105)
    for c in final_pack:
        fac_str = "/".join(c.get("factions", [])) if "factions" in c else ("双色" if c["id"]//100 in (4,5,6) else "常规")
        val_str = f"DP:{c['base_dp']}" if c["card_type"] == "MINION" else f"攻{c['atk_spell_val']}/防{c['def_spell_val']}"
        tags_str = ",".join(c["tags"]) if c["tags"] else "【白板/纯数值】"
        print(f"{c['id']:<6}{fac_str:<14}{c['name']:<16}{c['card_type']:<8}{c['cost']:<6}{val_str:<12}{tags_str:<28}")
    print("═" * 105 + "\n")

    # 异步触发后台卡图生成
    if auto_art and final_pack:
        try:
            from web_app.services.image_gen import art_queue
            art_queue.enqueue(final_pack)
            print(f"[后台生图队列] 已将本次印制的全部 {len(final_pack)} 张新卡加入后台排队出图队列。\n")
        except Exception as img_err:
            print(f"[后台生图队列] 提示: {img_err}\n")

    return current_pool, final_pack

# ==============================================================================
# Phase 2: 初始 30 张预构筑
# ==============================================================================
def step2_generate_prebuild_decks(cards_file: str, decks_file: str, new_cards: Optional[list] = None):
    print("\n" + "═" * 80)
    if new_cards and os.path.exists(decks_file):
        print("【阶段二】基于成熟基底与阵营痛点，执行新卡定向增量换入 (Pain-Point Incremental Substitution)")
    else:
        print("【阶段二】生成各阵营 30 张初始套牌构筑")
    print("═" * 80)

    deck_builder_script = resolve_path("deck_builder_deepseek.py")
    cmd = [
        sys.executable, deck_builder_script,
        "--cards", cards_file,
        "--output", decks_file,
        "--factions", "Red,Blue,Green"
    ]
    if new_cards:
        new_cards_path = resolve_path("last_expansion_pack.json")
        try:
            with open(new_cards_path, "w", encoding="utf-8") as f:
                json.dump(new_cards, f, indent=2, ensure_ascii=False)
            cmd.extend(["--new-cards", new_cards_path])
        except Exception as e:
            print(f"  [提示] 保存扩展包新卡临时文件异常: {e}")

    print(f"  [执行指令] {' '.join(cmd)}")
    run_command_safe(cmd)
    print("  [完成] 套牌构筑已更新至 decks_config.json")

# ==============================================================================
# Phase 3: PPO 智能体卡组微调
# ==============================================================================
def step3_ppo_deck_evolution(cards_file: str, decks_file: str, generations: int = 20, games_per_gen: int = 50, samples: int = 15):
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
    run_command_safe(cmd)
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
    run_command_safe(cmd)

    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    f_stats = metrics.get("faction_stats", {})
    factions = [f for f in ["Red", "Blue", "Green"] if f in f_stats]
    for f in f_stats.keys():
        if f not in factions:
            factions.append(f)
    if not factions:
        factions = ["Red", "Blue", "Green"]

    print("\n" + "─" * 60)
    print("【对战遥测统计结果】")
    for f_name in factions:
        s = f_stats.get(f_name, {"matches": 0, "wins": 0, "winrate": 50.0})
        print(f"  * {f_name:<6}: 对局 {s.get('matches', 0):<5} 胜场 {s.get('wins', 0):<5} 胜率: {s.get('winrate', 50.0):.2f}%")
    print("─" * 60)
    return metrics

# ==============================================================================
# Phase 5: 胜率偏离度判定与数值微调
# ==============================================================================
def check_balance_status(metrics: dict, target_tolerance: float = 5.0, target_pairwise_tolerance: float = 5.0) -> Tuple[bool, float, float, float]:
    f_stats = metrics.get("faction_stats", {})
    factions = [f for f in ["Red", "Blue", "Green"] if f in f_stats]
    for f in f_stats.keys():
        if f not in factions:
            factions.append(f)
    if not factions:
        factions = ["Red", "Blue", "Green"]

    deviations = {f: abs(f_stats.get(f, {}).get("winrate", 50.0) - 50.0) for f in factions}
    winrates = [f_stats.get(f, {}).get("winrate", 50.0) for f in factions]
    max_dev = max(deviations.values()) if deviations else 0.0
    spread = (max(winrates) - min(winrates)) if winrates else 0.0

    # 跨阵营两两对抗双向合并统计
    matchups = metrics.get("matchups", {})
    pairs = []
    for i in range(len(factions)):
        for j in range(i + 1, len(factions)):
            pairs.append((factions[i], factions[j]))

    pairwise_results = {}
    max_pairwise_dev = 0.0

    for f1, f2 in pairs:
        k1 = f"{f1}_vs_{f2}"
        k2 = f"{f2}_vs_{f1}"
        r1 = matchups.get(k1, {})
        r2 = matchups.get(k2, {})
        tot = r1.get("total", 0) + r2.get("total", 0)
        f1_wins = r1.get(f"{f1}_wins", 0) + r2.get(f"{f1}_wins", 0)
        wr = (f1_wins / max(1, tot)) * 100.0 if tot > 0 else 50.0
        p_dev = abs(wr - 50.0)
        if p_dev > max_pairwise_dev:
            max_pairwise_dev = p_dev
        pairwise_results[(f1, f2)] = (wr, tot, p_dev)

    is_overall_balanced = max_dev <= target_tolerance
    is_pairwise_balanced = (target_pairwise_tolerance <= 0) or (max_pairwise_dev <= target_pairwise_tolerance)
    is_balanced = is_overall_balanced and is_pairwise_balanced

    pairwise_info = f" | 两两对抗最大偏离: {max_pairwise_dev:.2f}% (容差: <={target_pairwise_tolerance:.2f}%)" if target_pairwise_tolerance > 0 else ""
    print(f"\n[胜率偏离检测] 阵营总偏离: {max_dev:.2f}% (容差: <={target_tolerance:.2f}%){pairwise_info} | 阵营极差: {spread:.2f}%")
    print("  【1. 各阵营综合胜率】:")
    for f in factions:
        wr = f_stats.get(f, {}).get("winrate", 50.0)
        dev = deviations[f]
        flag = "[OK]" if dev <= target_tolerance else "[! ]"
        print(f"    {flag} 【{f:<5}】当前胜率: {wr:5.1f}% (偏离 50% 达 {dev:4.1f}%)")

    if target_pairwise_tolerance > 0:
        print("  【2. 阵营两两对抗胜率 (双向合并实机对抗)】:")
        for (f1, f2), (wr, tot, p_dev) in pairwise_results.items():
            flag = "[OK]" if p_dev <= target_pairwise_tolerance else "[! ]"
            print(f"    {flag} {f1:<5} vs {f2:<5}: {wr:5.1f}% vs {100.0 - wr:5.1f}% (共 {tot} 局, 偏离 {p_dev:4.1f}%)")

    if is_balanced:
        print(f"[判定通过] 各阵营综合胜率与两两对抗均落入目标平衡区间！")
        return True, max_dev, spread, max_pairwise_dev
    else:
        reasons = []
        if not is_overall_balanced:
            reasons.append(f"阵营总偏离 {max_dev:.2f}% > {target_tolerance:.2f}%")
        if not is_pairwise_balanced:
            reasons.append(f"两两对抗最大偏离 {max_pairwise_dev:.2f}% > {target_pairwise_tolerance:.2f}%")
        print(f"[判定未达标] {', '.join(reasons)}")

    return False, max_dev, spread, max_pairwise_dev

def step5_deepseek_rebalance_cards(cards_file: str, metrics_file: str, target_balance: float = 5.0, target_pairwise_balance: float = 8.0, severe_threshold: float = 10.0):
    print("\n" + "═" * 80)
    print("【阶段五】根据遥测战报调整卡牌基础数值与费用")
    print("═" * 80)
    
    balancer_script = resolve_path("auto_balancer_deepseek.py")
    cmd = [
        sys.executable, balancer_script,
        "--cards", cards_file,
        "--metrics", metrics_file,
        "--output", cards_file,
        "--target-balance", str(target_balance),
        "--target-pairwise-balance", str(target_pairwise_balance),
        "--severe-threshold", str(severe_threshold)
    ]
    print(f"  [执行指令] {' '.join(cmd)}")
    run_command_safe(cmd)
    print("  [完成] 卡牌数值与费用调整完毕并保存至 cards_config.json")

# ==============================================================================
# Phase 6: 导出分析看板与对比图表
# ==============================================================================
def step6_export_reports_and_charts(cards_file: str, metrics_file: str, history_records: list):
    print("\n" + "═" * 80)
    print("【阶段六】导出分析数据与可视化对比图表")
    print("═" * 80)

    # 1. 导出供客户端 UI 使用的统一 JSON 数据
    ui_export_script = resolve_path("export_ui_data.py")
    if os.path.exists(ui_export_script):
        try:
            print("  [1/2] 正在导出前端 UI 接口数据 (ui_export_data.json)...")
            run_command_safe([sys.executable, ui_export_script])
        except Exception as e:
            print(f"  [警告] UI 数据导出异常: {e}")

    # 2. 生成前后对比大屏图表
    try:
        print("  [2/2] 正在生成对比大屏 (figure_brawl_comparison.png)...")
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

# ==============================================================================
# Main 流水线总控入口
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="TCG-AI 全自动扩展包印制与平衡调优流水线 (Pipeline Orchestrator)")
    parser.add_argument("--pack-name", type=str, default="新补充包", help="扩展包名称")
    parser.add_argument("--theme", type=str, default="根据环境数据调试", help="设计主题")
    parser.add_argument("--episodes", type=int, default=3000, help="每轮对局规模 (默认 3000 局，保证样本充分)")
    parser.add_argument("--target-balance", type=float, default=5.0, help="目标平衡偏离容差 (默认 5.0 百分点，即 45%%~55%% 平衡区间)")
    parser.add_argument("--target-pairwise-balance", type=float, default=8.0, help="两两跨阵营对抗胜率偏离容差 (默认 8.0 百分点，即 42%%~58%% 平衡区间)")
    parser.add_argument("--max-deck-attempts", type=int, default=2, help="同一卡池下 PPO 自主微调构筑的尝试次数 (默认 2 次)")
    parser.add_argument("--severe-imbalance-threshold", type=float, default=10.0, help="阵营胜率严重失衡偏离度阈值 (默认 10.0%%)")
    parser.add_argument("--severe-spread-threshold", type=float, default=15.0, help="阵营胜率极差严重失衡阈值 (默认 15.0%%)")
    parser.add_argument("--max-outer-iterations", type=int, default=10, help="最大 DeepSeek 数值微调迭代轮次 (默认 10 轮)")
    parser.add_argument("--max-iterations", type=int, default=None, help="(兼容旧参数) 等价于 --max-outer-iterations")
    parser.add_argument("--skip-print", action="store_true", help="跳过印卡阶段，直接基于现有卡池开始调优")
    parser.add_argument("--skip-art", action="store_true", help="跳过新卡原画插图自动后台生成")
    parser.add_argument("--eval-only", action="store_true", help="纯评估模式 (跳过 PPO 模型梯度更新)")
    parser.add_argument("--dry-run", action="store_true", help="快速测试模式 (小规模局数快速跑通流程)")
    parser.add_argument("--skip-final-audit", action="store_true", help="跳过达成平衡后的 3000 局终验对局 (用于高频压测加速)")
    args = parser.parse_args()

    max_outer = args.max_iterations if args.max_iterations is not None else args.max_outer_iterations
    cards_file = resolve_path("cards_config.json")
    decks_file = resolve_path("decks_config.json")
    metrics_file = resolve_path("training_metrics_brawl.json")

    brawl_episodes = 60 if args.dry_run else args.episodes
    ppo_gens = 2 if args.dry_run else 20
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

    api_key = get_api_key()
    if not api_key:
        print("\n" + "!" * 80)
        print("【终止】未检测到 DEEPSEEK_API_KEY！")
        print("  说明: 本调优流水线的核心是利用 DeepSeek 大模型对失衡卡牌进行诊断与参数微调。")
        print("        若未配置 API Key，将无法修改卡牌数值，导致空跑 10 轮上限。")
        print("  指引: 请先在 Web 界面左侧【AI 服务配置】中填入您的 DeepSeek 密钥并保存！")
        print("!" * 80 + "\n")
        sys.exit(1)

    # 1. 阶段一：卡牌扩展设计
    final_pack = []
    if not args.skip_print:
        _, final_pack = step1_print_expansion_pack(cards_file, metrics_file, args.pack_name, args.theme, auto_art=not args.skip_art)
    else:
        print("\n[*] 跳过印卡阶段，沿用当前卡池。")

    # 2. 阶段二：初始预构筑 (若印制了新扩展包，进行痛点增量换卡升级)
    if not args.skip_print:
        step2_generate_prebuild_decks(cards_file, decks_file, new_cards=final_pack)
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
            history_records.append({f: f_stats.get(f, {}).get("winrate", 50.0) for f in (list(f_stats.keys()) if f_stats else ["Red", "Blue", "Green"])})
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

            curr_stats = {f: metrics.get("faction_stats", {}).get(f, {}).get("winrate", 50.0) for f in (list(metrics.get("faction_stats", {}).keys()) if metrics.get("faction_stats") else ["Red", "Blue", "Green"])}
            history_records.append(curr_stats)

            # 3C: 平衡带审计判定
            is_balanced, max_dev, spread, max_pairwise_dev = check_balance_status(
                metrics,
                target_tolerance=args.target_balance,
                target_pairwise_tolerance=args.target_pairwise_balance
            )
            latest_max_dev = max_dev

            if is_balanced:
                print("\n" + "=" * 70)
                print(f"[判定] 胜率达成平衡收敛条件 (第 {outer_round} 轮数值调整，第 {attempt} 次卡组微调)")
                print(f"   各阵营最大偏离度: {max_dev:.2f}% <= 目标阈值: {args.target_balance:.2f}% (两两最大偏离: {max_pairwise_dev:.2f}% <= {args.target_pairwise_balance:.2f}%, 胜率极差: {spread:.2f}%)")
                if not args.dry_run and not args.skip_final_audit and current_brawl_episodes < 3000:
                    print("   [验证] 执行 3,000 局全量对战验收遥测...")
                    metrics = step4_run_brawl_audit(cards_file, decks_file, metrics_file, episodes=3000, eval_only=args.eval_only)
                    latest_metrics = metrics
                print("   已达成收敛，退出迭代循环。")
                print("=" * 70)
                break
            else:
                is_severe = (max_dev >= args.severe_imbalance_threshold) or (max_pairwise_dev >= args.severe_imbalance_threshold) or (spread >= args.severe_spread_threshold)
                # Suppress fusion on the first attempt so PPO gets at least one full chance to adapt
                if is_severe and attempt > 1:
                    print("\n" + "=" * 70)
                    print(f"[失衡触发] 检测到阵营胜率偏离较大:")
                    print(f"   阵营总偏离: {max_dev:.2f}% | 两两对抗最大偏离: {max_pairwise_dev:.2f}% (阈值: >= {args.severe_imbalance_threshold:.1f}%) | 阵营极差: {spread:.2f}%")
                    print(f"   原因: 存在卡牌数值/费用或克制差距，结束当前构筑调整，进入卡牌数值微调。")
                    print("=" * 70)
                    break
                elif attempt < deck_attempts:
                    print(f"\n[检测] 当前偏离度 (总偏离 {max_dev:.2f}%, 两两偏离 {max_pairwise_dev:.2f}%) > 目标，处于温和偏离区间，继续由 PPO 进行卡组微调 ({attempt + 1} / {deck_attempts})...")
                else:
                    print(f"\n[检测] PPO 已完成全部 {deck_attempts} 次卡组构筑探索，偏离度仍超标 (总偏离 {max_dev:.2f}%, 两两偏离 {max_pairwise_dev:.2f}%)，触发外环卡牌数值微调。")

        if is_balanced:
            break

        # 若内环尝试后依然失衡，执行数值调整
        if outer_round < max_outer:
            step5_deepseek_rebalance_cards(
                cards_file, metrics_file,
                target_balance=args.target_balance,
                target_pairwise_balance=args.target_pairwise_balance,
                severe_threshold=args.severe_imbalance_threshold
            )
            outer_round += 1
        else:
            print(f"\n[提示] 已达到最大迭代轮次 ({max_outer})，结束调优循环。")
            break

    # 5. 阶段六：导出成果与报表
    step6_export_reports_and_charts(cards_file, metrics_file, history_records)

    # 6. 后台生图状态检查与收尾
    if not args.skip_art:
        try:
            from web_app.services.image_gen import art_queue
            q_status = art_queue.get_status()
            if q_status.get("is_running"):
                rem = q_status["total"] - q_status["current"]
                print("\n" + "═" * 85)
                print(f" [后台生图队列] 平衡收敛调优已完成，后台当前剩余 {rem} 张新卡插图正在排队生成中。")
                print("               主流程已就绪，原画将在独立线程持续生成，不阻断前台操作。")
                print("═" * 85)
        except Exception:
            pass

    print("\n" + "═" * 85)
    print(" [完成] 流水线执行结束，所有数据与图表已生成。")
    print("═" * 85)
    
    if not is_balanced:
        print("\n[警告] 达到最大迭代轮次后未能达成平衡目标，以失败退出。")
        sys.exit(1)

if __name__ == "__main__":
    main()
