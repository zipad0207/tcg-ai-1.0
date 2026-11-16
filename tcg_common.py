import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

LEGAL_TAG_PATTERNS = [
    r"^RUSH$",
    r"^FORTIFY_[1-9]\d*$",
    r"^DEGRADE_[1-9]\d*$",
    r"^SUPPORT_ATK_[1-9]\d*$",
    r"^BONUS_SCORE_[1-9]\d*$",
    r"^SPAWN_[1-9]\d*_[1-9]\d*$",
    r"^DEATH_DRAW_[1-9]\d*$",
    r"^DEATH_MANA_[1-9]\d*$",
    r"^SACRIFICE_1_KILL_1$",
    r"^ATTACK_ONLY$",
    r"^DRAW_[1-9]\d*$",
    r"^RAMP_[1-9]\d*$",
    r"^TEMP_MANA_[1-9]\d*$",
    r"^DISCARD_[1-9]\d*$",
]

_DEFAULT_TAG_SUB_MAP = {
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

OUTCLASSING_RULE_BODY = """   - 严禁出现同费用、同阵营下的绝对完爆关系！
   - 严禁出现一张卡在同费用下，身材、词条以及机制全方位被另一张卡绝对碾压（如同样 3 费，一张 3 DP 生 1/1 能攻能守，另一张 1 DP 生 1/1 且仅能进攻）；
   - 同费卡牌之间必须有明确的属性权衡或功能差异（如高攻脆皮 vs 稳健肉盾、即时冲锋 vs 亡语后劲、单点突破 vs 横向多体），确保每张卡具有不可替代的构筑价值。"""

DEFICIT_RULE_BODY = """   - 规则不限于高费，严禁在任何费用区间（无论是 1~4 费低中费，还是 5~8+ 费高费）设计、保留或调出「身材/数值极度亏模且缺乏强力词条机制补偿」的垃圾废牌！
   - 随从身材模型基准：
     a. 纯白板随从（无任何词条）：必须严格遵守超模身材补偿基准线，底线为 DP >= 费用 + 1（例如 1费白板 >= 2 DP，2费白板 >= 3 DP，3费白板 >= 4 DP，4费白板 >= 5 DP，5费白板 >= 6 DP，6费白板 >= 7~8 DP，7费白板 >= 8~9 DP，8费白板 >= 9~10 DP）。绝对严禁出现 1费1DP、2费2DP、3费3DP、4费4DP 乃至 7~8费5DP 这类没有任何词条还严重亏模的废卡！
     b. 亏模随从（DP <= 费用）：必须携带足够强力或关键的战术词条（如 RUSH、FORTIFY、DEGRADE、BONUS_SCORE、DEATH_DRAW、SPAWN 等）作为亏模补偿。若 DP < 费用 - 1，其机制价值必须极其扎实，严禁低数值白板！
   - 法术数值模型基准：
     纯数值法术的总点数必须合理匹配费用（基础标准通常为 点数 >= 费用 * 1.5 到 2，如 1费直伤 2、2费直伤 3~4、3费护盾 5~6）。严禁出现 3 费仅提供 3 点护盾且无任何词条/跳费效果的极度亏模法术！
   - **严禁“倒亏法力”与逻辑错乱的法力法术（绝对核心红线，违者直接判废）：**
     - `TEMP_MANA_X` 为当前回合临时获得的法力水晶（打出当回合生效，如同炉石传说幸运币或激活）。
     - **纯临时法力法术**（仅包含 `TEMP_MANA_X`，无直伤、无护盾、无抽牌）：其法力消耗 (cost) **必须严格小于获得的临时法力数值**！标准设计只能是 **0 费消耗获得 1~2 点临时法力**（用于前期抢节奏、打连携爆发）；
     - **绝对严禁出现 `cost >= TEMP_MANA` 的反智倒亏设计**（例如：绝对严禁出现「5 费法术仅提供 TEMP_MANA_2」、「2 费法术仅提供 TEMP_MANA_1」这种花更多费用换取更少临时法力、净亏费用还白白浪费手牌的荒谬智商税废卡）！
     - 若一张法术的费用较高（如 2~5 费）且带有 `TEMP_MANA`，它必须作为高价值复合收益的返费润滑手段（例如：带有直伤打怪 `atk_spell_val`、强力护盾 `def_spell_val`，或者配合抽牌 `DRAW_2`），**绝不允许高费单挂一个 `TEMP_MANA` 却毫无其他任何正面效果**！"""


def to_int(val: Any) -> Optional[int]:
    if val is None or isinstance(val, bool):
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def to_int_default(val: Any, default: int = 0) -> int:
    parsed = to_int(val)
    return default if parsed is None else parsed


def to_float(val: Any, default: float = 0.0) -> float:
    if val is None or isinstance(val, bool):
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def resolve_path(p: str, base_dir: str = ROOT_DIR) -> str:
    if not p:
        return p
    if os.path.isabs(p) and os.path.exists(p):
        return p
    if os.path.exists(p):
        return p
    alt = os.path.join(base_dir, os.path.basename(p))
    if os.path.exists(alt):
        return alt
    if not os.path.isabs(p) and not os.path.dirname(p):
        return os.path.join(base_dir, p)
    return p


def load_json(filepath: str, default: Any = None) -> Any:
    fallback = {} if default is None else default
    if not filepath or not os.path.exists(filepath):
        return fallback
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(filepath: str, data: Any, indent: int = 2, ensure_ascii: bool = False) -> None:
    abs_path = os.path.abspath(filepath)
    parent = os.path.dirname(abs_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = abs_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
        os.replace(tmp_path, abs_path)
    except Exception:
        with open(abs_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def clean_secret(val: Any) -> str:
    if not isinstance(val, str):
        return ""
    s = val.strip()
    if not s or "••••" in s or s.startswith("sk-•••"):
        return ""
    return s


def load_llm_config(base_dir: str = ROOT_DIR) -> dict:
    cfg_path = os.path.join(base_dir, "llm_config.json")
    if not os.path.exists(cfg_path):
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_base_url(base_dir: str = ROOT_DIR) -> str:
    url = (os.environ.get("DEEPSEEK_BASE_URL") or "").strip()
    if not url:
        cfg = load_llm_config(base_dir)
        raw_u = cfg.get("base_url")
        url = raw_u.strip() if isinstance(raw_u, str) else ""
    return url or "https://api.deepseek.com"


def get_api_key(base_dir: str = ROOT_DIR) -> str:
    key = clean_secret(os.environ.get("DEEPSEEK_API_KEY", ""))
    if not key:
        cfg = load_llm_config(base_dir)
        key = clean_secret(cfg.get("deepseek_api_key")) or clean_secret(cfg.get("api_key"))
    if not key and sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as env_key:
                raw_k, _ = winreg.QueryValueEx(env_key, "DEEPSEEK_API_KEY")
                key = clean_secret(raw_k)
        except Exception:
            pass
    # Only reuse SILICONFLOW_API_KEY when base_url targets SiliconFlow; otherwise it is the image-gen key.
    if not key and "siliconflow" in get_base_url(base_dir).lower():
        key = clean_secret(os.environ.get("SILICONFLOW_API_KEY", ""))
    return key


def get_model_name(base_dir: str = ROOT_DIR) -> str:
    model = (os.environ.get("DEEPSEEK_MODEL") or "").strip()
    if not model:
        cfg = load_llm_config(base_dir)
        raw_m = cfg.get("model") or cfg.get("deepseek_model")
        model = raw_m.strip() if isinstance(raw_m, str) else ""
    if not model:
        base_url = get_base_url(base_dir)
        model = "deepseek-ai/DeepSeek-V3" if "siliconflow" in base_url.lower() else "deepseek-flash"
    return model


def clean_json_response(raw_text: str) -> str:
    if not raw_text or not isinstance(raw_text, str):
        return "{}"
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
    candidate = match.group(1).strip() if match else raw_text.strip()
    obj_start, obj_end = candidate.find("{"), candidate.rfind("}")
    arr_start, arr_end = candidate.find("["), candidate.rfind("]")
    if arr_start != -1 and arr_end > arr_start and (obj_start == -1 or arr_start < obj_start):
        return candidate[arr_start:arr_end + 1]
    if obj_start != -1 and obj_end > obj_start:
        return candidate[obj_start:obj_end + 1]
    return candidate or "{}"


def is_legal_tag(tag: str) -> bool:
    if not isinstance(tag, str):
        return False
    normalized = tag.strip().upper()
    return any(re.match(p, normalized) for p in LEGAL_TAG_PATTERNS)


def extract_valid_tags(c: dict) -> List[str]:
    raw_tags = c.get("tags")
    if raw_tags is None:
        raw_tags = c.get("keywords") or c.get("tag") or []
    if isinstance(raw_tags, str):
        raw_tags = [raw_tags]
    elif not isinstance(raw_tags, list):
        raw_tags = []

    kw_values = c.get("keyword_values", {})
    valid_tags: List[str] = []

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
                    k_clean = str(k).strip().upper()
                    if k_clean in t_clean or t_clean.startswith(k_clean):
                        val = v
                        break
            if val is not None:
                sub_tag = re.sub(r"_[XY]", f"_{val}", t_clean)
                if is_legal_tag(sub_tag) and sub_tag not in valid_tags:
                    valid_tags.append(sub_tag)
                    continue
            cand = _DEFAULT_TAG_SUB_MAP.get(t_clean)
            if cand and is_legal_tag(cand) and cand not in valid_tags:
                valid_tags.append(cand)

    return valid_tags


def enforce_temp_mana_cost(card_type: str, cost: int, atk_val: int, def_val: int, tags: list) -> Tuple[int, bool]:
    if card_type != "SPELL":
        return cost, False
    c_tags = tags or []
    t_val = 0
    for t in c_tags:
        if isinstance(t, str) and t.startswith("TEMP_MANA_"):
            parts = t.split("_")
            tv = to_int(parts[2]) if len(parts) == 3 else None
            if tv:
                t_val += tv
    has_other = (
        atk_val > 0
        or def_val > 0
        or any(
            isinstance(t, str) and (
                t.startswith("DRAW_") or t.startswith("RAMP_") or t.startswith("SPAWN_") or t == "SACRIFICE_1_KILL_1"
            )
            for t in c_tags
        )
    )
    if t_val > 0 and not has_other and cost >= t_val:
        return 0, True
    return cost, False


def ordered_factions(f_keys) -> List[str]:
    keys = list(f_keys) if f_keys else []
    factions = [f for f in ["Red", "Blue", "Green"] if f in keys]
    for f in keys:
        if f not in factions:
            factions.append(f)
    return factions or ["Red", "Blue", "Green"]


def compute_pairwise_matchups(metrics_data: dict) -> dict:
    raw_total = to_int(metrics_data.get("total_episodes"))
    total = max(1, 1000 if raw_total is None else raw_total)
    f_stats = metrics_data.get("faction_stats")
    factions: List[str] = []
    deviations: Dict[str, float] = {}
    winrates: List[float] = []
    pairwise_stats: Dict[Tuple[str, str], dict] = {}
    max_dev = 0.0
    max_pairwise_dev = 0.0

    if isinstance(f_stats, dict):
        factions = ordered_factions(f_stats.keys())
        for f in factions:
            s = f_stats.get(f) or {}
            wr = to_float(s.get("winrate"), 50.0)
            winrates.append(wr)
            deviations[f] = abs(wr - 50.0)
        max_dev = max(deviations.values()) if deviations else 0.0

        matchups = metrics_data.get("matchups")
        if isinstance(matchups, dict) and len(factions) >= 2:
            for i in range(len(factions)):
                for j in range(i + 1, len(factions)):
                    f1, f2 = factions[i], factions[j]
                    k1, k2 = f"{f1}_vs_{f2}", f"{f2}_vs_{f1}"
                    r1 = matchups.get(k1) or {}
                    r2 = matchups.get(k2) or {}
                    tot = to_int_default(r1.get("total"), 0) + to_int_default(r2.get("total"), 0)
                    f1_wins = to_int_default(r1.get(f"{f1}_wins"), 0) + to_int_default(r2.get(f"{f1}_wins"), 0)
                    f2_wins = max(0, tot - f1_wins)
                    wr1 = (f1_wins / max(1, tot)) * 100.0 if tot > 0 else 50.0
                    wr2 = 100.0 - wr1
                    p_dev = abs(wr1 - 50.0)
                    if p_dev > max_pairwise_dev:
                        max_pairwise_dev = p_dev
                    pairwise_stats[(f1, f2)] = {
                        "total": tot,
                        "f1_wins": f1_wins,
                        "f2_wins": f2_wins,
                        "wr1": wr1,
                        "wr2": wr2,
                        "dev": p_dev,
                    }
    else:
        p0_wins = to_int_default(metrics_data.get("p0_wins"), 500)
        p0_rate = (p0_wins / total) * 100.0
        max_dev = abs(p0_rate - 50.0)

    spread = (max(winrates) - min(winrates)) if winrates else 0.0
    return {
        "total": total,
        "factions": factions,
        "deviations": deviations,
        "pairwise_stats": pairwise_stats,
        "max_dev": max_dev,
        "spread": spread,
        "max_pairwise_dev": max_pairwise_dev,
    }


def set_low_priority_and_threads(threads: int = 2) -> None:
    t_str = str(threads)
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        os.environ[var] = t_str
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    os.environ["KMP_BLOCKTIME"] = "0"
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            k32.SetPriorityClass.restype = wintypes.BOOL
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)
        except Exception:
            pass
