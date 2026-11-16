import os
import sys
import json
from typing import List

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from openai import OpenAI
from tcg_common import (
    LEGAL_TAG_PATTERNS,
    atomic_write_json,
    clean_json_response as clean_json_text,
    enforce_temp_mana_cost,
    get_api_key,
    get_base_url,
    get_model_name,
    is_legal_tag,
)

SYSTEM_RESERVED_IDS = {990, 991, 992, 993, 994, 995, 996, 997, 998, 999}


class DeepSeekCardPrinter:
    def __init__(self):
        self.root_dir = _ROOT_DIR
        self.cards_path = os.path.join(self.root_dir, "cards_config.json")
        self.reload_config()

    def reload_config(self):
        self.api_key = get_api_key(self.root_dir)
        self.base_url = get_base_url(self.root_dir)
        self.model = get_model_name(self.root_dir)

    def _get_next_id(self, existing_cards: List[dict], default_start: int) -> int:
        if not existing_cards:
            return default_start
        curr_id = max(c.get("id", default_start) for c in existing_cards) + 1
        while curr_id in SYSTEM_RESERVED_IDS or (990 <= curr_id <= 999):
            curr_id += 1
        return curr_id

    def load_cards_config(self) -> dict:
        if os.path.exists(self.cards_path):
            with open(self.cards_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"Red": [], "Blue": [], "Green": [], "Neutral": []}

    def save_cards_config(self, config_data: dict):
        atomic_write_json(self.cards_path, config_data, indent=4, ensure_ascii=False)

        export_script = os.path.join(self.root_dir, "export_ui_data.py")
        if os.path.exists(export_script):
            import subprocess
            subprocess.run([sys.executable, export_script], check=False)

    def generate_new_cards(self, faction: str = "Red", count: int = 2, theme: str = "") -> dict:
        self.reload_config()
        if not self.api_key:
            raise ValueError("未配置有效的 DeepSeek API Key，请先在【AI 服务配置】或 .env 中配置 DEEPSEEK_API_KEY。")

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        cards_config = self.load_cards_config()

        faction_desc = {
            "Red": "红方：快攻冲锋、突袭（RUSH）、召唤衍生小兵（SPAWN_X_Y）、破甲削弱（DEGRADE_X）",
            "Blue": "蓝方：坚守高防（FORTIFY_X）、光环增益（SUPPORT_ATK_X）、控制阻挡",
            "Green": "绿方：跳费成长（RAMP_X/DEATH_MANA_X）、大身材随从、持续施压",
            "Neutral": "中立：过牌（DRAW_X）、通用攻防辅助"
        }.get(faction, f"{faction} 阵营")

        # Extract current card names to avoid duplicates
        existing_names = set()
        for f_cards in cards_config.values():
            if isinstance(f_cards, list):
                for c in f_cards:
                    if "name" in c:
                        existing_names.add(c["name"])

        prompt = f"""你是一名 TCG 卡牌设计与数值平衡工程师。
请为【{faction}】阵营 ({faction_desc}) 构想设计 {count} 张全新机制卡牌。
设计主题与战术方向: 【{theme if theme else '符合阵营核心战术，具备战术搭配与构筑价值的机制卡'}】

### 规则与数值平衡红线:
1. 绝对严禁与已有卡牌同名！现有部分卡牌名：{list(existing_names)[:25]}
2. 词条严格限制在以下支持列表中：
   - RUSH (突袭，立即冲锋)
   - FORTIFY_X (防守区战力+X)
   - DEGRADE_X (冲锋前削弱对方防守怪X战力)
   - SUPPORT_ATK_X (防守区为己方冲锋怪+X战力)
   - BONUS_SCORE_X (冲锋突破额外+X得分)
   - SPAWN_X_Y (召唤 Y 只战力为 X 的衍生随从)
   - DEATH_DRAW_X (亡语抽 X 张牌)
   - DEATH_MANA_X (亡语跳费 X 点)
   - SACRIFICE_1_KILL_1 (法术：献祭己方一怪强杀敌方一怪)
   - ATTACK_ONLY (只能打入进攻区)
   - DRAW_X (抽 X 张牌)
   - RAMP_X (永久法力上限+X)
   - TEMP_MANA_X (当回合临时法力+X)
   - DISCARD_X (负面补偿，弃 X 张手牌)
3. 数值平衡基准：
   - 白板随从基准身材: DP >= 费用 + 1
   - 亏模随从（DP <= 费用）必须有强力词条补偿
   - 法术点数匹配费用（如 1费打2，2费打3~4，3费打5或附带过牌）
   - 【严禁倒亏法力与负收益废卡（绝对红线）】：TEMP_MANA_X 代表当回合临时获得 X 点法力（如激活、幸运币）。纯临时法力法术费用必须为 0 费（如 0 费获得 1~2 临时法力）；绝对严禁生成「cost >= TEMP_MANA 数值」（如 5 费只给 2 临时法力）这类花钱买更少钱、倒亏法力还亏手牌的反智废卡！若法术费用 >= 2 且带有 TEMP_MANA，必须附带直伤、护盾或过牌（DRAW）等高额复合收益。
4. 单张卡字段定义:
   - "name": 中文卡牌名
   - "card_type": "MINION" 或 "SPELL"
   - "cost": 费用 (0~8 整数)
   - "base_dp": 基础战力 (SPELL 必须为 0)
   - "atk_spell_val": 直伤数值 (无为 0)
   - "def_spell_val": 护盾数值 (无为 0)
   - "tags": 词条列表 (如 ["RUSH", "DEGRADE_1"])
   - "flavor_text": 简短背景介绍或战术定位

严格输出合法 JSON 格式，如下:
{{
  "cards": [
    {{
      "name": "卡牌名",
      "card_type": "MINION",
      "cost": 3,
      "base_dp": 3,
      "atk_spell_val": 0,
      "def_spell_val": 0,
      "tags": ["RUSH"],
      "flavor_text": "战术描述"
    }}
  ]
}}
"""
        extra = {"thinking": {"type": "disabled"}} if "deepseek.com" in self.base_url.lower() else {}
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500,
            extra_body=extra if extra else None,
        )
        raw_content = response.choices[0].message.content or ""
        cleaned_json = clean_json_text(raw_content)
        data = json.loads(cleaned_json)
        new_cards_list = data.get("cards", [])

        if not new_cards_list:
            raise ValueError("DeepSeek 未返回有效的新卡列表")

        # Assign IDs and validate
        default_starts = {"Red": 100, "Blue": 200, "Green": 300, "Neutral": 900}
        def_start = default_starts.get(faction, 100)
        existing_faction_cards = cards_config.get(faction, [])

        approved_cards = []
        for card in new_cards_list:
            card_id = self._get_next_id(existing_faction_cards + approved_cards, def_start)
            c_type = card.get("card_type", "MINION").upper()
            cost = max(0, min(10, int(card.get("cost", 1))))
            base_dp = int(card.get("base_dp", 0)) if c_type == "MINION" else 0
            atk_spell = int(card.get("atk_spell_val", 0))
            def_spell = int(card.get("def_spell_val", 0))

            raw_tags = card.get("tags", [])
            valid_tags = [t.strip().upper() for t in raw_tags if is_legal_tag(t)]

            cost, _ = enforce_temp_mana_cost(c_type, cost, atk_spell, def_spell, valid_tags)

            final_card = {
                "id": card_id,
                "name": card.get("name", f"新卡_{card_id}"),
                "card_type": c_type,
                "cost": cost,
                "base_dp": base_dp,
                "atk_spell_val": atk_spell,
                "def_spell_val": def_spell,
                "tags": valid_tags,
                "factions": [faction],
                "flavor_text": card.get("flavor_text", "")
            }
            approved_cards.append(final_card)

        # Update cards_config.json
        if faction not in cards_config:
            cards_config[faction] = []
        cards_config[faction].extend(approved_cards)
        self.save_cards_config(cards_config)

        return {
            "success": True,
            "faction": faction,
            "generated_count": len(approved_cards),
            "cards": approved_cards
        }
