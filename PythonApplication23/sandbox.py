import os
import json
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
import numpy as np


class Faction(Enum):
    RED = "Red"
    BLUE = "Blue"
    GREEN = "Green"
    NEUTRAL = "Neutral"


class CardType(Enum):
    MINION = "MINION"
    SPELL = "SPELL"


@dataclass
class Card:
    id: int
    name: str
    card_type: CardType
    cost: int
    base_dp: int
    atk_spell_val: int
    def_spell_val: int
    tags: List[str] = field(default_factory=list)


@dataclass
class MinionInstance:
    card: Card
    current_dp: int
    owner: int
    ready_to_attack: bool = False  # 蓄势标记：为 True 时方可参与合击冲锋


@dataclass
class Lane:
    lane_id: int
    attackers: List[MinionInstance] = field(default_factory=list)
    defenders: List[MinionInstance] = field(default_factory=list)


@dataclass
class Player:
    player_id: int
    faction: Faction
    score: int = 0
    mana: int = 1
    max_mana: int = 1
    deck: List[Card] = field(default_factory=list)
    hand: List[Card] = field(default_factory=list)
    graveyard: List[Card] = field(default_factory=list)
    fatigue: int = 0


class DuelEnv:
    WIN_SCORE = 7
    MAX_HAND_SIZE = 7
    MAX_LANE_UNITS = 3
    MAX_TURNS = 100
    MAX_COPIES_PER_CARD = 3  # 同名卡上限 3 张
    DECK_SIZE = 30           # 卡组规模 30 张

    def __init__(self, p0_faction: Faction = Faction.RED, p1_faction: Faction = Faction.BLUE, 
                 cards_path: str = "cards_config.json", p0_decklist=None, p1_decklist=None):
        self.cards_path = cards_path
        self.card_database = self._load_card_database()
        self.p0_faction = p0_faction
        self.p1_faction = p1_faction
        self.p0_decklist = p0_decklist
        self.p1_decklist = p1_decklist

        # 动作空间：7张手牌 * 4种打出位置 + 1个结束回合动作 = 29
        self.action_space_size = self.MAX_HAND_SIZE * 4 + 1
        self.reset()

    def _load_card_database(self) -> Dict[int, Card]:
        if not os.path.exists(self.cards_path):
            alt_candidates = [
                os.path.join("PythonApplication23", self.cards_path),
                os.path.join(os.path.dirname(__file__), os.path.basename(self.cards_path)),
                os.path.join(os.path.dirname(__file__), self.cards_path)
            ]
            found = False
            for alt in alt_candidates:
                if os.path.exists(alt):
                    self.cards_path = alt
                    found = True
                    break
            if not found:
                raise FileNotFoundError(f"未找到配置文件: {self.cards_path}")
        with open(self.cards_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        db = {}
        for faction_key, card_list in raw_data.items():
            for c in card_list:
                card = Card(
                    id=c["id"],
                    name=c["name"],
                    card_type=CardType(c["card_type"]),
                    cost=c["cost"],
                    base_dp=c.get("base_dp", 0),
                    atk_spell_val=c.get("atk_spell_val", 0),
                    def_spell_val=c.get("def_spell_val", 0),
                    tags=c.get("tags", [])
                )
                db[card.id] = card
        return db

    def _clone_card(self, c: Card) -> Card:
        return Card(
            id=c.id,
            name=c.name,
            card_type=c.card_type,
            cost=c.cost,
            base_dp=c.base_dp,
            atk_spell_val=c.atk_spell_val,
            def_spell_val=c.def_spell_val,
            tags=list(c.tags)
        )

    def _build_deck(self, faction: Faction, custom_decklist: Optional[List[int]] = None) -> List[Card]:
        """
        构建对战牌库：
        1. 若提供了 custom_decklist，直接装配；
        2. 否则从可用卡池构建，每张同名卡最多 3 张。
        """
        if custom_decklist:
            deck = []
            for cid in custom_decklist:
                if cid in self.card_database:
                    c = self.card_database[cid]
                    deck.append(self._clone_card(c))
            if len(deck) == self.DECK_SIZE:
                random.shuffle(deck)
                return deck

        faction_code = 1 if faction == Faction.RED else 2 if faction == Faction.BLUE else 3
        faction_pool = [c for c in self.card_database.values() if c.id // 100 == faction_code]
        neutral_pool = [c for c in self.card_database.values() if c.id // 100 == 9]
        all_pool = faction_pool + neutral_pool

        if not all_pool:
            return []

        card_counts: Dict[int, int] = {}
        deck = []

        # 候选池：仅保留未达到携带上限 (<= 3 张) 的卡牌，随机抽取装配
        pool_candidates = list(all_pool)
        while len(deck) < self.DECK_SIZE and pool_candidates:
            picked = random.choice(pool_candidates)
            c_count = card_counts.get(picked.id, 0)
            if c_count < self.MAX_COPIES_PER_CARD:
                deck.append(self._clone_card(picked))
                card_counts[picked.id] = c_count + 1
                if card_counts[picked.id] >= self.MAX_COPIES_PER_CARD:
                    pool_candidates.remove(picked)
            else:
                if picked in pool_candidates:
                    pool_candidates.remove(picked)

        # 安全垫底：如果可用卡池总卡牌数不足 30（如初期卡池过小），循环填满
        while len(deck) < self.DECK_SIZE and all_pool:
            picked = random.choice(all_pool)
            deck.append(self._clone_card(picked))

        random.shuffle(deck)
        return deck

    def reset(self) -> np.ndarray:
        self.players = {
            0: Player(player_id=0, faction=self.p0_faction, deck=self._build_deck(self.p0_faction, self.p0_decklist)),
            1: Player(player_id=1, faction=self.p1_faction, deck=self._build_deck(self.p1_faction, self.p1_decklist))
        }
        self.lanes = {
            0: Lane(lane_id=0),
            1: Lane(lane_id=1)
        }
        self.current_player = 0
        self.turn_count = 1

        # 初始抽牌
        for _ in range(3):
            self._draw_card(self.players[0])
        for _ in range(4):
            self._draw_card(self.players[1])

        # 后手幸运币补给
        coin = Card(id=901, name="幸运币", card_type=CardType.SPELL, cost=0, base_dp=0, atk_spell_val=0, def_spell_val=0, tags=["TEMP_MANA_1"])
        if len(self.players[1].hand) < self.MAX_HAND_SIZE:
            self.players[1].hand.append(coin)

        return self.get_observation()

    def _draw_card(self, player: Player):
        if len(player.deck) > 0:
            card = player.deck.pop(0)
            if len(player.hand) < self.MAX_HAND_SIZE:
                player.hand.append(card)
            else:
                player.graveyard.append(card)
        else:
            player.fatigue += 1
            opp = self.players[1 - player.player_id]
            opp.score += player.fatigue

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space_size, dtype=np.float32)
        mask[-1] = 1.0  # PASS 动作永远合法

        curr = self.players[self.current_player]
        for idx, card in enumerate(curr.hand):
            if idx >= self.MAX_HAND_SIZE:
                break
            if card.cost <= curr.mana:
                base_idx = idx * 4
                is_atk_only = "ATTACK_ONLY" in card.tags
                # 纯法术（非衍生召唤类）不占用随从格子空间
                is_spell = (card.card_type == CardType.SPELL and not any(t.startswith("SPAWN_") for t in card.tags))

                # 左路进攻区 / 防守区容量判定
                l_atks = sum(1 for u in self.lanes[0].attackers if u.owner == curr.player_id)
                l_defs = sum(1 for u in self.lanes[0].defenders if u.owner == curr.player_id)
                if is_spell or l_atks < self.MAX_LANE_UNITS:
                    mask[base_idx + 0] = 1.0
                if not is_atk_only and (is_spell or l_defs < self.MAX_LANE_UNITS):
                    mask[base_idx + 1] = 1.0

                # 右路进攻区 / 防守区容量判定
                r_atks = sum(1 for u in self.lanes[1].attackers if u.owner == curr.player_id)
                r_defs = sum(1 for u in self.lanes[1].defenders if u.owner == curr.player_id)
                if is_spell or r_atks < self.MAX_LANE_UNITS:
                    mask[base_idx + 2] = 1.0
                if not is_atk_only and (is_spell or r_defs < self.MAX_LANE_UNITS):
                    mask[base_idx + 3] = 1.0

        return mask

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        prev_p0_score = self.players[0].score
        prev_p1_score = self.players[1].score
        acting_player = self.current_player  # 记录当前执行动作的玩家

        done = False
        if action == self.action_space_size - 1:
            # 结束回合 -> 触发同路合击冲锋结算与轮换
            done = self._resolve_turn_end()
        else:
            hand_idx = action // 4
            pos_choice = action % 4
            lane_id = 0 if pos_choice in (0, 1) else 1
            is_attack = pos_choice in (0, 2)
            self._play_card(hand_idx, lane_id, is_attack)

        # 获胜检查
        if not done:
            if self.players[0].score >= self.WIN_SCORE or self.players[1].score >= self.WIN_SCORE:
                done = True
            elif self.turn_count >= self.MAX_TURNS:
                done = True

        # 计算 P0 收益 (零和博弈)
        r0 = (self.players[0].score - prev_p0_score) - (self.players[1].score - prev_p1_score)
        if done:
            if self.players[0].score >= self.WIN_SCORE:
                r0 += 10.0
            elif self.players[1].score >= self.WIN_SCORE:
                r0 -= 10.0

        # 准确根据行动方 acting_player 返回奖励，避免换边轮换后将正向收益逆转为负惩罚
        step_reward = r0 if acting_player == 0 else -r0
        return self.get_observation(), step_reward, done, {}

    def _play_card(self, hand_idx: int, lane_id: int, is_attack: bool):
        player = self.players[self.current_player]
        card = player.hand.pop(hand_idx)
        player.mana -= card.cost
        lane = self.lanes[lane_id]

        if card.card_type == CardType.MINION:
            # 拥有 RUSH 词条可当回合冲锋，否则需蓄势一轮
            can_rush = "RUSH" in card.tags
            instance = MinionInstance(
                card=card,
                current_dp=card.base_dp,
                owner=player.player_id,
                ready_to_attack=can_rush
            )

            if is_attack:
                lane.attackers.append(instance)
            else:
                for tag in card.tags:
                    if tag.startswith("FORTIFY_"):
                        bonus = int(tag.split("_")[1])
                        instance.current_dp += bonus
                lane.defenders.append(instance)

            self._trigger_tags(card, player, lane_id, is_attack)

        elif card.card_type == CardType.SPELL:
            player.graveyard.append(card)
            opp_id = 1 - player.player_id

            if "SACRIFICE_1_KILL_1" in card.tags:
                my_units = [u for u in lane.defenders if u.owner == player.player_id] + [u for u in lane.attackers if u.owner == player.player_id]
                opp_units = [u for u in lane.defenders if u.owner == opp_id] + [u for u in lane.attackers if u.owner == opp_id]
                if my_units and opp_units:
                    sac = my_units[0]
                    tgt = opp_units[0]
                    if sac in lane.defenders:
                        lane.defenders.remove(sac)
                    elif sac in lane.attackers:
                        lane.attackers.remove(sac)
                    if tgt in lane.defenders:
                        lane.defenders.remove(tgt)
                    elif tgt in lane.attackers:
                        lane.attackers.remove(tgt)
                    player.graveyard.append(sac.card)
                    self.players[opp_id].graveyard.append(tgt.card)
                    self._trigger_deathrattle(sac)
                    self._trigger_deathrattle(tgt)

            for tag in card.tags:
                if tag.startswith("TEMP_MANA_"):
                    player.mana += int(tag.split("_")[2])
                elif tag.startswith("DRAW_"):
                    for _ in range(int(tag.split("_")[1])):
                        self._draw_card(player)
                elif tag.startswith("RAMP_"):
                    player.max_mana = min(10, player.max_mana + int(tag.split("_")[1]))

            # 直伤削弱法术
            if card.atk_spell_val > 0:
                opp_defs = [u for u in lane.defenders if u.owner == opp_id]
                if opp_defs:
                    opp_defs[0].current_dp = max(0, opp_defs[0].current_dp - card.atk_spell_val)
                    if opp_defs[0].current_dp == 0:
                        dead = opp_defs.pop(0)
                        lane.defenders.remove(dead)
                        self.players[opp_id].graveyard.append(dead.card)
                        self._trigger_deathrattle(dead)

    def _trigger_tags(self, card: Card, player: Player, lane_id: int, is_attack: bool):
        for tag in card.tags:
            if tag.startswith("DRAW_"):
                for _ in range(int(tag.split("_")[1])):
                    self._draw_card(player)
            elif tag.startswith("RAMP_"):
                player.max_mana = min(10, player.max_mana + int(tag.split("_")[1]))
            elif tag.startswith("DISCARD_"):
                for _ in range(int(tag.split("_")[1])):
                    if player.hand:
                        discarded = player.hand.pop(0)
                        player.graveyard.append(discarded)
            elif tag.startswith("SPAWN_"):
                # 召唤衍生小兵（严格受限于 MAX_LANE_UNITS = 3 上限）
                parts = tag.split("_")
                spawn_dp = int(parts[1])
                spawn_count = int(parts[2])

                target_lane = self.lanes[lane_id]
                container = target_lane.attackers if is_attack else target_lane.defenders

                for _ in range(spawn_count):
                    current_my_units = sum(1 for u in container if u.owner == player.player_id)
                    if current_my_units >= self.MAX_LANE_UNITS:
                        break

                    token_card = Card(
                        id=999,
                        name=f"{card.name}的小兵",
                        card_type=CardType.MINION,
                        cost=0,
                        base_dp=spawn_dp,
                        atk_spell_val=0,
                        def_spell_val=0,
                        tags=[]
                    )
                    can_token_rush = "RUSH" in card.tags
                    container.append(MinionInstance(
                        card=token_card,
                        current_dp=spawn_dp,
                        owner=player.player_id,
                        ready_to_attack=can_token_rush
                    ))

    def _trigger_deathrattle(self, minion: MinionInstance):
        owner_p = self.players[minion.owner]
        for tag in minion.card.tags:
            if tag.startswith("DEATH_DRAW_"):
                for _ in range(int(tag.split("_")[2])):
                    self._draw_card(owner_p)
            elif tag.startswith("DEATH_MANA_"):
                owner_p.max_mana = min(10, owner_p.max_mana + int(tag.split("_")[2]))

    def _resolve_turn_end(self) -> bool:
        """回合结束：同路已就绪单位发起合击冲锋"""
        curr_p = self.current_player
        opp_id = 1 - curr_p

        for lane_id, lane in self.lanes.items():
            # 筛选当前玩家中【已就绪】的冲锋单位
            ready_attackers = [u for u in lane.attackers if u.owner == curr_p and u.ready_to_attack]
            for u in ready_attackers:
                lane.attackers.remove(u)

            if not ready_attackers:
                continue

            # 统计己方防守区的 SUPPORT_ATK_X 增益
            support_bonus = 0
            for def_unit in lane.defenders:
                if def_unit.owner == curr_p:
                    for tag in def_unit.card.tags:
                        if tag.startswith("SUPPORT_ATK_"):
                            support_bonus += int(tag.split("_")[2])

            # 汇总总冲锋战力、削弱总值、加分总值
            total_atk_dp = sum(u.current_dp for u in ready_attackers) + support_bonus
            total_degrade = 0
            bonus_score_pts = 0

            for atk in ready_attackers:
                for tag in atk.card.tags:
                    if tag.startswith("DEGRADE_"):
                        total_degrade += int(tag.split("_")[1])
                    elif tag.startswith("BONUS_SCORE_"):
                        bonus_score_pts += int(tag.split("_")[2])

            total_award = 1 + bonus_score_pts

            # 碰撞判定
            opp_defenders = [u for u in lane.defenders if u.owner == opp_id]
            if opp_defenders:
                def_unit = opp_defenders[0]

                # 削弱词条生效
                if total_degrade > 0:
                    def_unit.current_dp = max(0, def_unit.current_dp - total_degrade)

                # 突破阈值判定
                if def_unit.current_dp == 0 or total_atk_dp > def_unit.current_dp:
                    lane.defenders.remove(def_unit)
                    self.players[opp_id].graveyard.append(def_unit.card)
                    self._trigger_deathrattle(def_unit)

                    self.players[curr_p].score += total_award
                    if self.players[curr_p].score >= self.WIN_SCORE:
                        return True
            else:
                # 空场突破
                self.players[curr_p].score += total_award
                if self.players[curr_p].score >= self.WIN_SCORE:
                    return True

            # 冲锋完毕进墓地
            for atk in ready_attackers:
                self.players[curr_p].graveyard.append(atk.card)
                self._trigger_deathrattle(atk)

        if any(p.score >= self.WIN_SCORE for p in self.players.values()):
            return True

        # 换边轮换与补给
        self.current_player = 1 - self.current_player
        self.turn_count += 1
        active_p = self.players[self.current_player]

        if active_p.max_mana < 10:
            active_p.max_mana += 1
        active_p.mana = active_p.max_mana
        self._draw_card(active_p)

        # 唤醒当前行动方场上蓄势完毕的单位
        for lane in self.lanes.values():
            for u in lane.attackers:
                if u.owner == self.current_player:
                    u.ready_to_attack = True

        if any(p.score >= self.WIN_SCORE for p in self.players.values()):
            return True

        return False

    def get_observation(self) -> np.ndarray:
        """状态特征张量输出: shape=(3, 13, 5)"""
        obs = np.zeros((3, 13, 5), dtype=np.float32)
        curr = self.players[self.current_player]
        opp = self.players[1 - self.current_player]

        # 玩家状态
        obs[0, 0, 0] = curr.mana / 10.0
        obs[0, 0, 1] = curr.max_mana / 10.0
        obs[0, 0, 2] = curr.score / 7.0
        obs[0, 0, 3] = len(curr.hand) / 7.0
        obs[0, 0, 4] = len(curr.deck) / 30.0

        obs[0, 1, 0] = opp.mana / 10.0
        obs[0, 1, 1] = opp.max_mana / 10.0
        obs[0, 1, 2] = opp.score / 7.0
        obs[0, 1, 3] = len(opp.hand) / 7.0
        obs[0, 1, 4] = len(opp.deck) / 30.0

        # 手牌特征
        for i, card in enumerate(curr.hand[:7]):
            obs[1, i, 0] = card.cost / 10.0
            obs[1, i, 1] = card.base_dp / 10.0
            obs[1, i, 2] = 1.0 if card.card_type == CardType.MINION else 0.0
            obs[1, i, 3] = 1.0 if "RUSH" in card.tags else 0.0
            obs[1, i, 4] = 1.0 if any(t.startswith("DEGRADE") for t in card.tags) else 0.0

        # 场面驻守特征: 完整编码战场 4 个区域 (己方进攻、敌方防守、己方防守、敌方蓄势)
        for lane_idx, lane in self.lanes.items():
            slot_offset = lane_idx * 6
            # 己方进攻区
            my_atks = [u for u in lane.attackers if u.owner == curr.player_id]
            for j, u in enumerate(my_atks[:3]):
                obs[2, slot_offset + j, 0] = u.current_dp / 10.0
                obs[2, slot_offset + j, 1] = 1.0 if u.ready_to_attack else 0.0
            # 敌方防守区
            opp_defs = [u for u in lane.defenders if u.owner == opp.player_id]
            for j, u in enumerate(opp_defs[:3]):
                obs[2, slot_offset + 3 + j, 0] = u.current_dp / 10.0
                obs[2, slot_offset + 3 + j, 2] = 1.0
            # 己方防守区 (补全: 通道3)
            my_defs = [u for u in lane.defenders if u.owner == curr.player_id]
            for j, u in enumerate(my_defs[:3]):
                obs[2, slot_offset + j, 3] = u.current_dp / 10.0
            # 敌方蓄势进攻区 (补全: 通道4)
            opp_atks = [u for u in lane.attackers if u.owner == opp.player_id]
            for j, u in enumerate(opp_atks[:3]):
                obs[2, slot_offset + j, 4] = u.current_dp / 10.0

        return obs