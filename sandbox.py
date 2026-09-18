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
    def_spell_val: int = 0
    tags: List[str] = field(default_factory=list)
    factions: List[str] = field(default_factory=list)
    is_token: bool = False


@dataclass
class MinionInstance:
    card: Card
    current_dp: int
    owner: int
    ready_to_attack: bool = False


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
    MAX_COPIES_PER_CARD = 3
    DECK_SIZE = 30

    def __init__(self, p0_faction: Faction = Faction.RED, p1_faction: Faction = Faction.BLUE, 
                 cards_path: str = "cards_config.json", p0_decklist=None, p1_decklist=None):
        self.cards_path = cards_path
        self.card_database = self._load_card_database()
        self.p0_faction = p0_faction
        self.p1_faction = p1_faction
        self.p0_decklist = p0_decklist
        self.p1_decklist = p1_decklist

        self.action_space_size = self.MAX_HAND_SIZE * 4 + 1
        self.reset()

    def _load_card_database(self) -> Dict[int, Card]:
        if not os.path.exists(self.cards_path):
            alt_candidates = [
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
                f_list = c.get("factions", [])
                if not f_list and faction_key != "Dual":
                    f_list = [faction_key]
                card = Card(
                    id=c["id"],
                    name=c["name"],
                    card_type=CardType(c["card_type"]),
                    cost=c["cost"],
                    base_dp=c.get("base_dp", 0),
                    atk_spell_val=c.get("atk_spell_val", 0),
                    def_spell_val=c.get("def_spell_val", 0),
                    tags=c.get("tags", []),
                    factions=f_list,
                    is_token=c.get("is_token", False)
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
            tags=list(c.tags),
            factions=list(c.factions),
            is_token=c.is_token
        )

    def _is_card_allowed_for_faction(self, c: Card, faction_code: int) -> bool:
        if c.is_token:
            return False
        f_target = "Red" if faction_code == 1 else ("Blue" if faction_code == 2 else "Green")
        if c.factions and (f_target in c.factions or "Neutral" in c.factions):
            return True
        c_f = c.id // 100
        if c_f == faction_code or c_f == 9:
            return True
        if c_f == 4 and faction_code in (1, 2):
            return True
        if c_f == 5 and faction_code in (2, 3):
            return True
        if c_f == 6 and faction_code in (1, 3):
            return True
        return False

    def _build_deck(self, faction: Faction, custom_decklist: Optional[List[int]] = None) -> List[Card]:
        faction_code = 1 if faction == Faction.RED else 2 if faction == Faction.BLUE else 3
        
        if custom_decklist:
            card_counts: Dict[int, int] = {}
            deck = []
            valid = True
            invalid_reasons = []

            for cid in custom_decklist:
                if cid not in self.card_database:
                    valid = False
                    invalid_reasons.append(f"卡牌 ID {cid} 不在数据库中")
                    break
                c = self.card_database[cid]
                if not self._is_card_allowed_for_faction(c, faction_code):
                    valid = False
                    invalid_reasons.append(f"卡牌 [{c.name}](ID:{cid}) 不属于该阵营可用池")
                    break
                card_counts[cid] = card_counts.get(cid, 0) + 1
                if card_counts[cid] > self.MAX_COPIES_PER_CARD:
                    valid = False
                    invalid_reasons.append(f"卡牌 [{c.name}] 超过同名卡上限")
                    break
                deck.append(self._clone_card(c))

            if valid and len(deck) == self.DECK_SIZE:
                random.shuffle(deck)
                return deck
            else:
                reason_str = " | ".join(invalid_reasons) if invalid_reasons else f"卡牌数量为 {len(deck)} 张"
                warn_key = (faction.value, reason_str)
                if not hasattr(self.__class__, '_warned_invalid_decks'):
                    self.__class__._warned_invalid_decks = set()
                if warn_key not in self.__class__._warned_invalid_decks:
                    print(f"\n[致命警告] 阵营 {faction.value} 预构筑卡组校验失败: {reason_str}")
                    print(f"[*] 系统正在执行随机兜底装配...")
                    self.__class__._warned_invalid_decks.add(warn_key)

        # 兜底装配：恢复纯随机抓取，不再按低费排序，防止造出极端快攻假想敌
        all_pool = [c for c in self.card_database.values() if self._is_card_allowed_for_faction(c, faction_code)]
        if not all_pool: return []
        
        card_counts: Dict[int, int] = {}
        deck = []
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
                    
        # 安全垫底：如果可用卡池总卡牌数极少，循环填满
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
        self.players[1].max_mana = 0
        self.players[1].mana = 0

        self.lanes = {
            0: Lane(lane_id=0),
            1: Lane(lane_id=1)
        }
        self.current_player = 0
        self.turn_count = 1

        for _ in range(3):
            self._draw_card(self.players[0])
        for _ in range(4):
            self._draw_card(self.players[1])

        coin = Card(id=996, name="幸运币", card_type=CardType.SPELL, cost=0, base_dp=0, atk_spell_val=0, def_spell_val=0, tags=["TEMP_MANA_1"], factions=["System"], is_token=True)
        if len(self.players[1].hand) < self.MAX_HAND_SIZE:
            self.players[1].hand.append(coin)

        self.last_bonus_score_pts = 0
        self.winner = None
        self.lane_spell_atk = np.zeros((2, 2), dtype=np.int32)
        self.lane_def_shield = np.zeros((2, 2), dtype=np.int32)
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
            opp.score += 1

    def _lane_has_enemy(self, lane_id: int) -> bool:
        opp_id = 1 - self.current_player
        lane = self.lanes[lane_id]
        return any(u.owner == opp_id for u in lane.attackers) or any(u.owner == opp_id for u in lane.defenders)

    def _lane_has_my_unit(self, lane_id: int) -> bool:
        lane = self.lanes[lane_id]
        return any(u.owner == self.current_player for u in lane.attackers) or any(u.owner == self.current_player for u in lane.defenders)

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space_size, dtype=np.float32)
        mask[-1] = 1.0

        curr = self.players[self.current_player]
        for idx, card in enumerate(curr.hand):
            if idx >= self.MAX_HAND_SIZE:
                break
            if card.cost <= curr.mana:
                base_idx = idx * 4
                is_atk_only = "ATTACK_ONLY" in card.tags
                is_spell = (card.card_type == CardType.SPELL and not any(t.startswith("SPAWN_") for t in card.tags))
                needs_sacrifice = "SACRIFICE_1_KILL_1" in card.tags
                is_atk_spell = card.atk_spell_val > 0
                is_def_spell = card.def_spell_val > 0

                for lane_id in [0, 1]:
                    lane = self.lanes[lane_id]
                    my_atks = sum(1 for u in lane.attackers if u.owner == curr.player_id)
                    my_defs = sum(1 for u in lane.defenders if u.owner == curr.player_id)
                    atk_slot = base_idx + (0 if lane_id == 0 else 2)
                    def_slot = base_idx + (1 if lane_id == 0 else 3)
                    
                    has_enemy = self._lane_has_enemy(lane_id)
                    has_my_atk = my_atks > 0
                    has_my_def = my_defs > 0
                    has_my = has_my_atk or has_my_def

                    can_play_atk_slot = False
                    can_play_def_slot = False

                    if is_spell:
                        if needs_sacrifice:
                            if has_enemy and has_my:
                                can_play_atk_slot = True
                                can_play_def_slot = not is_atk_only
                        else:
                            if is_atk_spell or is_def_spell:
                                if is_atk_spell and has_my_atk:
                                    can_play_atk_slot = True
                                if is_def_spell and has_my_def and not is_atk_only:
                                    can_play_def_slot = True
                            else:
                                can_play_atk_slot = True
                                can_play_def_slot = not is_atk_only
                    else:
                        if my_atks < self.MAX_LANE_UNITS:
                            can_play_atk_slot = True
                        if not is_atk_only and my_defs < self.MAX_LANE_UNITS:
                            can_play_def_slot = True
                            
                    if can_play_atk_slot:
                        mask[atk_slot] = 1.0
                    if can_play_def_slot:
                        mask[def_slot] = 1.0

        return mask

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        prev_p0_score = self.players[0].score
        prev_p1_score = self.players[1].score
        acting_player = self.current_player

        self.last_bonus_score_pts = 0
        done = False

        mask = self.get_action_mask()
        if action < 0 or action >= self.action_space_size or mask[action] <= 0.0:
            action = self.action_space_size - 1

        if action == self.action_space_size - 1:
            done = self._resolve_turn_end()
        else:
            hand_idx = action // 4
            pos_choice = action % 4
            lane_id = 0 if pos_choice in (0, 1) else 1
            is_attack = pos_choice in (0, 2)
            self._play_card(hand_idx, lane_id, is_attack)

        if not done:
            if self.players[0].score >= self.WIN_SCORE or self.players[1].score >= self.WIN_SCORE:
                done = True
            elif self.turn_count >= self.MAX_TURNS:
                done = True

        r0 = (self.players[0].score - prev_p0_score) - (self.players[1].score - prev_p1_score)
        if done:
            s0 = self.players[0].score
            s1 = self.players[1].score
            if s0 >= self.WIN_SCORE and s1 >= self.WIN_SCORE:
                if s0 > s1:
                    self.winner = 0
                elif s1 > s0:
                    self.winner = 1
                else:
                    self.winner = acting_player
            elif s0 >= self.WIN_SCORE:
                self.winner = 0
            elif s1 >= self.WIN_SCORE:
                self.winner = 1
            elif self.turn_count >= self.MAX_TURNS:
                if s0 > s1:
                    self.winner = 0
                elif s1 > s0:
                    self.winner = 1
                else:
                    self.winner = None  # 彻底消除平局偏心
            else:
                self.winner = None

            if self.winner == 0:
                r0 += 10.0
            elif self.winner == 1:
                r0 -= 10.0

        step_reward = r0 if acting_player == 0 else -r0
        return self.get_observation(), step_reward, done, {"bonus_score_pts": getattr(self, "last_bonus_score_pts", 0), "winner": self.winner}

    def _play_card(self, hand_idx: int, lane_id: int, is_attack: bool):
        player = self.players[self.current_player]
        card = player.hand.pop(hand_idx)
        player.mana -= card.cost
        lane = self.lanes[lane_id]

        if card.card_type == CardType.MINION:
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
                    self._apply_ramp(player, int(tag.split("_")[1]))
                elif tag.startswith("DISCARD_"):
                    for _ in range(int(tag.split("_")[1])):
                        if player.hand:
                            discarded = player.hand.pop(0)
                            player.graveyard.append(discarded)

            if card.atk_spell_val > 0:
                self.lane_spell_atk[lane_id, player.player_id] += card.atk_spell_val

            if card.def_spell_val > 0:
                self.lane_def_shield[lane_id, player.player_id] += card.def_spell_val

    def _add_mana_overload(self, player: Player):
        overload_card = Card(
            id=997,
            name="法力过载",
            card_type=CardType.SPELL,
            cost=0,
            base_dp=0,
            atk_spell_val=0,
            def_spell_val=0,
            tags=["DRAW_1"],
            factions=["System"],
            is_token=True
        )
        if len(player.hand) < self.MAX_HAND_SIZE:
            player.hand.append(overload_card)
        else:
            player.graveyard.append(overload_card)

    def _apply_ramp(self, player: Player, ramp_val: int):
        if player.max_mana >= 10:
            self._add_mana_overload(player)
        else:
            new_mana = player.max_mana + ramp_val
            if new_mana > 10:
                player.max_mana = 10
                self._add_mana_overload(player)
            else:
                player.max_mana = new_mana

    def _trigger_tags(self, card: Card, player: Player, lane_id: int, is_attack: bool):
        for tag in card.tags:
            if tag.startswith("DRAW_"):
                for _ in range(int(tag.split("_")[1])):
                    self._draw_card(player)
            elif tag.startswith("RAMP_"):
                self._apply_ramp(player, int(tag.split("_")[1]))
            elif tag.startswith("DISCARD_"):
                for _ in range(int(tag.split("_")[1])):
                    if player.hand:
                        discarded = player.hand.pop(0)
                        player.graveyard.append(discarded)
            elif tag.startswith("SPAWN_"):
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
                        tags=[],
                        factions=["System"],
                        is_token=True
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
                self._apply_ramp(owner_p, int(tag.split("_")[2]))

    def _resolve_turn_end(self) -> bool:
        curr_p = self.current_player
        opp_id = 1 - curr_p
        self.last_bonus_score_pts = 0

        for lane_id, lane in self.lanes.items():
            ready_attackers = [u for u in lane.attackers if u.owner == curr_p and u.ready_to_attack]
            for u in ready_attackers:
                lane.attackers.remove(u)

            if not ready_attackers:
                continue

            support_bonus = 0
            for def_unit in lane.defenders:
                if def_unit.owner == curr_p:
                    for tag in def_unit.card.tags:
                        if tag.startswith("SUPPORT_ATK_"):
                            support_bonus += int(tag.split("_")[2])

            spell_atk = self.lane_spell_atk[lane_id, curr_p]
            total_atk_dp = sum(u.current_dp for u in ready_attackers) + support_bonus + spell_atk
            total_degrade = 0
            bonus_score_pts = 0

            for atk in ready_attackers:
                for tag in atk.card.tags:
                    if tag.startswith("DEGRADE_"):
                        total_degrade += int(tag.split("_")[1])
                    elif tag.startswith("BONUS_SCORE_"):
                        bonus_score_pts += int(tag.split("_")[2])

            total_award = min(2, 1 + bonus_score_pts)
            remaining_atk = total_atk_dp
            breakthrough = True

            shield = self.lane_def_shield[lane_id, opp_id]
            if shield > 0:
                absorbed = min(shield, remaining_atk)
                remaining_atk -= absorbed
                self.lane_def_shield[lane_id, opp_id] -= absorbed
                if remaining_atk <= 0:
                    breakthrough = False

            opp_defenders = [u for u in lane.defenders if u.owner == opp_id]
            
            if opp_defenders and remaining_atk > 0:
                degrade_remaining = total_degrade
                for def_unit in list(opp_defenders):
                    if degrade_remaining <= 0:
                        break
                    reduction = min(degrade_remaining, def_unit.current_dp)
                    def_unit.current_dp = max(0, def_unit.current_dp - reduction)
                    degrade_remaining -= reduction
                    if def_unit.current_dp == 0:
                        lane.defenders.remove(def_unit)
                        self.players[opp_id].graveyard.append(def_unit.card)
                        self._trigger_deathrattle(def_unit)

                remaining_defenders = [u for u in lane.defenders if u.owner == opp_id]
                for def_unit in remaining_defenders:
                    if remaining_atk <= 0:
                        breakthrough = False
                        break

                    if remaining_atk >= def_unit.current_dp:
                        remaining_atk -= def_unit.current_dp
                        lane.defenders.remove(def_unit)
                        self.players[opp_id].graveyard.append(def_unit.card)
                        self._trigger_deathrattle(def_unit)
                        if remaining_atk == 0:
                            breakthrough = False
                            break
                    else:
                        def_unit.current_dp -= remaining_atk
                        remaining_atk = 0
                        breakthrough = False
                        break

            if breakthrough and remaining_atk > 0:
                self.players[curr_p].score += total_award
                if bonus_score_pts > 0:
                    self.last_bonus_score_pts += (total_award - 1)
                if self.players[curr_p].score >= self.WIN_SCORE:
                    return True

            for atk in ready_attackers:
                self.players[curr_p].graveyard.append(atk.card)
                self._trigger_deathrattle(atk)

        for lane_id in [0, 1]:
            self.lane_spell_atk[lane_id, curr_p] = 0
            self.lane_def_shield[lane_id, opp_id] = 0

        if any(p.score >= self.WIN_SCORE for p in self.players.values()):
            return True

        self.current_player = 1 - self.current_player
        self.turn_count += 1
        active_p = self.players[self.current_player]

        if active_p.max_mana < 10:
            active_p.max_mana += 1
        active_p.mana = active_p.max_mana
        self._draw_card(active_p)

        for lane in self.lanes.values():
            for u in lane.attackers:
                if u.owner == self.current_player:
                    u.ready_to_attack = True

        if any(p.score >= self.WIN_SCORE for p in self.players.values()):
            return True

        return False

    def get_observation(self) -> np.ndarray:
        # 完全恢复了 obs[0] 玩家状态和 obs[2] 场面状态，并扩展了 obs[1] 的 8 维视野
        obs = np.zeros((3, 13, 8), dtype=np.float32)
        curr = self.players[self.current_player]
        opp = self.players[1 - self.current_player]

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

        for i, card in enumerate(curr.hand[:7]):
            obs[1, i, 0] = card.cost / 10.0
            obs[1, i, 1] = card.base_dp / 10.0
            obs[1, i, 2] = 1.0 if card.card_type == CardType.MINION else 0.0
            obs[1, i, 3] = 1.0 if "RUSH" in card.tags else 0.0
            obs[1, i, 4] = 1.0 if any(t.startswith("DEGRADE") for t in card.tags) else 0.0
            obs[1, i, 5] = 1.0 if any(t.startswith("FORTIFY") for t in card.tags) else 0.0
            obs[1, i, 6] = 1.0 if any(t.startswith("RAMP") or t.startswith("TEMP_MANA") or t.startswith("DEATH_MANA") for t in card.tags) else 0.0
            obs[1, i, 7] = 1.0 if any(t.startswith("DRAW") or t.startswith("DEATH_DRAW") for t in card.tags) else 0.0

        for lane_idx, lane in self.lanes.items():
            slot_offset = lane_idx * 6
            my_atks = [u for u in lane.attackers if u.owner == curr.player_id]
            for j, u in enumerate(my_atks[:3]):
                obs[2, slot_offset + j, 0] = u.current_dp / 10.0
                obs[2, slot_offset + j, 1] = 1.0 if u.ready_to_attack else 0.0
            opp_defs = [u for u in lane.defenders if u.owner == opp.player_id]
            for j, u in enumerate(opp_defs[:3]):
                obs[2, slot_offset + 3 + j, 0] = u.current_dp / 10.0
                obs[2, slot_offset + 3 + j, 2] = 1.0
            my_defs = [u for u in lane.defenders if u.owner == curr.player_id]
            for j, u in enumerate(my_defs[:3]):
                obs[2, slot_offset + j, 3] = u.current_dp / 10.0
            opp_atks = [u for u in lane.attackers if u.owner == opp.player_id]
            for j, u in enumerate(opp_atks[:3]):
                obs[2, slot_offset + j, 4] = u.current_dp / 10.0
            
            obs[2, slot_offset, 5] = self.lane_spell_atk[lane_idx, curr.player_id] / 10.0
            obs[2, slot_offset, 6] = self.lane_def_shield[lane_idx, curr.player_id] / 10.0
            obs[2, slot_offset, 7] = self.lane_def_shield[lane_idx, opp.player_id] / 10.0

        return obs