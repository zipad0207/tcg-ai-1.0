import os
import sys
import json
import torch

# Ensure we can import from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from sandbox import DuelEnv, Faction
from agent import CardNet

class GameSession:
    def __init__(self, p0_faction="Red", p1_faction="Blue", p0_decklist=None, p1_decklist=None, mode="pve", started=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        f_map = {"Red": Faction.RED, "Blue": Faction.BLUE, "Green": Faction.GREEN}
        self.p0_faction = f_map.get(p0_faction, Faction.RED)
        self.p1_faction = f_map.get(p1_faction, Faction.BLUE)
        self.p0_decklist = p0_decklist
        self.p1_decklist = p1_decklist
        self.mode = mode
        self.started = started
        
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.cards_path = os.path.join(self.root_dir, "cards_config.json")
        self.env = DuelEnv(
            p0_faction=self.p0_faction, 
            p1_faction=self.p1_faction, 
            cards_path=self.cards_path,
            p0_decklist=self.p0_decklist,
            p1_decklist=self.p1_decklist
        )
        
        model_path = os.path.join(self.root_dir, "card_ppo_model_brawl.pth")
        self.model = CardNet(action_dim=self.env.action_space_size).to(self.device)
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        else:
            self.model = None

        self.logs = []
        if self.started:
            self.reset(p0_faction, p1_faction, p0_decklist, p1_decklist, mode)
        else:
            self.done = False
            self.obs = None
            self.logs = ["【演武场待命中】双方构筑准备就绪。点击上方【⚔️ 开始对局】进入战斗！"]

    def get_card_images_map(self):
        """Loads card image mappings dynamically from cards_config.json."""
        images = {}
        if os.path.exists(self.cards_path):
            try:
                with open(self.cards_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for faction, cards in data.items():
                        if isinstance(cards, list):
                            for c in cards:
                                if "id" in c and c.get("image_url"):
                                    images[c["id"]] = c["image_url"]
            except Exception:
                pass
        return images

    def reset(self, p0_faction=None, p1_faction=None, p0_decklist=None, p1_decklist=None, mode=None):
        self.started = True
        f_map = {"Red": Faction.RED, "Blue": Faction.BLUE, "Green": Faction.GREEN}
        if p0_faction:
            self.p0_faction = f_map.get(p0_faction, self.p0_faction)
        if p1_faction:
            self.p1_faction = f_map.get(p1_faction, self.p1_faction)
        if p0_decklist is not None:
            self.p0_decklist = p0_decklist
        if p1_decklist is not None:
            self.p1_decklist = p1_decklist
        if mode:
            self.mode = mode

        self.env = DuelEnv(
            p0_faction=self.p0_faction,
            p1_faction=self.p1_faction,
            cards_path=self.cards_path,
            p0_decklist=self.p0_decklist,
            p1_decklist=self.p1_decklist
        )
        self.obs = self.env.reset()
        self.done = False
        self.p0_initial_decklist = [int(c.id) for c in self.env.players[0].deck] + [int(c.id) for c in self.env.players[0].hand]
        self.p1_initial_decklist = [int(c.id) for c in self.env.players[1].deck] + [int(c.id) for c in self.env.players[1].hand]
        
        f0_name = "红方" if self.p0_faction == Faction.RED else ("蓝方" if self.p0_faction == Faction.BLUE else "绿方")
        f1_name = "红方" if self.p1_faction == Faction.RED else ("蓝方" if self.p1_faction == Faction.BLUE else "绿方")
        mode_name = "玩家对战 AI" if self.mode == "pve" else "AI 全自动推演"
        deck_info = " (自选卡组)" if self.p0_decklist else ""
        self.logs = [f"对局初始化完成【{mode_name}】：P0 {f0_name}{deck_info} VS P1 {f1_name} (AI)"]
        return self.get_state_dict()

    def format_action_desc(self, action_id, player_id):
        p_faction = self.env.players[player_id].faction
        f_name = "红方" if p_faction == Faction.RED else ("蓝方" if p_faction == Faction.BLUE else "绿方")
        role = "玩家" if (player_id == 0 and self.mode == "pve") else f"AI(P{player_id})"
        p_name = f"【{f_name} {role}】"
        if action_id == self.env.action_space_size - 1:
            return f"{p_name} 结束回合 (PASS)，发起全军交战冲锋！"
            
        hand_idx = action_id // 4
        pos_choice = action_id % 4
        lane_str = "【左路】" if pos_choice in (0, 1) else "【右路】"
        curr_hand = self.env.players[player_id].hand
        
        if hand_idx >= len(curr_hand):
            return f"{p_name} 做出未知指令 ({action_id})"
            
        card = curr_hand[hand_idx]
        if card.card_type.name == "SPELL":
            return f"{p_name} 释放法术 [{card.name} (费用:{card.cost})] 到 {lane_str}"
        else:
            pos_str = "进攻区" if pos_choice in (0, 2) else "防守区"
            return f"{p_name} 部署随从 [{card.name} (费用:{card.cost} 战力:{card.base_dp})] 到 {lane_str}{pos_str}"

    def step(self, action: int):
        if self.done:
            return self.get_state_dict()
            
        curr_p = self.env.current_player
        desc = self.format_action_desc(action, curr_p)
        self.logs.append(desc)
        
        prev_s0 = self.env.players[0].score
        prev_s1 = self.env.players[1].score
        
        self.obs, reward, self.done, info = self.env.step(action)
        
        # 记录献祭斩杀/舍身自爆事件日志
        sac_ev = info.get("sacrifice_event") if isinstance(info, dict) else None
        if sac_ev:
            lane_name = "【左路】" if sac_ev["lane_id"] == 0 else "【右路】"
            if sac_ev["is_self_destruct"]:
                self.logs.append(f"🩸 舍身引爆！{lane_name}【{sac_ev['sac_name']}】引爆自身，同归于尽消灭敌方大将【{sac_ev['tgt_name']}】(战力:{sac_ev['tgt_dp']})！")
            else:
                self.logs.append(f"🩸 献祭斩杀！{lane_name}牺牲己方单位【{sac_ev['sac_name']}】(战力:{sac_ev['sac_dp']})，精准抹杀敌方高危单位【{sac_ev['tgt_name']}】(战力:{sac_ev['tgt_dp']})！")

        # If score changed during clash, add a clash summary log
        cur_s0 = self.env.players[0].score
        cur_s1 = self.env.players[1].score
        if action == self.env.action_space_size - 1:
            diff0 = cur_s0 - prev_s0
            diff1 = cur_s1 - prev_s1
            if diff0 > 0 or diff1 > 0:
                self.logs.append(f"💥 冲锋交战结算：红方+{diff0}分，蓝方+{diff1}分！当前比分: {cur_s0} : {cur_s1}")
            else:
                self.logs.append("🛡️ 双方防线稳固，本轮交战未产生比分突破。")

        if self.done:
            if self.env.winner == 0:
                self.logs.append("🏆 演武结束！红方 (Player 0) 斩获胜利！")
            elif self.env.winner == 1:
                self.logs.append("🏆 演武结束！蓝方 AI (Player 1) 斩获胜利！")
            else:
                self.logs.append("⚖️ 演武结束！双方势均力敌，战成平局！")

        if len(self.logs) > 15:
            self.logs = self.logs[-15:]
            
        return self.get_state_dict()

    def get_ai_action(self):
        if self.done:
            return None
        mask = self.env.get_action_mask()
        if self.model is None:
            valid_actions = [i for i, m in enumerate(mask) if m > 0.5]
            import random
            return random.choice(valid_actions)
            
        state_t = torch.FloatTensor(self.obs).unsqueeze(0).to(self.device)
        mask_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
        curr_p_id = self.env.current_player
        player = self.env.players[curr_p_id]
        faction = player.faction

        with torch.no_grad():
            logits, val = self.model(state_t, mask_t)
            logits = logits.squeeze(0).clone()

            # 阵营战术个性与防守感知注入 (Tactical Heuristics)
            opp_id = 1 - curr_p_id
            for hand_idx, card in enumerate(player.hand[:7]):
                for pos_choice in range(4):
                    action_id = hand_idx * 4 + pos_choice
                    if mask[action_id] <= 0.5:
                        continue
                    lane_id = 0 if pos_choice in (0, 1) else 1
                    is_def = pos_choice in (1, 3)

                    opp_atks = [u for u in self.env.lanes[lane_id].attackers if u.owner == opp_id]
                    my_defs = [u for u in self.env.lanes[lane_id].defenders if u.owner == curr_p_id]
                    opp_atk_dp = sum(u.current_dp for u in opp_atks)
                    my_def_dp = sum(u.current_dp for u in my_defs)

                    bonus = 0.0

                    # 1. 蓝色阵营 (蔚蓝 - 控制/防守/坚守) 专属防守觉醒
                    if faction == Faction.BLUE:
                        if card.card_type.name == "MINION":
                            has_fortify = any(t.startswith("FORTIFY_") for t in card.tags)
                            has_support = any(t.startswith("SUPPORT_ATK_") for t in card.tags)

                            # 坚守词条：防守区战力有额外加成，优先驻防
                            if is_def and has_fortify:
                                bonus += 2.2
                            if is_def and has_support:
                                bonus += 1.8

                            # 敌情感知：若该路敌军战力高于我方防御，优先立盾防守阻截
                            if is_def and opp_atk_dp > my_def_dp:
                                deficit = opp_atk_dp - my_def_dp
                                bonus += min(3.0, 1.2 + deficit * 0.5)

                            # 杜绝空门：敌方在该路有冲锋威胁但我方防守区完全无人时，紧急驻守
                            if is_def and opp_atk_dp > 0 and len(my_defs) == 0:
                                bonus += 2.5

                            # 攻守平衡：如果该路我方防御已经很充裕，且自身有突袭能力，可尝试进攻
                            if not is_def and my_def_dp > opp_atk_dp and "RUSH" in card.tags:
                                bonus += 1.0
                        elif card.card_type.name == "SPELL":
                            # 防护法术优先加给受威胁战线
                            if card.def_spell_val > 0 and opp_atk_dp > my_def_dp:
                                bonus += 1.8

                    # 2. 绿色阵营 (翠绿 - 跳费/膨胀/中后期随从)
                    elif faction == Faction.GREEN:
                        if card.card_type.name == "SPELL":
                            # 早期优先跳费充能
                            if any(t.startswith("RAMP_") or t.startswith("TEMP_MANA_") for t in card.tags) and player.max_mana < 6:
                                bonus += 2.0
                        elif card.card_type.name == "MINION":
                            # 敌方进攻且我方防御为空时，做基础防守阻挡
                            if is_def and opp_atk_dp > my_def_dp and len(my_defs) == 0:
                                bonus += 1.5

                    # 3. 红色阵营 (赤红 - 快攻/突袭/破甲冲锋)
                    elif faction == Faction.RED:
                        if card.card_type.name == "MINION":
                            if not is_def and "RUSH" in card.tags:
                                bonus += 1.8
                            if not is_def and any(t.startswith("DEGRADE_") for t in card.tags):
                                bonus += 1.2

                    logits[action_id] += bonus

            # 再次通过 mask 抑制非法动作
            logits = logits - (1.0 - torch.FloatTensor(mask).to(self.device)) * 1e9
            action = torch.argmax(logits, dim=-1).item()
        return action


    def get_state_dict(self):
        if not getattr(self, "started", True):
            return {
                "is_started": False,
                "mode": self.mode,
                "turn": 0,
                "current_player": -1,
                "done": False,
                "winner": None,
                "logs": ["【演武场待命中】配置双方构筑后，点击上方【开始对局】进入战斗！"],
                "p0_faction": str(self.p0_faction.value),
                "p1_faction": str(self.p1_faction.value),
                "is_custom_deck": bool(self.p0_decklist),
                "p0_initial_decklist": getattr(self, "p0_initial_decklist", []),
                "p1_initial_decklist": getattr(self, "p1_initial_decklist", []),
                "p0": {
                    "score": 0, "mana": 0, "max_mana": 0, "faction": str(self.p0_faction.value),
                    "hand_count": 0, "deck_count": 30, "graveyard_count": 0,
                    "hand": [], "deck": [], "graveyard": []
                },
                "p1": {
                    "score": 0, "mana": 0, "max_mana": 0, "faction": str(self.p1_faction.value),
                    "hand_count": 0, "deck_count": 30, "graveyard_count": 0,
                    "hand": [], "deck": [], "graveyard": []
                },
                "lanes": [
                    {"p0_attackers": [], "p1_attackers": [], "p0_defenders": [], "p1_defenders": []},
                    {"p0_attackers": [], "p1_attackers": [], "p0_defenders": [], "p1_defenders": []}
                ],
                "action_mask": [0.0] * self.env.action_space_size
            }

        img_map = self.get_card_images_map()

        def serialize_player(p_id):
            p = self.env.players[p_id]
            return {
                "score": int(p.score), "mana": int(p.mana), "max_mana": int(p.max_mana), "faction": str(p.faction.value),
                "hand_count": int(len(p.hand)),
                "deck_count": int(len(p.deck)),
                "graveyard_count": int(len(p.graveyard)),
                "hand": [{
                    "id": int(c.id),
                    "name": str(c.name),
                    "cost": int(c.cost),
                    "dp": int(c.base_dp),
                    "type": str(c.card_type.name),
                    "tags": list(c.tags),
                    "atk_spell_val": int(getattr(c, "atk_spell_val", 0)),
                    "def_spell_val": int(getattr(c, "def_spell_val", 0)),
                    "image_url": img_map.get(c.id, "")
                } for c in p.hand],
                "deck": [{
                    "id": int(c.id),
                    "name": str(c.name),
                    "cost": int(c.cost),
                    "dp": int(c.base_dp),
                    "type": str(c.card_type.name),
                    "tags": list(c.tags),
                    "atk_spell_val": int(getattr(c, "atk_spell_val", 0)),
                    "def_spell_val": int(getattr(c, "def_spell_val", 0)),
                    "image_url": img_map.get(c.id, "")
                } for c in p.deck],
                "graveyard": [{
                    "id": int(c.id),
                    "name": str(c.name),
                    "cost": int(c.cost),
                    "dp": int(c.base_dp),
                    "type": str(c.card_type.name),
                    "tags": list(c.tags),
                    "image_url": img_map.get(c.id, "")
                } for c in p.graveyard]
            }

        def serialize_units(units):
            return [{
                "id": int(u.card.id),
                "name": str(u.card.name),
                "dp": int(u.current_dp),
                "ready": bool(u.ready_to_attack),
                "tags": list(u.card.tags),
                "atk_spell_val": int(getattr(u.card, "atk_spell_val", 0)),
                "def_spell_val": int(getattr(u.card, "def_spell_val", 0)),
                "image_url": img_map.get(u.card.id, "")
            } for u in units]

        state = {
            "is_started": True,
            "mode": self.mode,
            "turn": int(self.env.turn_count),
            "current_player": int(self.env.current_player),
            "done": bool(self.done),
            "winner": int(self.env.winner) if self.env.winner is not None else None,
            "logs": list(self.logs),
            "p0_faction": str(self.p0_faction.value),
            "p1_faction": str(self.p1_faction.value),
            "is_custom_deck": bool(self.p0_decklist),
            "p0_initial_decklist": getattr(self, "p0_initial_decklist", []),
            "p1_initial_decklist": getattr(self, "p1_initial_decklist", []),
            "p0": serialize_player(0),
            "p1": serialize_player(1),
            "lanes": [
                {
                    "p0_attackers": serialize_units([u for u in self.env.lanes[i].attackers if u.owner == 0]),
                    "p1_attackers": serialize_units([u for u in self.env.lanes[i].attackers if u.owner == 1]),
                    "p0_defenders": serialize_units([u for u in self.env.lanes[i].defenders if u.owner == 0]),
                    "p1_defenders": serialize_units([u for u in self.env.lanes[i].defenders if u.owner == 1]),
                } for i in range(2)
            ],
            "action_mask": [float(x) for x in self.env.get_action_mask()]
        }
        return state
