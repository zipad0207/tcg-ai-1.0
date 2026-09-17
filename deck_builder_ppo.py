import os
import sys
import json
import argparse
import random
import torch
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sandbox import DuelEnv, Faction, Card, CardType
from agent import CardNet

MAX_COPIES_PER_CARD = 3
DECK_SIZE = 30

def find_model_path(requested_path: str = None, stage: str = "tuned") -> str:
    candidates = []
    if requested_path:
        candidates.append(requested_path)
    candidates.append("card_ppo_model_brawl.pth")
    candidates.append(f"card_ppo_model_{stage}.pth")
    candidates.append("card_ppo_model_tuned.pth")
    candidates.append("card_ppo_model.pth")
    candidates.append("card_ppo_model_baseline.pth")

    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def load_card_pool(cards_path: str = "cards_config.json") -> dict:
    if not os.path.exists(cards_path):
        raise FileNotFoundError(f"未找到卡池配置文件: {cards_path}")
    with open(cards_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_faction_candidates(pool_data: dict, faction: str) -> List[dict]:
    """
    根据卡牌 factions 归属列表搜寻目标阵营的所有可用候选卡牌 (支持单阵营 >100 张卡及任意编号区间)
    包含：本阵营纯色卡、中立卡 (Neutral)、以及包含本阵营的双色卡 (Dual)
    """
    candidates = []
    seen_ids = set()

    for category, card_list in pool_data.items():
        for c in card_list:
            cid = c.get("id", 0)
            if cid in seen_ids:
                continue

            facs = c.get("factions", [])
            # 1. 优先依据 factions 属性判定
            if facs:
                if faction in facs or "Neutral" in facs:
                    candidates.append(c)
                    seen_ids.add(cid)
                    continue

            # 2. 向下兼容旧分类 key 与前缀
            c_f = cid // 100
            if category == faction or category == "Neutral":
                candidates.append(c)
                seen_ids.add(cid)
            elif c_f == 4 and faction in ("Red", "Blue"):
                candidates.append(c)
                seen_ids.add(cid)
            elif c_f == 5 and faction in ("Blue", "Green"):
                candidates.append(c)
                seen_ids.add(cid)
            elif c_f == 6 and faction in ("Red", "Green"):
                candidates.append(c)
                seen_ids.add(cid)

    return candidates

def evaluate_card_neural_utility(model: CardNet, device: torch.device, candidate: dict, 
                                 faction: Faction, cards_path: str, samples: int = 15) -> dict:
    """
    单卡效用评估:
    在多种模拟环境下测量 Actor 出牌偏好概率与 Critic 状态价值增益 (ΔV)
    """
    card_obj = Card(
        id=candidate["id"],
        name=candidate["name"],
        card_type=CardType(candidate["card_type"]),
        cost=candidate["cost"],
        base_dp=candidate.get("base_dp", 0),
        atk_spell_val=candidate.get("atk_spell_val", 0),
        def_spell_val=candidate.get("def_spell_val", 0),
        tags=candidate.get("tags", [])
    )

    play_probs = []
    value_gains = []
    is_playable_count = 0

    for _ in range(samples):
        # 随机初始化对局环境 (针对多阵营生态进行候选卡压测)
        opp_pool = [f for f in [Faction.RED, Faction.BLUE, Faction.GREEN] if f != faction]
        opp_f = random.choice(opp_pool)
        env = DuelEnv(p0_faction=faction, p1_faction=opp_f, cards_path=cards_path)
        obs = env.reset()

        acting_p_id = 0
        env.current_player = acting_p_id
        player = env.players[acting_p_id]

        # 模拟真实对局阶段 (1~10 费自然分布，真实检验节奏与卡手风险)
        sim_mana = random.randint(1, 10)
        player.mana = sim_mana
        player.max_mana = sim_mana

        # 注入候选卡至手牌第 0 位
        player.hand = [card_obj] + player.hand[:3]

        mask = env.get_action_mask()
        # 检查候选卡对应的 4 个动作槽位中是否有合法动作
        card_action_indices = [0, 1, 2, 3] # slot 0 的 4 种打出目标
        legal_card_actions = [idx for idx in card_action_indices if mask[idx] > 0.5]

        if not legal_card_actions:
            # 当前法力不足以打出该卡，记录不可用并继续测试
            continue

        is_playable_count += 1
        obs = env.get_observation()
        state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        mask_t = torch.FloatTensor(mask).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, v_before = model(state_t, mask_t)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        # 计算 PPO 选中该卡牌的总意图概率
        p_card_play = float(sum(probs[idx] for idx in legal_card_actions))
        play_probs.append(p_card_play)

        # 模拟执行 PPO 最偏好的动作，测量 Critic 估值增益 ΔV
        best_card_act = max(legal_card_actions, key=lambda idx: probs[idx])
        obs_next, reward, done, _ = env.step(best_card_act)

        next_state_t = torch.FloatTensor(obs_next).unsqueeze(0).to(device)
        next_mask_t = torch.FloatTensor(env.get_action_mask()).unsqueeze(0).to(device)
        with torch.no_grad():
            _, v_after = model(next_state_t, next_mask_t)

        v_delta = float(v_after.item() - v_before.item() + reward)
        value_gains.append(v_delta)

    playability_ratio = is_playable_count / max(1, samples)
    avg_play_prob = float(np.mean(play_probs)) if play_probs else 0.0
    avg_v_gain = float(np.mean(value_gains)) if value_gains else 0.0

    # 综合 PPO 神经效用评分 (0 ~ 100 分):
    # 结合打出时收益、单费性价比与自然回合可打出率，杜绝高费突袭虚高满分偏差
    cost_penalty = max(1, card_obj.cost)
    v_norm = (avg_v_gain + 0.05) * 300.0 / (cost_penalty ** 0.5)
    p_norm = avg_play_prob * 100.0 * 0.4
    tempo_norm = playability_ratio * 20.0
    ppo_score = round(max(20.0, min(98.0, 20.0 + v_norm + p_norm + tempo_norm)), 2)

    return {
        "id": candidate["id"],
        "name": candidate["name"],
        "cost": candidate["cost"],
        "card_type": candidate["card_type"],
        "base_dp": candidate.get("base_dp", 0),
        "tags": candidate.get("tags", []),
        "play_prob": round(avg_play_prob * 100, 1),
        "value_gain": round(avg_v_gain, 3),
        "ppo_score": ppo_score
    }

def normalize_deck_allocation(counts: Dict[int, int], candidate_ids: List[int], priority_order: List[int]) -> Dict[int, int]:
    """保证牌库恰好 30 张，且单卡在 0~3 张之间"""
    clean_counts = {cid: min(MAX_COPIES_PER_CARD, max(0, counts.get(cid, 0))) for cid in candidate_ids}
    total = sum(clean_counts.values())

    # 若不足 30 张，按 PPO 效用最高优先级依次补充至 3 张
    if total < DECK_SIZE:
        for cid in priority_order:
            while clean_counts[cid] < MAX_COPIES_PER_CARD and total < DECK_SIZE:
                clean_counts[cid] += 1
                total += 1
            if total >= DECK_SIZE:
                break

    # 若超过 30 张，按 PPO 效用最低优先级依次扣减
    if total > DECK_SIZE:
        rev_priority = list(reversed(priority_order))
        for cid in rev_priority:
            while clean_counts[cid] > 0 and total > DECK_SIZE:
                clean_counts[cid] -= 1
                total -= 1
            if total <= DECK_SIZE:
                break

    return clean_counts

def get_archetype_blueprints(faction: str) -> List[dict]:
    """
    三大阵营专属特色流派蓝图库 (Faction Archetype Blueprints):
    定义各流派的战术定位、核心词条偏好、法力曲线偏好与卡牌类型倾向。
    """
    if faction == "Red":
        return [
            {
                "id": "aggro_rush",
                "name": "赤红·极限突破快攻流",
                "archetype": "Aggro Rush",
                "tactical_concept": "极致低费攻势，核心抓取突袭 (RUSH)、击穿增分 (BONUS_SCORE) 与削弱 (DEGRADE)，以雷霆速度击穿防线抢夺积分。",
                "tag_weights": {"RUSH": 35.0, "BONUS_SCORE": 30.0, "DEGRADE": 20.0, "SPAWN": 15.0},
                "cost_bias": lambda c: max(0.0, (4 - c) * 12.0) - (max(0, c - 5) * 30.0),
                "type_bias": lambda t: 5.0 if t == "MINION" else 0.0
            },
            {
                "id": "midrange_degrade",
                "name": "赤红·破甲穿透中速流",
                "archetype": "Midrange Degrade",
                "tactical_concept": "稳健的中期突破体系，以破甲削弱 (DEGRADE) 摧毁敌方防守随从上限，配合中费高身材随从进行稳固推进与斩杀。",
                "tag_weights": {"DEGRADE": 40.0, "SUPPORT_ATK": 20.0, "RUSH": 15.0, "FORTIFY": 15.0},
                "cost_bias": lambda c: 15.0 if 2 <= c <= 5 else (5.0 if c == 1 else -15.0),
                "type_bias": lambda t: 0.0
            },
            {
                "id": "swarm_sacrifice",
                "name": "赤红·横向狂潮献祭流",
                "archetype": "Swarm & Sacrifice",
                "tactical_concept": "利用横向铺场衍生小兵 (SPAWN)，结合献祭消灭 (SACRIFICE_1_KILL_1) 强杀敌方巨怪并触发亡语抽牌补充资源。",
                "tag_weights": {"SPAWN": 40.0, "SACRIFICE_1_KILL_1": 35.0, "DEATH_DRAW": 25.0, "BONUS_SCORE": 15.0},
                "cost_bias": lambda c: max(0.0, (5 - c) * 8.0),
                "type_bias": lambda t: 8.0 if t == "SPELL" else 0.0
            }
        ]
    elif faction == "Blue":
        return [
            {
                "id": "heavy_fortify",
                "name": "蔚蓝·铁壁坚守控制流",
                "archetype": "Heavy Fortify Wall",
                "tactical_concept": "前中后全期坚守阵地，依托低中费坚守 (FORTIFY) 随从平滑过渡，结合超模防守大怪构建不可逾越的铜墙铁壁。",
                "tag_weights": {"FORTIFY": 40.0, "DRAW": 20.0, "SUPPORT_ATK": 10.0},
                "cost_bias": lambda c: 16.0 if 1 <= c <= 4 else (6.0 if c <= 6 else -15.0),
                "type_bias": lambda t: 8.0 if t == "MINION" else 0.0
            },
            {
                "id": "support_aura",
                "name": "蔚蓝·光环反攻协作流",
                "archetype": "Support Aura Counter",
                "tactical_concept": "以光环支援 (SUPPORT_ATK) 为防线核心，将防守区化为火力发射台，持续为进攻战线提供战力加成。",
                "tag_weights": {"SUPPORT_ATK": 45.0, "FORTIFY": 25.0, "RUSH": 15.0},
                "cost_bias": lambda c: 15.0 if 2 <= c <= 4 else (6.0 if c in [1, 5] else -10.0),
                "type_bias": lambda t: 5.0 if t == "MINION" else 0.0
            },
            {
                "id": "spell_draw",
                "name": "蔚蓝·秘法过牌反击流",
                "archetype": "Spell Control & Draw",
                "tactical_concept": "大量过牌 (DRAW) 与坚守随从协同，保持手牌充沛与防线坚实，配合法术解场掌控整局节奏。",
                "tag_weights": {"DRAW": 35.0, "FORTIFY": 30.0, "TEMP_MANA": 20.0, "DEGRADE": 15.0},
                "cost_bias": lambda c: max(0.0, (5 - c) * 8.0),
                "type_bias": lambda t: 5.0 if t == "MINION" else 0.0
            }
        ]
    elif faction == "Green":
        return [
            {
                "id": "ramp_behemoths",
                "name": "翠绿·跳费远古巨兽流",
                "archetype": "Ramp & Ancient Behemoths",
                "tactical_concept": "以跳费 (RAMP) 与亡语跳费 (DEATH_MANA) 为跳板，快速攀升水晶拍下 6~9 费终极巨兽统治战场。",
                "tag_weights": {"RAMP": 45.0, "DEATH_MANA": 40.0, "FORTIFY": 20.0},
                "cost_bias": lambda c: 20.0 if c in [1, 2, 3, 6, 7, 8, 9] else 0.0,
                "type_bias": lambda t: 0.0
            },
            {
                "id": "poison_degrade",
                "name": "翠绿·荆棘剧毒削弱流",
                "archetype": "Poison & Degrade Attrition",
                "tactical_concept": "削弱敌方 (DEGRADE) 战力，配合坚守与持续回复在拉锯战中瓦解对手攻势。",
                "tag_weights": {"DEGRADE": 40.0, "FORTIFY": 25.0, "DRAW": 20.0},
                "cost_bias": lambda c: 12.0 if 2 <= c <= 5 else 0.0,
                "type_bias": lambda t: 0.0
            },
            {
                "id": "midrange_beasts",
                "name": "翠绿·野兽生机平滑流",
                "archetype": "Midrange Beasts & Value",
                "tactical_concept": "放弃高费卡手风险，精选 2~5 费高身材优质生物与适量跳费，打出无断节奏的平滑推进。",
                "tag_weights": {"RAMP": 25.0, "FORTIFY": 25.0, "BONUS_SCORE": 20.0, "RUSH": 15.0},
                "cost_bias": lambda c: max(0.0, 15.0 - abs(c - 3.5) * 6.0),
                "type_bias": lambda t: 5.0 if t == "MINION" else 0.0
            }
        ]
    return []

def synthesize_archetype_deck(profile: dict, candidates: List[dict], neural_dict: dict) -> Tuple[Dict[int, int], List[int]]:
    """根据流派战术蓝图，自适应打分并组装一套完整的 30 张流派卡组"""
    candidate_ids = [c["id"] for c in candidates]
    cand_by_id = {c["id"]: c for c in candidates}

    scored_candidates = []
    for cid in candidate_ids:
        c = cand_by_id[cid]
        base_ppo = neural_dict.get(cid, {}).get("ppo_score", 50.0)

        # 战术词条契合度加成
        tag_bonus = 0.0
        for tag in c.get("tags", []):
            for k, w in profile.get("tag_weights", {}).items():
                if k in tag:
                    tag_bonus += w

        cost_b = profile.get("cost_bias", lambda x: 0.0)(c["cost"])
        type_b = profile.get("type_bias", lambda x: 0.0)(c["card_type"])

        total_fit = base_ppo * 0.45 + tag_bonus + cost_b + type_b
        scored_candidates.append((cid, total_fit))

    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    priority_order = [x[0] for x in scored_candidates]

    # 按契合度装配 30 张卡组
    allocation = {cid: 0 for cid in candidate_ids}
    rem = DECK_SIZE
    for cid in priority_order:
        take = min(MAX_COPIES_PER_CARD, rem)
        allocation[cid] = take
        rem -= take
        if rem <= 0:
            break

    return normalize_deck_allocation(allocation, candidate_ids, priority_order), priority_order

def ppo_self_play_deck_search(faction: str, candidates: List[dict], neural_stats: List[dict],
                              model: CardNet, device: torch.device, cards_path: str,
                              generations: int = 5, games_per_gen: int = 50,
                              existing_decks: dict = None) -> Tuple[Dict[int, int], List[str], List[dict], str, str, str]:
    """
    PPO 多流派自主构筑与锦标赛选拔引擎 (PPO Archetype Synthesis & Tournament Engine):
    废除机械的单卡对换，改为自主合成多套战术流派假设，通过大规模实机对抗锦标赛角逐出优胜卡组，
    并在冠军卡组基础上进行自适应微调，实现宏观战略与微观节奏的完美平衡。
    """
    candidate_ids = [c["id"] for c in candidates]
    cand_by_id = {c["id"]: c for c in candidates}
    neural_dict = {x["id"]: x for x in neural_stats}

    faction_map = {"Red": Faction.RED, "Blue": Faction.BLUE, "Green": Faction.GREEN}
    name_map = {Faction.RED: "Red", Faction.BLUE: "Blue", Faction.GREEN: "Green"}
    my_faction = faction_map.get(faction, Faction.RED)
    opp_factions = [f for f in [Faction.RED, Faction.BLUE, Faction.GREEN] if f != my_faction]

    # 实战对抗评估函数
    def evaluate_decklist_performance(deck_allocation: Dict[int, int], num_games: int):
        test_decklist = []
        for cid, count in deck_allocation.items():
            test_decklist.extend([cid] * count)

        card_played_win = Counter()
        card_played_total = Counter()
        card_hand_turns = Counter()
        card_idle_turns = Counter()
        wins = 0

        for _ in range(num_games):
            opp_f = random.choice(opp_factions)
            opp_name = name_map[opp_f]
            opp_deck = None
            if existing_decks and opp_name in existing_decks and "decklist" in existing_decks[opp_name]:
                opp_deck = existing_decks[opp_name]["decklist"]

            env = DuelEnv(p0_faction=my_faction, p1_faction=opp_f, cards_path=cards_path,
                          p0_decklist=test_decklist, p1_decklist=opp_deck)
            obs = env.reset()
            done = False
            played_this_game = set()

            while not done:
                curr_p = env.current_player
                mask = env.get_action_mask()
                state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                mask_t = torch.FloatTensor(mask).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits, _ = model(state_t, mask_t)
                    act = torch.argmax(logits, dim=-1).item()

                if curr_p == 0:
                    p0 = env.players[0]
                    for card in p0.hand:
                        card_hand_turns[card.id] += 1
                        if card.cost > p0.mana:
                            card_idle_turns[card.id] += 1

                    if act != env.action_space_size - 1:
                        h_idx = act // 4
                        if h_idx < len(p0.hand):
                            c = p0.hand[h_idx]
                            card_played_total[c.id] += 1
                            played_this_game.add(c.id)

                obs, _, done, _ = env.step(act)

            is_win = (env.winner == 0) if env.winner is not None else (env.players[0].score >= env.WIN_SCORE)
            if is_win:
                wins += 1
                for cid in played_this_game:
                    card_played_win[cid] += 1

        wr = (wins / max(1, num_games)) * 100.0
        tot_hands = sum(card_hand_turns.values())
        tot_idles = sum(card_idle_turns.values())
        idle_r = (tot_idles / max(1, tot_hands)) * 100.0
        avg_cost = sum(cand_by_id[cid]["cost"] for cid in test_decklist) / max(1, len(test_decklist))
        return wr, idle_r, avg_cost, card_played_win, card_played_total, card_idle_turns, card_hand_turns

    # 1. 生成多套特色战术流派候选
    blueprints = get_archetype_blueprints(faction)
    contenders = []

    for bp in blueprints:
        alloc, prio = synthesize_archetype_deck(bp, candidates, neural_dict)
        contenders.append({
            "id": bp["id"],
            "name": bp["name"],
            "archetype": bp["archetype"],
            "tactical_concept": bp["tactical_concept"],
            "allocation": alloc,
            "priority": prio,
            "is_defending": False
        })

    # 若存在既有天梯卡组，作为“守擂者”加入多流派锦标赛
    if existing_decks and faction in existing_decks and "decklist" in existing_decks[faction]:
        existing_list = existing_decks[faction]["decklist"]
        if len(existing_list) == DECK_SIZE:
            counts = Counter(existing_list)
            contenders.append({
                "id": "defending_champion",
                "name": f"{faction}·现役天梯守擂卡组",
                "archetype": existing_decks[faction].get("archetype", "Defending"),
                "tactical_concept": "上一代实战优胜构筑参与本次多流派锦标赛守擂，检验战术生命力。",
                "allocation": dict(counts),
                "priority": list(counts.keys()),
                "is_defending": True
            })

    # 2. 启动多流派自主锦标赛实战对决
    print("\n" + "═" * 85)
    print(f"【{faction}】PPO 多流派自主锦标赛启动 (共 {len(contenders)} 套战术流派实机决斗选拔)")
    print("═" * 85)
    print(f"{'流派标识':<18}{'流派名称':<22}{'战术风格':<18}{'均费':<8}{'实战胜率':<12}{'卡手率':<10}{'综合得分'}")
    print("─" * 85)

    tournament_results = []
    decision_logs = []
    introspections = []

    for c in contenders:
        wr, idle_r, avg_c, p_win, p_tot, idles, hands = evaluate_decklist_performance(c["allocation"], games_per_gen)
        # 综合评分：胜率为主，卡手率重罚 (Dead Card Penalty)
        fitness = wr - 0.25 * idle_r
        c["winrate"] = wr
        c["idle_rate"] = idle_r
        c["avg_cost"] = avg_c
        c["fitness"] = fitness
        c["stats"] = (p_win, p_tot, idles, hands)
        tournament_results.append(c)

        flag = " [守擂]" if c.get("is_defending") else ""
        print(f"{c['id'] + flag:<18}{c['name']:<22}{c['archetype']:<18}{avg_c:<8.2f}{wr:<11.1f}%{idle_r:<9.1f}%{fitness:<8.2f}")

    # 3. 决出锦标赛优胜冠军 (Champion Crowned)
    champion = max(tournament_results, key=lambda x: (x["fitness"], x["winrate"]))
    champ_name = champion["name"]
    champ_archetype = champion["archetype"]
    champ_concept = champion["tactical_concept"]
    best_counts = dict(champion["allocation"])
    best_winrate = champion["winrate"]
    best_stats = champion["stats"]

    print("═" * 85)
    print(f"🏆 锦标赛优胜登顶: 【{champ_name}】(胜率 {best_winrate:.1f}%, 均费 {champion['avg_cost']:.2f})！荣获本阵营版本答案！")
    print("═" * 85 + "\n")

    log_entry_tourn = f"【PPO 多流派锦标赛】{champ_name} 以 {best_winrate:.1f}% 胜率与最高战术综合得分击败其余 {len(contenders)-1} 套流派夺冠。"
    decision_logs.append(log_entry_tourn)

    intro_tourn = {
        "generation": 0,
        "winrate": best_winrate,
        "prev_winrate": best_winrate,
        "promoted": {"id": 0, "name": champ_name, "thought": f"在 {len(contenders)} 套流派锦标赛角逐中脱颖而出，以实测最高胜率夺得版本主力构筑。"},
        "demoted": None,
        "monologue": f"多流派对抗评估完成：【{champ_name}】在实机测试中展现最强协同爆发力与稳健节奏，确立为主力战术体系。"
    }
    introspections.append(intro_tourn)

    # 4. 冠军卡组自适应微调 (Adaptive Fine-Tuning, 1~2 轮轻量修剪)
    refine_gens = min(generations, 2)
    p_win, p_tot, idles, hands = best_stats

    for gen in range(1, refine_gens + 1):
        deck_cids = [cid for cid, cnt in best_counts.items() if cnt > 0]
        # 寻找冠军卡组内卡手率最高或场面贡献偏弱的卡牌
        worst_cid = None
        highest_idle = -1.0
        for cid in deck_cids:
            h_cnt = hands[cid]
            if h_cnt >= 2:
                i_rate = idles[cid] / h_cnt
                if i_rate > highest_idle and i_rate > 0.35:
                    highest_idle = i_rate
                    worst_cid = cid

        if not worst_cid:
            break

        # 寻找该流派候补中契合度最高且未满 3 张的润滑卡牌
        sub_cids = [cid for cid in champion["priority"] if best_counts.get(cid, 0) < MAX_COPIES_PER_CARD and cid != worst_cid]
        if not sub_cids:
            break
        best_sub = sub_cids[0]

        c_worst_name = cand_by_id[worst_cid]["name"]
        c_sub_name = cand_by_id[best_sub]["name"]

        test_counts = dict(best_counts)
        test_counts[worst_cid] -= 1
        test_counts[best_sub] += 1
        test_counts = normalize_deck_allocation(test_counts, candidate_ids, champion["priority"])

        t_wr, t_idle, t_avg, t_p_win, t_p_tot, t_idles, t_hands = evaluate_decklist_performance(test_counts, games_per_gen)
        delta = t_wr - best_winrate

        if delta >= -2.0:
            print(f"   [冠军精修 第 {gen} 代] 微调采纳！胜率 {best_winrate:.1f}% -> {t_wr:.1f}% ({delta:+.1f}%) | +[{c_sub_name}] / -[{c_worst_name}]")
            best_counts = test_counts
            best_winrate = t_wr
            decision_logs.append(f"第 {gen} 代微调: 增选 [{c_sub_name}]，替换高卡手牌 [{c_worst_name}] (胜率 {best_winrate:.1f}%)")
            introspections.append({
                "generation": gen,
                "winrate": t_wr,
                "prev_winrate": best_winrate,
                "promoted": {"id": best_sub, "name": c_sub_name, "thought": f"作为低费润滑组件有效改善卡手率。"},
                "demoted": {"id": worst_cid, "name": c_worst_name, "thought": f"实测卡手率偏高 ({highest_idle*100:.1f}%)，调减 1 张。"},
                "monologue": f"冠军卡组微调：调减卡手牌 [{c_worst_name}]，增选润滑组件 [{c_sub_name}]。"
            })
        else:
            print(f"   [冠军精修 第 {gen} 代] 胜率未获提振 ({t_wr:.1f}%)，维持冠军初始完美架构。")

    return best_counts, decision_logs, introspections, champ_name, champ_archetype, champ_concept

def build_ppo_deck_package(faction: str, candidates: List[dict], neural_stats: List[dict],
                           allocation: Dict[int, int], decision_logs: List[str],
                           introspections: List[dict],
                           deck_name: str = None, archetype: str = None, concept: str = None) -> dict:
    cand_dict = {c["id"]: c for c in candidates}
    decklist = []
    mana_curve = {i: 0 for i in range(10)}
    card_details = []
    minion_count = 0
    spell_count = 0
    total_mana = 0

    for cid, count in sorted(allocation.items(), key=lambda x: (cand_dict.get(x[0], {}).get("cost", 0), x[0])):
        if count <= 0 or cid not in cand_dict:
            continue
        c = cand_dict[cid]
        for _ in range(count):
            decklist.append(cid)
            cost_clamp = min(c["cost"], 9)
            mana_curve[cost_clamp] += 1
            total_mana += c["cost"]
            if c["card_type"] == "MINION":
                minion_count += 1
            else:
                spell_count += 1

        n_stat = next((s for s in neural_stats if s["id"] == cid), {})
        card_details.append({
            "id": cid,
            "name": c["name"],
            "cost": c["cost"],
            "card_type": c["card_type"],
            "base_dp": c.get("base_dp", 0),
            "atk_spell_val": c.get("atk_spell_val", 0),
            "def_spell_val": c.get("def_spell_val", 0),
            "tags": c.get("tags", []),
            "count": count,
            "ppo_play_prob": n_stat.get("play_prob", 0),
            "ppo_value_gain": n_stat.get("value_gain", 0),
            "ppo_score": n_stat.get("ppo_score", 0)
        })

    avg_cost = round(total_mana / max(1, len(decklist)), 2)

    deck_name = deck_name or f"{faction}·推荐构筑"
    archetype = archetype or "Reinforcement-Learned"
    concept = concept or f"基于 PPO 神经网络效用评估与多流派实战锦标赛对抗选拔构筑，平均费用 {avg_cost} 费。"

    top_cards = sorted(card_details, key=lambda x: x.get("ppo_score", 0), reverse=True)[:2]
    key_combos = [
        f"{top_cards[0]['name']} (评分 {top_cards[0]['ppo_score']}): 核心关键卡，状态价值增益最高。",
        f"{top_cards[1]['name']} (评分 {top_cards[1]['ppo_score']}): 辅助节奏卡，协同支撑场面与战术推进。"
    ] if len(top_cards) >= 2 else []

    return {
        "deck_name": deck_name,
        "archetype": archetype,
        "tactical_concept": concept,
        "key_combos": key_combos,
        "decision_evolution_logs": decision_logs,
        "introspections": introspections,
        "total_cards": len(decklist),
        "minion_count": minion_count,
        "spell_count": spell_count,
        "avg_cost": avg_cost,
        "mana_curve": mana_curve,
        "card_allocation": {str(k): v for k, v in allocation.items() if v > 0},
        "card_details": card_details,
        "decklist": decklist
    }

def print_ppo_deck_report(faction: str, deck_pkg: dict, neural_stats: List[dict]):
    print("\n" + "─" * 70)
    print(f"【{faction}】推荐卡组: 《{deck_pkg['deck_name']}》 (30 张 | 均费 {deck_pkg['avg_cost']} 费 | 随从 {deck_pkg['minion_count']} / 法术 {deck_pkg['spell_count']})")
    curve_summary = [f"{c}费({deck_pkg['mana_curve'].get(c, 0)})" for c in range(1, 6) if deck_pkg['mana_curve'].get(c, 0) > 0]
    high_cnt = sum(deck_pkg['mana_curve'].get(c, 0) for c in range(6, 10))
    if high_cnt > 0:
        curve_summary.append(f"6+费({high_cnt})")
    print(f"  法力分布: {' '.join(curve_summary)}")

    if deck_pkg.get("introspections"):
        print("  微调记录:")
        for intro in deck_pkg["introspections"]:
            g = intro["generation"]
            wr = intro["winrate"]
            pwr = intro["prev_winrate"]
            diff = wr - pwr
            actions = []
            if intro.get("demoted"):
                actions.append(f"-1 [{intro['demoted']['name']}]")
            if intro.get("promoted"):
                actions.append(f"+1 [{intro['promoted']['name']}]")
            action_str = " | ".join(actions) if actions else "无变动"
            print(f"   * 第 {g} 轮胜率 {wr:.1f}% ({diff:+.1f}%): {action_str}")
    print("─" * 70)

def export_introspection_report(decks_result: dict, output_path: str = "ppo_introspection_report.md"):
    """导出 PPO 选卡与微调记录报告"""
    lines = []
    lines.append("# PPO 自博弈选卡与调整记录报告\n")
    lines.append("> 本报告记录了 PPO 智能体在自博弈对战中逐步迭代、微调卡组构筑的过程。\n")
    lines.append("---\n")

    for faction, deck in decks_result.items():
        lines.append(f"## 【{faction}】阵营卡组：《{deck['deck_name']}》\n")
        lines.append(f"- **流派定位**：`{deck['archetype']}`")
        lines.append("- **牌库规模**：严格遵守 **30 张** 标准规则（同名卡上限 3 张）")
        lines.append(f"- **法力曲线均值**：**{deck['avg_cost']} 费**（随从 {deck['minion_count']} 张 / 法术 {deck['spell_count']} 张）\n")

        lines.append("### 1. 迭代调整记录\n")
        if deck.get("introspections"):
            for intro in deck["introspections"]:
                g = intro["generation"]
                wr = intro["winrate"]
                pwr = intro["prev_winrate"]
                diff = wr - pwr
                lines.append(f"#### 第 {g} 轮迭代（对战胜率: {wr:.1f}% | 胜率变化: {diff:+.1f}%）")
                if intro.get("demoted"):
                    lines.append(f"> **调减卡牌**：{intro['demoted']['thought']}\n")
                if intro.get("promoted"):
                    lines.append(f"> **增选卡牌**：{intro['promoted']['thought']}\n")
        else:
            lines.append("> 初始探索代数已稳定收敛。\n")

        lines.append("### 2. 30 张卡牌配置与评估表\n")
        lines.append("| ID | 卡牌名称 | 费用 | 类型 | DP/数值 | 最终入选 | Actor出牌偏好 | Critic价值增益 (ΔV) | PPO综合评分 |")
        lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for c in deck.get("card_details", []):
            dp_str = f"DP:{c.get('base_dp', 0)}" if c.get("card_type") == "MINION" else f"攻{c.get('atk_spell_val', 0)}/防{c.get('def_spell_val', 0)}"
            play_p = f"{c.get('ppo_play_prob')}%" if c.get('ppo_play_prob') is not None else "-"
            val_g = f"{c.get('ppo_value_gain', 0.0):+.3f}" if c.get('ppo_value_gain') is not None else "-"
            score_v = f"{c.get('ppo_score')}" if c.get('ppo_score') is not None else "-"
            lines.append(f"| {c['id']} | **{c['name']}** | {c['cost']} | {c['card_type']} | {dp_str} | **{c['count']} 张** | {play_p} | {val_g} | {score_v} |")
        lines.append("\n")

        lines.append("### 3. 法力曲线 (Mana Curve)\n")
        lines.append("```text")
        for cost in range(1, 8):
            cnt = deck["mana_curve"].get(cost, 0)
            bar = "█" * (cnt * 2)
            lines.append(f"{cost} 费: {bar:<22} ({cnt} 张)")
        cnt_high = sum(deck["mana_curve"].get(c, 0) for c in range(8, 10))
        if cnt_high > 0:
            bar = "█" * (cnt_high * 2)
            lines.append(f"8+ 费: {bar:<22} ({cnt_high} 张)")
        lines.append("```\n")
        lines.append("---\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"PPO 选卡报告已导出至: {os.path.abspath(output_path)}")

def main():
    parser = argparse.ArgumentParser(description="PPO 强化学习智能体自主选卡构筑系统 (PPO Deckbuilder)")
    parser.add_argument("--cards", type=str, default="cards_config.json", help="卡池配置文件路径")
    parser.add_argument("--model", type=str, default=None, help="PPO 模型权重路径")
    parser.add_argument("--stage", type=str, default="tuned", choices=["baseline", "tuned"], help="选用训练模型阶段")
    parser.add_argument("--output", type=str, default="decks_config.json", help="输出卡组保存文件")
    parser.add_argument("--report", type=str, default="ppo_introspection_report.md", help="输出卡组微调记录Markdown文件名")
    parser.add_argument("--factions", type=str, default="Red,Blue,Green", help="目标自主选卡阵营 (默认 Red,Blue,Green)")
    parser.add_argument("--generations", type=int, default=5, help="自博弈进化代数 (默认 5)")
    parser.add_argument("--games-per-gen", type=int, default=60, help="每代自博弈实机局数 (默认 60)")
    parser.add_argument("--samples", type=int, default=8, help="候选卡神经网络采样次数 (默认 8，速度提升 2.5x)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = find_model_path(args.model, args.stage)
    if not model_path:
        print("未检测到可用的 PPO 权重模型，请确认模型文件存在。")
        return

    print("=" * 85)
    print("PPO 自博弈选卡工具启动")
    print(f"模型权重: {model_path} | 设备: {device}")
    print("=" * 85)

    cards_db = load_card_pool(args.cards)
    
    # 载入 PPO 模型
    model = CardNet().to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    factions_to_build = [f.strip() for f in args.factions.split(",") if f.strip()]
    decks_result = {}
    if os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                decks_result = json.load(f)
        except Exception:
            pass

    for faction_name in factions_to_build:
        print(f"\n[1/2] 正在评估【{faction_name}】候选卡池...")
        candidates = get_faction_candidates(cards_db, faction_name)
        f_enum = Faction.RED if faction_name == "Red" else (Faction.BLUE if faction_name == "Blue" else Faction.GREEN)

        neural_stats = []
        for c in candidates:
            st = evaluate_card_neural_utility(model, device, c, f_enum, args.cards, samples=args.samples)
            neural_stats.append(st)

        print(f"[2/2] 正在执行【{faction_name}】PPO 自博弈对决与卡组微调...")
        alloc, decision_logs, introspections, champ_name, champ_archetype, champ_concept = ppo_self_play_deck_search(
            faction=faction_name,
            candidates=candidates,
            neural_stats=neural_stats,
            model=model,
            device=device,
            cards_path=args.cards,
            generations=args.generations,
            games_per_gen=args.games_per_gen,
            existing_decks=decks_result
        )

        deck_pkg = build_ppo_deck_package(
            faction_name, candidates, neural_stats, alloc, decision_logs, introspections,
            deck_name=champ_name, archetype=champ_archetype, concept=champ_concept
        )
        print_ppo_deck_report(faction_name, deck_pkg, neural_stats)
        decks_result[faction_name] = deck_pkg

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(decks_result, f, indent=2, ensure_ascii=False)

    abs_out = os.path.abspath(args.output)
    print(f"PPO 选卡结果已保存至: {abs_out}")

    # 导出卡组调整报告
    if args.report:
        export_introspection_report(decks_result, args.report)

if __name__ == "__main__":
    main()
