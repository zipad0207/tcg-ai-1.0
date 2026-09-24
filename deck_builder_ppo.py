import os
import sys
import json
import argparse
import random
import torch
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter
from torch.distributions.categorical import Categorical

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
    candidates = []
    seen_ids = set()
    f_code = {"Red": 1, "Blue": 2, "Green": 3}.get(faction, 0)

    for category, card_list in pool_data.items():
        for c in card_list:
            cid = c.get("id", 0)
            if cid in seen_ids:
                continue

            facs = c.get("factions", [])
            c_f = cid // 100
            allowed = False
            if facs and (faction in facs or "Neutral" in facs):
                allowed = True
            elif category == faction or category == "Neutral":
                allowed = True
            elif c_f == f_code or c_f == 9:
                allowed = True
            elif c_f == 4 and faction in ("Red", "Blue"):
                allowed = True
            elif c_f == 5 and faction in ("Blue", "Green"):
                allowed = True
            elif c_f == 6 and faction in ("Red", "Green"):
                allowed = True

            if allowed:
                candidates.append(c)
                seen_ids.add(cid)

    return candidates

def evaluate_card_neural_utility(model: CardNet, device: torch.device, candidate: dict, 
                                 faction: Faction, cards_path: str, samples: int = 15) -> dict:
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
        opp_pool = [f for f in [Faction.RED, Faction.BLUE, Faction.GREEN] if f != faction]
        opp_f = random.choice(opp_pool)
        env = DuelEnv(p0_faction=faction, p1_faction=opp_f, cards_path=cards_path)
        obs = env.reset()

        acting_p_id = 0
        env.current_player = acting_p_id
        player = env.players[acting_p_id]

        # 采样费用：保证该卡至少有一半概率能被打出，避免高费卡因为低费回合太多而被系统性低估
        min_fair_mana = max(1, card_obj.cost)
        sim_mana = random.randint(min_fair_mana, max(min_fair_mana, 10))
        player.mana = sim_mana
        player.max_mana = sim_mana

        player.hand = [card_obj] + player.hand[:3]

        mask = env.get_action_mask()
        card_action_indices = [0, 1, 2, 3] 
        legal_card_actions = [idx for idx in card_action_indices if mask[idx] > 0.5]

        if not legal_card_actions:
            continue

        is_playable_count += 1
        obs = env.get_observation()
        state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        mask_t = torch.FloatTensor(mask).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, v_before = model(state_t, mask_t)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        p_card_play = float(sum(probs[idx] for idx in legal_card_actions))
        play_probs.append(p_card_play)

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

    # 费用无关的公平评分：value_gain 和 play_prob 已在公平费用下采样，无需额外惩罚高费卡
    v_norm = (avg_v_gain + 0.05) * 300.0
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
    clean_counts = {cid: min(MAX_COPIES_PER_CARD, max(0, counts.get(cid, 0))) for cid in candidate_ids}
    total = sum(clean_counts.values())

    if total < DECK_SIZE:
        for cid in priority_order:
            while clean_counts[cid] < MAX_COPIES_PER_CARD and total < DECK_SIZE:
                clean_counts[cid] += 1
                total += 1
            if total >= DECK_SIZE:
                break

    if total > DECK_SIZE:
        rev_priority = list(reversed(priority_order))
        for cid in rev_priority:
            while clean_counts[cid] > 0 and total > DECK_SIZE:
                clean_counts[cid] -= 1
                total -= 1
            if total <= DECK_SIZE:
                break

    return clean_counts

def ppo_self_play_deck_search(faction: str, candidates: List[dict], neural_stats: List[dict],
                              model: CardNet, device: torch.device, cards_path: str,
                              generations: int = 20, games_per_gen: int = 50,
                              existing_decks: dict = None) -> Tuple[Dict[int, int], List[str], List[dict]]:
    candidate_ids = [c["id"] for c in candidates]
    cand_by_id = {c["id"]: c for c in candidates}
    sorted_by_ppo = sorted(neural_stats, key=lambda x: x["ppo_score"], reverse=True)
    priority_order = [x["id"] for x in sorted_by_ppo]
    neural_dict = {x["id"]: x for x in neural_stats}

    current_counts: Dict[int, int] = {cid: 0 for cid in candidate_ids}

    has_existing = False
    if existing_decks and faction in existing_decks and "decklist" in existing_decks[faction]:
        existing_list = existing_decks[faction]["decklist"]
        if len(existing_list) > 0:
            for cid in existing_list:
                if cid in current_counts:
                    current_counts[cid] += 1
            if sum(current_counts.values()) >= 15:
                # 费用健康度检查与平滑微创修剪
                total_inherited = sum(current_counts.values())
                min_high = 4 if faction == "Green" else 3
                max_low = 18  # 1~2 费卡上限（60%）

                high_cost_count = sum(cnt for cid, cnt in current_counts.items() if cand_by_id.get(cid, {}).get("cost", 0) >= 5)
                low_cost_count = sum(cnt for cid, cnt in current_counts.items() if cand_by_id.get(cid, {}).get("cost", 0) <= 2)

                # 1. 若高费不足，进行微创修剪补齐高费大哥
                if high_cost_count < min_high:
                    deficit = min_high - high_cost_count
                    trim_cands = sorted(
                        [cid for cid, cnt in current_counts.items() if cnt > 0 and cand_by_id.get(cid, {}).get("cost", 0) < 5],
                        key=lambda cid: neural_dict.get(cid, {}).get("ppo_score", 50.0)
                    )
                    high_add_pool = [
                        cid for cid in priority_order
                        if cand_by_id.get(cid, {}).get("cost", 0) >= 5 and current_counts.get(cid, 0) < MAX_COPIES_PER_CARD
                    ]
                    cut_done = 0
                    for cid in trim_cands:
                        if cut_done >= deficit: break
                        can_cut = min(current_counts[cid], deficit - cut_done)
                        current_counts[cid] -= can_cut
                        cut_done += can_cut
                    
                    add_done = 0
                    for cid in high_add_pool:
                        if add_done >= cut_done: break
                        can_add = min(MAX_COPIES_PER_CARD - current_counts.get(cid, 0), cut_done - add_done)
                        current_counts[cid] = current_counts.get(cid, 0) + can_add
                        add_done += can_add
                    
                    if add_done > 0:
                        print(f"[*] 【{faction}】既有卡组高费偏少 (5+费仅 {high_cost_count} 张)，已平滑补齐 {add_done} 张高费大哥。")

                # 重新计算低费张数
                low_cost_count = sum(cnt for cid, cnt in current_counts.items() if cand_by_id.get(cid, {}).get("cost", 0) <= 2)
                # 2. 若低费超标，进行微创修剪替换为 3~4 费中费卡
                if low_cost_count > max_low:
                    excess = low_cost_count - max_low
                    trim_low_cands = sorted(
                        [cid for cid, cnt in current_counts.items() if cnt > 0 and cand_by_id.get(cid, {}).get("cost", 0) <= 2],
                        key=lambda cid: neural_dict.get(cid, {}).get("ppo_score", 50.0)
                    )
                    mid_add_pool = [
                        cid for cid in priority_order
                        if 3 <= cand_by_id.get(cid, {}).get("cost", 0) <= 4 and current_counts.get(cid, 0) < MAX_COPIES_PER_CARD
                    ]
                    cut_done = 0
                    for cid in trim_low_cands:
                        if cut_done >= excess: break
                        can_cut = min(current_counts[cid], excess - cut_done)
                        current_counts[cid] -= can_cut
                        cut_done += can_cut

                    add_done = 0
                    for cid in mid_add_pool:
                        if add_done >= cut_done: break
                        can_add = min(MAX_COPIES_PER_CARD - current_counts.get(cid, 0), cut_done - add_done)
                        current_counts[cid] = current_counts.get(cid, 0) + can_add
                        add_done += can_add
                    
                    if add_done > 0:
                        print(f"[*] 【{faction}】既有卡组低费卡偏多 (1~2费 {low_cost_count} 张)，已平滑置换 {add_done} 张为优质中费卡。")

                # 最终校验总数与均费
                current_counts = normalize_deck_allocation(current_counts, candidate_ids, priority_order)
                total_inherited = sum(current_counts.values())
                avg_inherited = sum(cand_by_id.get(cid, {}).get("cost", 0) * cnt for cid, cnt in current_counts.items()) / max(1, total_inherited)
                has_existing = True
                print(f"[*] 【{faction}】成功继承既有实战卡组 (有效卡牌 {total_inherited} 张，均费 {avg_inherited:.1f})，启动 PPO 自主胜率检验与变异搜索...")

    if not has_existing:
        # 按合理法力曲线初始化，保证高中低费段都有足够卡牌
        curve_targets = {(1, 2): 10, (3, 4): 10, (5, 10): 10}
        rem = DECK_SIZE
        for cost_range, target in curve_targets.items():
            bracket_cands = [x["id"] for x in sorted_by_ppo if cost_range[0] <= cand_by_id[x["id"]]["cost"] <= cost_range[1]]
            bracket_rem = target
            for cid in bracket_cands:
                take = min(MAX_COPIES_PER_CARD, bracket_rem)
                current_counts[cid] = current_counts.get(cid, 0) + take
                bracket_rem -= take
                rem -= take
                if bracket_rem <= 0:
                    break
        for cid in priority_order:
            if rem <= 0: break
            if current_counts.get(cid, 0) < MAX_COPIES_PER_CARD:
                take = min(MAX_COPIES_PER_CARD - current_counts.get(cid, 0), rem)
                current_counts[cid] = current_counts.get(cid, 0) + take
                rem -= take

    current_counts = normalize_deck_allocation(current_counts, candidate_ids, priority_order)

    faction_map = {"Red": Faction.RED, "Blue": Faction.BLUE, "Green": Faction.GREEN}
    name_map = {Faction.RED: "Red", Faction.BLUE: "Blue", Faction.GREEN: "Green"}
    my_faction = faction_map.get(faction, Faction.RED)
    opp_factions = [f for f in [Faction.RED, Faction.BLUE, Faction.GREEN] if f != my_faction]

    def evaluate_decklist_performance(deck_allocation: Dict[int, int], num_games: int):
        test_decklist = []
        for cid, count in deck_allocation.items():
            test_decklist.extend([cid] * count)

        card_played_win = Counter()
        card_played_total = Counter()
        card_hand_turns = Counter()
        card_idle_turns = Counter()
        wins = 0

        for game_idx in range(num_games):
            # 对称均衡轮换对手阵营，确保对每个对手样本量严格相等，避免偏科
            opp_f = opp_factions[game_idx % len(opp_factions)]
            opp_name = name_map[opp_f]
            opp_deck = None
            if existing_decks and opp_name in existing_decks and "decklist" in existing_decks[opp_name]:
                opp_deck = existing_decks[opp_name]["decklist"]

            # 先手与后手严格交替对称分配，消除先后手胜率统计偏差
            is_p0 = (game_idx % 2 == 0)
            
            p0_f = my_faction if is_p0 else opp_f
            p1_f = opp_f if is_p0 else my_faction
            p0_deck = test_decklist if is_p0 else opp_deck
            p1_deck = opp_deck if is_p0 else test_decklist

            env = DuelEnv(p0_faction=p0_f, p1_faction=p1_f, cards_path=cards_path,
                          p0_decklist=p0_deck, p1_decklist=p1_deck)
            obs = env.reset()
            done = False
            played_this_game = set()

            while not done:
                curr_p = env.current_player
                is_my_turn = (curr_p == 0 and is_p0) or (curr_p == 1 and not is_p0)
                
                mask = env.get_action_mask()
                state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                mask_t = torch.FloatTensor(mask).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits, _ = model(state_t, mask_t)
                    dist = Categorical(logits=logits)
                    act = dist.sample().item()

                if is_my_turn:
                    my_player = env.players[curr_p]
                    for card in my_player.hand:
                        card_hand_turns[card.id] += 1
                        if card.cost > my_player.mana:
                            card_idle_turns[card.id] += 1

                    if act != env.action_space_size - 1:
                        h_idx = act // 4
                        if h_idx < len(my_player.hand):
                            c = my_player.hand[h_idx]
                            card_played_total[c.id] += 1
                            played_this_game.add(c.id)

                obs, _, done, _ = env.step(act)

            is_win = False
            if env.winner is not None:
                is_win = (env.winner == 0 and is_p0) or (env.winner == 1 and not is_p0)
            else:
                s0 = env.players[0].score
                s1 = env.players[1].score
                is_win = (s0 > s1 and is_p0) or (s1 > s0 and not is_p0)

            if is_win:
                wins += 1
                for cid in played_this_game:
                    card_played_win[cid] += 1

        wr = (wins / num_games) * 100.0
        return wr, card_played_win, card_played_total, card_idle_turns, card_hand_turns

    print(f"\n启动 PPO 自主学习卡组搜索 (共 {generations} 轮，每轮 {games_per_gen} 局实测)...")
    
    print(f"[*] 正在实测【{faction}】基准卡组性能...")
    best_winrate, best_p_win, best_p_tot, best_idles, best_hands = evaluate_decklist_performance(current_counts, games_per_gen)
    best_counts = dict(current_counts)
    print(f"[*] 【{faction}】初始基准实战胜率: {best_winrate:.1f}%")

    early_cards = sum(cnt for cid, cnt in best_counts.items() if cand_by_id[cid]["cost"] <= 2 and cand_by_id[cid]["card_type"] == "MINION")
    min_early = 4 if faction == "Green" else 6
    if early_cards < min_early:
        deficit = min_early - early_cards
        print(f"   [*] 检测到前期抗压随从偏少 ({early_cards} < {min_early} 张)，启动曲线微创修补 (定向补齐 {deficit} 个卡位)...")
        
        # 挑选可微调削减的卡牌：避开 1~2 费随从，避开底线高费大哥
        cnt_high = sum(cnt for cid, cnt in best_counts.items() if cand_by_id[cid]["cost"] >= 5)
        min_high = 5 if faction == "Green" else 3
        
        cut_candidates = []
        for cid, cnt in best_counts.items():
            if cnt <= 0: continue
            cost = cand_by_id[cid]["cost"]
            ctype = cand_by_id[cid]["card_type"]
            if cost <= 2 and ctype == "MINION":
                continue
            if cost >= 5 and cnt_high <= min_high:
                continue
            prior = neural_dict.get(cid, {}).get("ppo_score", 50.0)
            cut_candidates.append((cid, cost, cnt, prior))
        
        # 按 PPO 评分从低到高削减（优先调减相对低效的卡牌）
        cut_candidates.sort(key=lambda x: x[3])
        
        patch_counts = dict(best_counts)
        cut_done = 0
        for cid, cost, cnt, _ in cut_candidates:
            if cut_done >= deficit:
                break
            can_cut = min(cnt, deficit - cut_done)
            patch_counts[cid] -= can_cut
            cut_done += can_cut
            if cost >= 5:
                cnt_high -= can_cut

        # 寻找该阵营最优秀的 1~2 费随从补入
        add_candidates = [
            cid for cid in priority_order 
            if cand_by_id[cid]["cost"] <= 2 and cand_by_id[cid]["card_type"] == "MINION"
        ]
        
        add_done = 0
        for cid in add_candidates:
            if add_done >= cut_done:
                break
            curr_cnt = patch_counts.get(cid, 0)
            can_add = min(MAX_COPIES_PER_CARD - curr_cnt, cut_done - add_done)
            if can_add > 0:
                patch_counts[cid] = curr_cnt + can_add
                add_done += can_add

        cand_patch = normalize_deck_allocation(patch_counts, candidate_ids, priority_order)
        patch_wr, p_p_win, p_p_tot, p_idles, p_hands = evaluate_decklist_performance(cand_patch, games_per_gen)
        
        if patch_wr >= best_winrate or (patch_wr >= best_winrate - 2.0 and best_winrate < 40.0):
            delta = patch_wr - best_winrate
            best_counts = cand_patch
            best_winrate = patch_wr
            best_p_win, best_p_tot, best_idles, best_hands = p_p_win, p_p_tot, p_idles, p_hands
            print(f"   [*] 曲线微创修补成功采纳！基准胜率: {best_winrate:.1f}% ({delta:+.1f}%)，补足前期随从。")
        else:
            print(f"   [*] 曲线微创修补实测胜率 ({patch_wr:.1f}%) 未见明显改善，保留原构筑交由后续微调。")

    decision_logs = []
    introspections = []
    rejected_pairs = set()
    rejected_actions = set()

    for gen in range(1, generations + 1):
        prev_winrate = best_winrate

        deck_cids = [cid for cid, cnt in best_counts.items() if cnt > 0]
        fitness_map = {}
        for cid in deck_cids:
            tot = best_p_tot[cid]
            win_r = (best_p_win[cid] / tot) if tot >= 2 else (best_winrate / 100.0)
            h_turn = best_hands[cid]
            idle_r = (best_idles[cid] / h_turn) if h_turn >= 3 else 0.0
            prior = neural_dict.get(cid, {}).get("ppo_score", 50.0) / 100.0
            fit = (win_r * 0.6 + prior * 0.4) * (1.0 - 0.75 * idle_r)
            fitness_map[cid] = fit

        cnt_1_cost = sum(cnt for cid, cnt in best_counts.items() if cand_by_id[cid]["cost"] == 1)
        cnt_2_cost = sum(cnt for cid, cnt in best_counts.items() if cand_by_id[cid]["cost"] == 2)
        cnt_high_cost = sum(cnt for cid, cnt in best_counts.items() if cand_by_id[cid]["cost"] >= 5)
        min_high = 5 if faction == "Green" else 3
        
        eligible_worst_cids = []
        for cid in deck_cids:
            cost = cand_by_id[cid]["cost"]
            if cost == 1 and cnt_1_cost <= 3: continue
            if cost == 2 and cnt_2_cost <= 5: continue
            if cost >= 5 and cnt_high_cost <= min_high: continue
            eligible_worst_cids.append(cid)
            
        if not eligible_worst_cids:
            eligible_worst_cids = deck_cids

        sorted_worst = sorted(eligible_worst_cids, key=lambda cid: fitness_map[cid])

        candidate_pool_sorted = []
        for cid in candidate_ids:
            if best_counts.get(cid, 0) < MAX_COPIES_PER_CARD:
                prior = neural_dict.get(cid, {}).get("ppo_score", 50.0)
                # 纯 PPO 评分排序，不再给低费卡额外加分
                candidate_pool_sorted.append((cid, prior))
        candidate_pool_sorted.sort(key=lambda x: x[1], reverse=True)

        # 构建所有合规替换候选对
        valid_pairs = []
        for w_cid in sorted_worst:
            w_cost = cand_by_id[w_cid]["cost"]
            for b_cid, _ in candidate_pool_sorted:
                if w_cid == b_cid:
                    continue
                if (b_cid, w_cid) in rejected_pairs:
                    continue

                b_cost = cand_by_id[b_cid]["cost"]
                # 费用健康度防护 1：若当前低费卡 (1~2费) 已达上限 (18 张 / 60%)，禁止用低费卡替换中高费卡
                if b_cost <= 2 and w_cost > 2 and (cnt_1_cost + cnt_2_cost) >= 18:
                    continue
                # 费用健康度防护 2：若当前高费卡 (5+费) 已触底 (<= min_high)，禁止削减高费卡换入非高费卡
                if w_cost >= 5 and b_cost < 5 and cnt_high_cost <= min_high:
                    continue

                curr_w_cnt = best_counts.get(w_cid, 0)
                curr_b_cnt = best_counts.get(b_cid, 0)
                max_can_add = MAX_COPIES_PER_CARD - curr_b_cnt
                max_step = min(curr_w_cnt, max_can_add)
                if max_step <= 0:
                    continue

                # 备选步长选项：支持渐进平滑 1 张，或整批替换
                possible_steps = [max_step] if max_step == 1 else [1, max_step]
                avail_steps = [s for s in possible_steps if (b_cid, w_cid, s) not in rejected_actions]
                if not avail_steps:
                    continue

                fit_w = fitness_map.get(w_cid, 0.5)
                prior_b = neural_dict.get(b_cid, {}).get("ppo_score", 50.0) / 100.0
                pair_score = (1.0 - fit_w) * 0.55 + prior_b * 0.45

                valid_pairs.append({
                    "w_cid": w_cid,
                    "b_cid": b_cid,
                    "curr_w_cnt": curr_w_cnt,
                    "max_step": max_step,
                    "avail_steps": avail_steps,
                    "pair_score": pair_score
                })

        if not valid_pairs:
            print(f"   [代数 {gen}/{generations}] 所有可行组合已被探索或已达局部最优，停止变异。")
            break

        # 按综合评分排序
        valid_pairs.sort(key=lambda x: x["pair_score"], reverse=True)

        # 机制一：温度退火 Softmax 概率采样（代数越小温度越高，探索范围越广）
        top_k = min(len(valid_pairs), 12 if faction == "Red" else 8)
        top_pairs = valid_pairs[:top_k]
        temp = max(0.4, 2.0 * (1.0 - (gen - 1) / max(1, generations))) if faction == "Red" else max(0.2, 1.2 * (1.0 - (gen - 1) / max(1, generations)))

        scores_arr = np.array([p["pair_score"] for p in top_pairs], dtype=np.float64)
        exp_scores = np.exp((scores_arr - np.max(scores_arr)) / temp)
        sum_exp = np.sum(exp_scores)
        if sum_exp > 1e-12:
            probs = exp_scores / sum_exp
            probs = probs / np.sum(probs)  # 二次归一化保证浮点求和严格等于 1.0
        else:
            probs = np.ones(len(top_pairs), dtype=np.float64) / len(top_pairs)
        chosen_idx = int(np.random.choice(len(top_pairs), p=probs))
        chosen_pair = top_pairs[chosen_idx]

        worst_candidate = chosen_pair["w_cid"]
        best_candidate = chosen_pair["b_cid"]
        curr_w_cnt = chosen_pair["curr_w_cnt"]
        avail_steps = chosen_pair["avail_steps"]

        # 机制二：步长松动（退火与平滑单张微调）
        if len(avail_steps) > 1:
            swap_cnt = 1 if (random.random() < 0.55 or gen <= 3) else chosen_pair["max_step"]
        else:
            swap_cnt = avail_steps[0]

        is_smooth_step = (swap_cnt < curr_w_cnt)
        step_type_str = "平滑" if is_smooth_step else "整批"

        c_name_worst = cand_by_id[worst_candidate]["name"]
        c_name_best = cand_by_id[best_candidate]["name"]

        candidate_counts = dict(best_counts)
        candidate_counts[worst_candidate] -= swap_cnt
        candidate_counts[best_candidate] = candidate_counts.get(best_candidate, 0) + swap_cnt
        candidate_counts = normalize_deck_allocation(candidate_counts, candidate_ids, priority_order)

        test_winrate, t_p_win, t_p_tot, t_idles, t_hands = evaluate_decklist_performance(candidate_counts, games_per_gen)
        delta_wr = test_winrate - best_winrate
        
        gen_reflection = {
            "generation": gen, "winrate": test_winrate, "prev_winrate": prev_winrate,
            "promoted": None, "demoted": None, "monologue": ""
        }

        tolerance = -1.0 * (1.0 - (gen / generations))  # Much stricter tolerance, max -1.0 initially, approaching 0.0

        can_accept = False
        if best_winrate <= 0.0:
            can_accept = (test_winrate > 0.0)
        else:
            can_accept = (delta_wr >= tolerance)

        if can_accept:
            rejected_pairs.add((best_candidate, worst_candidate))
            remain_worst = candidate_counts.get(worst_candidate, 0)
            now_best = candidate_counts.get(best_candidate, 0)
            thought_demote = f"移出 [{c_name_worst}] x{swap_cnt} (余 {remain_worst} 张)"
            thought_promote = f"加入 [{c_name_best}] x{swap_cnt} (持 {now_best} 张)，胜率提升至 {test_winrate:.1f}% ({delta_wr:+.1f}%)"
            log_entry = f"第 {gen} 代微调 (胜率 {best_winrate:.1f}% -> {test_winrate:.1f}%, {delta_wr:+.1f}%): 采纳变异，+{swap_cnt}[{c_name_best}] / -{swap_cnt}[{c_name_worst}]"

            print(f"   [代数 {gen}/{generations}] 采纳新构筑: 胜率 {test_winrate:5.1f}% ({delta_wr:+5.1f}%) | +{swap_cnt}[{c_name_best}] / -{swap_cnt}[{c_name_worst}] (余 {remain_worst})")
            best_counts = candidate_counts
            best_winrate = test_winrate
            best_p_win, best_p_tot, best_idles, best_hands = t_p_win, t_p_tot, t_idles, t_hands
        else:
            rejected_actions.add((best_candidate, worst_candidate, swap_cnt))
            all_possible = [chosen_pair["max_step"]] if chosen_pair["max_step"] == 1 else [1, chosen_pair["max_step"]]
            if all((best_candidate, worst_candidate, s) in rejected_actions for s in all_possible):
                rejected_pairs.add((best_candidate, worst_candidate))

            thought_demote = f"尝试换入 [{c_name_best}] 胜率降至 {test_winrate:.1f}%，取消变动"
            thought_promote = f"维持原构筑，保留 [{c_name_worst}]"
            log_entry = f"第 {gen} 代微调 (实测胜率 {test_winrate:.1f}%, 变化 {delta_wr:+.1f}%): 放弃变异，回滚保留原构筑"

            print(f"   [代数 {gen}/{generations}] 拒绝调整: 实测胜率 {test_winrate:5.1f}% ({delta_wr:+5.1f}%) | 回滚保留原构筑")
            gen_reflection["winrate"] = best_winrate

        gen_reflection["promoted"] = {"id": best_candidate, "name": c_name_best, "thought": thought_promote}
        gen_reflection["demoted"] = {"id": worst_candidate, "name": c_name_worst, "thought": thought_demote}
        gen_reflection["monologue"] = f"{thought_demote}\n      {thought_promote}"

        decision_logs.append(log_entry)
        introspections.append(gen_reflection)

    return best_counts, decision_logs, introspections

def build_ppo_deck_package(faction: str, candidates: List[dict], neural_stats: List[dict],
                           allocation: Dict[int, int], decision_logs: List[str],
                           introspections: List[dict]) -> dict:
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
    deck_names = {
        "Red": "赤红·突破攻势流",
        "Blue": "蔚蓝·防御控制流",
        "Green": "翠绿·古树成长流"
    }
    archetypes = {
        "Red": "Aggro/Sacrifice",
        "Blue": "Control/Fortify",
        "Green": "Ramp/Behemoth"
    }
    deck_name = deck_names.get(faction, f"{faction}·推荐构筑")
    archetype = archetypes.get(faction, "Reinforcement-Learned")

    concept = f"基于 PPO 神经网络效用评估与实战博弈测试构筑，平均费用 {avg_cost} 费。"

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
    parser.add_argument("--generations", type=int, default=20, help="自博弈进化代数 (默认 20)")
    parser.add_argument("--games-per-gen", type=int, default=60, help="每代自博弈实机局数 (默认 60)")
    parser.add_argument("--samples", type=int, default=8, help="候选卡神经网络采样次数 (默认 8，速度提升 2.5x)")
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cuda", "cpu"], help="运算设备 (默认 cpu: 限制2线程且无GPU调度延迟；如需GPU可传 cuda)")
    args = parser.parse_args()

    req_dev = (args.device or "auto").strip().lower()
    if req_dev == "cpu":
        device = torch.device("cpu")
        print("[运算设备] 已指定使用 CPU 模式运行。")
    elif req_dev == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[运算设备] 已启用 CUDA GPU 加速: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("[运算设备] 提示: 未检测到可用 CUDA GPU，自动回退至 CPU 运算模式。")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[运算设备] 自动检测到 CUDA GPU 加速: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("[运算设备] 未检测到可用 CUDA GPU，自动采用 CPU 模式运行。")
    
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

    if device.type == "cpu":
        torch.set_num_threads(2)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
        print("[算力控制] 已启用后台静默优先级 (Below-Normal) 并严格限制 2 线程，绝不抢占前台与桌面资源。")

    model_path = find_model_path(args.model, args.stage)
    print("=" * 85)
    print("PPO 自博弈选卡工具启动")
    print(f"预期模型: {model_path} | 设备: {device}")
    print("=" * 85)

    cards_db = load_card_pool(args.cards)
    model = CardNet().to(device)
    
    if model_path and os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"[*] 成功加载 PPO 模型权重: {model_path}")
    else:
        print("[WARN] 未找到可用权重，已随机初始化神经网络进行冷启动探索！")
        
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
        alloc, decision_logs, introspections = ppo_self_play_deck_search(
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

        deck_pkg = build_ppo_deck_package(faction_name, candidates, neural_stats, alloc, decision_logs, introspections)
        print_ppo_deck_report(faction_name, deck_pkg, neural_stats)
        decks_result[faction_name] = deck_pkg

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(decks_result, f, indent=2, ensure_ascii=False)

    abs_out = os.path.abspath(args.output)
    print(f"PPO 选卡结果已保存至: {abs_out}")

    if args.report:
        export_introspection_report(decks_result, args.report)

if __name__ == "__main__":
    main()
