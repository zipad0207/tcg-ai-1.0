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
    f_cards = pool_data.get(faction, [])
    n_cards = pool_data.get("Neutral", [])
    return f_cards + n_cards

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

        # 随机设置合理法力阶段 (从卡牌费用至 10 费)
        sim_mana = random.randint(max(1, card_obj.cost), 10)
        player.mana = sim_mana
        player.max_mana = sim_mana

        # 注入候选卡至手牌第 0 位
        player.hand = [card_obj] + player.hand[:3]

        mask = env.get_action_mask()
        # 检查候选卡对应的 4 个动作槽位中是否有合法动作
        card_action_indices = [0, 1, 2, 3] # slot 0 的 4 种打出目标
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

    avg_play_prob = float(np.mean(play_probs)) if play_probs else 0.0
    avg_v_gain = float(np.mean(value_gains)) if value_gains else 0.0

    # 综合 PPO 神经效用评分 (0 ~ 100 分)
    # 强化 Critic 预期净收益 (ΔV) 权重，适度平衡 Actor 偏好，避免低费白板因容易打出而虚高
    v_norm = (avg_v_gain + 0.05) * 400.0  # ΔV 从 -0.05~+0.05 映射为 0~40
    p_norm = avg_play_prob * 100.0 * 0.6  # 出牌概率映射为 10~30
    ppo_score = round(max(20.0, min(98.0, 30.0 + v_norm + p_norm)), 2)

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
            if total == DECK_SIZE:
                break

    # 若超过 30 张，按 PPO 效用最低优先级依次削减
    elif total > DECK_SIZE:
        for cid in reversed(priority_order):
            while clean_counts[cid] > 0 and total > DECK_SIZE:
                clean_counts[cid] -= 1
                total -= 1
            if total == DECK_SIZE:
                break

    return clean_counts

def ppo_self_play_deck_search(faction: str, candidates: List[dict], neural_stats: List[dict],
                              model: CardNet, device: torch.device, cards_path: str,
                              generations: int = 5, games_per_gen: int = 60,
                              existing_decks: dict = None) -> Tuple[Dict[int, int], List[str], List[dict]]:
    """
    PPO 蒙特卡洛自博弈卡组进化搜索 (PPO Evolutionary Deck Optimizer):
    引入法力曲线与终结者保护机制，以神经网络效用评分为先验，在实机对抗中动态统计胜率贡献与卡牌卡手率，完成最终 30 张卡组自主遴选。
    """
    candidate_ids = [c["id"] for c in candidates]
    cand_by_id = {c["id"]: c for c in candidates}
    sorted_by_ppo = sorted(neural_stats, key=lambda x: x["ppo_score"], reverse=True)
    priority_order = [x["id"] for x in sorted_by_ppo]

    # 法力曲线分段 (低费 1-2 / 中费 3-4 / 高费 >= 5)
    sorted_low = [c["id"] for c in sorted_by_ppo if cand_by_id[c["id"]]["cost"] <= 2]
    sorted_mid = [c["id"] for c in sorted_by_ppo if 3 <= cand_by_id[c["id"]]["cost"] <= 4]
    sorted_high = [c["id"] for c in sorted_by_ppo if cand_by_id[c["id"]]["cost"] >= 5]

    # 各阵营天然法力曲线配额 (低/中/高)
    curve_quotas = {
        "Red": (14, 12, 4),
        "Blue": (10, 12, 8),
        "Green": (10, 12, 8)
    }.get(faction, (12, 12, 6))

    target_low, target_mid, target_high = curve_quotas
    current_counts: Dict[int, int] = {cid: 0 for cid in candidate_ids}

    # 1. 优先按分段装填最高 PPO 效用卡牌
    def fill_bracket(bracket_cids, quota):
        rem = quota
        for cid in bracket_cids:
            take = min(MAX_COPIES_PER_CARD, rem)
            current_counts[cid] = take
            rem -= take
            if rem <= 0:
                break

    fill_bracket(sorted_low, target_low)
    fill_bracket(sorted_mid, target_mid)
    fill_bracket(sorted_high, target_high)

    current_counts = normalize_deck_allocation(current_counts, candidate_ids, priority_order)
    decision_logs = []
    
    faction_map = {"Red": Faction.RED, "Blue": Faction.BLUE, "Green": Faction.GREEN}
    name_map = {Faction.RED: "Red", Faction.BLUE: "Blue", Faction.GREEN: "Green"}
    my_faction = faction_map.get(faction, Faction.RED)
    opp_factions = [f for f in [Faction.RED, Faction.BLUE, Faction.GREEN] if f != my_faction]

    introspections = []
    prev_winrate = 50.0

    print(f"\n启动 PPO 自博弈卡组搜索 (共 {generations} 轮，每轮 {games_per_gen} 局)...")

    for gen in range(1, generations + 1):
        # 组装当前测试卡组
        test_decklist = []
        for cid, count in current_counts.items():
            test_decklist.extend([cid] * count)

        card_played_win = Counter()
        card_played_total = Counter()
        wins = 0

        for _ in range(games_per_gen):
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

                if act != env.action_space_size - 1:
                    h_idx = act // 4
                    hand = env.players[curr_p].hand
                    if h_idx < len(hand) and curr_p == 0:
                        c = hand[h_idx]
                        card_played_total[c.id] += 1
                        played_this_game.add(c.id)

                obs, _, done, _ = env.step(act)

            is_win = (env.winner == 0) if env.winner is not None else (env.players[0].score >= env.WIN_SCORE)
            if is_win:
                wins += 1
                for cid in played_this_game:
                    card_played_win[cid] += 1

        gen_winrate = (wins / games_per_gen) * 100
        print(f"   [进化代数 {gen}/{generations}] PPO 实战自博弈胜率: {gen_winrate:5.1f}% (上一代: {prev_winrate:5.1f}%)")

        # 基于真实出场胜率进行多维考量（消除低费出场次数虚高偏差）
        best_candidate = None
        worst_candidate = None
        best_metric = -999.0
        worst_metric = 999.0

        # 当前卡组高费牌数量统计 (避免将终结核弹全部洗出卡组)
        curr_high_count = sum(cnt for cid, cnt in current_counts.items() if cand_by_id[cid]["cost"] >= 5)

        for cid in candidate_ids:
            tot = card_played_total[cid]
            c_cost = cand_by_id[cid]["cost"]
            if tot >= 3:
                win_ratio = card_played_win[cid] / tot
                metric = win_ratio * 10.0
            elif tot > 0:
                win_ratio = card_played_win[cid] / tot
                metric = win_ratio * 7.0 + 1.5
            else:
                metric = 4.5  # 样本过少时保护，不盲目剔除

            if current_counts[cid] < MAX_COPIES_PER_CARD and metric > best_metric:
                best_metric = metric
                best_candidate = cid

            # 约束：若高费牌数量已降至下限，则不再削减高费牌
            can_demote = True
            if c_cost >= 5 and curr_high_count <= (2 if faction == "Red" else 5):
                can_demote = False

            if current_counts[cid] > 0 and metric < worst_metric and can_demote:
                worst_metric = metric
                worst_candidate = cid

        # 生成 PPO 第一人称心智自省对话
        gen_reflection = {
            "generation": gen,
            "winrate": gen_winrate,
            "prev_winrate": prev_winrate,
            "promoted": None,
            "demoted": None,
            "monologue": ""
        }

        if best_candidate and worst_candidate and best_candidate != worst_candidate:
            current_counts[best_candidate] += 1
            current_counts[worst_candidate] -= 1
            current_counts = normalize_deck_allocation(current_counts, candidate_ids, priority_order)

            c_best = next((c for c in candidates if c["id"] == best_candidate), {})
            c_worst = next((c for c in candidates if c["id"] == worst_candidate), {})
            c_name_best = c_best.get("name", str(best_candidate))
            c_name_worst = c_worst.get("name", str(worst_candidate))

            best_played = card_played_total[best_candidate]
            best_win = card_played_win[best_candidate]
            best_win_pct = (best_win / best_played * 100) if best_played > 0 else 0.0

            worst_played = card_played_total[worst_candidate]
            worst_win = card_played_win[worst_candidate]
            worst_win_pct = (worst_win / worst_played * 100) if worst_played > 0 else 0.0

            thought_demote = (
                f"[{c_name_worst}] 本轮实测出战 {worst_played} 次，出场胜率仅 {worst_win_pct:.1f}%，表现不佳，调减 1 张（调整为 {current_counts[worst_candidate]} 张）。"
            )

            thought_promote = (
                f"[{c_name_best}] 本轮实测出战 {best_played} 次，出场胜率达 {best_win_pct:.1f}%，表现良好，追加 1 张（调整为 {current_counts[best_candidate]} 张）。"
            )

            gen_reflection["promoted"] = {
                "id": best_candidate, "name": c_name_best, "win_pct": best_win_pct, "play_count": best_played, "thought": thought_promote
            }
            gen_reflection["demoted"] = {
                "id": worst_candidate, "name": c_name_worst, "win_pct": worst_win_pct, "play_count": worst_played, "thought": thought_demote
            }
            gen_reflection["monologue"] = f"{thought_demote}\n      {thought_promote}"

            decision_logs.append(f"第 {gen} 代微调: 增选高效卡 [{c_name_best}] (+1张)，减选低效卡 [{c_name_worst}] (-1张)")
            introspections.append(gen_reflection)

        prev_winrate = gen_winrate

    return current_counts, decision_logs, introspections

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
        "Red": "赤红·PPO自主进化突破流",
        "Blue": "蔚蓝·PPO自主进化铁壁流",
        "Green": "翠绿·PPO自主进化古树流"
    }
    archetypes = {
        "Red": "PPO-Aggro/Sacrifice",
        "Blue": "PPO-Control/Fortify",
        "Green": "PPO-Ramp/Behemoth"
    }
    deck_name = deck_names.get(faction, f"{faction}·PPO自选流")
    archetype = archetypes.get(faction, "PPO-Reinforcement-Learned")

    concept = (
        f"由 PPO 强化学习智能体自主经过神经网络效用评估与实机对战进化遴选所得。"
        f"全套构筑最大化 PPO 决策置信度与状态价值增益 (ΔV)，平均费用 {avg_cost} 费。"
    )

    top_cards = sorted(card_details, key=lambda x: x.get("ppo_score", 0), reverse=True)[:2]
    key_combos = [
        f"{top_cards[0]['name']} (PPO评分 {top_cards[0]['ppo_score']}): 作为核心驱动点，提供最大状态价值增益。",
        f"{top_cards[1]['name']} (PPO评分 {top_cards[1]['ppo_score']}): 作为主力节奏支撑，协同完成场面压制与胜点累积。"
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
    print("\n" + "═" * 85)
    print(f"【{faction}】PPO 推荐卡组: 《{deck_pkg['deck_name']}》")
    print(f"牌库规模: {deck_pkg['total_cards']} 张 | 平均费用: {deck_pkg['avg_cost']} 费")
    print(f"随从 {deck_pkg['minion_count']} 张 / 法术 {deck_pkg['spell_count']} 张")
    print("─" * 85)
    print("单卡效用评分与入选结果:")
    print(f"{'ID':<6}{'名称':<12}{'费用':<6}{'类型':<8}{'PPO出牌偏好':<14}{'Critic收益(ΔV)':<16}{'PPO评分':<10}{'入选张数'}")
    print("─" * 85)

    sorted_stats = sorted(neural_stats, key=lambda x: x["ppo_score"], reverse=True)
    alloc = deck_pkg["card_allocation"]

    for st in sorted_stats:
        cid = st["id"]
        count = alloc.get(str(cid), 0)
        status = f"{count} 张" if count > 0 else "未入选"
        print(f"{cid:<6}{st['name']:<12}{st['cost']:<6}{st['card_type']:<8}{str(st['play_prob'])+'%':<14}{str(st['value_gain']):<16}{st['ppo_score']:<10}{status}")

    print("─" * 85)
    print("法力曲线分布:")
    for cost in range(1, 8):
        cnt = deck_pkg["mana_curve"].get(cost, 0)
        bar = "█" * (cnt * 2)
        print(f"   {cost} 费: {bar:<22} ({cnt} 张)")
    cnt_high = sum(deck_pkg["mana_curve"].get(c, 0) for c in range(8, 10))
    if cnt_high > 0:
        bar = "█" * (cnt_high * 2)
        print(f" 8+ 费: {bar:<22} ({cnt_high} 张)")

    if deck_pkg.get("introspections"):
        print("─" * 85)
        print("各轮对战调整记录:")
        for intro in deck_pkg["introspections"]:
            g = intro["generation"]
            wr = intro["winrate"]
            pwr = intro["prev_winrate"]
            print(f"\n   第 {g} 轮测试胜率: {wr:.1f}% (变化: {wr - pwr:+.1f}%)")
            if intro.get("demoted"):
                print(f"   │  {intro['demoted']['thought']}")
            if intro.get("promoted"):
                print(f"   │  {intro['promoted']['thought']}")
            

    print("═" * 85 + "\n")

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
    parser.add_argument("--report", type=str, default="ppo_introspection_report.md", help="输出心智自省报告Markdown文件名")
    parser.add_argument("--factions", type=str, default="Red,Blue,Green", help="目标自主选卡阵营 (默认 Red,Blue,Green)")
    parser.add_argument("--generations", type=int, default=5, help="自博弈进化代数 (默认 5)")
    parser.add_argument("--games-per-gen", type=int, default=60, help="每代自博弈实机局数 (默认 60)")
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
            st = evaluate_card_neural_utility(model, device, c, f_enum, args.cards, samples=20)
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

    # 导出心智自省报告
    if args.report:
        export_introspection_report(decks_result, args.report)

if __name__ == "__main__":
    main()
