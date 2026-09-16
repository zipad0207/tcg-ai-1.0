import os
import sys
import json
import argparse
import time
import copy
import random
import torch
import numpy as np
from typing import Dict, List, Any, Tuple
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
    PPO 神经网络单卡效用探针 (Neural Card Probe):
    在多种法力与场面环境下将候选卡置入手牌，测量 PPO Actor 的出牌偏好概率与 Critic 的状态价值增益 (ΔV)
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
        # 随机初始化对局环境
        p0_f = faction if faction == Faction.RED else Faction.RED
        p1_f = Faction.BLUE if faction == Faction.RED else faction
        env = DuelEnv(p0_faction=p0_f, p1_faction=p1_f, cards_path=cards_path)
        obs = env.reset()

        acting_p_id = 0 if faction == Faction.RED else 1
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
    # 结合出牌意愿 (Actor 偏好) 与 价值收益 (Critic 预期)
    ppo_score = round(avg_play_prob * 65.0 + max(0.0, avg_v_gain + 1.0) * 17.5, 2)

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
                              generations: int = 5, games_per_gen: int = 60) -> Tuple[Dict[int, int], List[str]]:
    """
    PPO 蒙特卡洛自博弈卡组进化搜索 (PPO Evolutionary Deck Optimizer):
    以神经网络效用评分为先验，在实机对抗中动态统计胜率贡献与卡牌卡手率，完成最终 30 张卡组自主遴选。
    """
    candidate_ids = [c["id"] for c in candidates]
    sorted_by_ppo = sorted(neural_stats, key=lambda x: x["ppo_score"], reverse=True)
    priority_order = [x["id"] for x in sorted_by_ppo]

    # 初始化先验卡组：PPO 评分最高的卡牌优先给 3 张或 2 张
    current_counts: Dict[int, int] = {cid: 0 for cid in candidate_ids}
    allocated = 0
    for idx, cinfo in enumerate(sorted_by_ppo):
        target_count = 3 if idx < 7 else (2 if idx < 12 else 0)
        current_counts[cinfo["id"]] = target_count
        allocated += target_count

    current_counts = normalize_deck_allocation(current_counts, candidate_ids, priority_order)
    decision_logs = []

    opp_faction = Faction.BLUE if faction == "Red" else Faction.RED
    my_faction = Faction.RED if faction == "Red" else Faction.BLUE

    print(f"\n🧬 启动 PPO 强化学习自博弈进化选卡算法 (共 {generations} 轮进化，每轮 {games_per_gen} 局实测)...")

    for gen in range(1, generations + 1):
        # 组装当前测试卡组
        test_decklist = []
        for cid, count in current_counts.items():
            test_decklist.extend([cid] * count)

        # 对手卡组使用全池或对称
        p0_deck = test_decklist if my_faction == Faction.RED else None
        p1_deck = test_decklist if my_faction == Faction.BLUE else None

        env = DuelEnv(p0_faction=Faction.RED, p1_faction=Faction.BLUE, cards_path=cards_path,
                      p0_decklist=p0_deck, p1_decklist=p1_deck)

        card_played_win = Counter()
        card_played_total = Counter()
        wins = 0

        for _ in range(games_per_gen):
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
                    if h_idx < len(hand):
                        c = hand[h_idx]
                        if (my_faction == Faction.RED and curr_p == 0) or (my_faction == Faction.BLUE and curr_p == 1):
                            card_played_total[c.id] += 1
                            played_this_game.add(c.id)

                obs, _, done, _ = env.step(act)

            my_player_idx = 0 if my_faction == Faction.RED else 1
            is_win = env.players[my_player_idx].score >= env.WIN_SCORE
            if is_win:
                wins += 1
                for cid in played_this_game:
                    card_played_win[cid] += 1

        gen_winrate = (wins / games_per_gen) * 100
        print(f"   [进化代数 {gen}/{generations}] PPO 自博弈实机胜率: {gen_winrate:5.1f}%")

        # 基于胜率贡献度与出场频次微调卡牌张数
        # 找出当前卡组中实测胜率最高和最低的卡
        best_candidate = None
        worst_candidate = None
        best_metric = -999.0
        worst_metric = 999.0

        for cid in candidate_ids:
            tot = card_played_total[cid]
            if tot > 0:
                win_ratio = card_played_win[cid] / tot
                metric = win_ratio * 10.0 + (tot / games_per_gen)
            else:
                metric = 0.0

            if current_counts[cid] < MAX_COPIES_PER_CARD and metric > best_metric:
                best_metric = metric
                best_candidate = cid

            if current_counts[cid] > 0 and metric < worst_metric:
                worst_metric = metric
                worst_candidate = cid

        # 变异微调
        if best_candidate and worst_candidate and best_candidate != worst_candidate:
            current_counts[best_candidate] += 1
            current_counts[worst_candidate] -= 1
            current_counts = normalize_deck_allocation(current_counts, candidate_ids, priority_order)
            c_name_best = next((c["name"] for c in candidates if c["id"] == best_candidate), str(best_candidate))
            c_name_worst = next((c["name"] for c in candidates if c["id"] == worst_candidate), str(worst_candidate))
            decision_logs.append(f"第 {gen} 代微调: 增选高效卡 [{c_name_best}] (+1张)，减选低效卡 [{c_name_worst}] (-1张)")

    return current_counts, decision_logs

def build_ppo_deck_package(faction: str, candidates: List[dict], neural_stats: List[dict],
                           allocation: Dict[int, int], decision_logs: List[str]) -> dict:
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
    deck_name = f"赤红·PPO特训突破流" if faction == "Red" else f"蔚蓝·PPO特训要塞流"
    concept = (
        f"由 PPO 强化学习智能体自主经过神经网络效用评估与实机对战进化遴选所得。"
        f"全套构筑最大化 PPO 决策置信度与状态价值增益 (ΔV)，平均费用 {avg_cost} 费。"
    )

    return {
        "deck_name": deck_name,
        "archetype": "PPO-Reinforcement-Learned",
        "tactical_concept": concept,
        "decision_evolution_logs": decision_logs,
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
    print(f"🧠 【{faction}】PPO 强化学习智能体自主选卡报告: 《{deck_pkg['deck_name']}》")
    print(f"📌 构筑属性: 智能体自研策略 | 牌库规模: {deck_pkg['total_cards']} 张 | 平均费用: {deck_pkg['avg_cost']} 费")
    print(f"📐 体系构成: 随从 {deck_pkg['minion_count']} 张 / 法术 {deck_pkg['spell_count']} 张")
    print("─" * 85)
    print("🎯 PPO 神经网络单卡效用评分与入选结果 (Actor 动作偏好度 & Critic 价值收益 ΔV):")
    print(f"{'ID':<6}{'名称':<12}{'费用':<6}{'类型':<8}{'PPO出牌偏好':<14}{'Critic收益(ΔV)':<16}{'PPO评分':<10}{'入选张数'}")
    print("─" * 85)

    sorted_stats = sorted(neural_stats, key=lambda x: x["ppo_score"], reverse=True)
    alloc = deck_pkg["card_allocation"]

    for st in sorted_stats:
        cid = st["id"]
        count = alloc.get(str(cid), 0)
        status = f"✅ {count} 张" if count > 0 else "❌ 弃选 (0张)"
        print(f"{cid:<6}{st['name']:<12}{st['cost']:<6}{st['card_type']:<8}{str(st['play_prob'])+'%':<14}{str(st['value_gain']):<16}{st['ppo_score']:<10}{status}")

    print("─" * 85)
    print("📊 PPO 自主规划法力曲线 (Mana Curve):")
    for cost in range(1, 8):
        cnt = deck_pkg["mana_curve"].get(cost, 0)
        bar = "█" * (cnt * 2)
        print(f"   {cost} 费: {bar:<22} ({cnt} 张)")
    cnt_high = sum(deck_pkg["mana_curve"].get(c, 0) for c in range(8, 10))
    if cnt_high > 0:
        bar = "█" * (cnt_high * 2)
        print(f" 8+ 费: {bar:<22} ({cnt_high} 张)")

    if deck_pkg.get("decision_evolution_logs"):
        print("─" * 85)
        print("🔄 PPO 蒙特卡洛实机进化微调记录:")
        for log in deck_pkg["decision_evolution_logs"]:
            print(f"   • {log}")

    print("═" * 85 + "\n")

def main():
    parser = argparse.ArgumentParser(description="PPO 强化学习智能体自主选卡构筑系统 (PPO Deckbuilder)")
    parser.add_argument("--cards", type=str, default="cards_config.json", help="卡池配置文件路径")
    parser.add_argument("--model", type=str, default=None, help="PPO 模型权重路径")
    parser.add_argument("--stage", type=str, default="tuned", choices=["baseline", "tuned"], help="选用训练模型阶段")
    parser.add_argument("--output", type=str, default="decks_config.json", help="输出卡组保存文件")
    parser.add_argument("--factions", type=str, default="Red,Blue", help="目标自主选卡阵营 (默认 Red,Blue)")
    parser.add_argument("--generations", type=int, default=5, help="自博弈进化代数 (默认 5)")
    parser.add_argument("--games-per-gen", type=int, default=60, help="每代自博弈实机局数 (默认 60)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = find_model_path(args.model, args.stage)
    if not model_path:
        print("❌ 未检测到可用的 PPO 权重模型！请先确认模型文件存在。")
        return

    print("=" * 85)
    print("🤖 PPO 强化学习智能体自主选卡系统启动 (Self-Play Autonomous Deckbuilding)")
    print(f"⚙️ 核心权重: {model_path} | 运行设备: {device}")
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
        print(f"\n🔍 [阶段 1/2] 正在提取【{faction_name}】候选卡池，启动神经网络效用探针...")
        candidates = get_faction_candidates(cards_db, faction_name)
        f_enum = Faction.RED if faction_name == "Red" else (Faction.BLUE if faction_name == "Blue" else Faction.GREEN)

        neural_stats = []
        for c in candidates:
            st = evaluate_card_neural_utility(model, device, c, f_enum, args.cards, samples=20)
            neural_stats.append(st)

        print(f"⚔️ [阶段 2/2] 正在执行 PPO 实机自博弈对抗与卡组进化筛选...")
        alloc, decision_logs = ppo_self_play_deck_search(
            faction=faction_name,
            candidates=candidates,
            neural_stats=neural_stats,
            model=model,
            device=device,
            cards_path=args.cards,
            generations=args.generations,
            games_per_gen=args.games_per_gen
        )

        deck_pkg = build_ppo_deck_package(faction_name, candidates, neural_stats, alloc, decision_logs)
        print_ppo_deck_report(faction_name, deck_pkg, neural_stats)
        decks_result[faction_name] = deck_pkg

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(decks_result, f, indent=2, ensure_ascii=False)

    abs_out = os.path.abspath(args.output)
    print(f"💾 PPO 自主选卡构筑成果已成功同步保存至: {abs_out}")

if __name__ == "__main__":
    main()
