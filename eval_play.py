import os
import sys
import argparse
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sandbox import DuelEnv, Faction
from agent import CardNet
def get_faction_meta(faction_val, player_id):
    f_str = str(faction_val).lower()
    if "green" in f_str or "绿" in f_str:
        return {"name": f"🟢 翠绿 (P{player_id} 林野跳费流)", "short": "🟢 绿方", "def_label": "🛡️ 绿防", "atk_label": "⚔️ 绿冲", "team_class": "green", "color": "#2ed573"}
    elif "blue" in f_str or "蓝" in f_str:
        return {"name": f"🔵 蔚蓝 (P{player_id} 守卫控制流)", "short": "🔵 蓝方", "def_label": "🛡️ 蓝防", "atk_label": "⚔️ 蓝冲", "team_class": "blue", "color": "#1e90ff"}
    else:
        return {"name": f"🔴 赤红 (P{player_id} 快攻突破流)", "short": "🔴 红方", "def_label": "🛡️ 红防", "atk_label": "⚔️ 红冲", "team_class": "red", "color": "#ff4757"}

def format_terminal_board(turn_count, acting_player, p0, p1, lanes, action_desc, result_log=None):
    lines = []
    w = 78
    m0 = get_faction_meta(p0.get("faction", "Red"), 0)
    m1 = get_faction_meta(p1.get("faction", "Blue"), 1)
    
    p0_score_bar = "■" * p0["score"] + "□" * (7 - p0["score"])
    p1_score_bar = "■" * p1["score"] + "□" * (7 - p1["score"])
    
    lines.append("╔" + "═" * (w - 2) + "╗")
    p1_header = f" {m1['name']}  得分: [{p1_score_bar}] {p1['score']}/7  法力: 💎 {p1['mana']}/{p1['max_mana']}  手牌: {len(p1['hand'])}张"
    lines.append(f"║{p1_header:<{w-2}}║")
    lines.append("╠" + "═" * 38 + "╦" + "═" * 37 + "╣")
    lines.append("║                【左路战场】          ║               【右路战场】          ║")
    
    def fmt_units(units, is_atk=False):
        if not units: return "空"
        res = []
        for u in units:
            if is_atk: res.append(f"{u['name']}({u['dp']})[{'⚡就绪' if u.get('ready', False) else '⏳蓄势'}]")
            else: res.append(f"{u['name']}(DP:{u['dp']})")
        return "、".join(res)

    lines.append(f"║ {m1['def_label']}: {fmt_units(lanes[0]['p1_defenders']):<27} ║ {m1['def_label']}: {fmt_units(lanes[1]['p1_defenders']):<26} ║")
    lines.append(f"║ {m1['atk_label']}: {fmt_units(lanes[0]['p1_attackers'], True):<27} ║ {m1['atk_label']}: {fmt_units(lanes[1]['p1_attackers'], True):<26} ║")
    lines.append("║ ┄┄┄┄┄┄┄┄ ⚡ 攻防对撞线 ┄┄┄┄┄┄┄┄ ╫ ┄┄┄┄┄┄┄┄ ⚡ 攻防对撞线 ┄┄┄┄┄┄┄┄ ║")
    lines.append(f"║ {m0['atk_label']}: {fmt_units(lanes[0]['p0_attackers'], True):<27} ║ {m0['atk_label']}: {fmt_units(lanes[1]['p0_attackers'], True):<26} ║")
    lines.append(f"║ {m0['def_label']}: {fmt_units(lanes[0]['p0_defenders']):<27} ║ {m0['def_label']}: {fmt_units(lanes[1]['p0_defenders']):<26} ║")
    
    lines.append("╠" + "═" * 38 + "╩" + "═" * 37 + "╣")
    p0_header = f" {m0['name']}  得分: [{p0_score_bar}] {p0['score']}/7  法力: 💎 {p0['mana']}/{p0['max_mana']}  手牌: {len(p0['hand'])}张"
    lines.append(f"║{p0_header:<{w-2}}║")
    lines.append("╚" + "═" * (w - 2) + "╝")
    
    act_p_str = m0['short'] if acting_player == 0 else m1['short']
    lines.append(f"👉 [第 {turn_count:02d} 回合] {act_p_str} 决策: {action_desc}")
    if result_log: lines.append(f"💥 {result_log}")
    return "\n".join(lines)
def format_action_desc(env, action_id):
    if action_id == env.action_space_size - 1:
        return "结束回合 (PASS - 触发冲锋交战)"
    hand_idx = action_id // 4
    target_code = action_id % 4
    lane_str = "【左路】" if target_code in [0, 1] else "【右路】"
    curr_hand = env.players[env.current_player].hand
    if hand_idx >= len(curr_hand):
        return f"动作编码 {action_id}"
    card = curr_hand[hand_idx]
    
    if card.card_type.name == "SPELL":
        return f"释放法术 [{card.name} (费用:{card.cost} 词条:{card.tags})] 到 {lane_str}"
    else:
        pos_str = "进攻区(发起冲锋)" if target_code in [0, 2] else "防守区(筑起防线)"
        return f"部署随从 [{card.name} (费用:{card.cost} DP:{card.base_dp} 词条:{card.tags})] 到 {lane_str}{pos_str}"

def find_model_path(requested_path: str = None, stage: str = "tuned") -> str:
    """智能查找最优可用的权重文件"""
    candidates = []
    if requested_path:
        candidates.append(requested_path)
    candidates.append(f"card_ppo_model_{stage}.pth")
    candidates.append("card_ppo_model_brawl.pth")
    candidates.append("card_ppo_model.pth")
    candidates.append("card_ppo_model_tuned.pth")
    candidates.append("card_ppo_model_baseline.pth")

    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def evaluate():
    parser = argparse.ArgumentParser(description="TCG 对局回放与评估")
    parser.add_argument("--stage", type=str, default="tuned", choices=["baseline", "tuned"],
                        help="选择评估模型阶段: tuned(调优后模型) 或 baseline(基准模型)")
    parser.add_argument("--model", type=str, default=None, help="自定义指定模型权重文件路径")
    parser.add_argument("--cards", type=str, default=None, help="自定义指定卡池配置文件路径")
    parser.add_argument("--decks", type=str, default="decks_config.json", help="卡组配置文件路径 (默认 decks_config.json)")
    parser.add_argument("--p0", "--f0", dest="p0_faction", type=str, default="Red", choices=["Red", "Blue", "Green"],
                        help="先手 P0 阵营 (默认 Red)")
    parser.add_argument("--p1", "--f1", dest="p1_faction", type=str, default="Blue", choices=["Red", "Blue", "Green"],
                        help="后手 P1 阵营 (默认 Blue)")
    parser.add_argument("--html", type=str, default="battle_replay.html", help="导出网页回放文件名")
    parser.add_argument("--tactical", action="store_true", default=True, help="启用阵营战术策略引导 (消除旧模型防守抑制偏差，默认开启)")
    parser.add_argument("--no-tactical", dest="tactical", action="store_false", help="禁用战术引导，使用纯网络原始输出")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 查找模型
    model_path = find_model_path(args.model, args.stage)
    if not model_path:
        print("未找到可用的权重文件，请先运行 train.py 进行训练。")
        return

    # 动态匹配卡池文件
    if args.cards:
        cards_path = args.cards
    else:
        stage_cards = f"cards_config_{args.stage}.json"
        cards_path = stage_cards if os.path.exists(stage_cards) else "cards_config.json"

    # 解析阵营枚举与元信息
    f_map = {"Red": Faction.RED, "Blue": Faction.BLUE, "Green": Faction.GREEN}
    p0_f = f_map.get(args.p0_faction, Faction.RED)
    p1_f = f_map.get(args.p1_faction, Faction.BLUE)

    faction_meta = {
        Faction.RED: ("红方", "快攻流"),
        Faction.BLUE: ("蓝方", "控制流"),
        Faction.GREEN: ("绿方", "跳费流")
    }
    p0_tag, p0_style = faction_meta.get(p0_f, ("红方", "赤红"))
    p1_tag, p1_style = faction_meta.get(p1_f, ("蓝方", "蔚蓝"))

    # 加载卡组
    p0_decklist, p1_decklist = None, None
    p0_deck_name = f"{args.p0_faction}卡组"
    p1_deck_name = f"{args.p1_faction}卡组"

    if args.decks and os.path.exists(args.decks):
        import json
        with open(args.decks, "r", encoding="utf-8") as df:
            decks_cfg = json.load(df)
            if args.p0_faction in decks_cfg:
                p0_decklist = decks_cfg[args.p0_faction].get("decklist")
                p0_deck_name = decks_cfg[args.p0_faction].get("deck_name", p0_deck_name)
            if args.p1_faction in decks_cfg:
                p1_decklist = decks_cfg[args.p1_faction].get("decklist")
                p1_deck_name = decks_cfg[args.p1_faction].get("deck_name", p1_deck_name)
        print(f"已加载卡组: {p0_tag}《{p0_deck_name}》 vs {p1_tag}《{p1_deck_name}》")

    print(f"加载模型权重: {model_path} | 卡池文件: {cards_path}")
    env = DuelEnv(p0_faction=p0_f, p1_faction=p1_f, cards_path=cards_path,
                  p0_decklist=p0_decklist, p1_decklist=p1_decklist)
    
    model = CardNet(action_dim=env.action_space_size).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    obs = env.reset()
    done = False
    snapshots = []
    
    print("\n" + "="*80)
    print(f"对局回放启动 ({p0_tag} vs {p1_tag}) | 模型: {os.path.basename(model_path)}")
    print(f"对战阵列: {p0_tag}《{p0_deck_name}》 VS {p1_tag}《{p1_deck_name}》")
    print("="*80)

    while not done:
        curr_p = env.current_player
        acting_turn = env.turn_count
        p_obj = env.players[curr_p]
        f_name = p0_tag if curr_p == 0 else p1_tag
        
        mask = env.get_action_mask()
        state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        mask_t = torch.FloatTensor(mask).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, val = model(state_t, mask_t)

            if args.tactical:
                opp_id = 1 - curr_p
                for idx, card in enumerate(p_obj.hand):
                    if idx >= env.MAX_HAND_SIZE:
                        break
                    for lane_id in [0, 1]:
                        def_slot = idx * 4 + (1 if lane_id == 0 else 3)
                        if mask[def_slot] > 0.5:
                            bonus = 0.0
                            # 1. 词条加成：FORTIFY 坚守在防守区获得额外生命/DP
                            for tag in card.tags:
                                if tag.startswith("FORTIFY_"):
                                    bonus += float(tag.split("_")[1]) * 0.8
                            # 2. 战场威胁感知：同路有敌方蓄势冲锋部队，防守拦截收益巨大
                            enemy_threat = sum(u.current_dp for u in env.lanes[lane_id].attackers if u.owner == opp_id)
                            if enemy_threat > 0:
                                bonus += 1.2
                            # 3. 阵营战术风格倾向：蔚蓝(防守控制)天然重视驻防，翠绿面对威胁时优先护脸
                            if p_obj.faction == Faction.BLUE:
                                bonus += 0.8
                            elif p_obj.faction == Faction.GREEN and enemy_threat > 0:
                                bonus += 0.5
                            logits[0, def_slot] += bonus

            action = torch.argmax(logits, dim=-1).item()

        action_desc = format_action_desc(env, action)
        prev_score = env.players[curr_p].score
        obs, reward, done, info = env.step(action)

        result_log = None
        if action == env.action_space_size - 1:
            gained = env.players[curr_p].score - prev_score
            s0 = env.players[0].score
            s1 = env.players[1].score
            bonus_score_pts = info.get("bonus_score_pts", getattr(env, "last_bonus_score_pts", 0))
            if gained > 0:
                if bonus_score_pts > 0:
                    base_pts = gained - bonus_score_pts
                    result_log = f"⚔️ 冲锋突破！{f_name} 本回合斩获 +{gained} 分 (基础{base_pts}分 + 词条额外加成{bonus_score_pts}分)！| 实时比分 -> {p0_tag} {s0} : {s1} {p1_tag}"
                else:
                    result_log = f"⚔️ 冲锋突破！{f_name} 本回合斩获 +{gained} 分！| 实时比分 -> {p0_tag} {s0} : {s1} {p1_tag}"
            else:
                result_log = f"🛡️ 防线阻挡/蓄势完成 | 实时比分 -> {p0_tag} {s0} : {s1} {p1_tag}"

            # 打印回合全景战局看板
            board_str = format_terminal_board(
                turn_count=acting_turn,
                acting_player=curr_p,
                p0={"score": env.players[0].score, "mana": env.players[0].mana, "max_mana": env.players[0].max_mana, "faction": env.p0_faction.value, "hand": env.players[0].hand},
                p1={"score": env.players[1].score, "mana": env.players[1].mana, "max_mana": env.players[1].max_mana, "faction": env.p1_faction.value, "hand": env.players[1].hand},
                lanes=[
                    {
                        "p0_attackers": [{"name": u.card.name, "dp": u.current_dp, "ready": u.ready_to_attack} for u in env.lanes[0].attackers if u.owner == 0],
                        "p1_attackers": [{"name": u.card.name, "dp": u.current_dp, "ready": u.ready_to_attack} for u in env.lanes[0].attackers if u.owner == 1],
                        "p0_defenders": [{"name": u.card.name, "dp": u.current_dp} for u in env.lanes[0].defenders if u.owner == 0],
                        "p1_defenders": [{"name": u.card.name, "dp": u.current_dp} for u in env.lanes[0].defenders if u.owner == 1]
                    },
                    {
                        "p0_attackers": [{"name": u.card.name, "dp": u.current_dp, "ready": u.ready_to_attack} for u in env.lanes[1].attackers if u.owner == 0],
                        "p1_attackers": [{"name": u.card.name, "dp": u.current_dp, "ready": u.ready_to_attack} for u in env.lanes[1].attackers if u.owner == 1],
                        "p0_defenders": [{"name": u.card.name, "dp": u.current_dp} for u in env.lanes[1].defenders if u.owner == 0],
                        "p1_defenders": [{"name": u.card.name, "dp": u.current_dp} for u in env.lanes[1].defenders if u.owner == 1]
                    }
                ],
                action_desc=action_desc,
                result_log=result_log
            )
            print("\n" + board_str)
        else:
            print(f"👉 [第 {acting_turn:02d} 回合] {f_name} 动作: {action_desc} (余法力: {p_obj.mana}/{p_obj.max_mana})")

    winner_idx = env.winner if env.winner is not None else (0 if env.players[0].score >= env.WIN_SCORE else 1)
    winner = f"{p0_tag} ({p0_style})" if winner_idx == 0 else f"{p1_tag} ({p1_style})"
    print("\n" + "="*80)
    print(f"🏁 对局结算完毕！获胜方: 【{winner}】")
    print(f"最终比分: {p0_tag} {env.players[0].score} : {env.players[1].score} {p1_tag} (总回合数: {env.turn_count} 轮)")
    print("="*80)


if __name__ == "__main__":
    evaluate()