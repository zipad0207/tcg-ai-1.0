import os
import sys
import argparse
import torch
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sandbox import DuelEnv, Faction
from agent import CardNet
from visualizer import format_terminal_board, export_html_replay

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
    candidates.append(f"PythonApplication23/card_ppo_model_{stage}.pth")
    candidates.append("card_ppo_model.pth")
    candidates.append("PythonApplication23/card_ppo_model.pth")
    candidates.append("card_ppo_model_tuned.pth")
    candidates.append("PythonApplication23/card_ppo_model_tuned.pth")
    candidates.append("card_ppo_model_baseline.pth")
    candidates.append("PythonApplication23/card_ppo_model_baseline.pth")

    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def pack_snapshot(env, turn_count, acting_player, action_desc, result_log=None):
    def pack_p(p):
        return {
            "score": p.score,
            "mana": p.mana,
            "max_mana": p.max_mana,
            "hand": [{"name": c.name, "cost": c.cost, "dp": c.base_dp, "tags": c.tags} for c in p.hand]
        }
    def pack_l(lanes):
        res = []
        for l_id in [0, 1]:
            lane = lanes[l_id]
            res.append({
                "lane_id": l_id,
                "p0_attackers": [{"name": u.card.name, "dp": u.current_dp, "ready": u.ready_to_attack} for u in lane.attackers if u.owner == 0],
                "p1_attackers": [{"name": u.card.name, "dp": u.current_dp, "ready": u.ready_to_attack} for u in lane.attackers if u.owner == 1],
                "p0_defenders": [{"name": u.card.name, "dp": u.current_dp} for u in lane.defenders if u.owner == 0],
                "p1_defenders": [{"name": u.card.name, "dp": u.current_dp} for u in lane.defenders if u.owner == 1]
            })
        return res

    return {
        "turn": turn_count,
        "acting_player": acting_player,
        "action_desc": action_desc,
        "result_log": result_log,
        "p0": pack_p(env.players[0]),
        "p1": pack_p(env.players[1]),
        "lanes": pack_l(env.lanes)
    }

def evaluate():
    parser = argparse.ArgumentParser(description="TCG AI 对局全景回放与评估")
    parser.add_argument("--stage", type=str, default="tuned", choices=["baseline", "tuned"],
                        help="选择评估模型阶段: tuned(调优后平衡模型) 或 baseline(基准模型)")
    parser.add_argument("--model", type=str, default=None, help="自定义指定模型权重文件路径")
    parser.add_argument("--cards", type=str, default=None, help="自定义指定卡池配置文件路径 (默认根据 stage 自动选择)")
    parser.add_argument("--html", type=str, default="battle_replay.html", help="导出可交互网页回放文件名")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 查找模型
    model_path = find_model_path(args.model, args.stage)
    if not model_path:
        print(f"❌ 未找到可用的权重文件！请先运行 train.py 进行训练。")
        return

    # 动态匹配卡池文件
    if args.cards:
        cards_path = args.cards
    else:
        stage_cards = f"cards_config_{args.stage}.json"
        cards_path = stage_cards if os.path.exists(stage_cards) else "cards_config.json"

    print(f"📦 正在加载智能体模型权重: {model_path} | 卡池文件: {cards_path}")
    env = DuelEnv(p0_faction=Faction.RED, p1_faction=Faction.BLUE, cards_path=cards_path)
    
    model = CardNet(action_dim=env.action_space_size).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    obs = env.reset()
    done = False
    snapshots = []
    
    print("\n" + "="*80)
    print(f"🎮 AI 对局全景回放启动 (红方 vs 蓝方) | 模型: {os.path.basename(model_path)}")
    print("="*80)

    while not done:
        curr_p = env.current_player
        acting_turn = env.turn_count
        p_obj = env.players[curr_p]
        f_name = "红方" if curr_p == 0 else "蓝方"
        
        mask = env.get_action_mask()
        state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        mask_t = torch.FloatTensor(mask).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, val = model(state_t, mask_t)
            action = torch.argmax(logits, dim=-1).item()

        action_desc = format_action_desc(env, action)
        prev_score = env.players[curr_p].score
        obs, reward, done, info = env.step(action)

        result_log = None
        if action == env.action_space_size - 1:
            gained = env.players[curr_p].score - prev_score
            if gained > 0:
                result_log = f"⚔️ 冲锋突破！{f_name} 本回合斩获 +{gained} 分！| 实时比分 -> 🔴 红 {env.players[0].score} : {env.players[1].score} 蓝 🔵"
            else:
                result_log = f"🛡️ 防线阻挡/蓄势完成 | 实时比分 -> 🔴 红 {env.players[0].score} : {env.players[1].score} 蓝 🔵"

            # 打印回合全景战局看板
            board_str = format_terminal_board(
                turn_count=acting_turn,
                acting_player=curr_p,
                p0={"score": env.players[0].score, "mana": env.players[0].mana, "max_mana": env.players[0].max_mana, "hand": env.players[0].hand},
                p1={"score": env.players[1].score, "mana": env.players[1].mana, "max_mana": env.players[1].max_mana, "hand": env.players[1].hand},
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
            print(f"👉 [第 {acting_turn:02d} 回合] {'🔴 红方' if curr_p==0 else '🔵 蓝方'} 动作: {action_desc} (余法力: {p_obj.mana}/{p_obj.max_mana})")

        # 记录每一步的快照用于生成交互式 HTML
        snap = pack_snapshot(env, acting_turn, curr_p, action_desc, result_log)
        snapshots.append(snap)

    winner = "红方 (P0 - 快攻突破流)" if env.players[0].score >= env.WIN_SCORE else "蓝方 (P1 - 控制防守流)"
    print("\n" + "="*80)
    print(f"🏁 对局结算完毕！获胜方: 【{winner}】")
    print(f"最终比分: 🔴 红方 {env.players[0].score} : {env.players[1].score} 蓝方 🔵 (总回合数: {env.turn_count} 轮)")
    print("="*80)

    # 导出可交互 HTML 网页回放器
    html_file = export_html_replay(snapshots, winner, output_path=args.html)
    abs_html = os.path.abspath(html_file)
    print(f"\n🌐 交互式战报回放网页已生成: file:///{abs_html.replace(os.sep, '/')}")
    print(f"💡 提示：双击该文件或在浏览器中打开，即可享受类似正式 TCG 游戏的动态播控回放！")

if __name__ == "__main__":
    evaluate()