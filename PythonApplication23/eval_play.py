import os
import sys
import argparse
import torch
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sandbox import DuelEnv, Faction
from agent import CardNet

def format_action_desc(env, action_id):
    if action_id == env.action_space_size - 1:
        return "结束回合 (PASS)"
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

def evaluate():
    parser = argparse.ArgumentParser(description="TCG AI 对局全景回放与评估")
    parser.add_argument("--stage", type=str, default="tuned", choices=["baseline", "tuned"],
                        help="选择评估模型阶段: tuned(调优后平衡模型) 或 baseline(基准模型)")
    parser.add_argument("--model", type=str, default=None, help="自定义指定模型权重文件路径")
    parser.add_argument("--cards", type=str, default="cards_config.json", help="指定卡池配置文件路径")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 查找模型
    model_path = find_model_path(args.model, args.stage)
    if not model_path:
        print(f"❌ 未找到可用的权重文件！请先运行 train.py 进行训练。")
        return

    print(f"📦 正在加载智能体模型权重: {model_path}")
    env = DuelEnv(p0_faction=Faction.RED, p1_faction=Faction.BLUE, cards_path=args.cards)
    
    model = CardNet(action_dim=env.action_space_size).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    obs = env.reset()
    done = False
    print("\n" + "="*60)
    print(f"🎮 AI 对局全景回放启动 (红方 vs 蓝方) | 模型: {os.path.basename(model_path)}")
    print("="*60)

    while not done:
        curr_p = env.current_player
        opp_p = 1 - curr_p
        p_obj = env.players[curr_p]
        f_name = "红方" if curr_p == 0 else "蓝方"
        
        mask = env.get_action_mask()
        state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        mask_t = torch.FloatTensor(mask).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, val = model(state_t, mask_t)
            # 评估时采用 argmax 贪婪选择，看 AI 认为最优的动作
            action = torch.argmax(logits, dim=-1).item()

        action_desc = format_action_desc(env, action)
        print(f"\n[第 {env.turn_count} 回合] {f_name}(P{curr_p}) 剩余法力: {p_obj.mana}/{p_obj.max_mana} | 执行: {action_desc}")

        prev_score = env.players[curr_p].score
        obs, reward, done, info = env.step(action)

        # 遇 Pass 回合打印交战与得分详细结算
        if action == env.action_space_size - 1:
            gained = env.players[curr_p].score - prev_score
            if gained > 0:
                print(f"  └─ ⚔️ 冲锋突破！{f_name} 本回合斩获 +{gained} 分！| 当前比分 -> 红方: {env.players[0].score} : {env.players[1].score} 蓝方")
            else:
                print(f"  └─ 🛡️ 防线阻挡/蓄势驻守 | 当前比分 -> 红方: {env.players[0].score} : {env.players[1].score} 蓝方")

            for l_id in [0, 1]:
                lane_name = "左路" if l_id == 0 else "右路"
                r_defs = [f"{u.card.name}(DP:{u.current_dp})" for u in env.lanes[l_id].defenders if u.owner == 0]
                b_defs = [f"{u.card.name}(DP:{u.current_dp})" for u in env.lanes[l_id].defenders if u.owner == 1]
                r_atks = [f"{u.card.name}(DP:{u.current_dp}{',就绪' if u.ready_to_attack else ',蓄势'})" for u in env.lanes[l_id].attackers if u.owner == 0]
                b_atks = [f"{u.card.name}(DP:{u.current_dp}{',就绪' if u.ready_to_attack else ',蓄势'})" for u in env.lanes[l_id].attackers if u.owner == 1]
                print(f"     [{lane_name}战况] 红方防线: {'、'.join(r_defs) if r_defs else '空'} | 蓝方防线: {'、'.join(b_defs) if b_defs else '空'}")

    winner = "红方(P0 - 快攻突破流)" if env.players[0].score >= env.WIN_SCORE else "蓝方(P1 - 控制防守流)"
    print("\n" + "="*60)
    print(f"🏁 对局结束！获胜方: 【{winner}】")
    print(f"最终比分: 红方 {env.players[0].score} : {env.players[1].score} 蓝方 (总步数: {env.turn_count} 回合)")
    print("="*60)

if __name__ == "__main__":
    evaluate()