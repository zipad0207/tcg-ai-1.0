import torch
import numpy as np
from sandbox import DuelEnv, Faction
from train import CardNet

def format_action_desc(env, action_id):
    if action_id == env.action_space_size - 1:
        return "结束回合 (PASS)"
    hand_idx = action_id // 4
    target_code = action_id % 4
    lane_str = "【左路】" if target_code in [0, 1] else "【右路】"
    card = env.players[env.current_player].hand[hand_idx]
    
    if card.card_type.name == "SPELL":
        return f"释放法术 [{card.name} (费用:{card.cost} 词条:{card.tags})] 到 {lane_str}"
    else:
        pos_str = "进攻区(发起冲锋)" if target_code in [0, 2] else "防守区(筑起防线)"
        return f"部署随从 [{card.name} (费用:{card.cost} DP:{card.base_dp} 词条:{card.tags})] 到 {lane_str}{pos_str}"

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = DuelEnv(p0_faction=Faction.RED, p1_faction=Faction.BLUE)
    
    model = CardNet(action_dim=env.action_space_size).to(device)
    model.load_state_dict(torch.load("card_ppo_model.pth", map_location=device))
    model.eval()

    obs = env.reset()
    done = False
    print("\n" + "="*50)
    print("🎮 AI 对局全景回放启动 (红方 vs 蓝方)")
    print("="*50)

    while not done:
        curr_p = env.current_player
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

        obs, reward, done, info = env.step(action)

        # 遇 Pass 回合打印结算状态
        if action == env.action_space_size - 1:
            print(f"  └─ ⚔️ 回合交战结算完毕 | 当前比分 -> 红方: {env.players[0].score} 分 | 蓝方: {env.players[1].score} 分")
            for l_id in [0, 1]:
                lane_name = "左路" if l_id == 0 else "右路"
                r_defs = [f"{u.card.name}(DP:{u.current_dp})" for u in env.lanes[l_id].defenders if u.owner == 0]
                b_defs = [f"{u.card.name}(DP:{u.current_dp})" for u in env.lanes[l_id].defenders if u.owner == 1]
                print(f"     [{lane_name}驻军] 红方防线: {'、'.join(r_defs) if r_defs else '无'} | 蓝方防线: {'、'.join(b_defs) if b_defs else '无'}")

    print("\n" + "="*50)
    print(f"🏁 对局结束！最终比分: 红方 {env.players[0].score} : {env.players[1].score} 蓝方")
    print("="*50)

if __name__ == "__main__":
    evaluate()