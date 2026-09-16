import json
import random
import torch
from sandbox import DuelEnv, Faction
from agent import CardNet

with open('cards_config.json', 'r', encoding='utf-8') as f:
    cards_data = json.load(f)

# 1. 红方微调：
# 裂甲掷斧手 (108): cost: 2, tags: ['RUSH'] (移除削弱，避免前期无限吃大怪)
# 赤红突击手 (100): cost: 1, base_dp: 1, tags: [] (1费1点DP正规身材)
for c in cards_data['Red']:
    if c['id'] == 108:
        c['tags'] = ['RUSH']
    elif c['id'] == 100:
        c['tags'] = []

# 2. 绿方微调：
# 翡翠幼龙 (309): cost: 5 -> 4 (4费3DP突袭+亡语水晶，中盘关键反打节奏)
# 翡翠巨熊 (308): cost: 6 -> 5 (5费6DP护甲2，扎实防守反推)
for c in cards_data['Green']:
    if c['id'] == 309:
        c['cost'] = 4
    elif c['id'] == 308:
        c['cost'] = 5

# 3. 蓝方微调：
# 寒晶护壁 (209): def_spell_val: 2 -> 3 (2费3点防守法术+抽1卡)
# 霜盾见习官 (207): base_dp: 1 -> 2
for c in cards_data['Blue']:
    if c['id'] == 209:
        c['def_spell_val'] = 3
    elif c['id'] == 207:
        c['base_dp'] = 2

with open('cards_config.json', 'w', encoding='utf-8') as f:
    json.dump(cards_data, f, indent=2, ensure_ascii=False)

with open('decks_config.json', 'r', encoding='utf-8') as f:
    decks_data = json.load(f)

env = DuelEnv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CardNet().to(device)
model.load_state_dict(torch.load('card_ppo_model_tuned.pth', map_location=device, weights_only=True))
model.eval()

stats = {'Red': {'m': 0, 'w': 0}, 'Blue': {'m': 0, 'w': 0}, 'Green': {'m': 0, 'w': 0}}
f_map = {Faction.RED: 'Red', Faction.BLUE: 'Blue', Faction.GREEN: 'Green'}
f_list = [Faction.RED, Faction.BLUE, Faction.GREEN]

for _ in range(800):
    f0, f1 = random.choice(f_list), random.choice(f_list)
    n0, n1 = f_map[f0], f_map[f1]
    env.p0_faction, env.p1_faction = f0, f1
    env.p0_decklist = decks_data[n0]['decklist']
    env.p1_decklist = decks_data[n1]['decklist']
    obs = env.reset()
    done = False
    while not done:
        mask = env.get_action_mask()
        s_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        m_t = torch.FloatTensor(mask).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = model(s_t, m_t)
            act = torch.argmax(logits, dim=-1).item()
        obs, _, done, _ = env.step(act)
    p0_w = (env.players[0].score >= env.WIN_SCORE)
    stats[n0]['m'] += 1
    stats[n1]['m'] += 1
    if p0_w:
        stats[n0]['w'] += 1
    else:
        stats[n1]['w'] += 1

for k, v in stats.items():
    wr = (v['w'] / v['m']) * 100 if v['m'] > 0 else 0
    print(f"{k}: {v['w']}/{v['m']} ({wr:.1f}%)")
