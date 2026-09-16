# 🏆 TCG-AI 竞技场卡牌大数据战力评级系统（按卡组分色专属榜）

> **系统设计说明**：在 TCG 标准规则中，各阵营（赤红/蔚蓝/翠绿）只能携带**本阵营专属卡 + 中立通用卡**。混排所有卡牌对单卡组构筑毫无指导意义！本榜单基于 PPO 深度强化学习智能体（`card_ppo_model_tuned.pth`）在 1000 局实机对抗中的**【带牌比例】**与**【对胜率的影响 (ΔWR)】**两大黄金指标，按卡组阵营分色独立建榜。

---

## 🔴 一、 【赤红 (Red) 卡组】战力评级与构筑分析
> **卡组定位**：快攻突破 · 压场爆发 · 斩杀续航  
> **牌库候选池**：12 张赤红专属卡 + 6 张中立通用卡（共 18 张候选，择优遴选 30 张入库）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | DP/属性 | **综合评分** | 梯队 | **携带比例** | **对胜率影响 (ΔWR)** | 推荐抓取 | AI 助手独家实战锐评 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **自爆** | Red专属 | 2费 | SPELL | 攻0/防0 `SACRIFICE_1_KILL_1,ATTACK_ONLY` | **98.0** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +13.5%</span> | 3 张 (拉满) | 两费换一杀还能打脸，红阵营解场兼输出的神卡。 |
| 2 | **射线** | Red专属 | 2费 | SPELL | 攻2/防0 | **98.0** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +13.5%</span> | 3 张 (拉满) | 两费灵活直伤，抢血斩杀两不误，红快攻必带。 |
| 3 | **血祭爆燃** | Red专属 | 3费 | SPELL | 攻3/防0 `SACRIFICE_1_KILL_1,ATTACK_ONLY` | **98.0** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +13.5%</span> | 3 张 (拉满) | 三费牺牲一杀一还打脸，节奏伤害拉满。 |
| 4 | **赤红突击手** | Red专属 | 1费 | MINION | DP:1 `DEATH_DRAW_1` | **76.3** | **B** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.7%</span> | 1~2 张 | 一费过牌不亏节奏，红快攻的完美润滑剂。 |
| 5 | **训练假人** | 中立 | 1费 | MINION | DP:1 | **57.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -4.6%</span> | 0 张 (坚决弃用) | 一费白板没威胁，红快攻要的是压力不是沙包。 |
| 6 | **佣兵斥候** | 中立 | 2费 | MINION | DP:1 `RUSH,DRAW_1` | **56.6** | **D** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -5.1%</span> | 0 张 (坚决弃用) | 两费突袭过牌，灵活不亏，红快攻优质曲线。 |
| 7 | **裂甲掷斧手** | Red专属 | 3费 | MINION | DP:1 `RUSH,DEGRADE_1` | **52.6** | **D** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -6.9%</span> | 0 张 (坚决弃用) | 三费突袭降攻，解不干净还亏身材，鸡肋。 |
| 8 | **破阵狂徒** | Red专属 | 4费 | MINION | DP:2 `RUSH,DEGRADE_1` | **51.8** | **D** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -7.3%</span> | 0 张 (坚决弃用) | 四费突袭降攻，交换吃亏，红阵营不需要。 |
| 9 | **商人** | 中立 | 2费 | MINION | DP:2 `DRAW_1` | **51.0** | **D** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -7.7%</span> | 0 张 (坚决弃用) | 两费抽一白板，节奏让给对手，红快攻大忌。 |
| 10 | **酒馆密账** | 中立 | 1费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **50.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -7.8%</span> | 0 张 (坚决弃用) | 一费抽二弃一，看似赚实则亏节奏，快攻用不上。 |
| 11 | **赤红掠袭者** | Red专属 | 5费 | MINION | DP:3 `RUSH,BONUS_SCORE_1` | **50.4** | **D** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -7.9%</span> | 0 张 (坚决弃用) | 五费突袭加分，太慢且亏节奏，快攻看不上。 |
| 12 | **红色小队长** | Red专属 | 3费 | MINION | DP:3 `SPAWN_1_1` | **46.6** | **D** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -9.6%</span> | 0 张 (坚决弃用) | 三费铺个11，节奏亏到姥姥家，胜率黑洞。 |
| 13 | **掠夺者** | Red专属 | 6费 | MINION | DP:4 `BONUS_SCORE_1` | **44.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -10.8%</span> | 0 张 (坚决弃用) | 六费才加分，太慢太笨，红快攻等不起。 |
| 14 | **拾荒盾卫** | 中立 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **42.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -11.5%</span> | 0 张 (坚决弃用) | 三费加固还过牌，但红阵营要进攻不要龟缩。 |
| 15 | **大块头** | Red专属 | 6费 | MINION | DP:10 `DISCARD_2` | **42.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -11.7%</span> | 0 张 (坚决弃用) | 六费弃两张，手牌打空还站不住，纯废件。 |
| 16 | **切割者** | Red专属 | 2费 | MINION | DP:2 `DEGRADE_1` | **40.0** | **D** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -12.6%</span> | 0 张 (坚决弃用) | 两费降攻软解，亏卡亏节奏，谁带谁输。 |
| 17 | **集结号手** | Red专属 | 2费 | MINION | DP:1 `SPAWN_1_1` | **38.3** | **D** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -13.4%</span> | 0 张 (坚决弃用) | 两费只出个11，铺场效率被一费完爆。 |
| 18 | **雇佣兵** | 中立 | 3费 | MINION | DP:5 | **38.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -13.5%</span> | 0 张 (坚决弃用) | 三费白板无特效，竞技场都嫌弱，天梯纯送。 |

---

## 🔵 二、 【蔚蓝 (Blue) 卡组】战力评级与构筑分析
> **卡组定位**：防守反击 · 护盾壁垒 · 资源消耗  
> **牌库候选池**：12 张蔚蓝专属卡 + 6 张中立通用卡（共 18 张候选）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | DP/属性 | **综合评分** | 梯队 | **携带比例** | **对胜率影响 (ΔWR)** | 推荐抓取 | AI 助手独家实战锐评 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **冰封禁制** | Blue专属 | 3费 | SPELL | 攻2/防2 | **98.0** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +13.5%</span> | 3 张 (拉满) | 三费硬控，蓝阵营唯一真神。 |
| 2 | **酒馆密账** | 中立 | 1费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **53.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -6.4%</span> | 0 张 (坚决弃用) | 一费抽二弃一，过牌最不亏。 |
| 3 | **防御！** | Blue专属 | 1费 | SPELL | 攻0/防2 | **52.2** | **D** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -7.1%</span> | 0 张 (坚决弃用) | 一费换一固守，亏牌亏到心碎。 |
| 4 | **霜盾见习官** | Blue专属 | 2费 | MINION | DP:1 `FORTIFY_1,SUPPORT_ATK_1` | **52.1** | **D** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -7.2%</span> | 0 张 (坚决弃用) | 二费双一，样样通样样松。 |
| 5 | **训练假人** | 中立 | 1费 | MINION | DP:1 | **48.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -8.9%</span> | 0 张 (坚决弃用) | 一费白板，连挡刀都嫌脆。 |
| 6 | **商人** | 中立 | 2费 | MINION | DP:2 `DRAW_1` | **47.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -9.0%</span> | 0 张 (坚决弃用) | 二费抽一，身材亏光节奏丢。 |
| 7 | **佣兵斥候** | 中立 | 2费 | MINION | DP:1 `RUSH,DRAW_1` | **47.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -9.1%</span> | 0 张 (坚决弃用) | 二费突袭抽一，勉强能用的节奏。 |
| 8 | **蔚蓝卫士** | Blue专属 | 2费 | MINION | DP:2 `FORTIFY_1` | **47.4** | **D** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -9.3%</span> | 0 张 (坚决弃用) | 两费一固守，节奏亏到姥姥家。 |
| 9 | **盾兵** | Blue专属 | 1费 | MINION | DP:1 `FORTIFY_1` | **44.8** | **D** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -10.4%</span> | 0 张 (坚决弃用) | 一费一固守，纯废件不如空过。 |
| 10 | **雇佣兵** | 中立 | 3费 | MINION | DP:5 | **42.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -11.5%</span> | 0 张 (坚决弃用) | 三费白板，没词条就是送。 |
| 11 | **蔚蓝要塞** | Blue专属 | 6费 | MINION | DP:5 `FORTIFY_3,SUPPORT_ATK_1` | **41.6** | **D** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -11.9%</span> | 0 张 (坚决弃用) | 六费三固守一加攻，太慢太笨。 |
| 12 | **石像鬼** | Blue专属 | 6费 | MINION | DP:8 | **41.3** | **D** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -12.0%</span> | 0 张 (坚决弃用) | 六费白板，拍下去等于投降。 |
| 13 | **壁垒工匠** | Blue专属 | 3费 | MINION | DP:3 `FORTIFY_2` | **41.1** | **D** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -12.1%</span> | 0 张 (坚决弃用) | 三费二固守，节奏让到没边。 |
| 14 | **弓箭手** | Blue专属 | 3费 | MINION | DP:3 `SUPPORT_ATK_1` | **40.9** | **D** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -12.2%</span> | 0 张 (坚决弃用) | 三费一加攻，站不住就是白给。 |
| 15 | **火铳手** | Blue专属 | 4费 | MINION | DP:4 `SUPPORT_ATK_2` | **39.3** | **D** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -12.9%</span> | 0 张 (坚决弃用) | 四费二加攻，被解就崩盘。 |
| 16 | **寒晶护壁** | Blue专属 | 2费 | SPELL | 攻0/防2 `DRAW_1` | **38.8** | **D** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -13.1%</span> | 0 张 (坚决弃用) | 二费抽一还亏，护壁在哪？ |
| 17 | **拾荒盾卫** | 中立 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **38.7** | **D** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -13.2%</span> | 0 张 (坚决弃用) | 三费一固守还亡语抽，太慢。 |
| 18 | **藤甲兵** | Blue专属 | 4费 | MINION | DP:4 `FORTIFY_2` | **38.0** | **D** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -13.5%</span> | 0 张 (坚决弃用) | 四费二固守，慢速环境也嫌蠢。 |

---

## 🟢 三、 【翠绿 (Green) 卡组】战力评级与构筑分析
> **卡组定位**：快速跳费 · 膨胀成长 · 终结核弹  
> **牌库候选池**：12 张翠绿专属卡 + 6 张中立通用卡（共 18 张候选）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | DP/属性 | **综合评分** | 梯队 | **携带比例** | **对胜率影响 (ΔWR)** | 推荐抓取 | AI 助手独家实战锐评 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **荆棘缠绕** | Green专属 | 2费 | SPELL | 攻2/防0 | **98.0** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +13.5%</span> | 3 张 (拉满) | 两费神解，节奏拉满，绿阵营唯一真神。 |
| 2 | **训练假人** | 中立 | 1费 | MINION | DP:1 | **67.2** | **C** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -0.4%</span> | 0~1 张 | 一费白板，几乎不影响胜率，纯凑数。 |
| 3 | **灭世翡翠巨龙** | Green专属 | 9费 | MINION | DP:10 `RUSH,BONUS_SCORE_1` | **56.0** | **D** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -5.4%</span> | 0 张 (坚决弃用) | 九费突袭加分，太慢，基本活不到出场。 |
| 4 | **佣兵斥候** | 中立 | 2费 | MINION | DP:1 `RUSH,DRAW_1` | **54.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -6.0%</span> | 0 张 (坚决弃用) | 两费突袭抽一，节奏极佳，绿阵营必带。  |
| 5 | **翡翠幼龙** | Green专属 | 5费 | MINION | DP:3 `RUSH,DEATH_MANA_1` | **52.9** | **D** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -6.8%</span> | 0 张 (坚决弃用) | 五费突袭回一费，勉强能用，但不够强。 |
| 6 | **商人** | 中立 | 2费 | MINION | DP:2 `DRAW_1` | **52.6** | **D** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -6.9%</span> | 0 张 (坚决弃用) | 两费抽一，不亏不赚，凑合过渡用。 |
| 7 | **剧毒花** | Green专属 | 2费 | MINION | DP:2 `DEGRADE_2` | **52.2** | **D** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -7.1%</span> | 0 张 (坚决弃用) | 两费降二攻，解不干净还亏卡，别碰。 |
| 8 | **翠绿萌芽** | Green专属 | 2费 | SPELL | 攻0/防0 `RAMP_1` | **48.0** | **D** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -9.0%</span> | 0 张 (坚决弃用) | 两费跳一费，亏牌亏节奏，胜率暴跌。 |
| 9 | **拾荒盾卫** | 中立 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **44.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -10.4%</span> | 0 张 (坚决弃用) | 三费固守加亡语抽一，勉强及格，但不够看。 |
| 10 | **雇佣兵** | 中立 | 3费 | MINION | DP:5 | **44.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -10.7%</span> | 0 张 (坚决弃用) | 三费白板，毫无亮点，竞技场都嫌弱。 |
| 11 | **远古巨树** | Green专属 | 7费 | MINION | DP:8 `FORTIFY_2` | **43.7** | **D** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -10.9%</span> | 0 张 (坚决弃用) | 七费才两固守，慢到死，早被快攻冲烂。 |
| 12 | **森林之狼** | Green专属 | 4费 | MINION | DP:4 | **42.2** | **D** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -11.6%</span> | 0 张 (坚决弃用) | 四费白板，被任何三费随从白吃，废卡。 |
| 13 | **翡翠巨熊** | Green专属 | 6费 | MINION | DP:6 `FORTIFY_2` | **41.3** | **D** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -12.0%</span> | 0 张 (坚决弃用) | 六费两固守，被解就崩，纯纯的陷阱。 |
| 14 | **芽苗祭司** | Green专属 | 3费 | MINION | DP:3 `RAMP_1` | **41.1** | **D** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -12.1%</span> | 0 张 (坚决弃用) | 三费跳费太慢，身材还亏，不如不带。 |
| 15 | **树人** | Green专属 | 3费 | MINION | DP:2 `FORTIFY_1` | **39.8** | **D** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -12.7%</span> | 0 张 (坚决弃用) | 三费一固守，站不住场，纯属送节奏。 |
| 16 | **酒馆密账** | 中立 | 1费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **39.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -12.7%</span> | 0 张 (坚决弃用) | 一费抽二弃一，看似赚实则亏节奏，别用。 |
| 17 | **狂暴生长** | Green专属 | 3费 | SPELL | 攻0/防3 | **39.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -13.0%</span> | 0 张 (坚决弃用) | 三费空过，没词条没场面，谁带谁输。 |
| 18 | **世界树恩泽** | Green专属 | 4费 | SPELL | 攻0/防3 `RAMP_1` | **38.0** | **D** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -13.5%</span> | 0 张 (坚决弃用) | 四费跳一费，亏到姥姥家，胜率垫底。 |

---

## ⚪ 四、 【中立 (Neutral) 卡牌】全卡组泛用性与效用异质性分析
> **学术亮点**：相同的中立卡在不同流派（快攻/控制/跳费）中具有显著的效用异质性。

| 中立卡名称 | 费用 | 类型 | 🔴 赤红卡组中评分 | 🔵 蔚蓝卡组中评分 | 🟢 翠绿卡组中评分 | 最优契合阵营 | AI 跨卡组机制定位 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **训练假人** | 1费 | MINION | **57.8** | **48.2** | **67.2** | **翠绿 (跳费)** | 专精型对策拼图 |
| **佣兵斥候** | 2费 | MINION | **56.6** | **47.7** | **54.6** | **赤红 (快攻)** | 专精型对策拼图 |
| **商人** | 2费 | MINION | **51.0** | **47.9** | **52.6** | **翠绿 (跳费)** | 专精型对策拼图 |
| **酒馆密账** | 1费 | SPELL | **50.7** | **53.8** | **39.8** | **蔚蓝 (防守)** | 专精型对策拼图 |
| **拾荒盾卫** | 3费 | MINION | **42.5** | **38.7** | **44.8** | **翠绿 (跳费)** | 专精型对策拼图 |
| **雇佣兵** | 3费 | MINION | **38.0** | **42.5** | **44.2** | **翠绿 (跳费)** | 专精型对策拼图 |
