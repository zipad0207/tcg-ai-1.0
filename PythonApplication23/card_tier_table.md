# 🏆 TCG-AI 竞技场卡牌大数据战力评级系统（按卡组分色专属榜）

> **系统设计说明**：在 TCG 标准规则中，各阵营（赤红/蔚蓝/翠绿）只能携带**本阵营专属卡 + 中立通用卡**。混排所有卡牌对单卡组构筑毫无指导意义！本榜单基于 PPO 深度强化学习智能体（`card_ppo_model_tuned.pth`）在 1000 局实机对抗中的**【带牌比例】**与**【对胜率的影响 (ΔWR)】**两大黄金指标，按卡组阵营分色独立建榜。

---

## 🔴 一、 【赤红 (Red) 卡组】战力评级与构筑分析
> **卡组定位**：快攻突破 · 压场爆发 · 斩杀续航  
> **牌库候选池**：12 张赤红专属卡 + 6 张中立通用卡（共 18 张候选，择优遴选 30 张入库）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | DP/属性 | **综合评分** | 梯队 | **携带比例** | **对胜率影响 (ΔWR)** | 推荐抓取 | AI 助手独家实战锐评 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **赤红突击手** | Red专属 | 1费 | MINION | DP:3 `RUSH,DEATH_DRAW_1` | **98.0** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">86.5%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +14.5%</span> | 3 张 (拉满) | 一费冲脸还补牌，红快攻的绝对核心。 |
| 2 | **射线** | Red专属 | 1费 | SPELL | 攻2/防0 | **69.5** | **C** | <span style="color:#b2bec3;">26.7%</span> | <span style="color:#dfe6e9;">⚪ -0.4%</span> | 0~1 张 | 一费打一不痛不痒，带它不如多带张地。 |
| 3 | **红色小队长** | Red专属 | 3费 | MINION | DP:3 `SPAWN_1_1` | **68.9** | **C** | <span style="color:#b2bec3;">29.1%</span> | <span style="color:#dfe6e9;">⚪ -2.1%</span> | 0~1 张 | 三费铺个11，节奏亏到姥姥家。 |
| 4 | **训练假人** | 中立 | 1费 | MINION | DP:1 | **68.8** | **C** | <span style="color:#b2bec3;">27.2%</span> | <span style="color:#dfe6e9;">⚪ -1.8%</span> | 0~1 张 | 一费纯肉盾，红快攻带它等于自断节奏。 |
| 5 | **切割者** | Red专属 | 2费 | MINION | DP:2 `DEGRADE_1` | **68.3** | **C** | <span style="color:#b2bec3;">26.6%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -4.2%</span> | 0~1 张 | 二费降一攻，节奏全无的废件。 |
| 6 | **商人** | 中立 | 2费 | MINION | DP:2 `DRAW_1` | **66.8** | **C** | <span style="color:#b2bec3;">21.2%</span> | <span style="color:#dfe6e9;">⚪ -0.7%</span> | 0~1 张 | 二费抽一勉强及格，但红不需要这种节奏。 |
| 7 | **自爆** | Red专属 | 2费 | SPELL | 攻0/防0 `SACRIFICE_1_KILL_1,ATTACK_ONLY` | **63.0** | **C** | <span style="color:#b2bec3;">20.8%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.6%</span> | 0~1 张 | 二费换一还只能打脸，纯纯的亏卡陷阱。 |
| 8 | **雇佣兵** | 中立 | 3费 | MINION | DP:5 | **62.4** | **C** | <span style="color:#b2bec3;">28.0%</span> | <span style="color:#dfe6e9;">⚪ -0.4%</span> | 0~1 张 | 三费白板，竞技场都嫌它平庸。 |
| 9 | **掠夺者** | Red专属 | 6费 | MINION | DP:4 `BONUS_SCORE_1` | **52.8** | **D** | <span style="color:#b2bec3;">3.0%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -18.2%</span> | 0 张 (坚决弃用) | 六费才加一分，慢速红卡组的自杀选项。 |
| 10 | **大块头** | Red专属 | 6费 | MINION | DP:10 `DISCARD_2` | **38.0** | **D** | <span style="color:#b2bec3;">4.1%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -11.1%</span> | 0 张 (坚决弃用) | 六费弃两张，手牌打空还站不住。 |

---

## 🔵 二、 【蔚蓝 (Blue) 卡组】战力评级与构筑分析
> **卡组定位**：防守反击 · 护盾壁垒 · 资源消耗  
> **牌库候选池**：12 张蔚蓝专属卡 + 6 张中立通用卡（共 18 张候选）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | DP/属性 | **综合评分** | 梯队 | **携带比例** | **对胜率影响 (ΔWR)** | 推荐抓取 | AI 助手独家实战锐评 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **蔚蓝卫士** | Blue专属 | 2费 | MINION | DP:2 `FORTIFY_1` | **98.0** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">81.4%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +17.0%</span> | 3 张 (拉满) | 两费固守一，蓝阵营的胜率定海神针。 |
| 2 | **防御！** | Blue专属 | 1费 | SPELL | 攻0/防2 | **93.4** | **S** | <span style="color:#ffeaa7; font-weight:bold;">89.4%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +14.0%</span> | 3 张 (拉满) | 一费保命神卡，蓝军苟活反打的胜负手。 |
| 3 | **商人** | 中立 | 2费 | MINION | DP:2 `DRAW_1` | **90.3** | **S** | <span style="color:#ffeaa7; font-weight:bold;">87.4%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +15.8%</span> | 3 张 (拉满) | 两费抽一，蓝阵营过牌引擎，胜率保障。 |
| 4 | **藤甲兵** | Blue专属 | 4费 | MINION | DP:4 `FORTIFY_2` | **86.7** | **A** | <span style="color:#ffeaa7; font-weight:bold;">61.5%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +6.7%</span> | 2~3 张 | 四费固守二，扎实肉盾，蓝军中期中流砥柱。 |
| 5 | **盾兵** | Blue专属 | 1费 | MINION | DP:1 | **78.3** | **B** | <span style="color:#81ecec;">48.7%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.6%</span> | 1~2 张 | 一费白板，能填曲线但别指望它翻盘。 |
| 6 | **训练假人** | 中立 | 1费 | MINION | DP:1 | **71.6** | **B** | <span style="color:#81ecec;">45.2%</span> | <span style="color:#dfe6e9;">⚪ +0.9%</span> | 1~2 张 | 一费炮灰，偶尔挡刀，聊胜于无。 |
| 7 | **火铳手** | Blue专属 | 4费 | MINION | DP:4 `SUPPORT_ATK_2` | **68.4** | **C** | <span style="color:#b2bec3;">26.2%</span> | <span style="color:#dfe6e9;">⚪ -1.6%</span> | 0~1 张 | 四费支援二攻，收益太低，不如直接下生物。 |
| 8 | **雇佣兵** | 中立 | 3费 | MINION | DP:5 | **67.9** | **C** | <span style="color:#b2bec3;">24.2%</span> | <span style="color:#dfe6e9;">⚪ -0.1%</span> | 0~1 张 | 三费白板，平庸到胜率几乎零影响。 |
| 9 | **弓箭手** | Blue专属 | 3费 | MINION | DP:3 `SUPPORT_ATK_1` | **65.8** | **C** | <span style="color:#b2bec3;">22.3%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0~1 张 | 三费支援一攻，节奏太亏，胜率负资产。 |
| 10 | **石像鬼** | Blue专属 | 6费 | MINION | DP:8 | **38.0** | **D** | <span style="color:#b2bec3;">5.8%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -13.3%</span> | 0 张 (坚决弃用) | 六费白板大傻个，拍下去等于投降。 |

---

## 🟢 三、 【翠绿 (Green) 卡组】战力评级与构筑分析
> **卡组定位**：快速跳费 · 膨胀成长 · 终结核弹  
> **牌库候选池**：12 张翠绿专属卡 + 6 张中立通用卡（共 18 张候选）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | DP/属性 | **综合评分** | 梯队 | **携带比例** | **对胜率影响 (ΔWR)** | 推荐抓取 | AI 助手独家实战锐评 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **翠绿萌芽** | Green专属 | 4费 | SPELL | 攻0/防0 `RAMP_1` | **98.0** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">89.6%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +15.2%</span> | 3 张 (拉满) | 四费跳一费，节奏神卡，绿阵营胜率发动机。 |
| 2 | **荆棘缠绕** | Green专属 | 2费 | SPELL | 攻2/防0 | **67.0** | **C** | <span style="color:#b2bec3;">25.1%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0~1 张 | 二费小解，不赚节奏，勉强凑数。 |
| 3 | **远古巨树** | Green专属 | 7费 | MINION | DP:8 `FORTIFY_2` | **65.7** | **C** | <span style="color:#b2bec3;">21.7%</span> | <span style="color:#dfe6e9;">⚪ -1.3%</span> | 0~1 张 | 七费双固守，勉强能拖，但太慢仍亏。 |
| 4 | **剧毒花** | Green专属 | 2费 | MINION | DP:1 `DEGRADE_1` | **50.4** | **D** | <span style="color:#b2bec3;">1.3%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -16.5%</span> | 0 张 (坚决弃用) | 二费降一攻，节奏全无，谁带谁掉分。 |
| 5 | **雇佣兵** | 中立 | 3费 | MINION | DP:5 | **46.8** | **D** | <span style="color:#b2bec3;">2.0%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -12.9%</span> | 0 张 (坚决弃用) | 三费白板，毫无压制力，直接淘汰。 |
| 6 | **训练假人** | 中立 | 1费 | MINION | DP:1 | **44.3** | **D** | <span style="color:#b2bec3;">4.2%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -11.4%</span> | 0 张 (坚决弃用) | 一费白板，连挡刀都嫌浪费卡位。 |
| 7 | **商人** | 中立 | 2费 | MINION | DP:2 `DRAW_1` | **42.7** | **D** | <span style="color:#b2bec3;">0.7%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -18.3%</span> | 0 张 (坚决弃用) | 二费抽一，亏到姥姥家，绿阵营最大陷阱。 |
| 8 | **树人** | Green专属 | 3费 | MINION | DP:2 `FORTIFY_1` | **41.9** | **D** | <span style="color:#b2bec3;">4.5%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -8.3%</span> | 0 张 (坚决弃用) | 三费一固守，亏节奏，防守不如直接解场。 |
| 9 | **狂暴生长** | Green专属 | 3费 | SPELL | 攻0/防3 | **40.7** | **D** | <span style="color:#b2bec3;">4.5%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -16.3%</span> | 0 张 (坚决弃用) | 三费空过，没词条没场面，纯废卡。 |
| 10 | **森林之狼** | Green专属 | 4费 | MINION | DP:4 | **38.0** | **D** | <span style="color:#b2bec3;">4.6%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -13.9%</span> | 0 张 (坚决弃用) | 四费白板，站不住也换不来节奏，别碰。 |

---

## ⚪ 四、 【中立 (Neutral) 卡牌】全卡组泛用性与效用异质性分析
> **学术亮点**：相同的中立卡在不同流派（快攻/控制/跳费）中具有显著的效用异质性。

| 中立卡名称 | 费用 | 类型 | 🔴 赤红卡组中评分 | 🔵 蔚蓝卡组中评分 | 🟢 翠绿卡组中评分 | 最优契合阵营 | AI 跨卡组机制定位 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **训练假人** | 1费 | MINION | **68.8** | **71.6** | **44.3** | **蔚蓝 (防守)** | 专精型对策拼图 |
| **商人** | 2费 | MINION | **66.8** | **90.3** | **42.7** | **蔚蓝 (防守)** | 专精型对策拼图 |
| **雇佣兵** | 3费 | MINION | **62.4** | **67.9** | **46.8** | **蔚蓝 (防守)** | 专精型对策拼图 |
