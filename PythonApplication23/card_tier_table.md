# 🏆 TCG-AI 竞技场卡牌大数据战力评级系统（按卡组分色专属榜）

> **系统设计说明**：在 TCG 标准规则中，各阵营（赤红/蔚蓝/翠绿）只能携带**本阵营专属卡 + 中立通用卡**。混排所有卡牌对单卡组构筑毫无指导意义！本榜单基于 PPO 深度强化学习智能体（`card_ppo_model_tuned.pth`）在 1000 局实机对抗中的**【带牌比例】**与**【对胜率的影响 (ΔWR)】**两大黄金指标，按卡组阵营分色独立建榜。

---

## 🔴 一、 【赤红 (Red) 卡组】战力评级与构筑分析
> **卡组定位**：快攻突破 · 压场爆发 · 斩杀续航  
> **牌库候选池**：12 张赤红专属卡 + 6 张中立通用卡（共 18 张候选，择优遴选 30 张入库）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | DP/属性 | **综合评分** | 梯队 | **携带比例** | **对胜率影响 (ΔWR)** | 推荐抓取 | AI 助手独家实战锐评 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **佣兵斥候** | 中立 | 2费 | MINION | DP:1 `RUSH,DRAW_1` | **98.0** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">89.9%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +14.1%</span> | 3 张 (拉满) | 二费突袭抽一，节奏过牌两不误，神卡。 |
| 2 | **裂甲掷斧手** | Red专属 | 2费 | MINION | DP:1 `RUSH,DEGRADE_1` | **95.1** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">88.6%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +15.9%</span> | 3 张 (拉满) | 二费突袭降攻，抢节奏神器胜率飙升。 |
| 3 | **破阵狂徒** | Red专属 | 3费 | MINION | DP:3 `RUSH,DEGRADE_1` | **92.1** | **S** | <span style="color:#ffeaa7; font-weight:bold;">84.6%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +16.7%</span> | 3 张 (拉满) | 三费突袭降攻，解场打脸两不误，强。 |
| 4 | **赤红掠袭者** | Red专属 | 5费 | MINION | DP:3 `RUSH,BONUS_SCORE_1` | **89.2** | **A** | <span style="color:#ffeaa7; font-weight:bold;">69.5%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +6.4%</span> | 2~3 张 | 五费突袭加分，中速红卡组核心必带。 |
| 5 | **赤红突击手** | Red专属 | 1费 | MINION | DP:1 `DEATH_DRAW_1` | **65.6** | **C** | <span style="color:#b2bec3;">25.5%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -4.0%</span> | 0~1 张 | 一费过牌却掉4%胜率，死抽太慢纯拖节奏。 |
| 6 | **集结号手** | Red专属 | 2费 | MINION | DP:1 `SPAWN_1_1` | **64.7** | **C** | <span style="color:#b2bec3;">23.9%</span> | <span style="color:#dfe6e9;">⚪ -0.3%</span> | 0~1 张 | 二费生1/1，铺场效率低得可怜。 |
| 7 | **训练假人** | 中立 | 1费 | MINION | DP:1 | **64.5** | **C** | <span style="color:#b2bec3;">21.8%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0~1 张 | 一费白板毫无威胁，站场等于空过。 |
| 8 | **红色小队长** | Red专属 | 3费 | MINION | DP:3 `SPAWN_1_1` | **62.5** | **C** | <span style="color:#b2bec3;">28.7%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -4.4%</span> | 0~1 张 | 三费只生个1/1，铺场太亏节奏白给。 |
| 9 | **商人** | 中立 | 2费 | MINION | DP:2 `DRAW_1` | **62.5** | **C** | <span style="color:#b2bec3;">25.1%</span> | <span style="color:#dfe6e9;">⚪ -1.9%</span> | 0~1 张 | 二费抽一，节奏太慢不如直接过。 |
| 10 | **酒馆密账** | 中立 | 1费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **61.6** | **C** | <span style="color:#b2bec3;">22.4%</span> | <span style="color:#dfe6e9;">⚪ -2.9%</span> | 0~1 张 | 一费抽二弃一，过滤还行但亏节奏。 |
| 11 | **射线** | Red专属 | 1费 | SPELL | 攻2/防0 | **61.4** | **C** | <span style="color:#b2bec3;">22.2%</span> | <span style="color:#dfe6e9;">⚪ -0.4%</span> | 0~1 张 | 一费白板毫无亮点，带它不如多带张地。 |
| 12 | **切割者** | Red专属 | 2费 | MINION | DP:2 `DEGRADE_1` | **60.6** | **C** | <span style="color:#b2bec3;">26.3%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.6%</span> | 0~1 张 | 二费降攻软解，节奏亏到姥姥家。 |
| 13 | **拾荒盾卫** | 中立 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **60.3** | **C** | <span style="color:#b2bec3;">26.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -4.5%</span> | 0~1 张 | 三费加固还死抽，防守太被动拖后腿。 |
| 14 | **自爆** | Red专属 | 2费 | SPELL | 攻0/防0 `SACRIFICE_1_KILL_1,ATTACK_ONLY` | **60.1** | **C** | <span style="color:#b2bec3;">28.9%</span> | <span style="color:#dfe6e9;">⚪ -0.4%</span> | 0~1 张 | 二费牺牲一换一还限攻，勉强能用的解。 |
| 15 | **雇佣兵** | 中立 | 3费 | MINION | DP:5 | **57.1** | **D** | <span style="color:#b2bec3;">1.2%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -15.1%</span> | 0 张 (坚决弃用) | 三费白板还掉15%胜率，纯属卡手废牌。 |
| 16 | **血祭爆燃** | Red专属 | 3费 | SPELL | 攻3/防0 `SACRIFICE_1_KILL_1,ATTACK_ONLY` | **54.9** | **D** | <span style="color:#b2bec3;">3.5%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -8.4%</span> | 0 张 (坚决弃用) | 三费牺牲一换一，亏牌亏节奏别带。 |
| 17 | **掠夺者** | Red专属 | 6费 | MINION | DP:4 `BONUS_SCORE_1` | **49.9** | **D** | <span style="color:#b2bec3;">5.5%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -11.9%</span> | 0 张 (坚决弃用) | 六费才加分，慢到被快攻踢死。 |
| 18 | **大块头** | Red专属 | 6费 | MINION | DP:10 `DISCARD_2` | **38.0** | **D** | <span style="color:#b2bec3;">3.3%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -8.7%</span> | 0 张 (坚决弃用) | 六费弃两张牌，手牌打空自断后路。 |

---

## 🔵 二、 【蔚蓝 (Blue) 卡组】战力评级与构筑分析
> **卡组定位**：防守反击 · 护盾壁垒 · 资源消耗  
> **牌库候选池**：12 张蔚蓝专属卡 + 6 张中立通用卡（共 18 张候选）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | DP/属性 | **综合评分** | 梯队 | **携带比例** | **对胜率影响 (ΔWR)** | 推荐抓取 | AI 助手独家实战锐评 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **佣兵斥候** | 中立 | 2费 | MINION | DP:1 `RUSH,DRAW_1` | **98.0** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">87.0%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +13.6%</span> | 3 张 (拉满) | 二费突袭抽一，蓝方唯一节奏神卡。 |
| 2 | **寒晶护壁** | Blue专属 | 2费 | SPELL | 攻0/防2 `DRAW_1` | **56.1** | **D** | <span style="color:#b2bec3;">1.6%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -11.1%</span> | 0 张 (坚决弃用) | 二费抽一还只加甲，节奏全无。 |
| 3 | **蔚蓝卫士** | Blue专属 | 2费 | MINION | DP:2 `FORTIFY_1` | **55.6** | **D** | <span style="color:#b2bec3;">1.7%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -16.1%</span> | 0 张 (坚决弃用) | 二费只加一甲，节奏亏到姥姥家。 |
| 4 | **拾荒盾卫** | 中立 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **54.5** | **D** | <span style="color:#b2bec3;">1.4%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -8.6%</span> | 0 张 (坚决弃用) | 三费一甲亡语抽一，勉强不亏但太慢。 |
| 5 | **防御！** | Blue专属 | 1费 | SPELL | 攻0/防2 | **53.8** | **D** | <span style="color:#b2bec3;">1.2%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -15.0%</span> | 0 张 (坚决弃用) | 纯防不加场，被动挨打必输节奏。 |
| 6 | **霜盾见习官** | Blue专属 | 2费 | MINION | DP:1 `FORTIFY_1,SUPPORT_ATK_1` | **53.8** | **D** | <span style="color:#b2bec3;">2.3%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -8.3%</span> | 0 张 (坚决弃用) | 二费双一效果，平庸到毫无存在感。 |
| 7 | **盾兵** | Blue专属 | 1费 | MINION | DP:1 `FORTIFY_1` | **53.2** | **D** | <span style="color:#b2bec3;">5.1%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -13.9%</span> | 0 张 (坚决弃用) | 一费一甲看似灵活，实则毫无压制力。 |
| 8 | **商人** | 中立 | 2费 | MINION | DP:2 `DRAW_1` | **52.4** | **D** | <span style="color:#b2bec3;">1.5%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -15.4%</span> | 0 张 (坚决弃用) | 二费抽一不站场，纯亏节奏的废牌。 |
| 9 | **壁垒工匠** | Blue专属 | 3费 | MINION | DP:3 `FORTIFY_2` | **52.3** | **D** | <span style="color:#b2bec3;">4.8%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -18.8%</span> | 0 张 (坚决弃用) | 三费两甲亏成马，胜率暴跌不冤。 |
| 10 | **藤甲兵** | Blue专属 | 4费 | MINION | DP:4 `FORTIFY_2` | **51.4** | **D** | <span style="color:#b2bec3;">2.1%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -11.8%</span> | 0 张 (坚决弃用) | 四费两甲太笨重，被解就崩盘。 |
| 11 | **蔚蓝要塞** | Blue专属 | 6费 | MINION | DP:5 `FORTIFY_3,SUPPORT_ATK_1` | **51.3** | **D** | <span style="color:#b2bec3;">1.6%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -14.3%</span> | 0 张 (坚决弃用) | 六费三甲一攻，太慢太贵太被动。 |
| 12 | **冰封禁制** | Blue专属 | 3费 | SPELL | 攻2/防2 | **49.7** | **D** | <span style="color:#b2bec3;">4.4%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -9.9%</span> | 0 张 (坚决弃用) | 三费控场不站场，蓝方缺的是压力。 |
| 13 | **酒馆密账** | 中立 | 1费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **48.5** | **D** | <span style="color:#b2bec3;">5.5%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -8.0%</span> | 0 张 (坚决弃用) | 一费抽二弃一，看似赚实则手牌越打越少。 |
| 14 | **训练假人** | 中立 | 1费 | MINION | DP:1 | **48.2** | **D** | <span style="color:#b2bec3;">4.6%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -13.0%</span> | 0 张 (坚决弃用) | 一费白板零作用，纯属浪费卡位。 |
| 15 | **雇佣兵** | 中立 | 3费 | MINION | DP:5 | **46.7** | **D** | <span style="color:#b2bec3;">5.1%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -17.6%</span> | 0 张 (坚决弃用) | 三费白板无词条，竞技场都不选。 |
| 16 | **火铳手** | Blue专属 | 4费 | MINION | DP:4 `SUPPORT_ATK_2` | **45.8** | **D** | <span style="color:#b2bec3;">3.4%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -8.5%</span> | 0 张 (坚决弃用) | 四费两攻光环，慢速环境也嫌亏。 |
| 17 | **弓箭手** | Blue专属 | 3费 | MINION | DP:3 `SUPPORT_ATK_1` | **45.6** | **D** | <span style="color:#b2bec3;">5.0%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -9.0%</span> | 0 张 (坚决弃用) | 三费才给一攻，输出效率低得可怜。 |
| 18 | **石像鬼** | Blue专属 | 6费 | MINION | DP:8 | **38.0** | **D** | <span style="color:#b2bec3;">5.3%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -9.8%</span> | 0 张 (坚决弃用) | 六费白板，后期抽到等于投降。 |

---

## 🟢 三、 【翠绿 (Green) 卡组】战力评级与构筑分析
> **卡组定位**：快速跳费 · 膨胀成长 · 终结核弹  
> **牌库候选池**：12 张翠绿专属卡 + 6 张中立通用卡（共 18 张候选）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | DP/属性 | **综合评分** | 梯队 | **携带比例** | **对胜率影响 (ΔWR)** | 推荐抓取 | AI 助手独家实战锐评 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **翡翠幼龙** | Green专属 | 5费 | MINION | DP:3 `RUSH,DEATH_MANA_1` | **98.0** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">88.3%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +16.2%</span> | 3 张 (拉满) | 五费突袭返一费，节奏神卡，翻盘利器。 |
| 2 | **灭世翡翠巨龙** | Green专属 | 9费 | MINION | DP:10 `RUSH,BONUS_SCORE_1` | **91.6** | **S** | <span style="color:#ffeaa7; font-weight:bold;">89.3%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +16.2%</span> | 3 张 (拉满) | 九费突袭加分，一锤定音，后期终结者。 |
| 3 | **佣兵斥候** | 中立 | 2费 | MINION | DP:1 `RUSH,DRAW_1` | **91.5** | **S** | <span style="color:#ffeaa7; font-weight:bold;">83.3%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +17.0%</span> | 3 张 (拉满) | 二费突袭抽一，节奏神卡，胜率飙升。 |
| 4 | **翠绿萌芽** | Green专属 | 2费 | SPELL | 攻0/防0 `RAMP_1` | **52.6** | **D** | <span style="color:#b2bec3;">3.4%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -12.2%</span> | 0 张 (坚决弃用) | 二费跳费亏节奏，胜率暴跌纯属自残。 |
| 5 | **剧毒花** | Green专属 | 2费 | MINION | DP:1 `DEGRADE_2` | **51.5** | **D** | <span style="color:#b2bec3;">4.3%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -17.6%</span> | 0 张 (坚决弃用) | 二费降二攻治标不治本，节奏全丢。 |
| 6 | **芽苗祭司** | Green专属 | 3费 | MINION | DP:2 `RAMP_1` | **50.6** | **D** | <span style="color:#b2bec3;">3.0%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -10.4%</span> | 0 张 (坚决弃用) | 三费跳费太迟，被先手压制成废人。 |
| 7 | **世界树恩泽** | Green专属 | 4费 | SPELL | 攻0/防3 `RAMP_1` | **49.7** | **D** | <span style="color:#b2bec3;">3.6%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -18.6%</span> | 0 张 (坚决弃用) | 四费跳费亏到爆，节奏全无，纯属陷阱。 |
| 8 | **训练假人** | 中立 | 1费 | MINION | DP:1 | **49.4** | **D** | <span style="color:#b2bec3;">5.6%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -16.3%</span> | 0 张 (坚决弃用) | 一费假人纯送，连挡刀都嫌浪费卡位。 |
| 9 | **酒馆密账** | 中立 | 1费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **44.8** | **D** | <span style="color:#b2bec3;">2.7%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -14.9%</span> | 0 张 (坚决弃用) | 一费抽二弃一，手牌事故制造机。 |
| 10 | **商人** | 中立 | 2费 | MINION | DP:2 `DRAW_1` | **44.3** | **D** | <span style="color:#b2bec3;">5.7%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -13.7%</span> | 0 张 (坚决弃用) | 二费抽一太亏身材，节奏让光不如不带。 |
| 11 | **远古巨树** | Green专属 | 7费 | MINION | DP:8 `FORTIFY_2` | **44.1** | **D** | <span style="color:#b2bec3;">1.3%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -16.5%</span> | 0 张 (坚决弃用) | 七费二固守慢如龟，被快攻当木桩拆。 |
| 12 | **荆棘缠绕** | Green专属 | 2费 | SPELL | 攻2/防0 | **42.2** | **D** | <span style="color:#b2bec3;">5.8%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -14.2%</span> | 0 张 (坚决弃用) | 二费无属性缠绕，控不住场还亏卡。 |
| 13 | **拾荒盾卫** | 中立 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **41.6** | **D** | <span style="color:#b2bec3;">3.9%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -9.2%</span> | 0 张 (坚决弃用) | 三费固守亡语抽一，赖场补牌，优质肉盾。 |
| 14 | **狂暴生长** | Green专属 | 3费 | SPELL | 攻0/防3 | **41.1** | **D** | <span style="color:#b2bec3;">1.2%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -16.8%</span> | 0 张 (坚决弃用) | 三费空过赌爆发，实战等于让先手。 |
| 15 | **雇佣兵** | 中立 | 3费 | MINION | DP:5 | **40.6** | **D** | <span style="color:#b2bec3;">4.2%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -16.5%</span> | 0 张 (坚决弃用) | 三费白板无特技，站场即被换，废牌一张。 |
| 16 | **树人** | Green专属 | 3费 | MINION | DP:2 `FORTIFY_1` | **39.8** | **D** | <span style="color:#b2bec3;">4.1%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -18.0%</span> | 0 张 (坚决弃用) | 三费一固守太笨重，站不住场就是废牌。 |
| 17 | **森林之狼** | Green专属 | 4费 | MINION | DP:4 | **39.5** | **D** | <span style="color:#b2bec3;">2.3%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -10.4%</span> | 0 张 (坚决弃用) | 四费白板狼毫无压制，亏到姥姥家。 |
| 18 | **翡翠巨熊** | Green专属 | 6费 | MINION | DP:6 `FORTIFY_2` | **38.0** | **D** | <span style="color:#b2bec3;">1.1%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -18.9%</span> | 0 张 (坚决弃用) | 六费二固守被解就崩，胜率垫底实至名归。 |

---

## ⚪ 四、 【中立 (Neutral) 卡牌】全卡组泛用性与效用异质性分析
> **学术亮点**：相同的中立卡在不同流派（快攻/控制/跳费）中具有显著的效用异质性。

| 中立卡名称 | 费用 | 类型 | 🔴 赤红卡组中评分 | 🔵 蔚蓝卡组中评分 | 🟢 翠绿卡组中评分 | 最优契合阵营 | AI 跨卡组机制定位 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **佣兵斥候** | 2费 | MINION | **98.0** | **98.0** | **91.5** | **赤红 (快攻)** | 泛用度极高的全体系核心 |
| **训练假人** | 1费 | MINION | **64.5** | **48.2** | **49.4** | **赤红 (快攻)** | 专精型对策拼图 |
| **商人** | 2费 | MINION | **62.5** | **52.4** | **44.3** | **赤红 (快攻)** | 专精型对策拼图 |
| **酒馆密账** | 1费 | SPELL | **61.6** | **48.5** | **44.8** | **赤红 (快攻)** | 专精型对策拼图 |
| **拾荒盾卫** | 3费 | MINION | **60.3** | **54.5** | **41.6** | **赤红 (快攻)** | 专精型对策拼图 |
| **雇佣兵** | 3费 | MINION | **57.1** | **46.7** | **40.6** | **赤红 (快攻)** | 专精型对策拼图 |
