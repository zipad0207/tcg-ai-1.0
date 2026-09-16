# 🏆 TCG-AI 竞技场卡牌大数据战力评级系统（按卡组分色专属榜）

> **系统设计说明**：在 TCG 标准规则中，各阵营（赤红/蔚蓝/翠绿）只能携带**本阵营专属卡 + 中立通用卡**。混排所有卡牌对单卡组构筑毫无指导意义！本榜单基于 PPO 深度强化学习智能体（`card_ppo_model_tuned.pth`）在 1000 局实机对抗中的**【带牌比例】**与**【对胜率的影响 (ΔWR)】**两大黄金指标，按卡组阵营分色独立建榜。

---

## 🔴 一、 【赤红 (Red) 卡组】战力评级与构筑分析
> **卡组定位**：快攻突破 · 压场爆发 · 斩杀续航  
> **牌库候选池**：12 张赤红专属卡 + 6 张中立通用卡（共 18 张候选，择优遴选 30 张入库）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | DP/属性 | **综合评分** | 梯队 | **携带比例** | **对胜率影响 (ΔWR)** | 推荐抓取 | AI 助手独家实战锐评 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **佣兵斥候** | 中立 | 2费 | MINION | DP:1 `RUSH,DRAW_1` | **98.0** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">88.3%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +15.0%</span> | 3 张 (拉满) | 【Red绝对幻神】突袭解场兼具过牌补手牌！完美抢回主动权，胜率净增+15.0%，无脑满编3张！ |
| 2 | **酒馆密账** | 中立 | 1费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **95.5** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">89.6%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +16.5%</span> | 3 张 (拉满) | 【快攻强力润滑】1费过2加速倾泻手牌，前期抢死对手的利器；快攻构筑推荐带满。 |
| 3 | **裂甲掷斧手** | Red专属 | 2费 | MINION | DP:1 `RUSH,DEGRADE_1` | **95.3** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">82.7%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +14.6%</span> | 3 张 (拉满) | 【Red破阵利器】突袭打乱敌方攻防节奏并削弱DP，抢节奏神卡，胜率净增+14.6%！ |
| 4 | **破阵狂徒** | Red专属 | 3费 | MINION | DP:3 `RUSH,DEGRADE_1` | **95.0** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">80.1%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +18.3%</span> | 3 张 (拉满) | 【Red破阵利器】突袭打乱敌方攻防节奏并削弱DP，抢节奏神卡，胜率净增+18.3%！ |
| 5 | **赤红突击手** | Red专属 | 1费 | MINION | DP:1 `DEATH_DRAW_1` | **94.6** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">81.3%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +18.3%</span> | 3 张 (拉满) | 【Red先锋核心】1费站场倒下不亏卡，为后续攻势源源不断续航，快攻必带。 |
| 6 | **集结号手** | Red专属 | 2费 | MINION | DP:1 `SPAWN_1_1` | **94.6** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">89.8%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +12.5%</span> | 3 张 (拉满) | 【Red场面核心】单卡提供双重铺场频率，完美契合攻防对撞机制，实测胜率超90%的进攻支柱！ |
| 7 | **训练假人** | 中立 | 1费 | MINION | DP:1 | **90.4** | **S** | <span style="color:#ffeaa7; font-weight:bold;">82.4%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +13.2%</span> | 3 张 (拉满) | 【Red天梯必备】综合契合度处于顶级水平，PPO智能体第一优先级选牌！ |
| 8 | **商人** | 中立 | 2费 | MINION | DP:2 `DRAW_1` | **90.0** | **S** | <span style="color:#ffeaa7; font-weight:bold;">84.2%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +14.2%</span> | 3 张 (拉满) | 【Red扎实拼图】2费标准身材还自带抽牌，不亏手牌的优质节奏基石，构筑万金油。 |
| 9 | **切割者** | Red专属 | 2费 | MINION | DP:2 `DEGRADE_1` | **89.4** | **A** | <span style="color:#ffeaa7; font-weight:bold;">65.1%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +6.4%</span> | 2~3 张 | 【Red主力中坚】身材扎实且效果契合该卡组定位，推荐编入2~3张。 |
| 10 | **射线** | Red专属 | 1费 | SPELL | 攻2/防0 | **89.3** | **A** | <span style="color:#ffeaa7; font-weight:bold;">64.8%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +8.0%</span> | 2~3 张 | 【Red主力中坚】身材扎实且效果契合该卡组定位，推荐编入2~3张。 |
| 11 | **拾荒盾卫** | 中立 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **88.3** | **A** | <span style="color:#ffeaa7; font-weight:bold;">60.5%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +7.4%</span> | 2~3 张 | 【Red主力中坚】身材扎实且效果契合该卡组定位，推荐编入2~3张。 |
| 12 | **赤红掠袭者** | Red专属 | 5费 | MINION | DP:3 `RUSH,BONUS_SCORE_1` | **87.9** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.3%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +8.5%</span> | 2~3 张 | 【Red主力中坚】身材扎实且效果契合该卡组定位，推荐编入2~3张。 |
| 13 | **红色小队长** | Red专属 | 3费 | MINION | DP:3 `SPAWN_1_1` | **87.2** | **A** | <span style="color:#ffeaa7; font-weight:bold;">68.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +9.4%</span> | 2~3 张 | 【Red场面核心】单卡提供双重铺场频率，完美契合攻防对撞机制，实测胜率超90%的进攻支柱！ |
| 14 | **自爆** | Red专属 | 2费 | SPELL | 攻0/防0 `SACRIFICE_1_KILL_1,ATTACK_ONLY` | **79.6** | **B** | <span style="color:#81ecec;">44.7%</span> | <span style="color:#dfe6e9;">⚪ +1.0%</span> | 1~2 张 | 【Red合格拼图】常规过渡组件，按费用曲线合理填充1~2张即可。 |
| 15 | **雇佣兵** | 中立 | 3费 | MINION | DP:5 | **78.2** | **B** | <span style="color:#81ecec;">46.5%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.8%</span> | 1~2 张 | 【Red合格拼图】常规过渡组件，按费用曲线合理填充1~2张即可。 |
| 16 | **血祭爆燃** | Red专属 | 4费 | SPELL | 攻3/防0 `SACRIFICE_1_KILL_1,ATTACK_ONLY` | **72.9** | **B** | <span style="color:#81ecec;">40.2%</span> | <span style="color:#dfe6e9;">⚪ +2.1%</span> | 1~2 张 | 【Red卡手亏牌】费用极高且需牺牲场面随从，逆风根本开不出来，严重拖累胜率。 |
| 17 | **掠夺者** | Red专属 | 6费 | MINION | DP:4 `BONUS_SCORE_1` | **67.0** | **C** | <span style="color:#b2bec3;">22.5%</span> | <span style="color:#dfe6e9;">⚪ -0.2%</span> | 0~1 张 | 【Red平庸备选】缺乏主动破局手段，在卡组中表现平平，有更好卡牌时先考虑替换。 |
| 18 | **大块头** | Red专属 | 6费 | MINION | DP:10 `DISCARD_2` | **38.0** | **D** | <span style="color:#b2bec3;">4.3%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -9.5%</span> | 0 张 (坚决弃用) | 【Red严重陷阱】虽有高身材，但强制弃2张手牌直接破产，胜率拉低9.5%，坚决0张！ |

---

## 🔵 二、 【蔚蓝 (Blue) 卡组】战力评级与构筑分析
> **卡组定位**：防守反击 · 护盾壁垒 · 资源消耗  
> **牌库候选池**：12 张蔚蓝专属卡 + 6 张中立通用卡（共 18 张候选）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | DP/属性 | **综合评分** | 梯队 | **携带比例** | **对胜率影响 (ΔWR)** | 推荐抓取 | AI 助手独家实战锐评 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **寒晶护壁** | Blue专属 | 2费 | SPELL | 攻0/防2 `DRAW_1` | **98.0** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">89.2%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +16.7%</span> | 3 张 (拉满) | 【Blue扎实拼图】2费标准身材还自带抽牌，不亏手牌的优质节奏基石，构筑万金油。 |
| 2 | **蔚蓝卫士** | Blue专属 | 2费 | MINION | DP:2 `FORTIFY_1` | **93.5** | **S** | <span style="color:#ffeaa7; font-weight:bold;">85.7%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +17.0%</span> | 3 张 (拉满) | 【Blue天梯必备】综合契合度处于顶级水平，PPO智能体第一优先级选牌！ |
| 3 | **酒馆密账** | 中立 | 1费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **91.5** | **S** | <span style="color:#ffeaa7; font-weight:bold;">84.7%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +18.1%</span> | 3 张 (拉满) | 【资源过牌】高效滤抽组件，但在控制套牌中弃牌存在微小风险，视手牌充裕度带1~2张。 |
| 4 | **商人** | 中立 | 2费 | MINION | DP:2 `DRAW_1` | **91.3** | **S** | <span style="color:#ffeaa7; font-weight:bold;">84.9%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +16.3%</span> | 3 张 (拉满) | 【Blue扎实拼图】2费标准身材还自带抽牌，不亏手牌的优质节奏基石，构筑万金油。 |
| 5 | **盾兵** | Blue专属 | 1费 | MINION | DP:1 `FORTIFY_1` | **90.9** | **S** | <span style="color:#ffeaa7; font-weight:bold;">89.7%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +15.8%</span> | 3 张 (拉满) | 【Blue天梯必备】综合契合度处于顶级水平，PPO智能体第一优先级选牌！ |
| 6 | **霜盾见习官** | Blue专属 | 2费 | MINION | DP:1 `FORTIFY_1,SUPPORT_ATK_1` | **90.8** | **S** | <span style="color:#ffeaa7; font-weight:bold;">86.7%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +15.9%</span> | 3 张 (拉满) | 【Blue天梯必备】综合契合度处于顶级水平，PPO智能体第一优先级选牌！ |
| 7 | **佣兵斥候** | 中立 | 2费 | MINION | DP:1 `RUSH,DRAW_1` | **90.7** | **S** | <span style="color:#ffeaa7; font-weight:bold;">87.4%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +15.2%</span> | 3 张 (拉满) | 【Blue绝对幻神】突袭解场兼具过牌补手牌！完美抢回主动权，胜率净增+15.2%，无脑满编3张！ |
| 8 | **藤甲兵** | Blue专属 | 4费 | MINION | DP:4 `FORTIFY_2` | **85.9** | **A** | <span style="color:#ffeaa7; font-weight:bold;">61.6%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +8.0%</span> | 2~3 张 | 【Blue主力中坚】身材扎实且效果契合该卡组定位，推荐编入2~3张。 |
| 9 | **壁垒工匠** | Blue专属 | 3费 | MINION | DP:3 `FORTIFY_2` | **83.7** | **A** | <span style="color:#ffeaa7; font-weight:bold;">60.6%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +5.7%</span> | 2~3 张 | 【Blue主力中坚】身材扎实且效果契合该卡组定位，推荐编入2~3张。 |
| 10 | **拾荒盾卫** | 中立 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **83.3** | **A** | <span style="color:#ffeaa7; font-weight:bold;">60.6%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +11.3%</span> | 2~3 张 | 【Blue主力中坚】身材扎实且效果契合该卡组定位，推荐编入2~3张。 |
| 11 | **蔚蓝要塞** | Blue专属 | 6费 | MINION | DP:5 `FORTIFY_3,SUPPORT_ATK_1` | **81.3** | **A** | <span style="color:#ffeaa7; font-weight:bold;">62.2%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +11.4%</span> | 2~3 张 | 【Blue高费叹息墙】超高护甲防线，但在面对快攻时费用过高容易被卡死在手里，属于环境对策卡。 |
| 12 | **防御！** | Blue专属 | 1费 | SPELL | 攻0/防2 | **80.7** | **A** | <span style="color:#ffeaa7; font-weight:bold;">63.2%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +7.3%</span> | 2~3 张 | 【Blue主力中坚】身材扎实且效果契合该卡组定位，推荐编入2~3张。 |
| 13 | **雇佣兵** | 中立 | 3费 | MINION | DP:5 | **75.3** | **B** | <span style="color:#81ecec;">45.6%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.6%</span> | 1~2 张 | 【Blue合格拼图】常规过渡组件，按费用曲线合理填充1~2张即可。 |
| 14 | **冰封禁制** | Blue专属 | 3费 | SPELL | 攻2/防2 | **74.3** | **B** | <span style="color:#81ecec;">47.7%</span> | <span style="color:#dfe6e9;">⚪ +2.2%</span> | 1~2 张 | 【Blue合格拼图】常规过渡组件，按费用曲线合理填充1~2张即可。 |
| 15 | **训练假人** | 中立 | 1费 | MINION | DP:1 | **66.5** | **C** | <span style="color:#b2bec3;">26.1%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0~1 张 | 【Blue平庸备选】缺乏主动破局手段，在卡组中表现平平，有更好卡牌时先考虑替换。 |
| 16 | **弓箭手** | Blue专属 | 3费 | MINION | DP:3 `SUPPORT_ATK_1` | **63.7** | **C** | <span style="color:#b2bec3;">22.4%</span> | <span style="color:#dfe6e9;">⚪ -1.9%</span> | 0~1 张 | 【Blue平庸备选】缺乏主动破局手段，在卡组中表现平平，有更好卡牌时先考虑替换。 |
| 17 | **火铳手** | Blue专属 | 4费 | MINION | DP:4 `SUPPORT_ATK_2` | **61.8** | **C** | <span style="color:#b2bec3;">28.9%</span> | <span style="color:#dfe6e9;">⚪ -2.4%</span> | 0~1 张 | 【Blue平庸备选】缺乏主动破局手段，在卡组中表现平平，有更好卡牌时先考虑替换。 |
| 18 | **石像鬼** | Blue专属 | 6费 | MINION | DP:8 | **38.0** | **D** | <span style="color:#b2bec3;">3.2%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -18.8%</span> | 0 张 (坚决弃用) | 【Blue低效避坑】费用偏高或机制负收益，实测负贡献频发，建议放弃。 |

---

## 🟢 三、 【翠绿 (Green) 卡组】战力评级与构筑分析
> **卡组定位**：快速跳费 · 膨胀成长 · 终结核弹  
> **牌库候选池**：12 张翠绿专属卡 + 6 张中立通用卡（共 18 张候选）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | DP/属性 | **综合评分** | 梯队 | **携带比例** | **对胜率影响 (ΔWR)** | 推荐抓取 | AI 助手独家实战锐评 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **翠绿萌芽** | Green专属 | 2费 | SPELL | 攻0/防0 `RAMP_1` | **98.0** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">88.0%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +16.6%</span> | 3 张 (拉满) | 【Green跳费引擎】翠绿体系的命脉，先手跳费能让你提前打出高费大哥。 |
| 2 | **世界树恩泽** | Green专属 | 4费 | SPELL | 攻0/防3 `RAMP_1` | **88.6** | **A** | <span style="color:#ffeaa7; font-weight:bold;">60.7%</span> | <span style="color:#00b894; font-weight:bold;">🟢 +10.9%</span> | 2~3 张 | 【Green跳费引擎】翠绿体系的命脉，先手跳费能让你提前打出高费大哥。 |
| 3 | **芽苗祭司** | Green专属 | 3费 | MINION | DP:2 `RAMP_1` | **83.0** | **A** | <span style="color:#ffeaa7; font-weight:bold;">65.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +8.3%</span> | 2~3 张 | 【Green跳费引擎】翠绿体系的命脉，先手跳费能让你提前打出高费大哥。 |
| 4 | **灭世翡翠巨龙** | Green专属 | 9费 | MINION | DP:11 `RUSH,BONUS_SCORE_1` | **69.0** | **C** | <span style="color:#b2bec3;">22.5%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.3%</span> | 0~1 张 | 【Green终结核弹】单卡制胜手段，但极度依赖前期跳费与法力储备，没有跳费容易卡手到死。 |
| 5 | **翡翠幼龙** | Green专属 | 5费 | MINION | DP:3 `RUSH,DEATH_MANA_1` | **67.7** | **C** | <span style="color:#b2bec3;">20.8%</span> | <span style="color:#dfe6e9;">⚪ -0.9%</span> | 0~1 张 | 【Green平庸备选】缺乏主动破局手段，在卡组中表现平平，有更好卡牌时先考虑替换。 |
| 6 | **远古巨树** | Green专属 | 7费 | MINION | DP:8 `FORTIFY_2` | **61.2** | **C** | <span style="color:#b2bec3;">20.6%</span> | <span style="color:#dfe6e9;">⚪ -0.4%</span> | 0~1 张 | 【Green高费叹息墙】超高护甲防线，但在面对快攻时费用过高容易被卡死在手里，属于环境对策卡。 |
| 7 | **酒馆密账** | 中立 | 1费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **59.8** | **D** | <span style="color:#b2bec3;">5.7%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -14.2%</span> | 0 张 (坚决弃用) | 【资源过牌】高效滤抽组件，但在控制套牌中弃牌存在微小风险，视手牌充裕度带1~2张。 |
| 8 | **剧毒花** | Green专属 | 2费 | MINION | DP:1 `DEGRADE_2` | **56.6** | **D** | <span style="color:#b2bec3;">2.4%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -18.9%</span> | 0 张 (坚决弃用) | 【Green低效避坑】费用偏高或机制负收益，实测负贡献频发，建议放弃。 |
| 9 | **荆棘缠绕** | Green专属 | 2费 | SPELL | 攻2/防0 | **52.1** | **D** | <span style="color:#b2bec3;">2.3%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -14.1%</span> | 0 张 (坚决弃用) | 【Green低效避坑】费用偏高或机制负收益，实测负贡献频发，建议放弃。 |
| 10 | **狂暴生长** | Green专属 | 3费 | SPELL | 攻0/防3 | **51.5** | **D** | <span style="color:#b2bec3;">4.9%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -9.7%</span> | 0 张 (坚决弃用) | 【Green低效避坑】费用偏高或机制负收益，实测负贡献频发，建议放弃。 |
| 11 | **雇佣兵** | 中立 | 3费 | MINION | DP:5 | **48.0** | **D** | <span style="color:#b2bec3;">2.0%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -9.6%</span> | 0 张 (坚决弃用) | 【Green低效避坑】费用偏高或机制负收益，实测负贡献频发，建议放弃。 |
| 12 | **拾荒盾卫** | 中立 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **46.9** | **D** | <span style="color:#b2bec3;">5.7%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -8.8%</span> | 0 张 (坚决弃用) | 【Green低效避坑】费用偏高或机制负收益，实测负贡献频发，建议放弃。 |
| 13 | **森林之狼** | Green专属 | 4费 | MINION | DP:4 | **46.7** | **D** | <span style="color:#b2bec3;">1.9%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -13.7%</span> | 0 张 (坚决弃用) | 【Green低效避坑】费用偏高或机制负收益，实测负贡献频发，建议放弃。 |
| 14 | **商人** | 中立 | 2费 | MINION | DP:2 `DRAW_1` | **46.6** | **D** | <span style="color:#b2bec3;">2.7%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -14.1%</span> | 0 张 (坚决弃用) | 【Green扎实拼图】2费标准身材还自带抽牌，不亏手牌的优质节奏基石，构筑万金油。 |
| 15 | **训练假人** | 中立 | 1费 | MINION | DP:1 | **45.2** | **D** | <span style="color:#b2bec3;">2.1%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -11.8%</span> | 0 张 (坚决弃用) | 【Green低效避坑】费用偏高或机制负收益，实测负贡献频发，建议放弃。 |
| 16 | **树人** | Green专属 | 3费 | MINION | DP:2 `FORTIFY_1` | **45.0** | **D** | <span style="color:#b2bec3;">2.0%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -11.0%</span> | 0 张 (坚决弃用) | 【Green低效避坑】费用偏高或机制负收益，实测负贡献频发，建议放弃。 |
| 17 | **佣兵斥候** | 中立 | 2费 | MINION | DP:1 `RUSH,DRAW_1` | **44.7** | **D** | <span style="color:#b2bec3;">3.4%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -16.0%</span> | 0 张 (坚决弃用) | 【Green绝对幻神】突袭解场兼具过牌补手牌！完美抢回主动权，胜率净增+-16.0%，无脑满编3张！ |
| 18 | **翡翠巨熊** | Green专属 | 6费 | MINION | DP:7 `FORTIFY_2` | **38.0** | **D** | <span style="color:#b2bec3;">4.8%</span> | <span style="color:#d63031; font-weight:bold;">🔴 -11.9%</span> | 0 张 (坚决弃用) | 【Green高费叹息墙】超高护甲防线，但在面对快攻时费用过高容易被卡死在手里，属于环境对策卡。 |

---

## ⚪ 四、 【中立 (Neutral) 卡牌】全卡组泛用性与效用异质性分析
> **学术亮点**：相同的中立卡在不同流派（快攻/控制/跳费）中具有显著的效用异质性。

| 中立卡名称 | 费用 | 类型 | 🔴 赤红卡组中评分 | 🔵 蔚蓝卡组中评分 | 🟢 翠绿卡组中评分 | 最优契合阵营 | AI 跨卡组机制定位 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **佣兵斥候** | 2费 | MINION | **98.0** | **90.7** | **44.7** | **赤红 (快攻)** | 专精型对策拼图 |
| **酒馆密账** | 1费 | SPELL | **95.5** | **91.5** | **59.8** | **赤红 (快攻)** | 专精型对策拼图 |
| **训练假人** | 1费 | MINION | **90.4** | **66.5** | **45.2** | **赤红 (快攻)** | 专精型对策拼图 |
| **商人** | 2费 | MINION | **90.0** | **91.3** | **46.6** | **蔚蓝 (防守)** | 专精型对策拼图 |
| **拾荒盾卫** | 3费 | MINION | **88.3** | **83.3** | **46.9** | **赤红 (快攻)** | 专精型对策拼图 |
| **雇佣兵** | 3费 | MINION | **78.2** | **75.3** | **48.0** | **赤红 (快攻)** | 专精型对策拼图 |
