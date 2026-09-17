# 🏆 TCG 卡牌战力评级与构筑指南（阵营分色专榜）

> **构筑规则说明**：在 TCG 标准规则中，各阵营（赤红/蔚蓝/翠绿）由**阵营专属卡 + 双色协同卡 + 中立通用卡**构筑。本指南基于 PPO 强化学习智能体在对战环境中的实战数据，综合**【卡组携带率】**与**【局势胜率贡献 (ΔWR)】**两大维度，按阵营分色独立建榜，提供客观、严谨的构筑参考与单卡解析。

---

## 🔴 一、 【赤红 (Red) 卡组】单卡战力与构筑指南
> **战术核心**：快攻压制 · 牺牲协同 · 节奏斩杀  
> **候选牌池**：32 张Red专属卡 + 8 张双色协同卡 + 6 张中立通用卡（共 46 张候选，择优遴选 30 张入套）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | 属性/数值 | **综合评分** | 梯队 | **携带率** | **胜率贡献 (ΔWR)** | 推荐配置 | 实战构筑解析 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **佣兵斥候** | 中立通用 | 2费 | MINION | DP:1 `RUSH` | **98.0** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.2%</span> | 3 张 (核心满编) | 两费突袭一点，解场夺先手，满编补低费互动与伤害延伸。 |
| 2 | **赤焰哨卫** | Red专属 | 4费 | MINION | DP:3 `RUSH,FORTIFY_1` | **97.5** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.1%</span> | 3 张 (核心满编) | 四费突袭并加固，解场同时站住频率，满编撑中期攻防转换。 |
| 3 | **熔岩破阵者** | Red专属 | 5费 | MINION | DP:4 `RUSH,BONUS_SCORE_1` | **97.2** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.0%</span> | 3 张 (核心满编) | 五费突袭得分，中期抢血与交换一体，满编保证核心节奏。 |
| 4 | **赤红突破手** | Red专属 | 8费 | MINION | DP:1 `RUSH,BONUS_SCORE_1` | **96.9** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.0%</span> | 3 张 (核心满编) | 八费突袭得分终结，满编确保后期抽到即能补足斩杀与分数。 |
| 5 | **赤红新兵** | Red专属 | 1费 | MINION | DP:1 | **95.3** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.7%</span> | 3 张 (核心满编) | 一费一点场攻，最顺滑的早期曲线，满编支撑铺场与牺牲。 |
| 6 | **红色小队长** | Red专属 | 3费 | MINION | DP:3 `SPAWN_1_1` | **94.5** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.6%</span> | 3 张 (核心满编) | 三费铺两个频率，持续给压力，满编保证三费节奏与牺牲素材。 |
| 7 | **商人** | 中立通用 | 2费 | MINION | DP:2 `DRAW_1` | **91.3** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.1%</span> | 3 张 (核心满编) | 两费站场抽一，润滑手牌并延续攻势，满编保证资源不断。 |
| 8 | **集结号手** | Red专属 | 2费 | MINION | DP:1 `SPAWN_1_1,ATTACK_ONLY` | **90.5** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.0%</span> | 3 张 (核心满编) | 两费生两个频率，低费铺场核心，满编保证开局压制与交换。 |
| 9 | **赤红破阵兵** | Red专属 | 5费 | MINION | DP:3 `RUSH` | **89.7** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 2 张 (主力配置) | 五费突袭三点，解场后保留站场，二张稳固中期争夺先手。 |
| 10 | **裂甲掷斧手** | Red专属 | 6费 | MINION | DP:1 `RUSH,DEGRADE_1` | **87.6** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.5%</span> | 2 张 (主力配置) | 六费突袭降级，解场后留一点场攻，二张补中后期夺回先手。 |
| 11 | **赤红献祭** | Red专属 | 2费 | SPELL | 攻0/防0 `SACRIFICE_1_KILL_1` | **84.9** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.1%</span> | 2 张 (主力配置) | 两费牺牲杀单位，灵活解掉关键墙，二张兼顾节奏与手牌资源。 |
| 12 | **红绿共生** | Red/Green双色 | 5费 | MINION | DP:5 `RUSH,DEATH_DRAW_1` | **58.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.1%</span> | 0 张 (暂不推荐) | 五费突袭亡语抽牌，交换后补牌但攻击偏低，节奏偏中速。 |
| 13 | **赤红掠袭者** | Red专属 | 9费 | MINION | DP:1 `RUSH,BONUS_SCORE_1` | **58.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.2%</span> | 0 张 (暂不推荐) | 九费突袭仅一点场攻，终结太慢，高费位有更优得分选择。 |
| 14 | **赤红中期突破手** | Red专属 | 8费 | MINION | DP:5 `RUSH,BONUS_SCORE_1` | **58.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.2%</span> | 0 张 (暂不推荐) | 八费突袭得分，身材更高却仍慢，同费突破手更符合斩杀需求。 |
| 15 | **破阵先锋** | Red专属 | 6费 | MINION | DP:4 `RUSH,BONUS_SCORE_1` | **57.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.3%</span> | 0 张 (暂不推荐) | 六费突袭得分尚可，但五费熔岩破阵者更早，卡位被挤占。 |
| 16 | **训练假人** | 中立通用 | 1费 | MINION | DP:1 | **57.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.4%</span> | 0 张 (暂不推荐) | 一费白板无攻击延伸，赤红新兵同费更契合铺场与牺牲。 |
| 17 | **切割者** | Red专属 | 2费 | MINION | DP:2 `DEGRADE_1` | **56.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.4%</span> | 0 张 (暂不推荐) | 二费降级身材普通，压制力不足，前期更需要站场与打脸。 |
| 18 | **赤红战盾** | Red专属 | 2费 | SPELL | 攻0/防4 | **56.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.5%</span> | 0 张 (暂不推荐) | 两费四点防御，纯防守牌，抢血卡组不需要拖延对局。 |
| 19 | **牺牲祭师** | Red专属 | 4费 | MINION | DP:3 `SACRIFICE_1_KILL_1,DEATH_DRAW_1` | **56.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.6%</span> | 0 张 (暂不推荐) | 四费牺牲解场加亡语抽牌，节奏偏慢，快攻不需要中期换牌。 |
| 20 | **赤蓝战术交换** | Red/Blue双色 | 2费 | SPELL | 攻3/防0 `DRAW_1` | **55.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.6%</span> | 0 张 (暂不推荐) | 两费打三抽一，效率尚可但占直伤卡位，快攻更愿带生物。 |
| 21 | **烈焰清算** | Red专属 | 4费 | SPELL | 攻4/防0 `DEGRADE_1` | **55.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.6%</span> | 0 张 (暂不推荐) | 四费打四降级，解场尚可但直伤不足，快攻更需低费打脸。 |
| 22 | **献祭之焰** | Red专属 | 3费 | SPELL | 攻0/防0 `SACRIFICE_1_KILL_1,DISCARD_1` | **55.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.7%</span> | 0 张 (暂不推荐) | 三费牺牲杀牌还要弃牌，资源亏损过大，快攻无法承担。 |
| 23 | **血祭爆燃** | Red专属 | 3费 | SPELL | 攻3/防0 `SACRIFICE_1_KILL_1,ATTACK_ONLY` | **54.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.8%</span> | 0 张 (暂不推荐) | 三费牺牲换三攻解场，亏牌又限攻击，抢血体系无法承受。 |
| 24 | **射线** | Red专属 | 2费 | SPELL | 攻2/防0 | **54.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.8%</span> | 0 张 (暂不推荐) | 两费打二直伤效率平庸，抢血不如低费生物，难以进入构筑。 |
| 25 | **自爆** | Red专属 | 2费 | SPELL | 攻0/防0 `SACRIFICE_1_KILL_1,ATTACK_ONLY` | **54.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.9%</span> | 0 张 (暂不推荐) | 两费牺牲解场且限攻击，亏节奏，快攻不愿用场面换单解。 |
| 26 | **赤红战意** | Red专属 | 5费 | SPELL | 攻0/防0 `TEMP_MANA_2` | **53.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.9%</span> | 0 张 (暂不推荐) | 五费换临时法力，当回合难兑现，快攻不愿为爆发牺牲场面。 |
| 27 | **赤红突击手** | Red专属 | 3费 | MINION | DP:1 `DEATH_DRAW_1` | **53.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.9%</span> | 0 张 (暂不推荐) | 三费一点场攻，亡语抽一太慢，抢血卡组不缺这类滞后资源，空。 |
| 28 | **赤蓝交织者** | Red/Blue双色 | 4费 | MINION | DP:4 `SUPPORT_ATK_1` | **53.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -3.0%</span> | 0 张 (暂不推荐) | 四费支援攻一，中速增益与抢血节奏不合，基础版不带。 |
| 29 | **拾荒盾卫** | 中立通用 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **52.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.1%</span> | 0 张 (暂不推荐) | 三费防守亡语抽牌，节奏滞后，抢血卡组不愿投入中期防御。 |
| 30 | **赤红铁卫** | Red专属 | 3费 | MINION | DP:3 `FORTIFY_2` | **52.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 三费固守两点偏防守，缺乏进攻压力，不适合抢血思路。 |
| 31 | **雇佣兵** | 中立通用 | 3费 | MINION | DP:5 | **52.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 三费五点身材扎实，但无词条无压力，快攻需要即时威胁。 |
| 32 | **酒馆密账** | 中立通用 | 4费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **51.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 四费抽二弃一，费用过重且过滤有限，快攻不需要慢速补牌。 |
| 33 | **赤蓝交织者** | Red/Blue双色 | 4费 | MINION | DP:4 `SUPPORT_ATK_1` | **51.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.3%</span> | 0 张 (暂不推荐) | 四费支援攻一，中速增益与抢血节奏不合，基础版不带。 |
| 34 | **余烬反击官** | Red专属 | 5费 | MINION | DP:5 `SUPPORT_ATK_1,DEATH_DRAW_1` | **51.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.3%</span> | 0 张 (暂不推荐) | 五费支援亡语抽牌，身材扎实但偏慢，不利于快速压低血线。 |
| 35 | **牺牲角斗士** | Red专属 | 4费 | MINION | DP:4 `SACRIFICE_1_KILL_1` | **50.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0 张 (暂不推荐) | 四费牺牲解场虽稳，但快攻更愿直接铺场，费用与卡位不合。 |
| 36 | **献祭狂徒** | Red专属 | 4费 | MINION | DP:4 `SACRIFICE_1_KILL_1` | **50.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0 张 (暂不推荐) | 四费牺牲解场，进攻端平庸，快攻不愿用生物换单次移除。 |
| 37 | **战地督军** | Red专属 | 5费 | MINION | DP:4 `SUPPORT_ATK_1` | **50.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 五费支援加攻，需场面配合且启动慢，不如直接突袭抢血。 |
| 38 | **红绿古树桥梁** | Red/Green双色 | 6费 | MINION | DP:6 `FORTIFY_2,DEATH_DRAW_1` | **50.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 六费防守亡语抽牌，厚度足却拖节奏，不适合快速抢血。 |
| 39 | **余烬传令官** | Red专属 | 5费 | MINION | DP:5 `DEATH_DRAW_1` | **49.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 五费亡语抽牌，身材尚可但节奏滞后，抢血卡组无暇经营。 |
| 40 | **红绿共生体** | Red/Green双色 | 6费 | MINION | DP:6 `FORTIFY_2,BONUS_SCORE_1` | **49.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.6%</span> | 0 张 (暂不推荐) | 六费支援加固偏防守，抢血卡组卡位不足，不采用。 |
| 41 | **赤蓝交织者** | Red/Blue双色 | 4费 | MINION | DP:4 `SUPPORT_ATK_1,DEATH_MANA_1` | **49.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.6%</span> | 0 张 (暂不推荐) | 四费支援攻一，中速增益与抢血节奏不合，基础版不带。 |
| 42 | **破阵狂徒** | Red专属 | 8费 | MINION | DP:2 `DEGRADE_1` | **49.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.7%</span> | 0 张 (暂不推荐) | 八费降级仅两点场攻，费用沉重，远逊同曲线突袭得分单位。 |
| 43 | **大块头** | Red专属 | 6费 | MINION | DP:10 `DISCARD_2` | **48.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.7%</span> | 0 张 (暂不推荐) | 六费大墙弃两张手牌，快攻手牌即伤害，拖节奏故不采用。 |
| 44 | **红绿共生体** | Red/Green双色 | 6费 | MINION | DP:6 `SUPPORT_ATK_2,FORTIFY_2` | **48.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 六费支援加固偏防守，抢血卡组卡位不足，不采用。 |
| 45 | **掠夺者** | Red专属 | 7费 | MINION | DP:3 `BONUS_SCORE_1` | **48.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 七费三点场攻太迟，得分词条救不了节奏，快攻不会等它。 |
| 46 | **红莲驻防长** | Red专属 | 6费 | MINION | DP:6 `FORTIFY_2` | **48.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 六费固守大墙，防守价值高却拖慢进攻，抢血体系不采用。 |

---

## 🔵 二、 【蔚蓝 (Blue) 卡组】单卡战力与构筑指南
> **战术核心**：防守反击 · 固守护盾 · 资源消耗  
> **候选牌池**：32 张Blue专属卡 + 8 张双色协同卡 + 6 张中立通用卡（共 46 张候选，择优遴选 30 张入套）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | 属性/数值 | **综合评分** | 梯队 | **携带率** | **胜率贡献 (ΔWR)** | 推荐配置 | 实战构筑解析 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **冰封禁制** | Blue专属 | 2费 | SPELL | 攻6/防2 | **95.1** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.7%</span> | 3 张 (核心满编) | 二费打六带二防，高效解场兼保命，核心满编三张不疑。 |
| 2 | **佣兵斥候** | 中立通用 | 2费 | MINION | DP:1 `RUSH` | **90.0** | **S** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.9%</span> | 3 张 (核心满编) | 二费突袭，解场抢节奏，核心满编三张，前期主动权关键。 |
| 3 | **冰潮突袭者** | Blue专属 | 3费 | MINION | DP:3 `RUSH` | **89.7** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 2 张 (主力配置) | 三费突袭三攻，解小怪抢先手，主力两张补防守反攻。 |
| 4 | **破甲潮汐兵** | Blue专属 | 2费 | MINION | DP:3 `RUSH,DEGRADE_1` | **89.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 2 张 (主力配置) | 二费突袭降攻一，解场抢节奏，主力两张应对快攻与中速。 |
| 5 | **训练假人** | 中立通用 | 1费 | MINION | DP:1 | **89.2** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 2 张 (主力配置) | 一费一防白板，拖延快攻攻击频率，主力两张作炮灰。 |
| 6 | **盾兵** | Blue专属 | 1费 | MINION | DP:1 `FORTIFY_1` | **88.9** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.7%</span> | 2 张 (主力配置) | 一费固守吸收前期伤害，为主力配置，两张开局稳定防线。 |
| 7 | **蔚蓝智慧** | Blue专属 | 3费 | SPELL | 攻0/防0 `DRAW_2,TEMP_MANA_1` | **86.8** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.4%</span> | 2 张 (主力配置) | 三费抽二返一费，过牌续航兼跳费，主力两张调资源。 |
| 8 | **商人** | 中立通用 | 2费 | MINION | DP:2 `DRAW_1` | **86.3** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.3%</span> | 2 张 (主力配置) | 二费站场抽一，兼顾曲线与资源，主力两张润滑前期。 |
| 9 | **寒晶护壁** | Blue专属 | 2费 | SPELL | 攻0/防2 `DRAW_1` | **84.9** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.1%</span> | 2 张 (主力配置) | 二费抽一加二防，润滑手牌与护脸，主力配置两张稳定。 |
| 10 | **蔚蓝守卫** | Blue专属 | 2费 | MINION | DP:2 `FORTIFY_2` | **84.4** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.0%</span> | 2 张 (主力配置) | 二费二防固守二，前期合格墙，主力两张挡住快攻攻势。 |
| 11 | **蔚蓝护盾反击** | Blue专属 | 3费 | SPELL | 攻0/防4 `FORTIFY_1` | **79.9** | **B** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.3%</span> | 2 张 (按需携带) | 三费加四防附固守，按需两张，强化低费墙对抗爆发。 |
| 12 | **蔚蓝盾卫** | Blue专属 | 5费 | MINION | DP:6 `FORTIFY_3,SUPPORT_ATK_1` | **79.1** | **B** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.1%</span> | 2 张 (按需携带) | 5费DP 6随从，附带【FORTIFY_3,SUPPORT_ATK_1】，承担Blue阵营核心战术组件。 |
| 13 | **蔚蓝要塞** | Blue专属 | 5费 | MINION | DP:6 `FORTIFY_3,SUPPORT_ATK_1` | **78.5** | **B** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.0%</span> | 2 张 (按需携带) | 五费六防固守三，附支援攻一，按需两张撑中期与反打。 |
| 14 | **石像鬼** | Blue专属 | 6费 | MINION | DP:8 | **78.0** | **B** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.0%</span> | 2 张 (按需携带) | 六费八防白板，中后期高质量墙，吸收直伤后为大哥铺路。 |
| 15 | **防御！** | Blue专属 | 1费 | SPELL | 攻0/防2 | **76.0** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +0.6%</span> | 1 张 (按需携带) | 一费加二防，针对快攻的临时解，按环境作一张对策。 |
| 16 | **蔚蓝守壁** | Blue专属 | 2费 | MINION | DP:2 `FORTIFY_2` | **58.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.1%</span> | 0 张 (暂不推荐) | 二费二防固守二，与蔚蓝守卫重复，卡位紧张故舍弃。 |
| 17 | **潮汐学者** | Blue专属 | 2费 | MINION | DP:2 `DEATH_DRAW_1` | **58.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.2%</span> | 0 张 (暂不推荐) | 2费DP 2随从，附带【DEATH_DRAW_1】，承担Blue阵营核心战术组件。 |
| 18 | **赤蓝战术交换** | Red/Blue双色 | 2费 | SPELL | 攻3/防0 `DRAW_1` | **58.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.2%</span> | 0 张 (暂不推荐) | 二费打三抽一，伤害偏低且过牌延迟，不符防守需求。 |
| 19 | **寒冰解离** | Blue专属 | 3费 | SPELL | 攻4/防0 | **57.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.3%</span> | 0 张 (暂不推荐) | 三费打四白板，效率低于冰封禁制，防守卡组无空闲。 |
| 20 | **寒霜破甲** | Blue专属 | 2费 | SPELL | 攻3/防0 `DEGRADE_1` | **57.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.3%</span> | 0 张 (暂不推荐) | 二费打三降攻一，看似灵活，实际解场深度不足，不推荐。 |
| 21 | **蔚蓝冲击** | Blue专属 | 2费 | SPELL | 攻4/防0 | **57.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.4%</span> | 0 张 (暂不推荐) | 二费打四无附加，解场效率一般，防守体系不优先选用。 |
| 22 | **霜盾见习官** | Blue专属 | 2费 | MINION | DP:1 `FORTIFY_1,SUPPORT_ATK_1` | **56.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.5%</span> | 0 张 (暂不推荐) | 二费身材差，固守与支援攻一收益低，易被解，不配卡位。 |
| 23 | **蓝晶守御者** | Blue专属 | 3费 | MINION | DP:3 `FORTIFY_1,ATTACK_ONLY` | **55.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.6%</span> | 0 张 (暂不推荐) | 三费仅固守一且只能攻击，防御速度差，难以承担护脸。 |
| 24 | **蔚蓝卫士** | Blue专属 | 2费 | MINION | DP:2 `FORTIFY_1` | **55.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.6%</span> | 0 张 (暂不推荐) | 二费仅一固守，身材平庸，卡位竞争激烈，构筑无需投入。 |
| 25 | **壁垒工匠** | Blue专属 | 3费 | MINION | DP:3 `FORTIFY_2` | **55.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.7%</span> | 0 张 (暂不推荐) | 三费三防固守二尚可，但同费有更优选择，构筑暂不进。 |
| 26 | **弓箭手** | Blue专属 | 3费 | MINION | DP:3 `SUPPORT_ATK_1` | **54.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.8%</span> | 0 张 (暂不推荐) | 三费支援攻一，防御端太弱，防守体系难保它站场，弃用。 |
| 27 | **蔚蓝破甲师** | Blue专属 | 3费 | MINION | DP:3 `DEGRADE_1` | **54.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.9%</span> | 0 张 (暂不推荐) | 三费降攻一，防守卡组不需要软解，占卡位且威胁不足。 |
| 28 | **赤蓝交织者** | Red/Blue双色 | 4费 | MINION | DP:4 `SUPPORT_ATK_1` | **53.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.9%</span> | 0 张 (暂不推荐) | 4费DP 4随从，附带【SUPPORT_ATK_1】，承担Blue阵营核心战术组件。 |
| 29 | **蔚蓝盾卫** | Blue专属 | 2费 | MINION | DP:2 `FORTIFY_1` | **53.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.9%</span> | 0 张 (暂不推荐) | 2费DP 2随从，附带【FORTIFY_1】，承担Blue阵营核心战术组件。 |
| 30 | **拾荒盾卫** | 中立通用 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **53.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -3.0%</span> | 0 张 (暂不推荐) | 三费固守一亡语抽一，站场与过牌皆慢，竞争不过商人。 |
| 31 | **潮汐学者** | Blue专属 | 4费 | MINION | DP:4 `DEATH_DRAW_1` | **53.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -3.0%</span> | 0 张 (暂不推荐) | 4费DP 4随从，附带【DEATH_DRAW_1】，承担Blue阵营核心战术组件。 |
| 32 | **赤蓝交织者** | Red/Blue双色 | 4费 | MINION | DP:4 `SUPPORT_ATK_1` | **52.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.1%</span> | 0 张 (暂不推荐) | 4费DP 4随从，附带【SUPPORT_ATK_1】，承担Blue阵营核心战术组件。 |
| 33 | **藤甲兵** | Blue专属 | 4费 | MINION | DP:4 `FORTIFY_2` | **52.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.1%</span> | 0 张 (暂不推荐) | 四费四防固守二，数值不及同费曲线，防守效率低不采用。 |
| 34 | **秘蓝回溯** | Blue专属 | 3费 | SPELL | 攻0/防3 `DRAW_1` | **52.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 三费加三防抽一，费用偏高，同类低费法术更易衔接。 |
| 35 | **雇佣兵** | 中立通用 | 3费 | MINION | DP:5 | **52.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 三费五防白板，虽耐打但无战术价值，卡位留给功能件。 |
| 36 | **蓝绿潮涌** | Blue/Green双色 | 3费 | SPELL | 攻0/防0 `DRAW_2` | **51.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 三费抽二无护脸，防守卡组需即时影响，纯过牌太慢。 |
| 37 | **酒馆密账** | 中立通用 | 4费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **51.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.3%</span> | 0 张 (暂不推荐) | 四费抽二弃一，过滤代价大，防守卡组手牌不容弃。 |
| 38 | **蓝绿潮汐织法者** | Blue/Green双色 | 5费 | MINION | DP:4 `SUPPORT_ATK_1,RAMP_1` | **51.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.3%</span> | 0 张 (暂不推荐) | 五费支援攻一并跳费，进攻与跳费混合，防守卡组不取。 |
| 39 | **赤蓝交织者** | Red/Blue双色 | 4费 | MINION | DP:4 `SUPPORT_ATK_1,DEATH_MANA_1` | **50.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0 张 (暂不推荐) | 4费DP 4随从，附带【SUPPORT_ATK_1,DEATH_MANA_1】，承担Blue阵营核心战术组件。 |
| 40 | **蓝绿潮汐者** | Blue/Green双色 | 5费 | MINION | DP:5 `DEATH_MANA_2,FORTIFY_2` | **50.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0 张 (暂不推荐) | 五费亡语回费带固守二，节奏拖沓，防守端不如直接大墙。 |
| 41 | **碧波灵鳍** | Blue专属 | 4费 | MINION | DP:4 `SPAWN_1_1,FORTIFY_1` | **50.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 四费铺一崽带固守一，场面收益低，防守卡组不需要铺场。 |
| 42 | **火铳手** | Blue专属 | 3费 | MINION | DP:4 `SUPPORT_ATK_2` | **50.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 三费四防支援攻二，进攻偏慢，防守卡组难靠它取胜。 |
| 43 | **法力回流哨兵** | Blue专属 | 4费 | MINION | DP:4 `DEATH_MANA_2` | **49.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.6%</span> | 0 张 (暂不推荐) | 四费亡语回二费，节奏太慢，防守卡组不缺此等返费。 |
| 44 | **蓝绿潮涌使** | Blue/Green双色 | 5费 | MINION | DP:5 `RAMP_1,DEATH_DRAW_1` | **49.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.6%</span> | 0 张 (暂不推荐) | 五费跳费亡语抽一，偏资源展开，防守端节奏不合拍。 |
| 45 | **深海守望者** | Blue专属 | 5费 | MINION | DP:5 `FORTIFY_3` | **48.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.7%</span> | 0 张 (暂不推荐) | 五费五防固守三，五费优质墙众多，它性价比不够突出。 |
| 46 | **深流变奏师** | Blue专属 | 5费 | MINION | DP:5 `SUPPORT_ATK_1,DEATH_MANA_2` | **48.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 五费支援攻一亡语回二费，偏进攻且节奏滞后，不带。 |

---

## 🟢 三、 【翠绿 (Green) 卡组】单卡战力与构筑指南
> **战术核心**：法力跳费 · 质量成长 · 终结大哥  
> **候选牌池**：32 张Green专属卡 + 8 张双色协同卡 + 6 张中立通用卡（共 46 张候选，择优遴选 30 张入套）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | 属性/数值 | **综合评分** | 梯队 | **携带率** | **胜率贡献 (ΔWR)** | 推荐配置 | 实战构筑解析 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **灭世翡翠巨龙** | Green专属 | 9费 | MINION | DP:4 `RUSH,BONUS_SCORE_1` | **96.7** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.0%</span> | 3 张 (核心满编) | 九费突袭加额外得分，跳费后终结比赛，核心满编不怕卡手。 |
| 2 | **训练假人** | 中立通用 | 1费 | MINION | DP:1 | **96.4** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.9%</span> | 3 张 (核心满编) | 一费低费站场挡突袭，保护跳费单位，核心满编润滑前期。 |
| 3 | **萌芽跳费使** | Green专属 | 3费 | MINION | DP:3 `RAMP_1` | **94.5** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.6%</span> | 3 张 (核心满编) | 三费跳费兼站场，衔接四五六费曲线，核心满编提速。 |
| 4 | **佣兵斥候** | 中立通用 | 2费 | MINION | DP:1 `RUSH` | **90.0** | **S** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.9%</span> | 3 张 (核心满编) | 两费突袭解小随从，抢回先手兼护跳费，核心满编保前期。 |
| 5 | **荆棘反制者** | Green专属 | 3费 | MINION | DP:3 `RUSH,DEGRADE_1` | **89.7** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 2 张 (主力配置) | 三费突袭削弱敌随从，解场兼站场，主力两张争夺前期节奏。 |
| 6 | **藤蔓突袭者** | Green专属 | 4费 | MINION | DP:4 `RUSH` | **89.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 2 张 (主力配置) | 四费突袭即时解场，夺回先手后继续跳费，主力两张灵活。 |
| 7 | **红绿共生** | Red/Green双色 | 5费 | MINION | DP:5 `RUSH,DEATH_DRAW_1` | **89.2** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 2 张 (主力配置) | 五费突袭解场，亡语补手牌，主力两张兼顾返场与续航。 |
| 8 | **翡翠幼龙** | Green专属 | 7费 | MINION | DP:4 `RUSH,DEATH_MANA_1` | **88.9** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.7%</span> | 2 张 (主力配置) | 七费突袭解场，死亡返费支撑后续核弹，主力两张稳定过渡。 |
| 9 | **世界树恩泽** | Green专属 | 5费 | SPELL | 攻0/防3 `RAMP_1` | **88.9** | **A** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.7%</span> | 3 张 (主力配置) | 五费跳费并加防，护脸同时扩张法力，满编保证高费准时启动。 |
| 10 | **翠绿幼苗** | Green专属 | 1费 | MINION | DP:1 `FORTIFY_1` | **87.9** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.5%</span> | 2 张 (主力配置) | 一费低攻防御，前期吸收伤害，带两张润滑跳费曲线。 |
| 11 | **翡翠藤盾卫** | Green专属 | 2费 | MINION | DP:2 `FORTIFY_2` | **86.8** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.4%</span> | 2 张 (主力配置) | 两费高防御站场，抵御快攻并保护跳费，主力两张稳固前期。 |
| 12 | **剧毒花** | Green专属 | 2费 | MINION | DP:2 `DEGRADE_2` | **86.3** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.3%</span> | 2 张 (主力配置) | 两费削弱敌随从，前期交换能拖慢快攻，带两张补防守短板。 |
| 13 | **翠绿跳费者** | Green专属 | 4费 | MINION | DP:3 `RAMP_1` | **73.6** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +0.3%</span> | 1 张 (按需携带) | 四费跳费身材一般，节奏不顺时补一张，顺境不必多带。 |
| 14 | **翠绿资源滋长** | Green专属 | 4费 | SPELL | 攻0/防0 `RAMP_1,DRAW_1` | **73.1** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +0.2%</span> | 1 张 (按需携带) | 四费跳费加过牌，后期补资源可用，前期节奏偏慢只挂一。 |
| 15 | **商人** | 中立通用 | 2费 | MINION | DP:2 `DRAW_1` | **58.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.2%</span> | 0 张 (暂不推荐) | 两费过一牌身材弱，跳费体系需法力扩张，过牌位不优先。 |
| 16 | **自然献祭** | Green专属 | 4费 | SPELL | 攻3/防0 `SACRIFICE_1_KILL_1` | **57.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.3%</span> | 0 张 (暂不推荐) | 四费牺牲一随从换解，成本高且亏节奏，跳费不缺此对策。 |
| 17 | **荆棘缠绕** | Green专属 | 2费 | SPELL | 攻4/防0 | **57.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.4%</span> | 0 张 (暂不推荐) | 两费直伤仅四点，解不掉主流随从，还挤占跳费过牌卡位。 |
| 18 | **翠绿哨兵** | Green专属 | 3费 | MINION | DP:2 `FORTIFY_2,DEATH_DRAW_1` | **57.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.4%</span> | 0 张 (暂不推荐) | 三费低攻虽亡语过牌，但交换差节奏慢，跳费更需即时扩张。 |
| 19 | **拾荒盾卫** | 中立通用 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **56.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.6%</span> | 0 张 (暂不推荐) | 三费防御亡语过牌，交换尚可但慢，核心跳费位轮不到。 |
| 20 | **蓝绿潮涌** | Blue/Green双色 | 3费 | SPELL | 攻0/防0 `DRAW_2` | **55.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.6%</span> | 0 张 (暂不推荐) | 三费抽二只过牌，不扩张法力，跳费链不需要纯滤抽。 |
| 21 | **翠绿驻防藤蔓** | Green专属 | 2费 | MINION | DP:2 `FORTIFY_1` | **55.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.6%</span> | 0 张 (暂不推荐) | 两费防御随从，身材与词条皆平庸，前期卡位让给功能牌。 |
| 22 | **树人** | Green专属 | 3费 | MINION | DP:2 `FORTIFY_1` | **55.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.7%</span> | 0 张 (暂不推荐) | 三费两攻防御普通，站场与交换均无优势，跳费体系不采用。 |
| 23 | **狂暴生长** | Green专属 | 3费 | SPELL | 攻0/防3 | **54.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.7%</span> | 0 张 (暂不推荐) | 三费仅加三点防御，无法跳费或交换，防御法术位已被取代。 |
| 24 | **古树庇护** | Green专属 | 3费 | SPELL | 攻0/防5 `FORTIFY_1` | **54.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.8%</span> | 0 张 (暂不推荐) | 三费防御法术收益低，既不解场也不跳费，构筑优先舍弃。 |
| 25 | **翠绿守林人** | Green专属 | 3费 | MINION | DP:3 `FORTIFY_2` | **54.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.8%</span> | 0 张 (暂不推荐) | 三费防御随从交换尚可，但无跳费突袭，核心卡位竞争失败。 |
| 26 | **自然生长** | Green专属 | 3费 | SPELL | 攻0/防0 `RAMP_1` | **54.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.9%</span> | 0 张 (暂不推荐) | 三费单跳无过牌，与低费跳费重复，满编核心后无需追加。 |
| 27 | **翠绿萌芽** | Green专属 | 3费 | SPELL | 攻0/防0 `RAMP_1` | **53.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.9%</span> | 0 张 (暂不推荐) | 三费只跳一费，节奏亏损明显，翠绿已有更低费跳费，卡位不采纳。 |
| 28 | **森林之狼** | Green专属 | 4费 | MINION | DP:4 | **53.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -3.0%</span> | 0 张 (暂不推荐) | 四费白板攻防平平，缺乏突袭与跳费联动，难进竞速构筑。 |
| 29 | **翠绿守护** | Green专属 | 3费 | SPELL | 攻0/防5 | **52.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.1%</span> | 0 张 (暂不推荐) | 三费五防纯防守，不跳费不返场，进攻体系里拖节奏。 |
| 30 | **酒馆密账** | 中立通用 | 4费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **52.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.1%</span> | 0 张 (暂不推荐) | 四费抽二弃一，弃牌风险不合跳费，资源计划不依赖此牌。 |
| 31 | **古木护林官** | Green专属 | 5费 | MINION | DP:5 `SUPPORT_ATK_1,DEATH_DRAW_1` | **52.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 五费辅助加攻且亡语过牌，回报偏慢，跳费体系不需中期杂项。 |
| 32 | **翠绿滋养** | Green专属 | 3费 | SPELL | 攻0/防0 `RAMP_1,DRAW_1` | **52.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 三费跳费过一张，效率不差但卡位紧，已有更优跳费核心。 |
| 33 | **翠绿复苏使** | Green专属 | 4费 | MINION | DP:4 `DEATH_MANA_2,FORTIFY_1` | **51.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 四费亡语返费但身材普通，延迟收益不适合抢跳节奏。 |
| 34 | **蓝绿潮汐织法者** | Blue/Green双色 | 5费 | MINION | DP:4 `SUPPORT_ATK_1,RAMP_1` | **51.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.3%</span> | 0 张 (暂不推荐) | 五费跳费带辅助，身材偏弱且延迟，三费跳费更高效。 |
| 35 | **雇佣兵** | 中立通用 | 3费 | MINION | DP:5 | **51.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.3%</span> | 0 张 (暂不推荐) | 三费五攻纯站场，无突袭跳费联动，易被解后丢节奏。 |
| 36 | **林地衍生兽** | Green专属 | 4费 | MINION | DP:3 `SPAWN_1_1` | **50.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0 张 (暂不推荐) | 四费铺两个小身板，扩张速度慢，无法替代跳费或突袭位。 |
| 37 | **芽苗祭司** | Green专属 | 4费 | MINION | DP:3 `RAMP_1` | **50.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 四费才跳一费且身材差，慢于三费跳费使，不推荐投入。 |
| 38 | **蓝绿潮汐者** | Blue/Green双色 | 5费 | MINION | DP:5 `DEATH_MANA_2,FORTIFY_2` | **50.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 五费防御亡语返费，面板扎实但节奏延迟，双色卡位紧张。 |
| 39 | **红绿共生体** | Red/Green双色 | 6费 | MINION | DP:6 `FORTIFY_2,BONUS_SCORE_1` | **49.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 6费DP 6随从，附带【FORTIFY_2,BONUS_SCORE_1】，承担Green阵营核心战术组件。 |
| 40 | **萌芽巨兽** | Green专属 | 7费 | MINION | DP:7 `FORTIFY_3` | **49.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.6%</span> | 0 张 (暂不推荐) | 七费高防御但缺突袭，终结回合太慢，九费核弹更直接。 |
| 41 | **古树智者** | Green专属 | 6费 | MINION | DP:6 `DEATH_MANA_2` | **49.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.6%</span> | 0 张 (暂不推荐) | 六费亡语返两费太慢，站场无突袭，无法解决中期压力。 |
| 42 | **蓝绿潮涌使** | Blue/Green双色 | 5费 | MINION | DP:5 `RAMP_1,DEATH_DRAW_1` | **49.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.7%</span> | 0 张 (暂不推荐) | 五费跳费亡语过牌，身材一般且慢，已有三费跳费使替代。 |
| 43 | **翡翠巨熊** | Green专属 | 6费 | MINION | DP:6 `FORTIFY_2` | **48.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.7%</span> | 0 张 (暂不推荐) | 六费防御厚但无突袭，不能即时返场，中速位被功能卡挤掉。 |
| 44 | **红绿古树桥梁** | Red/Green双色 | 6费 | MINION | DP:6 `FORTIFY_2,DEATH_DRAW_1` | **48.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 六费防御亡语过牌，稳定但不提速，无法帮助高费核弹提前。 |
| 45 | **红绿共生体** | Red/Green双色 | 6费 | MINION | DP:6 `SUPPORT_ATK_2,FORTIFY_2` | **48.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 6费DP 6随从，附带【SUPPORT_ATK_2,FORTIFY_2】，承担Green阵营核心战术组件。 |
| 46 | **远古巨树** | Green专属 | 7费 | MINION | DP:8 `FORTIFY_2` | **48.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 七费八攻虽厚，但无突袭终结慢，九费核弹面前定位重叠。 |

---

## ⚪ 四、 【中立 (Neutral) 卡牌】全阵营适配性与战术表现分析
> **机制说明**：同一张中立卡在快攻、控制、跳费等不同战术体系下具有截然不同的战术价值与契合度。

| 中立卡名称 | 费用 | 类型 | 🔴 赤红卡组评分 | 🔵 蔚蓝卡组评分 | 🟢 翠绿卡组评分 | 最佳契合卡组 | 跨阵营战术定位 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **佣兵斥候** | 2费 | MINION | **98.0** | **90.0** | **90.0** | **赤红 (快攻)** | 多体系通用的高质量拼图 |
| **商人** | 2费 | MINION | **91.3** | **86.3** | **58.1** | **赤红 (快攻)** | 偏向赤红 (快攻)体系的针对性组件 |
| **训练假人** | 1费 | MINION | **57.1** | **89.2** | **96.4** | **翠绿 (跳费)** | 偏向翠绿 (跳费)体系的针对性组件 |
| **拾荒盾卫** | 3费 | MINION | **52.8** | **53.3** | **56.0** | **翠绿 (跳费)** | 特定战局下的可选备编卡 |
| **雇佣兵** | 3费 | MINION | **52.0** | **52.0** | **51.2** | **赤红 (快攻)** | 特定战局下的可选备编卡 |
| **酒馆密账** | 4费 | SPELL | **51.7** | **51.5** | **52.5** | **翠绿 (跳费)** | 特定战局下的可选备编卡 |
