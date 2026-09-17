# 🏆 TCG 卡牌战力评级与构筑指南（阵营分色专榜）

> **构筑规则说明**：在 TCG 标准规则中，各阵营（赤红/蔚蓝/翠绿）由**阵营专属卡 + 双色协同卡 + 中立通用卡**构筑。本指南基于 PPO 强化学习智能体在对战环境中的实战数据，综合**【卡组携带率】**与**【局势胜率贡献 (ΔWR)】**两大维度，按阵营分色独立建榜，提供客观、严谨的构筑参考与单卡解析。

---

## 🔴 一、 【赤红 (Red) 卡组】单卡战力与构筑指南
> **战术核心**：快攻压制 · 牺牲协同 · 节奏斩杀  
> **候选牌池**：27 张Red专属卡 + 6 张双色协同卡 + 6 张中立通用卡（共 39 张候选，择优遴选 30 张入套）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | 属性/数值 | **综合评分** | 梯队 | **携带率** | **胜率贡献 (ΔWR)** | 推荐配置 | 实战构筑解析 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **佣兵斥候** | 中立通用 | 2费 | MINION | DP:1 `RUSH` | **98.0** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.2%</span> | 3 张 (核心满编) | 2费突袭，解小怪并抢回先手，满编强化前期节奏。 |
| 2 | **赤红破阵兵** | Red专属 | 5费 | MINION | DP:3 `RUSH` | **97.4** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.1%</span> | 3 张 (核心满编) | 5费突袭，及时解场并保留压力，满编稳住中期先手。 |
| 3 | **赤红中期突破手** | Red专属 | 8费 | MINION | DP:5 `RUSH,BONUS_SCORE_1` | **97.1** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.0%</span> | 3 张 (核心满编) | 8费突袭带得分，终结与返场兼备，满编抢血上限。 |
| 4 | **破阵先锋** | Red专属 | 6费 | MINION | DP:4 `RUSH,BONUS_SCORE_1` | **96.7** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.0%</span> | 3 张 (核心满编) | 6费突袭得分，中期强攻核心，满编保证上手与斩杀线。 |
| 5 | **裂甲掷斧手** | Red专属 | 6费 | MINION | DP:1 `RUSH,DEGRADE_1` | **95.8** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.8%</span> | 3 张 (核心满编) | 6费突袭降阶，解场兼抢血，满编保证中期返场与压制。 |
| 6 | **赤红新兵** | Red专属 | 1费 | MINION | DP:1 | **94.5** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.6%</span> | 3 张 (核心满编) | 1费铺场起点，吸收火力并给牺牲协同，满编保前期密度。 |
| 7 | **赤红突击手** | Red专属 | 3费 | MINION | DP:1 `DEATH_DRAW_1` | **94.2** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.6%</span> | 3 张 (核心满编) | 3费死亡抽牌，交换后不亏手，快攻续航基石，满编抢节奏。 |
| 8 | **商人** | 中立通用 | 2费 | MINION | DP:2 `DRAW_1` | **93.9** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.5%</span> | 3 张 (核心满编) | 2费抽牌站场，润滑手牌并延续攻势，满编保证资源。 |
| 9 | **红绿共生** | Red/Green双色 | 5费 | MINION | DP:5 `RUSH,DEATH_DRAW_1` | **89.7** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 2 张 (主力配置) | 5费突袭死亡抽，换牌抢血兼备，主力补中期续航。 |
| 10 | **集结号手** | Red专属 | 2费 | MINION | DP:1 `SPAWN_1_1,ATTACK_ONLY` | **86.8** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.4%</span> | 2 张 (主力配置) | 2费铺两个攻击点，配合牺牲与突袭，主力配置够用。 |
| 11 | **赤红掠袭者** | Red专属 | 9费 | MINION | DP:1 `RUSH,BONUS_SCORE_1` | **78.4** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +1.0%</span> | 1 张 (按需携带) | 9费突袭得分，作为高费补伤挂件，仅特定对局需要。 |
| 12 | **赤红突破手** | Red专属 | 8费 | MINION | DP:1 `RUSH,BONUS_SCORE_1` | **78.1** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +1.0%</span> | 1 张 (按需携带) | 8费突袭得分，费用笨重，按需补一张后期斩杀。 |
| 13 | **训练假人** | 中立通用 | 1费 | MINION | DP:1 | **57.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.3%</span> | 0 张 (暂不推荐) | 1费白板，无压力无收益，前期铺场选择里最差。 |
| 14 | **拾荒盾卫** | 中立通用 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **57.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.4%</span> | 0 张 (暂不推荐) | 3费加固死亡抽，偏防守续航，抢血体系不取。 |
| 15 | **赤蓝战术交换** | Red/Blue双色 | 2费 | SPELL | 攻3/防0 `DRAW_1` | **55.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.6%</span> | 0 张 (暂不推荐) | 2费打3抽1，效率可但卡位紧，直伤有更优解。 |
| 16 | **献祭之焰** | Red专属 | 3费 | SPELL | 攻0/防0 `SACRIFICE_1_KILL_1,DISCARD_1` | **55.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.7%</span> | 0 张 (暂不推荐) | 3费牺牲解场还弃牌，双重亏牌，快攻绝不能带。 |
| 17 | **赤红献祭** | Red专属 | 2费 | SPELL | 攻0/防0 `SACRIFICE_1_KILL_1` | **54.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.7%</span> | 0 张 (暂不推荐) | 2费牺牲解场，亏损站场，抢血计划中难以接受。 |
| 18 | **血祭爆燃** | Red专属 | 3费 | SPELL | 攻3/防0 `SACRIFICE_1_KILL_1,ATTACK_ONLY` | **54.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.8%</span> | 0 张 (暂不推荐) | 3费牺牲加打3，解场成本过高，抢血体系不接纳。 |
| 19 | **射线** | Red专属 | 2费 | SPELL | 攻2/防0 | **54.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.8%</span> | 0 张 (暂不推荐) | 2费打2直伤，效率平庸，低费解场与抢血均有更优选择。 |
| 20 | **自爆** | Red专属 | 2费 | SPELL | 攻0/防0 `SACRIFICE_1_KILL_1,ATTACK_ONLY` | **54.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.9%</span> | 0 张 (暂不推荐) | 2费牺牲单位才能解场，快攻不愿丢场面，直伤节奏里格格不入。 |
| 21 | **献祭狂徒** | Red专属 | 4费 | MINION | DP:4 `SACRIFICE_1_KILL_1` | **53.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.9%</span> | 0 张 (暂不推荐) | 4费牺牲解场，普通身材，解场手段不如突袭灵活。 |
| 22 | **牺牲祭师** | Red专属 | 3费 | MINION | DP:3 `SACRIFICE_1_KILL_1,DEATH_DRAW_1` | **53.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -3.0%</span> | 0 张 (暂不推荐) | 3费牺牲解场带死亡抽，节奏偏慢，快攻无需二手解。 |
| 23 | **赤蓝交织者** | Red/Blue双色 | 4费 | MINION | DP:4 `SUPPORT_ATK_1` | **53.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -3.0%</span> | 0 张 (暂不推荐) | 同名重复，4费支援加攻仍偏慢，不进入快攻构筑。 |
| 24 | **雇佣兵** | 中立通用 | 3费 | MINION | DP:5 | **52.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.1%</span> | 0 张 (暂不推荐) | 3费5血白板，无突袭无协同，快攻宁愿要攻击频率。 |
| 25 | **大块头** | Red专属 | 6费 | MINION | DP:10 `DISCARD_2` | **52.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.1%</span> | 0 张 (暂不推荐) | 6费弃两张换10血，快攻手牌珍贵，弃牌代价压垮节奏。 |
| 26 | **赤红铁卫** | Red专属 | 3费 | MINION | DP:3 `FORTIFY_2` | **52.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 3费加固防守，思路偏控，快攻需要攻击频率而非护脸。 |
| 27 | **切割者** | Red专属 | 2费 | MINION | DP:2 `DEGRADE_1` | **51.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 2费降阶效果不稳定，身材普通，无法巩固前期先手。 |
| 28 | **牺牲角斗士** | Red专属 | 4费 | MINION | DP:4 `SACRIFICE_1_KILL_1` | **51.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.3%</span> | 0 张 (暂不推荐) | 4费牺牲解场，身材尚可但拖节奏，不如突袭直接。 |
| 29 | **红色小队长** | Red专属 | 3费 | MINION | DP:3 `SPAWN_1_1` | **51.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.3%</span> | 0 张 (暂不推荐) | 3费仅召唤1/1，身材与铺场效率均不及集结号手，卡位不选。 |
| 30 | **酒馆密账** | 中立通用 | 4费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **50.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0 张 (暂不推荐) | 4费抽二弃一，费用高且亏节奏，快攻不需过滤。 |
| 31 | **战地督军** | Red专属 | 5费 | MINION | DP:4 `SUPPORT_ATK_1` | **50.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0 张 (暂不推荐) | 5费支援加攻，不具突袭，铺场收益慢，卡位紧张。 |
| 32 | **余烬传令官** | Red专属 | 5费 | MINION | DP:5 `DEATH_DRAW_1` | **50.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 5费死亡抽牌，身材合格但攻速慢，快攻用低费过牌。 |
| 33 | **红绿共生体** | Red/Green双色 | 6费 | MINION | DP:6 `SUPPORT_ATK_2,FORTIFY_2` | **49.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 6费支援加固，防守与大身材思路，快攻不需。 |
| 34 | **赤蓝交织者** | Red/Blue双色 | 4费 | MINION | DP:4 `SUPPORT_ATK_1` | **49.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.6%</span> | 0 张 (暂不推荐) | 同名重复，4费支援加攻仍偏慢，不进入快攻构筑。 |
| 35 | **赤红战意** | Red专属 | 5费 | SPELL | 攻0/防0 `TEMP_MANA_2` | **49.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.6%</span> | 0 张 (暂不推荐) | 5费仅换临时法力，节奏空过，快攻无法承担。 |
| 36 | **赤红战盾** | Red专属 | 2费 | SPELL | 攻0/防4 | **48.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.7%</span> | 0 张 (暂不推荐) | 2费加4防，纯防守牌，快攻抢血时手牌效率低。 |
| 37 | **掠夺者** | Red专属 | 7费 | MINION | DP:3 `BONUS_SCORE_1` | **48.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.7%</span> | 0 张 (暂不推荐) | 7费仅3点场面且奖励分滞后，快攻不等高费终结。 |
| 38 | **破阵狂徒** | Red专属 | 8费 | MINION | DP:2 `DEGRADE_1` | **48.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 8费降阶且仅2点，费用过高，终结效率远逊突袭得分。 |
| 39 | **红绿古树桥梁** | Red/Green双色 | 6费 | MINION | DP:6 `FORTIFY_2,DEATH_DRAW_1` | **48.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 6费加固死亡抽，偏防守资源，快攻节奏不合。 |

---

## 🔵 二、 【蔚蓝 (Blue) 卡组】单卡战力与构筑指南
> **战术核心**：防守反击 · 固守护盾 · 资源消耗  
> **候选牌池**：27 张Blue专属卡 + 6 张双色协同卡 + 6 张中立通用卡（共 39 张候选，择优遴选 30 张入套）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | 属性/数值 | **综合评分** | 梯队 | **携带率** | **胜率贡献 (ΔWR)** | 推荐配置 | 实战构筑解析 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **佣兵斥候** | 中立通用 | 2费 | MINION | DP:1 `RUSH` | **90.0** | **S** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.9%</span> | 3 张 (核心满编) | 两费突袭一，满编抢节奏解小怪，前中期主动权的关键。 |
| 2 | **冰潮突袭者** | Blue专属 | 3费 | MINION | DP:3 `RUSH` | **89.7** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 2 张 (主力配置) | 三费突袭三可解小怪，两张补中坚节奏，攻防转换顺畅。 |
| 3 | **破甲潮汐兵** | Blue专属 | 2费 | MINION | DP:3 `RUSH,DEGRADE_1` | **89.4** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 2 张 (主力配置) | 两费突袭带破甲，两张抢回先手并削弱大怪，节奏关键。 |
| 4 | **盾兵** | Blue专属 | 1费 | MINION | DP:1 `FORTIFY_1` | **89.1** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.7%</span> | 2 张 (主力配置) | 一费固守挡一次伤害，两张保证起手有盾，不宜满编。 |
| 5 | **蔚蓝守壁** | Blue专属 | 2费 | MINION | DP:2 `FORTIFY_2` | **86.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.3%</span> | 2 张 (主力配置) | 两费固守二防守高效，两张稳定护脸，为高费大哥争取回合。 |
| 6 | **商人** | 中立通用 | 2费 | MINION | DP:2 `DRAW_1` | **85.9** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.2%</span> | 2 张 (主力配置) | 两费站场抽一，两张兼顾曲线与资源，防守体系润滑剂。 |
| 7 | **蔚蓝智慧** | Blue专属 | 2费 | SPELL | 攻0/防0 `DRAW_2,TEMP_MANA_1` | **85.3** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.1%</span> | 2 张 (主力配置) | 两费抽二并临时加费，两张关键过牌，助中期连续施压。 |
| 8 | **霜盾见习官** | Blue专属 | 2费 | MINION | DP:1 `FORTIFY_1,SUPPORT_ATK_1` | **84.9** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.1%</span> | 2 张 (主力配置) | 两费兼具固守与辅助，两张润滑曲线，别当主力肉盾。 |
| 9 | **雇佣兵** | 中立通用 | 3费 | MINION | DP:5 | **84.0** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.9%</span> | 2 张 (主力配置) | 三费五生命身材厚，两张扛前期交换，无词条但能护脸。 |
| 10 | **蓝绿潮汐织法者** | Blue/Green双色 | 5费 | MINION | DP:4 `SUPPORT_ATK_1,RAMP_1` | **82.4** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.7%</span> | 2 张 (主力配置) | 五费辅助攻击并跳费，两张加速大哥登场，但身材偏脆。 |
| 11 | **蔚蓝要塞** | Blue专属 | 5费 | MINION | DP:6 `FORTIFY_3,SUPPORT_ATK_1` | **80.2** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.3%</span> | 2 张 (主力配置) | 五费固守三带辅助，两张撑中期并反打，高质量支点。 |
| 12 | **蓝绿潮汐者** | Blue/Green双色 | 5费 | MINION | DP:5 `DEATH_MANA_2,FORTIFY_2` | **79.3** | **B** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.2%</span> | 2 张 (按需携带) | 五费固守二亡语回费，两张续航按环境选用，非必满。 |
| 13 | **赤蓝战术交换** | Red/Blue双色 | 2费 | SPELL | 攻3/防0 `DRAW_1` | **78.1** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +1.0%</span> | 1 张 (按需携带) | 两费三攻并抽一，灵活解小怪补资源，一张即可，不宜多。 |
| 14 | **冰封禁制** | Blue专属 | 2费 | SPELL | 攻6/防2 | **76.8** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +0.8%</span> | 1 张 (按需携带) | 两费六攻可解关键小怪，一张针对环境，不宜多带。 |
| 15 | **蔚蓝守卫** | Blue专属 | 2费 | MINION | DP:2 `FORTIFY_2` | **75.6** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +0.6%</span> | 1 张 (按需携带) | 两费固守二防守扎实，一张补曲线，多了挤终端卡位。 |
| 16 | **蔚蓝盾卫** | Blue专属 | 2费 | MINION | DP:2 `FORTIFY_1` | **73.4** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +0.2%</span> | 1 张 (按需携带) | 2费DP 2随从，附带【FORTIFY_1】，承担Blue阵营核心战术组件。 |
| 17 | **蔚蓝卫士** | Blue专属 | 2费 | MINION | DP:2 `FORTIFY_1` | **72.7** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +0.1%</span> | 1 张 (按需携带) | 二费固守随从，一张补足前期防守，后期易被替代。 |
| 18 | **潮汐学者** | Blue专属 | 4费 | MINION | DP:4 `DEATH_DRAW_1` | **70.8** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ -0.2%</span> | 1 张 (按需携带) | 四费亡语抽一，交换后不亏牌，一张润滑，多了节奏偏慢。 |
| 19 | **训练假人** | 中立通用 | 1费 | MINION | DP:1 | **58.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.1%</span> | 0 张 (暂不推荐) | 一费一生命无固守，挡刀效率太低，远不如带盾兵。 |
| 20 | **防御！** | Blue专属 | 1费 | SPELL | 攻0/防2 | **58.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.2%</span> | 0 张 (暂不推荐) | 单次加防不如站场随从，卡差亏节奏，故不投入。 |
| 21 | **寒冰解离** | Blue专属 | 3费 | SPELL | 攻4/防0 | **57.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.3%</span> | 0 张 (暂不推荐) | 三费四攻解场平庸，无法过牌护脸，卡位让给优质随从。 |
| 22 | **寒霜破甲** | Blue专属 | 2费 | SPELL | 攻3/防0 `DEGRADE_1` | **57.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.3%</span> | 0 张 (暂不推荐) | 两费三攻破甲解场尚可，但效率一般，同类随从更赚节奏。 |
| 23 | **蔚蓝冲击** | Blue专属 | 2费 | SPELL | 攻4/防0 | **57.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.4%</span> | 0 张 (暂不推荐) | 两费四攻直伤普通，防守反击更愿用随从交换，故舍。 |
| 24 | **弓箭手** | Blue专属 | 3费 | MINION | DP:3 `SUPPORT_ATK_1` | **56.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.5%</span> | 0 张 (暂不推荐) | 三费辅助攻击收益偏低，防守体系更需固守与过牌。 |
| 25 | **蓝绿潮涌** | Blue/Green双色 | 3费 | SPELL | 攻0/防0 `DRAW_2` | **54.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.8%</span> | 0 张 (暂不推荐) | 三费抽二无场面，防守卡组嫌慢，已有低费过牌替代。 |
| 26 | **拾荒盾卫** | 中立通用 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **54.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.8%</span> | 0 张 (暂不推荐) | 三费固守一亡语抽一，数值偏低，交换后节奏仍显拖沓。 |
| 27 | **蔚蓝破甲师** | Blue专属 | 3费 | MINION | DP:3 `DEGRADE_1` | **53.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.9%</span> | 0 张 (暂不推荐) | 三费破甲一威胁有限，防守体系不需慢速削甲，放弃。 |
| 28 | **深海守望者** | Blue专属 | 5费 | MINION | DP:5 `FORTIFY_3` | **53.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -3.0%</span> | 0 张 (暂不推荐) | 五费固守三仅算合格，缺辅助与即时影响，竞争不过要塞。 |
| 29 | **法力回流哨兵** | Blue专属 | 4费 | MINION | DP:4 `DEATH_MANA_2` | **52.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 四费亡语回费太慢，站场无固守，卡位应留给中期支点。 |
| 30 | **寒晶护壁** | Blue专属 | 2费 | SPELL | 攻0/防2 `DRAW_1` | **51.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 两费加防并过牌，看似润滑，实际让节奏且护脸不足。 |
| 31 | **藤甲兵** | Blue专属 | 4费 | MINION | DP:4 `FORTIFY_2` | **51.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.3%</span> | 0 张 (暂不推荐) | 四费固守二但身材普通，同类低费卡更易抢先站场。 |
| 32 | **壁垒工匠** | Blue专属 | 3费 | MINION | DP:3 `FORTIFY_2` | **51.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.3%</span> | 0 张 (暂不推荐) | 三费固守二数值合格，但卡位紧，低费盾兵更灵活。 |
| 33 | **火铳手** | Blue专属 | 3费 | MINION | DP:4 `SUPPORT_ATK_2` | **50.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0 张 (暂不推荐) | 辅助攻击二却无防守词条，站场后难保住，节奏偏慢。 |
| 34 | **赤蓝交织者** | Red/Blue双色 | 4费 | MINION | DP:4 `SUPPORT_ATK_1` | **49.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 四费辅助攻击一，身材平庸且无防守词条，卡位不优先。 |
| 35 | **蔚蓝盾卫** | Blue专属 | 4费 | MINION | DP:6 `FORTIFY_3,SUPPORT_ATK_1` | **49.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.6%</span> | 0 张 (暂不推荐) | 4费DP 6随从，附带【FORTIFY_3,SUPPORT_ATK_1】，承担Blue阵营核心战术组件。 |
| 36 | **酒馆密账** | 中立通用 | 4费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **48.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.7%</span> | 0 张 (暂不推荐) | 四费抽二弃一太慢，弃牌风险高，防守卡组不需此过牌。 |
| 37 | **赤蓝交织者** | Red/Blue双色 | 4费 | MINION | DP:4 `SUPPORT_ATK_1` | **48.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.7%</span> | 0 张 (暂不推荐) | 四费辅助攻击一，身材平庸且无防守词条，卡位不优先。 |
| 38 | **石像鬼** | Blue专属 | 6费 | MINION | DP:8 | **48.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 六费仅高生命无词条，太笨重，不如高质量终端。 |
| 39 | **蔚蓝护盾反击** | Blue专属 | 3费 | SPELL | 攻0/防4 `FORTIFY_1` | **48.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 三费加防四并固守一，偏防守却亏卡，不如直接下盾卫。 |

---

## 🟢 三、 【翠绿 (Green) 卡组】单卡战力与构筑指南
> **战术核心**：法力跳费 · 质量成长 · 终结大哥  
> **候选牌池**：27 张Green专属卡 + 6 张双色协同卡 + 6 张中立通用卡（共 39 张候选，择优遴选 30 张入套）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | 属性/数值 | **综合评分** | 梯队 | **携带率** | **胜率贡献 (ΔWR)** | 推荐配置 | 实战构筑解析 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **佣兵斥候** | 中立通用 | 2费 | MINION | DP:1 `RUSH` | **98.0** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.2%</span> | 3 张 (核心满编) | 两费突袭，解小怪保跳费，抢回先手，低费节奏核心满编。 |
| 2 | **藤蔓突袭者** | Green专属 | 4费 | MINION | DP:4 `RUSH` | **97.7** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.1%</span> | 3 张 (核心满编) | 四费四攻突袭，返场解场夺先手，中期节奏核心，满编。 |
| 3 | **灭世翡翠巨龙** | Green专属 | 10费 | MINION | DP:4 `RUSH,BONUS_SCORE_1` | **97.4** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.1%</span> | 3 张 (核心满编) | 十费突袭终结，奖励得分拉满，跳费核弹核心，后期一锤定音。 |
| 4 | **红绿共生** | Red/Green双色 | 5费 | MINION | DP:5 `RUSH,DEATH_DRAW_1` | **97.1** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.0%</span> | 3 张 (核心满编) | 五费突袭亡语过牌，解场补手一体，中期核心，满编。 |
| 5 | **剧毒花** | Green专属 | 2费 | MINION | DP:2 `DEGRADE_2` | **93.3** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.4%</span> | 3 张 (核心满编) | 两费降攻二，压制快攻生物，低费防守兼软解，前期满编。 |
| 6 | **翡翠幼龙** | Green专属 | 7费 | MINION | DP:4 `RUSH,DEATH_MANA_1` | **88.7** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.7%</span> | 2 张 (主力配置) | 七费突袭，死亡回一费，解场后仍可铺垫，但身材偏脆。 |
| 7 | **翠绿幼苗** | Green专属 | 1费 | MINION | DP:1 `FORTIFY_1` | **88.1** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.6%</span> | 2 张 (主力配置) | 一费加固站场，吸收早期伤害，保护跳费生物，开局润滑。 |
| 8 | **训练假人** | 中立通用 | 1费 | MINION | DP:1 | **87.8** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.5%</span> | 2 张 (主力配置) | 一费一攻挡刀，吸收早期伤害，保护跳费，低费防守主力。 |
| 9 | **商人** | 中立通用 | 2费 | MINION | DP:2 `DRAW_1` | **86.8** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.4%</span> | 2 张 (主力配置) | 两费站场过牌，润滑手牌找跳费与大哥，前期节奏主力。 |
| 10 | **树人** | Green专属 | 3费 | MINION | DP:2 `FORTIFY_1` | **86.2** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.3%</span> | 2 张 (主力配置) | 三费二攻带加固，前期挡快攻，为跳费与大哥争取回合。 |
| 11 | **翠绿哨兵** | Green专属 | 3费 | MINION | DP:2 `FORTIFY_2,DEATH_DRAW_1` | **84.6** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.0%</span> | 2 张 (主力配置) | 三费加固亡语过牌，防守不亏手，适合拖入中后期。 |
| 12 | **芽苗祭司** | Green专属 | 4费 | MINION | DP:3 `RAMP_1` | **81.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.5%</span> | 2 张 (主力配置) | 四费三攻跳费，站场同时扩张法力，衔接中期大哥。 |
| 13 | **翠绿资源滋长** | Green专属 | 4费 | SPELL | 攻0/防0 `RAMP_1,DRAW_1` | **73.7** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +0.3%</span> | 1 张 (按需携带) | 四费跳费过牌，提速补手，可按环境作一张润滑，不宜多带。 |
| 14 | **翠绿驻防藤蔓** | Green专属 | 2费 | MINION | DP:2 `FORTIFY_1` | **58.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.2%</span> | 0 张 (暂不推荐) | 两费加固二攻，防守尚可，但同类低费更需过牌或跳费。 |
| 15 | **翠绿守林人** | Green专属 | 3费 | MINION | DP:3 `FORTIFY_2` | **57.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.3%</span> | 0 张 (暂不推荐) | 三费三攻加固二，站场稳固，但缺乏过牌跳费，难进核心。 |
| 16 | **荆棘缠绕** | Green专属 | 2费 | SPELL | 攻2/防0 | **57.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.4%</span> | 0 张 (暂不推荐) | 两费打二，伤效有限，无法跳费或过牌，环境内解场不足。 |
| 17 | **雇佣兵** | 中立通用 | 3费 | MINION | DP:5 | **56.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.5%</span> | 0 张 (暂不推荐) | 三费五攻身材扎实，但无词条，中速交换不如带突袭或跳费。 |
| 18 | **萌芽跳费使** | Green专属 | 3费 | MINION | DP:3 `RAMP_1` | **55.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.6%</span> | 0 张 (暂不推荐) | 三费三攻跳费，站场提速，但四费芽苗祭司更稳，卡位不足。 |
| 19 | **狂暴生长** | Green专属 | 3费 | SPELL | 攻0/防3 | **55.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.6%</span> | 0 张 (暂不推荐) | 三费仅加三防，无跳费过牌，防守效率低，不契合核弹节奏。 |
| 20 | **蓝绿潮涌** | Blue/Green双色 | 3费 | SPELL | 攻0/防0 `DRAW_2` | **54.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.7%</span> | 0 张 (暂不推荐) | 三费过二，手牌补充尚可，但无跳费站场，前期节奏偏慢。 |
| 21 | **翠绿跳费者** | Green专属 | 4费 | MINION | DP:3 `RAMP_1` | **54.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.8%</span> | 0 张 (暂不推荐) | 四费三攻跳费，节奏与芽苗祭司重叠，身材偏弱故不选用。 |
| 22 | **拾荒盾卫** | 中立通用 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **54.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.9%</span> | 0 张 (暂不推荐) | 三费加固亡语过牌，但身材一般，三费位被哨兵等竞争。 |
| 23 | **古树庇护** | Green专属 | 3费 | SPELL | 攻0/防5 `FORTIFY_1` | **53.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -3.0%</span> | 0 张 (暂不推荐) | 三费五防加固，防守厚但无提速，三费跳费节奏更优先。 |
| 24 | **自然生长** | Green专属 | 3费 | SPELL | 攻0/防0 `RAMP_1` | **53.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -3.0%</span> | 0 张 (暂不推荐) | 三费纯跳费，当回合无场面，易被快攻惩罚，构筑不取。 |
| 25 | **翠绿守护** | Green专属 | 3费 | SPELL | 攻0/防5 | **52.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.1%</span> | 0 张 (暂不推荐) | 三费五防，纯防守无跳费，面对直伤与铺场均显被动。 |
| 26 | **森林之狼** | Green专属 | 4费 | MINION | DP:4 | **52.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.1%</span> | 0 张 (暂不推荐) | 四费四攻白板，无突袭跳费，争夺场面平庸，被同类取代。 |
| 27 | **翠绿萌芽** | Green专属 | 3费 | SPELL | 攻0/防0 `RAMP_1` | **52.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 三费纯跳费，当回合无场面，快攻压力下易崩，卡位不足。 |
| 28 | **翠绿滋养** | Green专属 | 3费 | SPELL | 攻0/防0 `RAMP_1,DRAW_1` | **51.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 三费跳费过一，效率尚可，但挤占防守卡位，当前不取。 |
| 29 | **古树智者** | Green专属 | 6费 | MINION | DP:6 `DEATH_MANA_2` | **51.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.3%</span> | 0 张 (暂不推荐) | 六费死亡回二费，亡语滞后，六费更需要即时影响场面。 |
| 30 | **酒馆密账** | 中立通用 | 4费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **50.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0 张 (暂不推荐) | 四费过二弃一，手牌质量受损，节奏偏慢，不符合跳费计划。 |
| 31 | **翡翠巨熊** | Green专属 | 6费 | MINION | DP:6 `FORTIFY_2` | **50.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0 张 (暂不推荐) | 六费六攻加固二，质量尚可，但无突袭，六费位更需跳费或解场。 |
| 32 | **蓝绿潮汐织法者** | Blue/Green双色 | 5费 | MINION | DP:4 `SUPPORT_ATK_1,RAMP_1` | **50.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 五费跳费带光环，身材偏弱，提速太晚，难与四费跳费竞争。 |
| 33 | **林地衍生兽** | Green专属 | 4费 | MINION | DP:3 `SPAWN_1_1` | **49.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 四费铺两个小生物，横向展开尚可，但无跳费突袭，节奏偏慢。 |
| 34 | **红绿共生体** | Red/Green双色 | 6费 | MINION | DP:6 `SUPPORT_ATK_2,FORTIFY_2` | **49.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.6%</span> | 0 张 (暂不推荐) | 六费光环加固，团战强但无突袭，六费回合应直接威胁。 |
| 35 | **红绿古树桥梁** | Red/Green双色 | 6费 | MINION | DP:6 `FORTIFY_2,DEATH_DRAW_1` | **49.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.6%</span> | 0 张 (暂不推荐) | 六费加固亡语过牌，价值滞后，六费位更需终结或即时解场。 |
| 36 | **蓝绿潮汐者** | Blue/Green双色 | 5费 | MINION | DP:5 `DEATH_MANA_2,FORTIFY_2` | **48.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.7%</span> | 0 张 (暂不推荐) | 五费死亡回二费加固，亡语慢，五费更需即时跳费或压制。 |
| 37 | **远古巨树** | Green专属 | 7费 | MINION | DP:8 `FORTIFY_2` | **48.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.7%</span> | 0 张 (暂不推荐) | 七费八攻加固二，防守够厚但缺突袭，终端回合太慢。 |
| 38 | **萌芽巨兽** | Green专属 | 7费 | MINION | DP:7 `FORTIFY_3` | **48.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 七费七攻加固三，防守强但无突袭，终结速度不符合核弹思路。 |
| 39 | **世界树恩泽** | Green专属 | 5费 | SPELL | 攻0/防3 `RAMP_1` | **48.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 五费加防跳费，防守与提速混杂，节奏拖沓，不如直接跳费。 |

---

## ⚪ 四、 【中立 (Neutral) 卡牌】全阵营适配性与战术表现分析
> **机制说明**：同一张中立卡在快攻、控制、跳费等不同战术体系下具有截然不同的战术价值与契合度。

| 中立卡名称 | 费用 | 类型 | 🔴 赤红卡组评分 | 🔵 蔚蓝卡组评分 | 🟢 翠绿卡组评分 | 最佳契合卡组 | 跨阵营战术定位 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **佣兵斥候** | 2费 | MINION | **98.0** | **90.0** | **98.0** | **赤红 (快攻)** | 多体系通用的高质量拼图 |
| **商人** | 2费 | MINION | **93.9** | **85.9** | **86.8** | **赤红 (快攻)** | 多体系通用的高质量拼图 |
| **训练假人** | 1费 | MINION | **57.5** | **58.7** | **87.8** | **翠绿 (跳费)** | 偏向翠绿 (跳费)体系的针对性组件 |
| **拾荒盾卫** | 3费 | MINION | **57.2** | **54.3** | **54.0** | **赤红 (快攻)** | 特定战局下的可选备编卡 |
| **雇佣兵** | 3费 | MINION | **52.7** | **84.0** | **56.5** | **蔚蓝 (防守)** | 偏向蔚蓝 (防守)体系的针对性组件 |
| **酒馆密账** | 4费 | SPELL | **50.8** | **48.9** | **50.8** | **赤红 (快攻)** | 特定战局下的可选备编卡 |
