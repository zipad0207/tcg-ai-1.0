# 🏆 TCG 卡牌战力评级与构筑指南（阵营分色专榜）

> **构筑规则说明**：在 TCG 标准规则中，各阵营（赤红/蔚蓝/翠绿）由**阵营专属卡 + 双色协同卡 + 中立通用卡**构筑。本指南基于 PPO 强化学习智能体在对战环境中的实战数据，综合**【卡组携带率】**与**【局势胜率贡献 (ΔWR)】**两大维度，按阵营分色独立建榜，提供客观、严谨的构筑参考与单卡解析。

---

## 🔴 一、 【赤红 (Red) 卡组】单卡战力与构筑指南
> **战术核心**：快攻压制 · 牺牲协同 · 节奏斩杀  
> **候选牌池**：32 张Red专属卡 + 8 张双色协同卡 + 6 张中立通用卡（共 46 张候选，择优遴选 30 张入套）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | 属性/数值 | **综合评分** | 梯队 | **携带率** | **胜率贡献 (ΔWR)** | 推荐配置 | 实战构筑解析 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **赤焰哨卫** | Red专属 | 4费 | MINION | DP:3 `RUSH,FORTIFY_1` | **97.7** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.1%</span> | 3 张 (核心满编) | 四费突袭加固一，解场后站住护脸，满编衔接中期节奏。 |
| 2 | **熔岩破阵者** | Red专属 | 6费 | MINION | DP:4 `RUSH,BONUS_SCORE_1` | **96.4** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.9%</span> | 3 张 (核心满编) | 六费突袭加分，返场与推进兼备，满编保证中期压力。 |
| 3 | **赤红掠袭者** | Red专属 | 9费 | MINION | DP:1 `RUSH,BONUS_SCORE_1` | **95.9** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.8%</span> | 3 张 (核心满编) | 九费突袭加分，斩杀终端；赤红需稳定高费收尾，满编保证后期抽到。 |
| 4 | **赤红突破手** | Red专属 | 9费 | MINION | DP:1 `RUSH,BONUS_SCORE_1` | **95.6** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.8%</span> | 3 张 (核心满编) | 九费突袭加分，第二张终结终端，满编提高斩杀组合稳定性。 |
| 5 | **商人** | 中立通用 | 2费 | MINION | DP:2 `DRAW_1` | **95.3** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.7%</span> | 3 张 (核心满编) | 二费二攻过一，补资源不亏节奏，满编润滑手牌与曲线。 |
| 6 | **红色小队长** | Red专属 | 3费 | MINION | DP:3 `SPAWN_1_1` | **94.8** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.6%</span> | 3 张 (核心满编) | 三费铺两个攻击频率，牺牲协同与抢血核心，满编保证前期展开。 |
| 7 | **集结号手** | Red专属 | 2费 | MINION | DP:1 `SPAWN_1_1,ATTACK_ONLY` | **92.9** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.3%</span> | 3 张 (核心满编) | 二费生两个频率，攻击限定契合牺牲与抢血，满编撑起前期。 |
| 8 | **佣兵斥候** | 中立通用 | 2费 | MINION | DP:1 `RUSH` | **90.0** | **S** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.9%</span> | 3 张 (核心满编) | 二费突袭，前期解场抢节奏，满编提高先手与交换能力。 |
| 9 | **赤红破阵兵** | Red专属 | 5费 | MINION | DP:3 `RUSH` | **89.2** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 2 张 (主力配置) | 五费突袭三攻，返场解小怪尚可，两张补中程先手。 |
| 10 | **赤红新兵** | Red专属 | 1费 | MINION | DP:2 | **86.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.3%</span> | 2 张 (主力配置) | 一费二攻，前期站场与抢血都合格，两张补足低费曲线。 |
| 11 | **赤红献祭** | Red专属 | 2费 | SPELL | 攻0/防0 `SACRIFICE_1_KILL_1` | **81.2** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.5%</span> | 2 张 (主力配置) | 两费牺牲杀一，灵活解场工具，两张应对关键威胁且不卡手。 |
| 12 | **赤红中期突破手** | Red专属 | 8费 | MINION | DP:5 `RUSH,BONUS_SCORE_1` | **78.1** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +1.0%</span> | 1 张 (按需携带) | 八费突袭加分，单张作为中后期终结，避免高费上手卡滞。 |
| 13 | **红绿共生** | Red/Green双色 | 5费 | MINION | DP:5 `RUSH,DEATH_DRAW_1` | **59.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.0%</span> | 0 张 (暂不推荐) | 五费突袭亡语过牌，返场过牌兼顾，但五费争场不如直接得分。 |
| 14 | **破阵先锋** | Red专属 | 6费 | MINION | DP:4 `RUSH,BONUS_SCORE_1` | **58.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.1%</span> | 0 张 (暂不推荐) | 六费突袭加分四攻，能返场抢分，但同类六费竞争激烈，优先级低。 |
| 15 | **裂甲掷斧手** | Red专属 | 6费 | MINION | DP:1 `RUSH,DEGRADE_1` | **58.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.1%</span> | 0 张 (暂不推荐) | 六费突袭仅一攻，削弱一收益低，解不掉关键威胁。 |
| 16 | **雇佣兵** | 中立通用 | 3费 | MINION | DP:5 | **57.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.4%</span> | 0 张 (暂不推荐) | 三费五攻白板，数值扎实但无协同，快攻更需词条频率。 |
| 17 | **训练假人** | 中立通用 | 1费 | MINION | DP:1 | **56.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.5%</span> | 0 张 (暂不推荐) | 一费一防白板，仅能拖延，无攻击压力且妨碍抽到有效卡。 |
| 18 | **切割者** | Red专属 | 2费 | MINION | DP:2 `DEGRADE_1` | **56.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.6%</span> | 0 张 (暂不推荐) | 二费二攻削弱一，节奏平庸，既不能高效解场也难持续抢血。 |
| 19 | **拾荒盾卫** | 中立通用 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **55.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.6%</span> | 0 张 (暂不推荐) | 三费加固一亡语过牌，防守与延迟过牌，速度太慢。 |
| 20 | **赤红铁卫** | Red专属 | 3费 | MINION | DP:3 `FORTIFY_2` | **55.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.6%</span> | 0 张 (暂不推荐) | 三费三防加固二，防守向卡拖慢抢血，快攻内战也不如直接铺场。 |
| 21 | **赤红战盾** | Red专属 | 2费 | SPELL | 攻0/防4 | **55.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.7%</span> | 0 张 (暂不推荐) | 两费加四防，纯防守法术与快攻计划相悖，浪费进攻卡位。 |
| 22 | **赤蓝交织者** | Red/Blue双色 | 4费 | MINION | DP:4 `SUPPORT_ATK_1` | **54.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.8%</span> | 0 张 (暂不推荐) | 4费DP 4随从，附带【SUPPORT_ATK_1】，承担Red阵营核心战术组件。 |
| 23 | **赤蓝交织者** | Red/Blue双色 | 4费 | MINION | DP:4 `SUPPORT_ATK_1,DEATH_MANA_1` | **54.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.8%</span> | 0 张 (暂不推荐) | 4费DP 4随从，附带【SUPPORT_ATK_1,DEATH_MANA_1】，承担Red阵营核心战术组件。 |
| 24 | **赤红突击手** | Red专属 | 3费 | MINION | DP:1 `DEATH_DRAW_1` | **54.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.9%</span> | 0 张 (暂不推荐) | 三费仅一攻，亡语过牌太慢，快攻争场不需要迟到的资源。 |
| 25 | **大块头** | Red专属 | 6费 | MINION | DP:10 `DISCARD_2` | **53.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.9%</span> | 0 张 (暂不推荐) | 六费十防却要弃两张，防守端与快攻抢血思路相悖且亏牌。 |
| 26 | **牺牲角斗士** | Red专属 | 4费 | MINION | DP:4 `SACRIFICE_1_KILL_1` | **53.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.9%</span> | 0 张 (暂不推荐) | 四费四攻，牺牲杀一但费用卡手，与低费牺牲收益冲突。 |
| 27 | **酒馆密账** | 中立通用 | 4费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **53.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -3.0%</span> | 0 张 (暂不推荐) | 四费抽二弃一，过滤手牌但亏节奏，快攻不愿空过四费。 |
| 28 | **余烬反击官** | Red专属 | 5费 | MINION | DP:5 `SUPPORT_ATK_1` | **53.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -3.0%</span> | 0 张 (暂不推荐) | 五费五攻光环加攻，需铺场支持，单独登场威胁不足。 |
| 29 | **红绿共生体** | Red/Green双色 | 6费 | MINION | DP:6 `FORTIFY_2,BONUS_SCORE_1` | **52.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.1%</span> | 0 张 (暂不推荐) | 6费DP 6随从，附带【FORTIFY_2,BONUS_SCORE_1】，承担Red阵营核心战术组件。 |
| 30 | **献祭狂徒** | Red专属 | 4费 | MINION | DP:4 `SACRIFICE_1_KILL_1` | **52.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.1%</span> | 0 张 (暂不推荐) | 四费四攻牺牲杀一，交换效率一般，费用高不如低费解。 |
| 31 | **战地督军** | Red专属 | 5费 | MINION | DP:4 `SUPPORT_ATK_1` | **52.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 五费四攻光环加攻，需要场面配合，赤红更愿直接输出。 |
| 32 | **赤蓝战术交换** | Red/Blue双色 | 2费 | SPELL | 攻3/防0 `DRAW_1` | **52.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 两费打三抽一，效率不错，但快攻低费法术位更需铺场或直伤。 |
| 33 | **烈焰清算** | Red专属 | 4费 | SPELL | 攻4/防0 `DEGRADE_1` | **51.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 四费打四削弱一，去除效率尚可，但法术卡位紧，铺场优先。 |
| 34 | **献祭之焰** | Red专属 | 3费 | SPELL | 攻0/防0 `SACRIFICE_1_KILL_1,DISCARD_1` | **51.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.3%</span> | 0 张 (暂不推荐) | 牺牲杀一还弃一，双资源消耗，解场代价过重，不宜携带。 |
| 35 | **血祭爆燃** | Red专属 | 3费 | SPELL | 攻3/防0 `SACRIFICE_1_KILL_1,ATTACK_ONLY` | **50.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0 张 (暂不推荐) | 三费牺牲打三还限攻击，资源换伤害效率差，卡手风险高。 |
| 36 | **射线** | Red专属 | 2费 | SPELL | 攻3/防0 | **50.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0 张 (暂不推荐) | 两费打三直伤，效率尚可，但赤红低费曲线拥挤，卡位让给铺场。 |
| 37 | **自爆** | Red专属 | 2费 | SPELL | 攻0/防0 `SACRIFICE_1_KILL_1,ATTACK_ONLY` | **50.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 牺牲一换一但仅能攻击，限制过多，快攻浪费场面换解不划算。 |
| 38 | **赤蓝交织者** | Red/Blue双色 | 4费 | MINION | DP:4 `SUPPORT_ATK_1` | **50.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 4费DP 4随从，附带【SUPPORT_ATK_1】，承担Red阵营核心战术组件。 |
| 39 | **牺牲祭师** | Red专属 | 4费 | MINION | DP:3 `SACRIFICE_1_KILL_1,DEATH_DRAW_1` | **49.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 四费三攻，牺牲解场后亡语过牌，但费用偏重，节奏延后。 |
| 40 | **余烬传令官** | Red专属 | 5费 | MINION | DP:5 `DEATH_DRAW_1` | **49.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.6%</span> | 0 张 (暂不推荐) | 五费五攻亡语过牌，身材合格但过牌慢，快攻不靠亡语续航。 |
| 41 | **赤红战意** | Red专属 | 5费 | SPELL | 攻0/防0 `TEMP_MANA_2` | **49.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.6%</span> | 0 张 (暂不推荐) | 五费临时两费，纯亏节奏，快攻不需要高费法术回蓝。 |
| 42 | **红绿古树桥梁** | Red/Green双色 | 6费 | MINION | DP:6 `FORTIFY_2,DEATH_DRAW_1` | **49.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.7%</span> | 0 张 (暂不推荐) | 六费六防加固亡语过牌，防守资源卡，快攻曲线不容。 |
| 43 | **掠夺者** | Red专属 | 7费 | MINION | DP:3 `BONUS_SCORE_1` | **48.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.7%</span> | 0 张 (暂不推荐) | 七费仅三攻，加分收益太靠后，快攻到七费已该结束对局。 |
| 44 | **红莲驻防长** | Red专属 | 6费 | MINION | DP:6 `FORTIFY_2` | **48.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 六费六防加固二，防守型高费，拖慢赤红斩杀节奏。 |
| 45 | **红绿共生体** | Red/Green双色 | 6费 | MINION | DP:6 `SUPPORT_ATK_2,FORTIFY_2` | **48.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 6费DP 6随从，附带【SUPPORT_ATK_2,FORTIFY_2】，承担Red阵营核心战术组件。 |
| 46 | **破阵狂徒** | Red专属 | 8费 | MINION | DP:2 `DEGRADE_1` | **48.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 八费二攻削弱一，费用过高，出场前节奏已崩。 |

---

## 🔵 二、 【蔚蓝 (Blue) 卡组】单卡战力与构筑指南
> **战术核心**：防守反击 · 固守护盾 · 资源消耗  
> **候选牌池**：32 张Blue专属卡 + 8 张双色协同卡 + 6 张中立通用卡（共 46 张候选，择优遴选 30 张入套）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | 属性/数值 | **综合评分** | 梯队 | **携带率** | **胜率贡献 (ΔWR)** | 推荐配置 | 实战构筑解析 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **佣兵斥候** | 中立通用 | 2费 | MINION | DP:1 `RUSH` | **98.0** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.2%</span> | 3 张 (核心满编) | 两费突袭解场，抢回先手并保护壁垒，满编保证前期节奏。 |
| 2 | **冰潮突袭者** | Blue专属 | 3费 | MINION | DP:3 `RUSH` | **89.7** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 2 张 (主力配置) | 三费突袭解小怪，抢回先手并掩护壁垒，双张灵活补曲线。 |
| 3 | **破甲潮汐兵** | Blue专属 | 3费 | MINION | DP:3 `RUSH,DEGRADE_1` | **89.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 2 张 (主力配置) | 三费突袭破甲，解场夺回先手，双张应对高防随从与快攻。 |
| 4 | **盾兵** | Blue专属 | 1费 | MINION | DP:1 `FORTIFY_1` | **89.2** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 2 张 (主力配置) | 一费固守一，前期拖延快攻，双张保证低费曲线流畅。 |
| 5 | **石像鬼** | Blue专属 | 6费 | MINION | DP:8 | **89.2** | **A** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 3 张 (主力配置) | 六费八防终端，中后期稳定站场，满编保证大哥上手率。 |
| 6 | **蔚蓝守卫** | Blue专属 | 2费 | MINION | DP:2 `FORTIFY_2` | **88.9** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.7%</span> | 2 张 (主力配置) | 两费固守二，前期挡住快攻，双张稳定链接中后期大哥。 |
| 7 | **防御！** | Blue专属 | 1费 | SPELL | 攻0/防2 | **88.4** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.6%</span> | 2 张 (主力配置) | 一费加二防，低费护脸保随从，双张稳定对抗快攻。 |
| 8 | **商人** | 中立通用 | 2费 | MINION | DP:2 `DRAW_1` | **86.8** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.4%</span> | 2 张 (主力配置) | 两费抽一，低费过牌润滑手牌，双张保证中期资源衔接。 |
| 9 | **蔚蓝护盾反击** | Blue专属 | 3费 | SPELL | 攻0/防4 `FORTIFY_1` | **86.3** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.3%</span> | 2 张 (主力配置) | 三费四防并固守一，守住中期攻势，双张支撑反击回合。 |
| 10 | **寒晶护壁** | Blue专属 | 2费 | SPELL | 攻0/防2 `DRAW_1` | **85.2** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.1%</span> | 2 张 (主力配置) | 两费二防附抽一，防守同时润牌，双张保证中期不断资源。 |
| 11 | **蔚蓝智慧** | Blue专属 | 3费 | SPELL | 攻0/防0 `DRAW_2,TEMP_MANA_1` | **84.7** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.0%</span> | 2 张 (主力配置) | 三费抽二回临时法力，过牌提速找大哥，双张保障中期展开。 |
| 12 | **冰封禁制** | Blue专属 | 2费 | SPELL | 攻6/防2 | **83.1** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.8%</span> | 2 张 (主力配置) | 两费六攻带二防，解场护脸兼备，双张补足中期反击火力。 |
| 13 | **蔚蓝要塞** | Blue专属 | 6费 | MINION | DP:6 `FORTIFY_3,SUPPORT_ATK_1` | **79.9** | **B** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.3%</span> | 2 张 (按需携带) | 六费固守三并支援攻击，中后期攻防兼备，按环境带两张。 |
| 14 | **潮汐学者** | Blue专属 | 2费 | MINION | DP:2 `DEATH_DRAW_1` | **75.5** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +0.6%</span> | 1 张 (按需携带) | 2费DP 2随从，附带【DEATH_DRAW_1】，承担Blue阵营核心战术组件。 |
| 15 | **蔚蓝盾卫** | Blue专属 | 6费 | MINION | DP:6 `FORTIFY_3,SUPPORT_ATK_1` | **68.8** | **C** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ -0.5%</span> | 0~1 张 (可选备编) | 6费DP 6随从，附带【FORTIFY_3,SUPPORT_ATK_1】，承担Blue阵营核心战术组件。 |
| 16 | **蔚蓝卫士** | Blue专属 | 2费 | MINION | DP:2 `FORTIFY_1` | **58.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.1%</span> | 0 张 (暂不推荐) | 两费固守一，抵挡效率一般，低费位更需过牌与突袭。 |
| 17 | **霜盾见习官** | Blue专属 | 2费 | MINION | DP:1 `FORTIFY_1,SUPPORT_ATK_1` | **58.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.2%</span> | 0 张 (暂不推荐) | 两费一防太脆，双词条难发挥，前期交换容易亏节奏。 |
| 18 | **蔚蓝盾卫** | Blue专属 | 2费 | MINION | DP:2 `FORTIFY_1` | **57.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.3%</span> | 0 张 (暂不推荐) | 2费DP 2随从，附带【FORTIFY_1】，承担Blue阵营核心战术组件。 |
| 19 | **雇佣兵** | 中立通用 | 3费 | MINION | DP:5 | **57.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.3%</span> | 0 张 (暂不推荐) | 三费五防纯身材，无固守与功能，难以替代体系随从。 |
| 20 | **拾荒盾卫** | 中立通用 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **57.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.4%</span> | 0 张 (暂不推荐) | 三费固守一亡语抽一，防守薄且延迟过牌，卡位不足。 |
| 21 | **蔚蓝守壁** | Blue专属 | 2费 | MINION | DP:2 `FORTIFY_2` | **57.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.4%</span> | 0 张 (暂不推荐) | 同费同身材，功能重复且竞争不过蔚蓝守卫，故不携带。 |
| 22 | **训练假人** | 中立通用 | 1费 | MINION | DP:1 | **56.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.5%</span> | 0 张 (暂不推荐) | 一费一防无词条，防守效率低于盾兵，无法进入构筑。 |
| 23 | **弓箭手** | Blue专属 | 3费 | MINION | DP:3 `SUPPORT_ATK_1` | **56.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.6%</span> | 0 张 (暂不推荐) | 三费身材普通，支援攻击偏慢，防守体系无需此类输出。 |
| 24 | **蔚蓝破甲师** | Blue专属 | 3费 | MINION | DP:3 `DEGRADE_1` | **55.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.6%</span> | 0 张 (暂不推荐) | 三费破甲随从，身材普通且防守弱，解场不如突袭兵。 |
| 25 | **蓝绿潮涌** | Blue/Green双色 | 3费 | SPELL | 攻0/防0 `DRAW_2` | **54.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.7%</span> | 0 张 (暂不推荐) | 三费抽二无防守，虽润牌但让节奏，优先选择蔚蓝智慧。 |
| 26 | **壁垒工匠** | Blue专属 | 3费 | MINION | DP:3 `FORTIFY_2` | **54.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.8%</span> | 0 张 (暂不推荐) | 三费固守二合格，但同费竞争激烈，无法进入主力构筑。 |
| 27 | **赤蓝战术交换** | Red/Blue双色 | 2费 | SPELL | 攻3/防0 `DRAW_1` | **54.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.9%</span> | 0 张 (暂不推荐) | 两费三攻抽一，看似灵活但防守端薄弱，卡位已满。 |
| 28 | **寒冰解离** | Blue专属 | 3费 | SPELL | 攻4/防0 | **53.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.9%</span> | 0 张 (暂不推荐) | 三费四攻单解，费用偏高且无附加，防守体系不优先选用。 |
| 29 | **寒霜破甲** | Blue专属 | 2费 | SPELL | 攻3/防0 `DEGRADE_1` | **53.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.9%</span> | 0 张 (暂不推荐) | 两费三攻带破甲，解场效率一般，卡位优先给抽牌与防守。 |
| 30 | **蔚蓝冲击** | Blue专属 | 2费 | SPELL | 攻4/防0 | **53.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -3.0%</span> | 0 张 (暂不推荐) | 两费四攻直伤，防守体系更需站场，纯输出不占卡位。 |
| 31 | **赤蓝交织者** | Red/Blue双色 | 4费 | MINION | DP:4 `SUPPORT_ATK_1` | **52.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.1%</span> | 0 张 (暂不推荐) | 4费DP 4随从，附带【SUPPORT_ATK_1】，承担Blue阵营核心战术组件。 |
| 32 | **火铳手** | Blue专属 | 3费 | MINION | DP:4 `SUPPORT_ATK_2` | **52.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.1%</span> | 0 张 (暂不推荐) | 三费支援攻击二偏进攻，与护盾反击思路不合，故舍弃。 |
| 33 | **藤甲兵** | Blue专属 | 4费 | MINION | DP:4 `FORTIFY_2` | **52.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 四费固守二尚可，但中期卡位紧，优先高质量终端与过牌。 |
| 34 | **赤蓝交织者** | Red/Blue双色 | 4费 | MINION | DP:4 `SUPPORT_ATK_1` | **52.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 4费DP 4随从，附带【SUPPORT_ATK_1】，承担Blue阵营核心战术组件。 |
| 35 | **赤蓝交织者** | Red/Blue双色 | 4费 | MINION | DP:4 `SUPPORT_ATK_1,DEATH_MANA_1` | **51.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 4费DP 4随从，附带【SUPPORT_ATK_1,DEATH_MANA_1】，承担Blue阵营核心战术组件。 |
| 36 | **碧波灵鳍** | Blue专属 | 4费 | MINION | DP:4 `SPAWN_1_1,FORTIFY_1` | **51.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.3%</span> | 0 张 (暂不推荐) | 四费固守一召唤小随从，铺场尚可但防线质量不足。 |
| 37 | **秘蓝回溯** | Blue专属 | 3费 | SPELL | 攻0/防3 `DRAW_1` | **50.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0 张 (暂不推荐) | 三费三防抽一，费用高于寒晶护壁，同类竞争落败。 |
| 38 | **法力回流哨兵** | Blue专属 | 4费 | MINION | DP:4 `DEATH_MANA_2` | **50.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0 张 (暂不推荐) | 四费亡语回两费，节奏偏慢，防守体系无需此类延迟资源。 |
| 39 | **潮汐学者** | Blue专属 | 4费 | MINION | DP:4 `DEATH_DRAW_1` | **50.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 4费DP 4随从，附带【DEATH_DRAW_1】，承担Blue阵营核心战术组件。 |
| 40 | **蓝晶守御者** | Blue专属 | 3费 | MINION | DP:3 `FORTIFY_1,ATTACK_ONLY` | **50.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 三费固守一且只能攻击，防守灵活性差，不适合体系。 |
| 41 | **蓝绿潮涌使** | Blue/Green双色 | 5费 | MINION | DP:5 `RAMP_1,DEATH_DRAW_1` | **49.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.6%</span> | 0 张 (暂不推荐) | 五费增幅亡语抽一，延迟收益太慢，中后期大哥更直接。 |
| 42 | **蓝绿潮汐者** | Blue/Green双色 | 5费 | MINION | DP:5 `DEATH_MANA_2,FORTIFY_2` | **49.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.6%</span> | 0 张 (暂不推荐) | 五费固守二亡语回费，节奏偏慢，资源收益不如直接过牌。 |
| 43 | **酒馆密账** | 中立通用 | 4费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **49.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.7%</span> | 0 张 (暂不推荐) | 四费抽二弃一，节奏亏损且费用过高，不如蔚蓝智慧。 |
| 44 | **深海守望者** | Blue专属 | 5费 | MINION | DP:5 `FORTIFY_3` | **48.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 五费固守三虽厚，但缺乏压制力，六费终端更值得卡位。 |
| 45 | **蓝绿潮汐织法者** | Blue/Green双色 | 5费 | MINION | DP:4 `SUPPORT_ATK_1,RAMP_1` | **48.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 五费支援攻击加增幅，身材亏且偏进攻，不适合防守反击。 |
| 46 | **深流变奏师** | Blue专属 | 5费 | MINION | DP:5 `SUPPORT_ATK_1,DEATH_MANA_2` | **48.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 五费支援攻击加亡语回费，节奏混乱，防守端收益太低。 |

---

## 🟢 三、 【翠绿 (Green) 卡组】单卡战力与构筑指南
> **战术核心**：法力跳费 · 质量成长 · 终结大哥  
> **候选牌池**：32 张Green专属卡 + 8 张双色协同卡 + 6 张中立通用卡（共 46 张候选，择优遴选 30 张入套）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | 属性/数值 | **综合评分** | 梯队 | **携带率** | **胜率贡献 (ΔWR)** | 推荐配置 | 实战构筑解析 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **佣兵斥候** | 中立通用 | 2费 | MINION | DP:1 `RUSH` | **98.0** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.2%</span> | 3 张 (核心满编) | 两费突袭补刀，灵活处理小生物，低费节奏核心满编。 |
| 2 | **灭世翡翠巨龙** | Green专属 | 9费 | MINION | DP:5 `RUSH,BONUS_SCORE_1` | **96.7** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.0%</span> | 3 张 (核心满编) | 九费核弹，突袭抢节奏并加分，跳费成功后满编终结比赛。 |
| 3 | **训练假人** | 中立通用 | 1费 | MINION | DP:1 | **96.4** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.9%</span> | 3 张 (核心满编) | 一费站场吸收伤害，保护跳费曲线，低费回合核心起手。 |
| 4 | **荆棘反制者** | Green专属 | 3费 | MINION | DP:3 `RUSH,DEGRADE_1` | **89.7** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 2 张 (主力配置) | 三费突袭附带削弱，解小怪兼干扰，前期争夺节奏主力。 |
| 5 | **藤蔓突袭者** | Green专属 | 4费 | MINION | DP:4 `RUSH` | **89.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 2 张 (主力配置) | 四费突袭四点，能即时解场抢回先手，中期节奏主力。 |
| 6 | **红绿共生** | Red/Green双色 | 5费 | MINION | DP:5 `RUSH,DEATH_DRAW_1` | **89.2** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 2 张 (主力配置) | 五费突袭五点，死亡补牌，中期解场兼续航，主力配置。 |
| 7 | **翡翠幼龙** | Green专属 | 7费 | MINION | DP:4 `RUSH,DEATH_MANA_1` | **88.9** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.7%</span> | 2 张 (主力配置) | 七费突袭仅四点攻，死亡回一费；中期解场并衔接跳费，主力配置。 |
| 8 | **翠绿幼苗** | Green专属 | 1费 | MINION | DP:1 `FORTIFY_1` | **88.1** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.6%</span> | 2 张 (主力配置) | 一费站场带强化，吸收早期伤害，为主力跳费争取空间。 |
| 9 | **剧毒花** | Green专属 | 2费 | MINION | DP:2 `DEGRADE_2` | **87.6** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.5%</span> | 2 张 (主力配置) | 二费削减两点攻，能压制低攻生物，主力配置补前期防守。 |
| 10 | **翡翠藤盾卫** | Green专属 | 2费 | MINION | DP:2 `FORTIFY_2` | **87.1** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.4%</span> | 2 张 (主力配置) | 两费强化二防，前期抗压扎实，为主力跳费争取回合。 |
| 11 | **世界树恩泽** | Green专属 | 5费 | SPELL | 攻0/防3 `RAMP_1` | **86.8** | **A** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.4%</span> | 3 张 (主力配置) | 五费跳费兼三防，撑过中期并提前大哥登场，核心满编。 |
| 12 | **萌芽跳费使** | Green专属 | 4费 | MINION | DP:3 `RAMP_1` | **84.1** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.9%</span> | 2 张 (主力配置) | 四费随从跳费，虽身材弱但能持续扩张，衔接中后期大哥。 |
| 13 | **翠绿资源滋长** | Green专属 | 4费 | SPELL | 攻0/防0 `RAMP_1,DRAW_1` | **80.7** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.4%</span> | 2 张 (主力配置) | 四费跳费并抽一，弥补手牌，稳定过渡到高费回合。 |
| 14 | **雇佣兵** | 中立通用 | 3费 | MINION | DP:5 | **57.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.3%</span> | 0 张 (暂不推荐) | 三费五点身材扎实，但无词条无跳费，卡位让给突袭与跳费。 |
| 15 | **翠绿驻防藤蔓** | Green专属 | 2费 | MINION | DP:2 `FORTIFY_1` | **57.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.4%</span> | 0 张 (暂不推荐) | 两费两点带强化，防守普通且无跳费，卡位让给功能性二费。 |
| 16 | **商人** | 中立通用 | 2费 | MINION | DP:2 `DRAW_1` | **56.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.4%</span> | 0 张 (暂不推荐) | 两费抽牌随从，身材合格但拖节奏，跳费体系另有法术过牌。 |
| 17 | **翠绿守林人** | Green专属 | 3费 | MINION | DP:3 `FORTIFY_2` | **56.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.5%</span> | 0 张 (暂不推荐) | 三费三点强化身材，纯防守无法施压，同费跳费或突袭更优先。 |
| 18 | **翠绿萌芽** | Green专属 | 3费 | SPELL | 攻0/防0 `RAMP_1` | **56.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.5%</span> | 0 张 (暂不推荐) | 三费仅跳一费，节奏亏牌亏场，被低费跳费曲线完全压制。 |
| 19 | **拾荒盾卫** | 中立通用 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **56.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.6%</span> | 0 张 (暂不推荐) | 三费防守亡语抽牌，节奏太慢，前期不如低费站场与突袭。 |
| 20 | **翠绿滋养** | Green专属 | 3费 | SPELL | 攻0/防0 `RAMP_1,DRAW_1` | **55.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.6%</span> | 0 张 (暂不推荐) | 三费跳费加抽牌看似润滑，实则节奏后置，四费资源滋长更稳。 |
| 21 | **翠绿复苏使** | Green专属 | 4费 | MINION | DP:4 `DEATH_MANA_2,FORTIFY_1` | **55.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.6%</span> | 0 张 (暂不推荐) | 四费亡语回费，延迟且身材平庸，无法帮助当下跳费或解场。 |
| 22 | **蓝绿潮涌** | Blue/Green双色 | 3费 | SPELL | 攻0/防0 `DRAW_2` | **55.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.7%</span> | 0 张 (暂不推荐) | 三费抽二不错，但纯过牌不跳费，减缓大哥登场速度。 |
| 23 | **自然献祭** | Green专属 | 4费 | SPELL | 攻3/防0 `SACRIFICE_1_KILL_1` | **54.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.7%</span> | 0 张 (暂不推荐) | 四费牺牲换杀，条件苛刻且亏牌，抢节奏不如突袭随从。 |
| 24 | **荆棘缠绕** | Green专属 | 2费 | SPELL | 攻5/防0 | **54.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.8%</span> | 0 张 (暂不推荐) | 两费打五直伤可观，但本阵营缺补刀配合，卡位让给跳费。 |
| 25 | **翠绿哨兵** | Green专属 | 3费 | MINION | DP:2 `FORTIFY_2,DEATH_DRAW_1` | **54.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.8%</span> | 0 张 (暂不推荐) | 三费两点攻太软，亡语抽牌延迟，防守端不如低费强化随从。 |
| 26 | **自然生长** | Green专属 | 3费 | SPELL | 攻0/防0 `RAMP_1` | **53.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.9%</span> | 0 张 (暂不推荐) | 三费单跳太慢，前期让节奏，后期抽到无用，卡位不足。 |
| 27 | **翠绿守护** | Green专属 | 3费 | SPELL | 攻0/防5 | **53.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.9%</span> | 0 张 (暂不推荐) | 三费五点防御只能拖延，无跳费无解场，偏离快速扩张计划。 |
| 28 | **森林之狼** | Green专属 | 4费 | MINION | DP:4 | **53.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -3.0%</span> | 0 张 (暂不推荐) | 四费白板四点身材，缺乏突袭与跳费，竞争不过同费功能卡。 |
| 29 | **古树庇护** | Green专属 | 3费 | SPELL | 攻0/防5 `FORTIFY_1` | **53.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -3.0%</span> | 0 张 (暂不推荐) | 三费防御加强化，仍缺跳费与解场，防守牌过多会拖慢。 |
| 30 | **翠绿跳费者** | Green专属 | 4费 | MINION | DP:3 `RAMP_1` | **52.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.1%</span> | 0 张 (暂不推荐) | 四费三点身材跳费慢，易被解，同费萌芽跳费使更优。 |
| 31 | **狂暴生长** | Green专属 | 3费 | SPELL | 攻0/防3 | **52.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.1%</span> | 0 张 (暂不推荐) | 三费只给三点防御，无法跳费也不解场，拖慢核心计划。 |
| 32 | **树人** | Green专属 | 3费 | MINION | DP:2 `FORTIFY_1` | **52.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 身材平庸，强化一防难换频；三费更需跳费或突袭，占卡位无收益。 |
| 33 | **林地衍生兽** | Green专属 | 4费 | MINION | DP:3 `SPAWN_1_1` | **52.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 四费铺场收益低，召唤物不解决跳费需求，拖累核心曲线。 |
| 34 | **芽苗祭司** | Green专属 | 4费 | MINION | DP:3 `RAMP_1` | **51.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 四费跳费随从身材差，落地延迟收益，同费法术跳费更稳。 |
| 35 | **蓝绿潮涌使** | Blue/Green双色 | 5费 | MINION | DP:5 `RAMP_1,DEATH_DRAW_1` | **51.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.3%</span> | 0 张 (暂不推荐) | 五费跳费带亡语抽牌，收益延迟，同费红绿共生即时性更强。 |
| 36 | **古木护林官** | Green专属 | 5费 | MINION | DP:5 `SUPPORT_ATK_1,DEATH_DRAW_1` | **51.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.3%</span> | 0 张 (暂不推荐) | 五费光环加攻，亡语抽牌偏慢，跳费体系更需即时大哥。 |
| 37 | **酒馆密账** | 中立通用 | 4费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **50.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0 张 (暂不推荐) | 四费抽二弃一，过滤但不赚牌，跳费期更需要上限而非换牌。 |
| 38 | **红绿共生体** | Red/Green双色 | 6费 | MINION | DP:6 `FORTIFY_2,BONUS_SCORE_1` | **50.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 6费DP 6随从，附带【FORTIFY_2,BONUS_SCORE_1】，承担Green阵营核心战术组件。 |
| 39 | **古树智者** | Green专属 | 6费 | MINION | DP:6 `DEATH_MANA_2` | **50.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 六费身材普通，亡语回费延迟，无法即时影响战局，不推荐。 |
| 40 | **蓝绿潮汐织法者** | Blue/Green双色 | 5费 | MINION | DP:4 `SUPPORT_ATK_1,RAMP_1` | **49.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 五费跳费带光环，身材偏弱，收益慢且卡位输给优质跳费。 |
| 41 | **蓝绿潮汐者** | Blue/Green双色 | 5费 | MINION | DP:5 `DEATH_MANA_2,FORTIFY_2` | **49.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.6%</span> | 0 张 (暂不推荐) | 五费亡语回费加防，延迟收益，卡组已有更稳定跳费组件。 |
| 42 | **红绿古树桥梁** | Red/Green双色 | 6费 | MINION | DP:6 `FORTIFY_2,DEATH_DRAW_1` | **49.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.6%</span> | 0 张 (暂不推荐) | 六费防守亡语抽牌，过于被动，无法在中期给足压力。 |
| 43 | **远古巨树** | Green专属 | 7费 | MINION | DP:8 `FORTIFY_2` | **49.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.7%</span> | 0 张 (暂不推荐) | 七费笨重，八点身材无突袭，无法即时返场，拖慢终结节奏。 |
| 44 | **萌芽巨兽** | Green专属 | 7费 | MINION | DP:7 `FORTIFY_3` | **48.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 七费高防无突袭，仅能站场挨打，终结端不如突袭巨龙。 |
| 45 | **翡翠巨熊** | Green专属 | 6费 | MINION | DP:6 `FORTIFY_2` | **48.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 六费仅防守，六点身材缺突袭，无法给对手压力，卡位紧张。 |
| 46 | **红绿共生体** | Red/Green双色 | 6费 | MINION | DP:6 `SUPPORT_ATK_2,FORTIFY_2` | **48.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 6费DP 6随从，附带【SUPPORT_ATK_2,FORTIFY_2】，承担Green阵营核心战术组件。 |

---

## ⚪ 四、 【中立 (Neutral) 卡牌】全阵营适配性与战术表现分析
> **机制说明**：同一张中立卡在快攻、控制、跳费等不同战术体系下具有截然不同的战术价值与契合度。

| 中立卡名称 | 费用 | 类型 | 🔴 赤红卡组评分 | 🔵 蔚蓝卡组评分 | 🟢 翠绿卡组评分 | 最佳契合卡组 | 跨阵营战术定位 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **商人** | 2费 | MINION | **95.3** | **86.8** | **56.8** | **赤红 (快攻)** | 偏向赤红 (快攻)体系的针对性组件 |
| **佣兵斥候** | 2费 | MINION | **90.0** | **98.0** | **98.0** | **蔚蓝 (防守)** | 多体系通用的高质量拼图 |
| **雇佣兵** | 3费 | MINION | **57.1** | **57.6** | **57.9** | **翠绿 (跳费)** | 特定战局下的可选备编卡 |
| **训练假人** | 1费 | MINION | **56.3** | **56.5** | **96.4** | **翠绿 (跳费)** | 偏向翠绿 (跳费)体系的针对性组件 |
| **拾荒盾卫** | 3费 | MINION | **55.7** | **57.3** | **56.0** | **蔚蓝 (防守)** | 特定战局下的可选备编卡 |
| **酒馆密账** | 4费 | SPELL | **53.3** | **49.1** | **50.9** | **赤红 (快攻)** | 特定战局下的可选备编卡 |
