# 🏆 TCG 卡牌战力评级与构筑指南（阵营分色专榜）

> **构筑规则说明**：在 TCG 标准规则中，各阵营（赤红/蔚蓝/翠绿）由**阵营专属卡 + 中立通用卡**构筑。本指南基于 PPO 强化学习智能体（`card_ppo_model_tuned.pth`）在对战环境中的实战数据，综合**【卡组携带率】**与**【局势胜率贡献 (ΔWR)】**两大维度，按阵营分色独立建榜，提供客观、严谨的构筑参考与单卡解析。

---

## 🔴 一、 【赤红 (Red) 卡组】单卡战力与构筑指南
> **战术核心**：快攻压制 · 牺牲协同 · 节奏斩杀  
> **候选牌池**：12 张赤红专属卡 + 6 张中立通用卡（共 18 张候选，择优遴选 30 张入套）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | 属性/数值 | **综合评分** | 梯队 | **携带率** | **胜率贡献 (ΔWR)** | 推荐配置 | 实战构筑解析 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **射线** | Red专属 | 2费 | SPELL | 攻2/防0 | **97.3** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.0%</span> | 3 张 (核心满编) | 两费打二直伤，补刀解小怪压血线，快攻高效终结手段，满编保证上手。 |
| 2 | **自爆** | Red专属 | 2费 | SPELL | 攻0/防0 `SACRIFICE_1_KILL_1,ATTACK_ONLY` | **96.6** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.9%</span> | 3 张 (核心满编) | 两费牺牲一卒点杀一怪，抢节奏解前场，攻击限定需算目标，满编保解场上限。 |
| 3 | **赤红突击手** | Red专属 | 1费 | MINION | DP:1 `DEATH_DRAW_1` | **95.2** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.7%</span> | 3 张 (核心满编) | 一费亡语过牌，铺场与续资源兼顾，快攻起手保留，满编稳固前期节奏。 |
| 4 | **破阵狂徒** | Red专属 | 4费 | MINION | DP:2 `RUSH,DEGRADE_1` | **93.8** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.5%</span> | 3 张 (核心满编) | 四费突袭削弱，处理中期屏障并继续压血，快攻断节奏时靠它返场，满编。 |
| 5 | **裂甲掷斧手** | Red专属 | 3费 | MINION | DP:1 `RUSH,DEGRADE_1` | **93.1** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.4%</span> | 3 张 (核心满编) | 三费突袭带削弱，进场即刻换怪并降血，先手争夺关键，满编保证上手。 |
| 6 | **血祭爆燃** | Red专属 | 3费 | SPELL | 攻3/防0 `SACRIFICE_1_KILL_1,ATTACK_ONLY` | **90.0** | **S** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.9%</span> | 3 张 (核心满编) | 三费牺牲一卒打三并点杀，解场兼直伤，斩杀回合极强，满编提高关键抽。 |
| 7 | **红色小队长** | Red专属 | 3费 | MINION | DP:3 `SPAWN_1_1` | **88.1** | **A** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.6%</span> | 3 张 (主力配置) | 三费召两个一攻，横向展开压场，配合牺牲与增益，三张保证中期铺场密度。 |
| 8 | **集结号手** | Red专属 | 2费 | MINION | DP:1 `SPAWN_1_1` | **86.7** | **A** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.4%</span> | 3 张 (主力配置) | 两费召一攻小生物，二费横向铺场，为牺牲和群体增益供料，三张稳定展开。 |
| 9 | **赤红掠袭者** | Red专属 | 5费 | MINION | DP:3 `RUSH,BONUS_SCORE_1` | **84.4** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.0%</span> | 2 张 (主力配置) | 五费突袭计分，可解场并额外得分，后期补伤害，带两张避免卡手。 |
| 10 | **商人** | 中立 | 2费 | MINION | DP:2 `DRAW_1` | **83.6** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.9%</span> | 2 张 (主力配置) | 两费过一，身材可交换，润滑手牌找直伤与突袭，两张兼顾节奏和资源。 |
| 11 | **切割者** | Red专属 | 2费 | MINION | DP:2 `DEGRADE_1` | **82.9** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.7%</span> | 2 张 (主力配置) | 两费二攻削弱一，交换不亏且破高血随从，卡位略紧带两张，够用不挤核心。 |
| 12 | **佣兵斥候** | 中立 | 2费 | MINION | DP:1 `RUSH,DRAW_1` | **76.5** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +0.7%</span> | 1 张 (按需携带) | 两费突袭过一，能补刀换牌，但身材弱，按需挂一张针对缺过牌局。 |
| 13 | **训练假人** | 中立 | 1费 | MINION | DP:1 | **57.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.3%</span> | 0 张 (暂不推荐) | 一费一攻无词条，无法过牌也无法压制，快攻低费位不塞纯站场。 |
| 14 | **大块头** | Red专属 | 6费 | MINION | DP:10 `DISCARD_2` | **52.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 六费十攻打手需弃两张，资源消耗过大，快攻无力承担，暂不进入构筑。 |
| 15 | **雇佣兵** | 中立 | 3费 | MINION | DP:5 | **51.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.3%</span> | 0 张 (暂不推荐) | 三费五攻白板，无突袭无过牌，快攻更需即时影响，卡位不留给它。 |
| 16 | **拾荒盾卫** | 中立 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **50.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0 张 (暂不推荐) | 三费守备亡语过牌，偏防守拖速，快攻节奏卡组不需要此类缓冲。 |
| 17 | **掠夺者** | Red专属 | 6费 | MINION | DP:4 `BONUS_SCORE_1` | **49.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.6%</span> | 0 张 (暂不推荐) | 六费仅四攻且计分滞后，快攻六费已需斩杀，费用太重，直接放弃。 |
| 18 | **酒馆密账** | 中立 | 1费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **48.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 一费抽二弃一，手牌质量易崩，快攻不愿弃关键直伤，故不进构筑。 |

---

## 🔵 二、 【蔚蓝 (Blue) 卡组】单卡战力与构筑指南
> **战术核心**：防守反击 · 固守护盾 · 资源消耗  
> **候选牌池**：12 张蔚蓝专属卡 + 6 张中立通用卡（共 18 张候选，择优遴选 30 张入套）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | 属性/数值 | **综合评分** | 梯队 | **携带率** | **胜率贡献 (ΔWR)** | 推荐配置 | 实战构筑解析 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **壁垒工匠** | Blue专属 | 3费 | MINION | DP:3 `FORTIFY_2` | **93.1** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.4%</span> | 3 张 (核心满编) | 三费固守二，强化阵地吸收，三张保障中期防线。 |
| 2 | **蔚蓝卫士** | Blue专属 | 2费 | MINION | DP:2 `FORTIFY_1` | **91.6** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.1%</span> | 3 张 (核心满编) | 两费固守一，前期挡刀稳节奏，满编保证防线连续。 |
| 3 | **石像鬼** | Blue专属 | 6费 | MINION | DP:8 | **90.9** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.0%</span> | 3 张 (核心满编) | 六费八血站场，高耐久吸收伤害，三张支撑后期终结。 |
| 4 | **冰封禁制** | Blue专属 | 3费 | SPELL | 攻2/防2 | **90.0** | **S** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.9%</span> | 3 张 (核心满编) | 三费二攻二防，解场兼护脸，三张应对关键威胁。 |
| 5 | **藤甲兵** | Blue专属 | 4费 | MINION | DP:4 `FORTIFY_2` | **89.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 3 张 (主力配置) | 四费固守二，中期抗线优质，三张稳住交换节奏。 |
| 6 | **蔚蓝要塞** | Blue专属 | 6费 | MINION | DP:5 `FORTIFY_3,SUPPORT_ATK_1` | **88.1** | **A** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.6%</span> | 3 张 (主力配置) | 六费固守三并加攻，后期质量终端，三张终结对局。 |
| 7 | **霜盾见习官** | Blue专属 | 2费 | MINION | DP:1 `FORTIFY_1,SUPPORT_ATK_1` | **87.9** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.5%</span> | 2 张 (主力配置) | 两费兼顾固守与加攻，攻防转换灵活，两张填曲线。 |
| 8 | **防御！** | Blue专属 | 1费 | SPELL | 攻0/防2 | **87.2** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.4%</span> | 2 张 (主力配置) | 一费加二防，低费保护随从，两张开防爆发伤害。 |
| 9 | **弓箭手** | Blue专属 | 3费 | MINION | DP:3 `SUPPORT_ATK_1` | **85.8** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.2%</span> | 2 张 (主力配置) | 三费支援加攻，后排持续压血，两张开给中速对局。 |
| 10 | **盾兵** | Blue专属 | 1费 | MINION | DP:1 `FORTIFY_1` | **84.4** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.0%</span> | 2 张 (主力配置) | 一费固守一，低费补盾护脸，两张防快攻起手。 |
| 11 | **火铳手** | Blue专属 | 4费 | MINION | DP:4 `SUPPORT_ATK_2` | **78.7** | **B** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.1%</span> | 2 张 (按需携带) | 四费支援加二攻，增幅后排输出，两张按环境编入。 |
| 12 | **寒晶护壁** | Blue专属 | 2费 | SPELL | 攻0/防2 `DRAW_1` | **78.0** | **B** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.0%</span> | 2 张 (按需携带) | 两费加二防并过牌，润滑手牌，两张应对消耗战。 |
| 13 | **拾荒盾卫** | 中立 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **69.4** | **C** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ -0.4%</span> | 0~1 张 (可选备编) | 三费固守一死亡过牌，可作消耗对局备编，通常零或一张。 |
| 14 | **酒馆密账** | 中立 | 1费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **59.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.0%</span> | 0 张 (暂不推荐) | 一费过二弃一，手牌消耗大，防守体系不宜采用。 |
| 15 | **佣兵斥候** | 中立 | 2费 | MINION | DP:1 `RUSH,DRAW_1` | **58.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.1%</span> | 0 张 (暂不推荐) | 两费突袭过牌，身材太差，防守思路无需投入。 |
| 16 | **商人** | 中立 | 2费 | MINION | DP:2 `DRAW_1` | **56.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.5%</span> | 0 张 (暂不推荐) | 两费过一，身材平庸且挤占防线卡位，无需携带。 |
| 17 | **训练假人** | 中立 | 1费 | MINION | DP:1 | **52.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 一费一血无固守，无法有效挡刀，不列入防线。 |
| 18 | **雇佣兵** | 中立 | 3费 | MINION | DP:5 | **50.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0 张 (暂不推荐) | 三费五血无词条，虽耐打但无协同，构筑不带。 |

---

## 🟢 三、 【翠绿 (Green) 卡组】单卡战力与构筑指南
> **战术核心**：法力跳费 · 质量成长 · 终结大哥  
> **候选牌池**：12 张翠绿专属卡 + 6 张中立通用卡（共 18 张候选，择优遴选 30 张入套）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | 属性/数值 | **综合评分** | 梯队 | **携带率** | **胜率贡献 (ΔWR)** | 推荐配置 | 实战构筑解析 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **灭世翡翠巨龙** | Green专属 | 9费 | MINION | DP:10 `RUSH,BONUS_SCORE_1` | **97.3** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.0%</span> | 3 张 (核心满编) | 九费十攻突袭，即时解场并追加得分，跳费成功后核心胜点，满编。 |
| 2 | **翠绿萌芽** | Green专属 | 2费 | SPELL | 攻0/防0 `RAMP_1` | **94.5** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.6%</span> | 3 张 (核心满编) | 两费跳费法术，尽早抬高法力上限，让大哥提前登场，节奏基石满编。 |
| 3 | **世界树恩泽** | Green专属 | 4费 | SPELL | 攻0/防3 `RAMP_1` | **90.9** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.0%</span> | 3 张 (核心满编) | 四费跳费并给三点防御，抗快攻同时加速大哥，节奏基石满编。 |
| 4 | **远古巨树** | Green专属 | 7费 | MINION | DP:8 `FORTIFY_2` | **90.2** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.9%</span> | 3 张 (核心满编) | 七费八攻带加固，跳费后压制力极强，中后期终结核心，必须满编。 |
| 5 | **荆棘缠绕** | Green专属 | 2费 | SPELL | 攻2/防0 | **90.0** | **S** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.9%</span> | 3 张 (核心满编) | 两费打二，前期解场抢节奏，后期补刀压血，低费高效必须满编。 |
| 6 | **芽苗祭司** | Green专属 | 3费 | MINION | DP:3 `RAMP_1` | **89.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 3 张 (主力配置) | 三费站场跳费，法力领先衔接大哥，中期扩张核心，满编稳固节奏。 |
| 7 | **翡翠巨熊** | Green专属 | 6费 | MINION | DP:6 `FORTIFY_2` | **87.4** | **A** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.5%</span> | 3 张 (主力配置) | 六费六攻加固，跳费后提前站场，攻守兼备的质量终端，建议满编。 |
| 8 | **翡翠幼龙** | Green专属 | 5费 | MINION | DP:3 `RUSH,DEATH_MANA_1` | **85.8** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.2%</span> | 2 张 (主力配置) | 五费突袭解场，死亡返费可续节奏，灵活但身材偏弱，两张够用。 |
| 9 | **商人** | 中立 | 2费 | MINION | DP:2 `DRAW_1` | **85.1** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.1%</span> | 2 张 (主力配置) | 两费站场过牌，帮助找跳费与大哥，润滑但不抢节奏，带两张。 |
| 10 | **森林之狼** | Green专属 | 4费 | MINION | DP:4 | **84.4** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.0%</span> | 2 张 (主力配置) | 四费四攻白板，曲线过渡尚可，但缺少词条上限，补两张即可。 |
| 11 | **树人** | Green专属 | 3费 | MINION | DP:2 `FORTIFY_1` | **83.6** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.9%</span> | 2 张 (主力配置) | 三费站场带加固，前期护脸换小随从，衔接跳费后的中期曲线，带两张。 |
| 12 | **剧毒花** | Green专属 | 2费 | MINION | DP:2 `DEGRADE_2` | **78.7** | **B** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.1%</span> | 2 张 (按需携带) | 两费降攻随从，针对敌方低攻核心或快攻小怪，按环境投入两张。 |
| 13 | **训练假人** | 中立 | 1费 | MINION | DP:1 | **58.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.1%</span> | 0 张 (暂不推荐) | 一费一攻白板，无法跳费过牌，前期卡位价值过低，不予采用。 |
| 14 | **佣兵斥候** | 中立 | 2费 | MINION | DP:1 `RUSH,DRAW_1` | **57.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.3%</span> | 0 张 (暂不推荐) | 两费突袭过牌，身材太差，挤占跳费与解场卡位，不值得投入。 |
| 15 | **拾荒盾卫** | 中立 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **57.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.4%</span> | 0 张 (暂不推荐) | 三费防守亡语过牌，功能可用，但与跳费曲线冲突，暂不投入。 |
| 16 | **酒馆密账** | 中立 | 1费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **50.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0 张 (暂不推荐) | 一费过二弃一，弃牌易断跳费链，节奏风险高，卡位不足不带。 |
| 17 | **雇佣兵** | 中立 | 3费 | MINION | DP:5 | **50.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 三费五攻白板，进攻端尚可，但无防守与跳费收益，体系外不带。 |
| 18 | **狂暴生长** | Green专属 | 3费 | SPELL | 攻0/防3 | **48.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 三费只给三点防御，不跳费不过牌，无法解决前期压力，暂不投入。 |

---

## ⚪ 四、 【中立 (Neutral) 卡牌】全阵营适配性与战术表现分析
> **机制说明**：同一张中立卡在快攻、控制、跳费等不同战术体系下具有截然不同的战术价值与契合度。

| 中立卡名称 | 费用 | 类型 | 🔴 赤红卡组评分 | 🔵 蔚蓝卡组评分 | 🟢 翠绿卡组评分 | 最佳契合卡组 | 跨阵营战术定位 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **商人** | 2费 | MINION | **83.6** | **56.5** | **85.1** | **翠绿 (跳费)** | 偏向翠绿 (跳费)体系的针对性组件 |
| **佣兵斥候** | 2费 | MINION | **76.5** | **58.6** | **57.9** | **赤红 (快攻)** | 偏向赤红 (快攻)体系的针对性组件 |
| **训练假人** | 1费 | MINION | **57.9** | **52.2** | **58.6** | **翠绿 (跳费)** | 特定战局下的可选备编卡 |
| **雇佣兵** | 3费 | MINION | **51.5** | **50.8** | **50.1** | **赤红 (快攻)** | 特定战局下的可选备编卡 |
| **拾荒盾卫** | 3费 | MINION | **50.8** | **69.4** | **57.2** | **蔚蓝 (防守)** | 特定战局下的可选备编卡 |
| **酒馆密账** | 1费 | SPELL | **48.0** | **59.3** | **50.8** | **蔚蓝 (防守)** | 特定战局下的可选备编卡 |
