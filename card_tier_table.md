# 🏆 TCG 卡牌战力评级与构筑指南（阵营分色专榜）

> **构筑规则说明**：在 TCG 标准规则中，各阵营（赤红/蔚蓝/翠绿）由**阵营专属卡 + 双色协同卡 + 中立通用卡**构筑。本指南基于 PPO 强化学习智能体在对战环境中的实战数据，综合**【卡组携带率】**与**【局势胜率贡献 (ΔWR)】**两大维度，按阵营分色独立建榜，提供客观、严谨的构筑参考与单卡解析。

---

## 🔴 一、 【赤红 (Red) 卡组】单卡战力与构筑指南
> **战术核心**：快攻压制 · 牺牲协同 · 节奏斩杀  
> **候选牌池**：27 张Red专属卡 + 6 张双色协同卡 + 6 张中立通用卡（共 39 张候选，择优遴选 30 张入套）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | 属性/数值 | **综合评分** | 梯队 | **携带率** | **胜率贡献 (ΔWR)** | 推荐配置 | 实战构筑解析 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **红绿共生** | Red/Green双色 | 5费 | MINION | DP:5 `RUSH,DEATH_DRAW_1` | **97.7** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.1%</span> | 3 张 (核心满编) | 五费五攻突袭亡语过牌，兼顾解场续航。满编补中期节奏与资源。 |
| 2 | **破阵先锋** | Red专属 | 6费 | MINION | DP:4 `RUSH,BONUS_SCORE_1` | **97.4** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.1%</span> | 3 张 (核心满编) | 六费四攻突袭，额外得分补伤害。中期抢节奏兼终结，满编保证连贯。 |
| 3 | **赤红中期突破手** | Red专属 | 8费 | MINION | DP:5 `RUSH,BONUS_SCORE_1` | **97.1** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.0%</span> | 3 张 (核心满编) | 八费五攻突袭，额外得分加速斩杀。核心终结手段，满编提高后期上手率。 |
| 4 | **赤红破阵兵** | Red专属 | 5费 | MINION | DP:3 `RUSH` | **96.7** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.0%</span> | 3 张 (核心满编) | 五费三攻突袭，解场兼抢血。费用适中且即时生效，满编保中期压制。 |
| 5 | **裂甲掷斧手** | Red专属 | 6费 | MINION | DP:1 `RUSH,DEGRADE_1` | **96.4** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.9%</span> | 3 张 (核心满编) | 六费突袭带削弱，可解场夺先手。中期节奏卡，满编保证上手。 |
| 6 | **赤红新兵** | Red专属 | 1费 | MINION | DP:1 | **95.2** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.7%</span> | 3 张 (核心满编) | 一费一攻白板，低费铺场与献祭素材。起手展开关键，满编稳节奏。 |
| 7 | **商人** | 中立通用 | 2费 | MINION | DP:2 `DRAW_1` | **94.8** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.6%</span> | 3 张 (核心满编) | 二费二攻过一，站场兼润滑手牌。快攻续航关键，满编保展开。 |
| 8 | **集结号手** | Red专属 | 2费 | MINION | DP:1 `SPAWN_1_1,ATTACK_ONLY` | **93.3** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.4%</span> | 3 张 (核心满编) | 二费铺两个攻击频率，滚雪球与献祭载体。低费展开基石，必须满编。 |
| 9 | **佣兵斥候** | 中立通用 | 2费 | MINION | DP:1 `RUSH` | **90.0** | **S** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.9%</span> | 3 张 (核心满编) | 二费突袭抢节奏，补刀解小怪皆宜。低费即时输出，满编稳前期。 |
| 10 | **赤红突破手** | Red专属 | 8费 | MINION | DP:1 `RUSH,BONUS_SCORE_1` | **88.1** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.6%</span> | 2 张 (主力配置) | 八费突袭补刀，额外得分作终结。两张防卡手，后期上手可斩杀。 |
| 11 | **赤红突击手** | Red专属 | 3费 | MINION | DP:1 `DEATH_DRAW_1` | **86.2** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.3%</span> | 2 张 (主力配置) | 三费一攻带死亡过牌，交换后补资源。两张防卡手，满编收益递减。 |
| 12 | **赤红掠袭者** | Red专属 | 9费 | MINION | DP:1 `RUSH,BONUS_SCORE_1` | **57.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.3%</span> | 0 张 (暂不推荐) | 九费一攻突袭，额外得分太慢。赤红九费前应抢死，高费拖累节奏。 |
| 13 | **训练假人** | 中立通用 | 1费 | MINION | DP:1 | **57.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.3%</span> | 0 张 (暂不推荐) | 一费一防无词条，功能不如赤红新兵。快攻需要进攻频率，此卡太弱。 |
| 14 | **牺牲祭师** | Red专属 | 3费 | MINION | DP:3 `SACRIFICE_1_KILL_1,DEATH_DRAW_1` | **56.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.5%</span> | 0 张 (暂不推荐) | 三费身材合格，但牺牲杀与死亡过牌冲突。快攻要直接输出，无空余费。 |
| 15 | **拾荒盾卫** | 中立通用 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **55.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.6%</span> | 0 张 (暂不推荐) | 三费三攻带加固亡语，防守向组合。快攻无需拖后过牌，不如直接铺场。 |
| 16 | **赤红战盾** | Red专属 | 2费 | SPELL | 攻0/防4 | **55.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.6%</span> | 0 张 (暂不推荐) | 二费加四防，只护脸不返场。快攻内战需争夺节奏，防御法术太被动。 |
| 17 | **红色小队长** | Red专属 | 3费 | MINION | DP:3 `SPAWN_1_1` | **54.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.7%</span> | 0 张 (暂不推荐) | 三费铺一衍生，身材平庸且无突袭。快攻更需即时伤害，卡位紧不投入。 |
| 18 | **牺牲角斗士** | Red专属 | 4费 | MINION | DP:4 `SACRIFICE_1_KILL_1` | **54.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.8%</span> | 0 张 (暂不推荐) | 四费四攻身材尚可，牺牲杀拖慢进攻。快攻四费应铺场或打脸，不带。 |
| 19 | **雇佣兵** | 中立通用 | 3费 | MINION | DP:5 | **54.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.8%</span> | 0 张 (暂不推荐) | 三费五防白板，纯防守难施压。快攻卡位宝贵，白板墙不如突袭生物。 |
| 20 | **切割者** | Red专属 | 2费 | MINION | DP:2 `DEGRADE_1` | **54.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.9%</span> | 0 张 (暂不推荐) | 二费二攻带削弱，站场收益偏低。快攻要即时输出，防御向词条拖节奏。 |
| 21 | **赤蓝交织者** | Red/Blue双色 | 4费 | MINION | DP:4 `SUPPORT_ATK_1` | **53.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.9%</span> | 0 张 (暂不推荐) | 四费四攻支援加攻，依赖己方场面。快攻四费要即战，配合卡卡手。 |
| 22 | **赤蓝交织者** | Red/Blue双色 | 4费 | MINION | DP:4 `SUPPORT_ATK_1` | **53.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -3.0%</span> | 0 张 (暂不推荐) | 四费四攻支援加攻，依赖己方场面。快攻四费要即战，配合卡卡手。 |
| 23 | **赤红战意** | Red专属 | 5费 | SPELL | 攻0/防0 `TEMP_MANA_2` | **53.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -3.0%</span> | 0 张 (暂不推荐) | 五费换两临时法力，当回合收益难兑现。快攻五费需威胁，法术让节奏。 |
| 24 | **余烬传令官** | Red专属 | 5费 | MINION | DP:5 `DEATH_DRAW_1` | **52.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.1%</span> | 0 张 (暂不推荐) | 五费五攻死亡过牌，节奏偏慢。快攻需要即时压力，亡语补牌太迟。 |
| 25 | **赤红铁卫** | Red专属 | 3费 | MINION | DP:3 `FORTIFY_2` | **52.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.1%</span> | 0 张 (暂不推荐) | 三费三攻带加固，偏防御交换。赤红要抢血，此类站场卡卡位不足。 |
| 26 | **献祭狂徒** | Red专属 | 4费 | MINION | DP:4 `SACRIFICE_1_KILL_1` | **52.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 四费四攻配牺牲杀，交换思路偏慢。快攻要打脸施压，不愿亏随从。 |
| 27 | **酒馆密账** | 中立通用 | 4费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **51.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 四费抽二弃一，净手牌有限且让节奏。快攻四费要出威胁，过牌太慢。 |
| 28 | **大块头** | Red专属 | 6费 | MINION | DP:10 `DISCARD_2` | **51.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.3%</span> | 0 张 (暂不推荐) | 六费十防弃两张，防守身材与快攻相悖。弃牌伤资源，卡手时不救场。 |
| 29 | **赤蓝战术交换** | Red/Blue双色 | 2费 | SPELL | 攻3/防0 `DRAW_1` | **51.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.3%</span> | 0 张 (暂不推荐) | 二费打三过一，看似灵活但快攻不缺血。卡位留给生物更持续。 |
| 30 | **献祭之焰** | Red专属 | 3费 | SPELL | 攻0/防0 `SACRIFICE_1_KILL_1,DISCARD_1` | **50.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0 张 (暂不推荐) | 三费牺牲杀还弃牌，资源亏损严重。快攻经不起双重消耗，完全不合。 |
| 31 | **赤红献祭** | Red专属 | 2费 | SPELL | 攻0/防0 `SACRIFICE_1_KILL_1` | **50.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0 张 (暂不推荐) | 二费牺牲一换一杀，条件被动且亏节奏。快攻需铺场，不选此类解。 |
| 32 | **血祭爆燃** | Red专属 | 3费 | SPELL | 攻3/防0 `SACRIFICE_1_KILL_1,ATTACK_ONLY` | **50.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 三费打三且牺牲一杀，双重条件过苛。仅能进攻，快攻不愿如此换牌。 |
| 33 | **射线** | Red专属 | 2费 | SPELL | 攻2/防0 | **49.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 二费打二效率普通，缺过牌与解场价值。赤红更愿用生物持续施压。 |
| 34 | **自爆** | Red专属 | 2费 | SPELL | 攻0/防0 `SACRIFICE_1_KILL_1,ATTACK_ONLY` | **49.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.6%</span> | 0 张 (暂不推荐) | 牺牲随从换一杀，仅能进攻限制大。快攻不愿亏场面，二费不如直接出人。 |
| 35 | **红绿古树桥梁** | Red/Green双色 | 6费 | MINION | DP:6 `FORTIFY_2,DEATH_DRAW_1` | **49.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.6%</span> | 0 张 (暂不推荐) | 六费六防亡语过牌，纯防守拖节奏。快攻六费需终结，不会投入。 |
| 36 | **红绿共生体** | Red/Green双色 | 6费 | MINION | DP:6 `SUPPORT_ATK_2,FORTIFY_2` | **48.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.7%</span> | 0 张 (暂不推荐) | 六费六攻支援加固，偏中速站场。赤红六费应完成斩杀，不需大型配合。 |
| 37 | **战地督军** | Red专属 | 5费 | MINION | DP:4 `SUPPORT_ATK_1` | **48.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.7%</span> | 0 张 (暂不推荐) | 五费四攻支援加攻，需场面配合。快攻五费要立即威胁，顺风才可用。 |
| 38 | **掠夺者** | Red专属 | 7费 | MINION | DP:3 `BONUS_SCORE_1` | **48.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 七费仅三攻，额外得分需拖到后期。快攻七费已定胜负，不会携带。 |
| 39 | **破阵狂徒** | Red专属 | 8费 | MINION | DP:2 `DEGRADE_1` | **48.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 八费二攻仅削弱，威胁太低。高费卡需终结比赛，此牌难担重任。 |

---

## 🔵 二、 【蔚蓝 (Blue) 卡组】单卡战力与构筑指南
> **战术核心**：防守反击 · 固守护盾 · 资源消耗  
> **候选牌池**：27 张Blue专属卡 + 6 张双色协同卡 + 6 张中立通用卡（共 39 张候选，择优遴选 30 张入套）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | 属性/数值 | **综合评分** | 梯队 | **携带率** | **胜率贡献 (ΔWR)** | 推荐配置 | 实战构筑解析 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **破甲潮汐兵** | Blue专属 | 2费 | MINION | DP:3 `RUSH,DEGRADE_1` | **98.0** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.2%</span> | 3 张 (核心满编) | 两费三攻突袭带破甲，解场抢先手并削厚墙，前期核心，满编。 |
| 2 | **佣兵斥候** | 中立通用 | 2费 | MINION | DP:1 `RUSH` | **89.7** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 2 张 (主力配置) | 两费突袭一攻，补刀解场保节奏，低费灵活，主力两张。 |
| 3 | **盾兵** | Blue专属 | 1费 | MINION | DP:1 `FORTIFY_1` | **88.7** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.7%</span> | 2 张 (主力配置) | 一费固守一，前期吸收小伤害，护住后排并润滑曲线，主力两张。 |
| 4 | **蔚蓝守壁** | Blue专属 | 2费 | MINION | DP:2 `FORTIFY_2` | **87.8** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.5%</span> | 2 张 (主力配置) | 两费固守二，低费高效护脸，吸收快攻伤害，主力两张稳前期。 |
| 5 | **蔚蓝智慧** | Blue专属 | 2费 | SPELL | 攻0/防0 `DRAW_2,TEMP_MANA_1` | **87.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.5%</span> | 2 张 (主力配置) | 两费抽二并回一费，过牌同时润滑费用，找大哥与解牌，主力两张。 |
| 6 | **商人** | 中立通用 | 2费 | MINION | DP:2 `DRAW_1` | **86.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.3%</span> | 2 张 (主力配置) | 两费站场抽一，防守同时补资源，曲线顺滑，主力两张。 |
| 7 | **蔚蓝盾卫** | Blue专属 | 2费 | MINION | DP:2 `FORTIFY_1` | **85.6** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.2%</span> | 2 张 (主力配置) | 二费固守一可作主力两张；四费固守三支援一太笨重，不推荐。 |
| 8 | **霜盾见习官** | Blue专属 | 2费 | MINION | DP:1 `FORTIFY_1,SUPPORT_ATK_1` | **85.3** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.1%</span> | 2 张 (主力配置) | 两费兼具固守与支援，低费曲线润滑，保护核心并提供攻击，主力两张。 |
| 9 | **雇佣兵** | 中立通用 | 3费 | MINION | DP:5 | **84.0** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.9%</span> | 2 张 (主力配置) | 三费五点防，白板但扎实，前期挡快攻，中期换资源，主力两张。 |
| 10 | **蓝绿潮汐者** | Blue/Green双色 | 5费 | MINION | DP:5 `DEATH_MANA_2,FORTIFY_2` | **79.6** | **B** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.2%</span> | 2 张 (按需携带) | 五费固守二亡语回二，吸收伤害并衔接高费，按需两张跳质量。 |
| 11 | **冰潮突袭者** | Blue专属 | 3费 | MINION | DP:3 `RUSH` | **79.4** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +1.2%</span> | 1 张 (按需携带) | 三费突袭解场，能夺回先手，但无破甲，按需一张补节奏。 |
| 12 | **蔚蓝要塞** | Blue专属 | 5费 | MINION | DP:6 `FORTIFY_3,SUPPORT_ATK_1` | **78.6** | **B** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.1%</span> | 2 张 (按需携带) | 五费固守三且支援加攻，中期吸收伤害并反打，按需两张撑质量。 |
| 13 | **蓝绿潮汐织法者** | Blue/Green双色 | 5费 | MINION | DP:4 `SUPPORT_ATK_1,RAMP_1` | **78.3** | **B** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.0%</span> | 2 张 (按需携带) | 五费跳费支援，加速拍大哥，但身板偏薄，按需两张抢科技。 |
| 14 | **蔚蓝守卫** | Blue专属 | 2费 | MINION | DP:2 `FORTIFY_2` | **77.2** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +0.8%</span> | 1 张 (按需携带) | 两费固守二，低费高防挡快攻，但后期抽到偏弱，按需一张。 |
| 15 | **赤蓝战术交换** | Red/Blue双色 | 2费 | SPELL | 攻3/防0 `DRAW_1` | **71.8** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ -0.0%</span> | 1 张 (按需携带) | 两费三攻抽一，解小怪补资源，但效率中庸，按需一张。 |
| 16 | **冰封禁制** | Blue专属 | 2费 | SPELL | 攻6/防2 | **70.5** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ -0.2%</span> | 1 张 (按需携带) | 两费六攻配防二，可解关键随从或补刀，环境厚墙多时挂一。 |
| 17 | **潮汐学者** | Blue专属 | 4费 | MINION | DP:4 `DEATH_DRAW_1` | **69.3** | **C** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ -0.4%</span> | 0~1 张 (可选备编) | 四费亡语抽一，交换后补牌，但节奏偏慢，可选备编一张。 |
| 18 | **防御！** | Blue专属 | 1费 | SPELL | 攻0/防2 | **59.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.1%</span> | 0 张 (暂不推荐) | 一费防二不补牌不站场，纯叠甲效率低，卡位应留给过牌与随从。 |
| 19 | **训练假人** | 中立通用 | 1费 | MINION | DP:1 | **58.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.2%</span> | 0 张 (暂不推荐) | 一费一防无词条，挡不住也换不到，低费卡位不如盾兵。 |
| 20 | **寒晶护壁** | Blue专属 | 2费 | SPELL | 攻0/防2 `DRAW_1` | **58.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.2%</span> | 0 张 (暂不推荐) | 两费防二抽一，看似润滑，但防守方二费过牌易失节奏，不带。 |
| 21 | **赤蓝交织者** | Red/Blue双色 | 4费 | MINION | DP:4 `SUPPORT_ATK_1` | **56.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.4%</span> | 0 张 (暂不推荐) | 四费支援加攻一，光环收益低，本体四点防普通，防守体系不带。 |
| 22 | **弓箭手** | Blue专属 | 3费 | MINION | DP:3 `SUPPORT_ATK_1` | **56.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.5%</span> | 0 张 (暂不推荐) | 三费支援加攻却只三点身板，站不住场，光环易被解，节奏太亏。 |
| 23 | **蔚蓝卫士** | Blue专属 | 2费 | MINION | DP:2 `FORTIFY_1` | **55.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.6%</span> | 0 张 (暂不推荐) | 两费仅两点固守一，数值平庸，卡位被盾卫与守壁挤占，不带。 |
| 24 | **蔚蓝护盾反击** | Blue专属 | 3费 | SPELL | 攻0/防4 `FORTIFY_1` | **54.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.7%</span> | 0 张 (暂不推荐) | 三费防四固守一，纯防御效率低，无法返场，防守不靠单张叠甲。 |
| 25 | **壁垒工匠** | Blue专属 | 3费 | MINION | DP:3 `FORTIFY_2` | **54.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.8%</span> | 0 张 (暂不推荐) | 三费固守二，防得住但无反制，三费争不过破甲潮汐兵与突袭者。 |
| 26 | **拾荒盾卫** | 中立通用 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **54.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.8%</span> | 0 张 (暂不推荐) | 三费固守一亡语抽一，交换后补牌，但三点防太脆，节奏亏。 |
| 27 | **蔚蓝盾卫** | Blue专属 | 4费 | MINION | DP:6 `FORTIFY_3,SUPPORT_ATK_1` | **53.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.9%</span> | 0 张 (暂不推荐) | 二费固守一可作主力两张；四费固守三支援一太笨重，不推荐。 |
| 28 | **酒馆密账** | 中立通用 | 4费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **53.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -3.0%</span> | 0 张 (暂不推荐) | 四费抽二弃一，节奏太慢且弃牌风险高，防守方不缺此类过牌。 |
| 29 | **火铳手** | Blue专属 | 3费 | MINION | DP:4 `SUPPORT_ATK_2` | **53.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -3.0%</span> | 0 张 (暂不推荐) | 三费光环加攻二，本体四点防可站，但防守体系无需攻击光环，卡位紧。 |
| 30 | **法力回流哨兵** | Blue专属 | 4费 | MINION | DP:4 `DEATH_MANA_2` | **52.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.1%</span> | 0 张 (暂不推荐) | 四费亡语回二，延迟收益且身材平庸，防守节奏等不起，不带。 |
| 31 | **藤甲兵** | Blue专属 | 4费 | MINION | DP:4 `FORTIFY_2` | **52.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.1%</span> | 0 张 (暂不推荐) | 四费四点固守二，防御尚可但无交换能力，同费雇佣兵更扎实。 |
| 32 | **蓝绿潮涌** | Blue/Green双色 | 3费 | SPELL | 攻0/防0 `DRAW_2` | **52.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 三费抽二，纯过牌费用偏高，蔚蓝智慧更润滑，无卡位。 |
| 33 | **寒冰解离** | Blue专属 | 3费 | SPELL | 攻4/防0 | **51.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.3%</span> | 0 张 (暂不推荐) | 三费四攻解场，效率一般，破甲与突袭随从更灵活，不带。 |
| 34 | **寒霜破甲** | Blue专属 | 2费 | SPELL | 攻3/防0 `DEGRADE_1` | **51.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.3%</span> | 0 张 (暂不推荐) | 两费三攻破甲一，解小怪尚可，但破甲潮汐兵能站场，法术位紧。 |
| 35 | **蔚蓝冲击** | Blue专属 | 2费 | SPELL | 攻4/防0 | **50.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0 张 (暂不推荐) | 两费四攻打脸有限，防守体系不缺直伤，解场不如破甲与突袭。 |
| 36 | **赤蓝交织者** | Red/Blue双色 | 4费 | MINION | DP:4 `SUPPORT_ATK_1` | **50.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 四费支援加攻一，光环收益低，本体四点防普通，防守体系不带。 |
| 37 | **蔚蓝破甲师** | Blue专属 | 3费 | MINION | DP:3 `DEGRADE_1` | **49.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 三费破甲一，功能被二费突袭破甲兵覆盖，费用更重，无卡位。 |
| 38 | **石像鬼** | Blue专属 | 6费 | MINION | DP:8 | **48.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.7%</span> | 0 张 (暂不推荐) | 六费八防白板，缺固守与返场，六费回合拍下难扭转，终端不合格。 |
| 39 | **深海守望者** | Blue专属 | 5费 | MINION | DP:5 `FORTIFY_3` | **48.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 五费固守三，身材合格却无支援返场，五费卡位留给要塞更优。 |

---

## 🟢 三、 【翠绿 (Green) 卡组】单卡战力与构筑指南
> **战术核心**：法力跳费 · 质量成长 · 终结大哥  
> **候选牌池**：27 张Green专属卡 + 6 张双色协同卡 + 6 张中立通用卡（共 39 张候选，择优遴选 30 张入套）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | 属性/数值 | **综合评分** | 梯队 | **携带率** | **胜率贡献 (ΔWR)** | 推荐配置 | 实战构筑解析 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **藤蔓突袭者** | Green专属 | 4费 | MINION | DP:4 `RUSH` | **97.7** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.1%</span> | 3 张 (核心满编) | 四费突袭解场兼压血，夺回先手，满编提升中期压制力。 |
| 2 | **红绿共生** | Red/Green双色 | 5费 | MINION | DP:5 `RUSH,DEATH_DRAW_1` | **97.4** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.1%</span> | 3 张 (核心满编) | 五费突袭解场并亡语补牌，中期核心，满编保证压制续航。 |
| 3 | **灭世翡翠巨龙** | Green专属 | 10费 | MINION | DP:4 `RUSH,BONUS_SCORE_1` | **96.4** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.9%</span> | 3 张 (核心满编) | 终结核心，突袭夺分兼斩杀，跳费顺利时满编确保上手。 |
| 4 | **剧毒花** | Green专属 | 2费 | MINION | DP:2 `DEGRADE_2` | **94.5** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.6%</span> | 3 张 (核心满编) | 低费减益压制对手随从，节奏顺滑且干扰强，满编稳定上手。 |
| 5 | **芽苗祭司** | Green专属 | 4费 | MINION | DP:3 `RAMP_1` | **91.1** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.1%</span> | 3 张 (核心满编) | 四费站场同时涨费，衔接高费回合，是跳费体系核心满编。 |
| 6 | **佣兵斥候** | 中立通用 | 2费 | MINION | DP:1 `RUSH` | **90.0** | **S** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.9%</span> | 3 张 (核心满编) | 低费突袭解小随从，保跳费回合，满编抢节奏。 |
| 7 | **翠绿幼苗** | Green专属 | 1费 | MINION | DP:1 `FORTIFY_1` | **88.7** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.7%</span> | 2 张 (主力配置) | 一费早期站场吸收伤害，保护跳费随从，主力两张润滑。 |
| 8 | **商人** | 中立通用 | 2费 | MINION | DP:2 `DRAW_1` | **88.1** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.6%</span> | 2 张 (主力配置) | 二费站场并补牌，缓解手牌消耗，两张维持跳费链稳定。 |
| 9 | **训练假人** | 中立通用 | 1费 | MINION | DP:1 | **87.8** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.5%</span> | 2 张 (主力配置) | 一费吸收伤害，保护跳费随从，前期垫节奏的主力屏障。 |
| 10 | **萌芽跳费使** | Green专属 | 3费 | MINION | DP:3 `RAMP_1` | **86.2** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.3%</span> | 2 张 (主力配置) | 三费跳费并站场，提速关键，两张保证上手且不卡高费。 |
| 11 | **树人** | Green专属 | 3费 | MINION | DP:2 `FORTIFY_1` | **84.0** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.9%</span> | 2 张 (主力配置) | 三费两攻带加固，能换小随从并吸收伤害，主力二张稳前期。 |
| 12 | **翠绿哨兵** | Green专属 | 3费 | MINION | DP:2 `FORTIFY_2,DEATH_DRAW_1` | **81.2** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.5%</span> | 2 张 (主力配置) | 亡语补牌且带加固，交换后不亏手牌，两张稳住中期资源。 |
| 13 | **翡翠幼龙** | Green专属 | 7费 | MINION | DP:4 `RUSH,DEATH_MANA_1` | **79.1** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +1.1%</span> | 1 张 (按需携带) | 突袭解场并回费，特定对局可作过渡，一张按需补曲线。 |
| 14 | **翠绿驻防藤蔓** | Green专属 | 2费 | MINION | DP:2 `FORTIFY_1` | **57.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.3%</span> | 0 张 (暂不推荐) | 二费加固随从质量偏低，前期更需跳费或过牌，故舍弃。 |
| 15 | **雇佣兵** | 中立通用 | 3费 | MINION | DP:5 | **57.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.4%</span> | 0 张 (暂不推荐) | 三费高攻白板，无词条易被解，争夺节奏不如突袭随从。 |
| 16 | **蓝绿潮涌** | Blue/Green双色 | 3费 | SPELL | 攻0/防0 `DRAW_2` | **56.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.4%</span> | 0 张 (暂不推荐) | 三费过两张不站场，跳费体系更需随从占场，故不推荐。 |
| 17 | **翠绿萌芽** | Green专属 | 3费 | SPELL | 攻0/防0 `RAMP_1` | **55.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.6%</span> | 0 张 (暂不推荐) | 单独跳费不站场，前期亏节奏且卡位紧，不列入构筑。 |
| 18 | **林地衍生兽** | Green专属 | 4费 | MINION | DP:3 `SPAWN_1_1` | **55.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.6%</span> | 0 张 (暂不推荐) | 四费铺场效率一般，衍生物难影响战局，不如突袭争夺先手。 |
| 19 | **翠绿守护** | Green专属 | 3费 | SPELL | 攻0/防5 | **55.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.7%</span> | 0 张 (暂不推荐) | 三费仅五点防御，不能解场也不能返场，防守端卡位过剩。 |
| 20 | **翠绿滋养** | Green专属 | 3费 | SPELL | 攻0/防0 `RAMP_1,DRAW_1` | **54.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.7%</span> | 0 张 (暂不推荐) | 三费跳费加过牌看似全能，当回合让节奏，高费前易崩。 |
| 21 | **酒馆密账** | 中立通用 | 4费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **54.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.8%</span> | 0 张 (暂不推荐) | 四费过二弃一，手牌质量受损，节奏回合不适合。 |
| 22 | **翠绿跳费者** | Green专属 | 4费 | MINION | DP:3 `RAMP_1` | **54.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.8%</span> | 0 张 (暂不推荐) | 同费同效重复，卡组只需一种跳费随从，芽苗祭司已占满。 |
| 23 | **翠绿守林人** | Green专属 | 3费 | MINION | DP:3 `FORTIFY_2` | **53.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.9%</span> | 0 张 (暂不推荐) | 三费纯防守随从，交换尚可但缺乏跳费，拖慢终结计划。 |
| 24 | **荆棘缠绕** | Green专属 | 2费 | SPELL | 攻2/防0 | **53.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -3.0%</span> | 0 张 (暂不推荐) | 二费打二效率普通，解不掉关键随从，还挤占跳费与过牌。 |
| 25 | **拾荒盾卫** | 中立通用 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **52.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.1%</span> | 0 张 (暂不推荐) | 亡语过牌尚可，但三费身材防守一般，卡位让给跳费。 |
| 26 | **森林之狼** | Green专属 | 4费 | MINION | DP:4 | **52.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.1%</span> | 0 张 (暂不推荐) | 四费白板身材平庸，既无突袭也无成长，竞争不过同费曲线。 |
| 27 | **狂暴生长** | Green专属 | 3费 | SPELL | 攻0/防3 | **52.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 纯防御法术，无法返场或跳费，主动体系里没有空位。 |
| 28 | **自然生长** | Green专属 | 3费 | SPELL | 攻0/防0 `RAMP_1` | **51.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 三费纯跳费不站场，被随从跳费完全压制，不推荐。 |
| 29 | **古树庇护** | Green专属 | 3费 | SPELL | 攻0/防5 `FORTIFY_1` | **51.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.3%</span> | 0 张 (暂不推荐) | 防御与加固叠加仍不解场，无法应对突袭，构筑价值低。 |
| 30 | **世界树恩泽** | Green专属 | 5费 | SPELL | 攻0/防3 `RAMP_1` | **50.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0 张 (暂不推荐) | 五费才跳费太慢，防御属性也不解场，拖累关键回合。 |
| 31 | **翠绿资源滋长** | Green专属 | 4费 | SPELL | 攻0/防0 `RAMP_1,DRAW_1` | **50.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0 张 (暂不推荐) | 四费跳费过牌太迟，关键回合让出主动权，不如低费组件。 |
| 32 | **蓝绿潮汐织法者** | Blue/Green双色 | 5费 | MINION | DP:4 `SUPPORT_ATK_1,RAMP_1` | **50.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 五费跳费兼辅助，身材偏弱且当回合让节奏，难进主力。 |
| 33 | **蓝绿潮汐者** | Blue/Green双色 | 5费 | MINION | DP:5 `DEATH_MANA_2,FORTIFY_2` | **49.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 五费亡语回费太慢，防御虽高难返场，跳费已有优质曲线。 |
| 34 | **红绿共生体** | Red/Green双色 | 6费 | MINION | DP:6 `SUPPORT_ATK_2,FORTIFY_2` | **49.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.6%</span> | 0 张 (暂不推荐) | 六费辅助攻击无突袭，需场面配合，不如直接高攻终结。 |
| 35 | **红绿古树桥梁** | Red/Green双色 | 6费 | MINION | DP:6 `FORTIFY_2,DEATH_DRAW_1` | **49.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.6%</span> | 0 张 (暂不推荐) | 六费防守亡语过牌，节奏偏慢，与跳费抢终结思路冲突。 |
| 36 | **翡翠巨熊** | Green专属 | 6费 | MINION | DP:6 `FORTIFY_2` | **48.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.7%</span> | 0 张 (暂不推荐) | 六费仅加固无突袭，被解后节奏大亏，不如低费跳费组件。 |
| 37 | **古树智者** | Green专属 | 6费 | MINION | DP:6 `DEATH_MANA_2` | **48.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.7%</span> | 0 张 (暂不推荐) | 亡语回费需先阵亡，六费节奏太慢，跳费链已足够。 |
| 38 | **萌芽巨兽** | Green专属 | 7费 | MINION | DP:7 `FORTIFY_3` | **48.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 七费高防无突袭，易被解场法术处理，难承担终结任务。 |
| 39 | **远古巨树** | Green专属 | 7费 | MINION | DP:8 `FORTIFY_2` | **48.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 高费笨重缺突袭，登场前易被解，不如直接拍终结巨龙。 |

---

## ⚪ 四、 【中立 (Neutral) 卡牌】全阵营适配性与战术表现分析
> **机制说明**：同一张中立卡在快攻、控制、跳费等不同战术体系下具有截然不同的战术价值与契合度。

| 中立卡名称 | 费用 | 类型 | 🔴 赤红卡组评分 | 🔵 蔚蓝卡组评分 | 🟢 翠绿卡组评分 | 最佳契合卡组 | 跨阵营战术定位 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **商人** | 2费 | MINION | **94.8** | **86.5** | **88.1** | **赤红 (快攻)** | 多体系通用的高质量拼图 |
| **佣兵斥候** | 2费 | MINION | **90.0** | **89.7** | **90.0** | **赤红 (快攻)** | 多体系通用的高质量拼图 |
| **训练假人** | 1费 | MINION | **57.5** | **58.4** | **87.8** | **翠绿 (跳费)** | 偏向翠绿 (跳费)体系的针对性组件 |
| **拾荒盾卫** | 3费 | MINION | **55.9** | **54.3** | **52.7** | **赤红 (快攻)** | 特定战局下的可选备编卡 |
| **雇佣兵** | 3费 | MINION | **54.3** | **84.0** | **57.2** | **蔚蓝 (防守)** | 偏向蔚蓝 (防守)体系的针对性组件 |
| **酒馆密账** | 4费 | SPELL | **51.8** | **53.4** | **54.6** | **翠绿 (跳费)** | 特定战局下的可选备编卡 |
