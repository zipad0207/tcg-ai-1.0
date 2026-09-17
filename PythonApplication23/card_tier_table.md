# 🏆 TCG 卡牌战力评级与构筑指南（阵营分色专榜）

> **构筑规则说明**：在 TCG 标准规则中，各阵营（赤红/蔚蓝/翠绿）由**阵营专属卡 + 中立通用卡**构筑。本指南基于 PPO 强化学习智能体（`card_ppo_model_tuned.pth`）在对战环境中的实战数据，综合**【卡组携带率】**与**【局势胜率贡献 (ΔWR)】**两大维度，按阵营分色独立建榜，提供客观、严谨的构筑参考与单卡解析。

---

## 🔴 一、 【赤红 (Red) 卡组】单卡战力与构筑指南
> **战术核心**：快攻压制 · 牺牲协同 · 节奏斩杀  
> **候选牌池**：12 张赤红专属卡 + 6 张中立通用卡（共 18 张候选，择优遴选 30 张入套）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | 属性/数值 | **综合评分** | 梯队 | **携带率** | **胜率贡献 (ΔWR)** | 推荐配置 | 实战构筑解析 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **佣兵斥候** | 中立 | 2费 | MINION | DP:1 `RUSH,DRAW_1` | **95.9** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.8%</span> | 3 张 (核心满编) | 二费突袭抽一，解场补牌一体，节奏极佳，满编保证前期轮转。 |
| 2 | **裂甲掷斧手** | Red专属 | 4费 | MINION | DP:1 `RUSH,DEGRADE_1` | **95.2** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.7%</span> | 3 张 (核心满编) | 四费突袭降攻，进场处理威胁并抢回先手，满编支撑中期交换。 |
| 3 | **商人** | 中立 | 2费 | MINION | DP:2 `DRAW_1` | **94.5** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.6%</span> | 3 张 (核心满编) | 二费二攻抽一，站场同时润滑手牌，快攻续航关键，满编。 |
| 4 | **酒馆密账** | 中立 | 1费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **93.8** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.5%</span> | 3 张 (核心满编) | 一费抽二弃一，极速过滤手牌，弃牌可喂牺牲，满编不减。 |
| 5 | **赤红掠袭者** | Red专属 | 5费 | MINION | DP:3 `RUSH,BONUS_SCORE_1` | **93.1** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.4%</span> | 3 张 (核心满编) | 五费突袭带得分，解场抢血兼顾计分，满编作为中后期核心。 |
| 6 | **破阵狂徒** | Red专属 | 4费 | MINION | DP:2 `RUSH,DEGRADE_1` | **92.4** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.3%</span> | 3 张 (核心满编) | 四费突袭二攻降攻，解场同时站场，核心节奏点，满编保证上手。 |
| 7 | **训练假人** | 中立 | 1费 | MINION | DP:1 | **90.9** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.0%</span> | 3 张 (核心满编) | 一费一攻，填补一费曲线并供牺牲，满编保证开局展开。 |
| 8 | **掠夺者** | Red专属 | 6费 | MINION | DP:4 `BONUS_SCORE_1` | **87.4** | **A** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.5%</span> | 3 张 (主力配置) | 六费四攻带额外得分，中期终结比赛，满编确保高费曲线和收官能力。 |
| 9 | **赤红突击手** | Red专属 | 2费 | MINION | DP:1 `DEATH_DRAW_1` | **86.7** | **A** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.4%</span> | 3 张 (主力配置) | 二费一攻死亡抽一，交换后补牌，快攻不断节奏，满编保证前期密度。 |
| 10 | **雇佣兵** | 中立 | 3费 | MINION | DP:5 | **78.0** | **B** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.0%</span> | 2 张 (按需携带) | 三费五攻纯身材，交换力强但无词条，按曲线需求带两张。 |
| 11 | **拾荒盾卫** | 中立 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **73.6** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +0.3%</span> | 1 张 (按需携带) | 三费加固死亡抽一，防守交换稳，但偏慢，仅作一张针对挂件。 |
| 12 | **血祭爆燃** | Red专属 | 3费 | SPELL | 攻3/防0 `SACRIFICE_1_KILL_1,ATTACK_ONLY` | **60.0** | **C** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -1.9%</span> | 0~1 张 (可选备编) | 三费牺牲单位打三，可越墙补伤，但亏卡，仅特定对局挂一。 |
| 13 | **射线** | Red专属 | 2费 | SPELL | 攻2/防0 | **59.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.0%</span> | 0 张 (暂不推荐) | 二费打二直伤，效率低于单位与过牌，抢血卡位紧，通常舍弃。 |
| 14 | **自爆** | Red专属 | 2费 | SPELL | 攻0/防0 `SACRIFICE_1_KILL_1,ATTACK_ONLY` | **58.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.1%</span> | 0 张 (暂不推荐) | 二费牺牲己方单位换单体解，快攻损节奏，解场非本职，不推荐。 |
| 15 | **大块头** | Red专属 | 6费 | MINION | DP:10 `DISCARD_2` | **52.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 六费十防但弃两张手牌，防守型高费，快攻手牌宝贵，完全不合。 |
| 16 | **切割者** | Red专属 | 2费 | MINION | DP:2 `DEGRADE_1` | **51.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.3%</span> | 0 张 (暂不推荐) | 二费二攻附带降攻，身材与效果均不赚，快攻不需此功能，放弃。 |
| 17 | **集结号手** | Red专属 | 2费 | MINION | DP:1 `SPAWN_1_1` | **50.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0 张 (暂不推荐) | 二费一攻生一攻，铺场滞后且身材弱，二费位竞争激烈，不采用。 |
| 18 | **红色小队长** | Red专属 | 3费 | MINION | DP:3 `SPAWN_1_1` | **50.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 三费仅三攻并衍生一攻，节奏偏慢，快攻曲线不需要，暂不投入。 |

---

## 🔵 二、 【蔚蓝 (Blue) 卡组】单卡战力与构筑指南
> **战术核心**：防守反击 · 固守护盾 · 资源消耗  
> **候选牌池**：12 张蔚蓝专属卡 + 6 张中立通用卡（共 18 张候选，择优遴选 30 张入套）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | 属性/数值 | **综合评分** | 梯队 | **携带率** | **胜率贡献 (ΔWR)** | 推荐配置 | 实战构筑解析 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **佣兵斥候** | 中立 | 2费 | MINION | DP:1 `RUSH,DRAW_1` | **97.3** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.0%</span> | 3 张 (核心满编) | 2费突袭过牌，即时解场并保持手牌，抢回先手；核心满编。 |
| 2 | **石像鬼** | Blue专属 | 6费 | MINION | DP:8 | **93.1** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.4%</span> | 3 张 (核心满编) | 6费8防大白板，吸收大量伤害，为中后期大哥争取回合；核心满编。 |
| 3 | **蔚蓝要塞** | Blue专属 | 6费 | MINION | DP:5 `FORTIFY_3,SUPPORT_ATK_1` | **92.4** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.3%</span> | 3 张 (核心满编) | 6费5防固守3并加攻，高质量壁垒，稳固后期并反打；核心满编。 |
| 4 | **拾荒盾卫** | 中立 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **89.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 3 张 (主力配置) | 3费3防固守1，亡语抽牌，防守同时补资源；满编润滑中期。 |
| 5 | **防御！** | Blue专属 | 1费 | SPELL | 攻0/防2 | **88.6** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.7%</span> | 2 张 (主力配置) | 1费加2防，低费保随从与护脸，节奏灵活；两张支撑防守反击。 |
| 6 | **雇佣兵** | 中立 | 3费 | MINION | DP:5 | **88.1** | **A** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.6%</span> | 3 张 (主力配置) | 3费5防，优质肉盾，无词条但身材扎实，满编撑起中期防线。 |
| 7 | **酒馆密账** | 中立 | 1费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **87.9** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.5%</span> | 2 张 (主力配置) | 1费抽二弃一，过滤手牌找关键防守与大哥，两张维持资源。 |
| 8 | **火铳手** | Blue专属 | 3费 | MINION | DP:4 `SUPPORT_ATK_2` | **87.4** | **A** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.5%</span> | 3 张 (主力配置) | 3费4防支援加2攻，攻守兼备，强化交换与直伤；满编稳固中期。 |
| 9 | **寒晶护壁** | Blue专属 | 2费 | SPELL | 攻0/防2 `DRAW_1` | **86.0** | **A** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.2%</span> | 3 张 (主力配置) | 2费加2防并过牌，兼顾防守与手牌润滑，防守体系满编不嫌多。 |
| 10 | **训练假人** | 中立 | 1费 | MINION | DP:1 | **77.2** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +0.8%</span> | 1 张 (按需携带) | 1费1防白板，可拖延快攻或喂给固守交换，按需带一张即可。 |
| 11 | **盾兵** | Blue专属 | 1费 | MINION | DP:1 `FORTIFY_1` | **75.8** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +0.6%</span> | 1 张 (按需携带) | 1费固守盾，能挡一次小伤害，保护关键随从；按需带一张即可。 |
| 12 | **弓箭手** | Blue专属 | 3费 | MINION | DP:3 `SUPPORT_ATK_1` | **72.9** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +0.1%</span> | 1 张 (按需携带) | 3费后排支援加攻，配合防御单位换怪，但自身脆弱，带一张补伤害。 |
| 13 | **商人** | 中立 | 2费 | MINION | DP:2 `DRAW_1` | **72.2** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ 0.0%</span> | 1 张 (按需携带) | 2费2防过一牌，润滑曲线，但身材一般；按需带一张补充资源。 |
| 14 | **蔚蓝卫士** | Blue专属 | 2费 | MINION | DP:2 `FORTIFY_1` | **68.7** | **C** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ -0.5%</span> | 0~1 张 (可选备编) | 2费2防固守1，前期护脸尚可，但后期身材偏弱，仅作备选补位。 |
| 15 | **冰封禁制** | Blue专属 | 3费 | SPELL | 攻2/防2 | **60.0** | **C** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -1.9%</span> | 0~1 张 (可选备编) | 3费2攻2防，灵活解小怪或补防，但效率一般，备编一张针对。 |
| 16 | **霜盾见习官** | Blue专属 | 2费 | MINION | DP:1 `FORTIFY_1,SUPPORT_ATK_1` | **56.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.5%</span> | 0 张 (暂不推荐) | 2费1防双词条，身材太差，固守与加攻收益低，不推荐携带。 |
| 17 | **藤甲兵** | Blue专属 | 4费 | MINION | DP:4 `FORTIFY_2` | **53.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.9%</span> | 0 张 (暂不推荐) | 4费4防固守2，防线扎实但费用偏高，卡位竞争激烈，暂不投入。 |
| 18 | **壁垒工匠** | Blue专属 | 3费 | MINION | DP:3 `FORTIFY_2` | **50.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0 张 (暂不推荐) | 3费3防固守2，防御合格但无返场，易被解，卡位不足暂弃。 |

---

## 🟢 三、 【翠绿 (Green) 卡组】单卡战力与构筑指南
> **战术核心**：法力跳费 · 质量成长 · 终结大哥  
> **候选牌池**：12 张翠绿专属卡 + 6 张中立通用卡（共 18 张候选，择优遴选 30 张入套）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | 属性/数值 | **综合评分** | 梯队 | **携带率** | **胜率贡献 (ΔWR)** | 推荐配置 | 实战构筑解析 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **灭世翡翠巨龙** | Green专属 | 9费 | MINION | DP:10 `RUSH,BONUS_SCORE_1` | **96.6** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.9%</span> | 3 张 (核心满编) | 九费十血突袭终结，返场兼抢分；满编确保跳费后稳定斩杀。 |
| 2 | **佣兵斥候** | 中立 | 2费 | MINION | DP:1 `RUSH,DRAW_1` | **95.9** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.8%</span> | 3 张 (核心满编) | 二费突袭过牌，解小怪同时找跳费；满编抢节奏，前期核心润滑。 |
| 3 | **翡翠幼龙** | Green专属 | 5费 | MINION | DP:3 `RUSH,DEATH_MANA_1` | **95.2** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.7%</span> | 3 张 (核心满编) | 五费突袭解场并回费，衔接九费大哥；满编保证中期节奏不断。 |
| 4 | **雇佣兵** | 中立 | 3费 | MINION | DP:5 | **89.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 3 张 (主力配置) | 三费五血优质前排，吸收伤害掩护跳费；满编保证前期站场。 |
| 5 | **酒馆密账** | 中立 | 1费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **89.3** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 2 张 (主力配置) | 一费过二弃一，筛选跳费与终端；弃牌有风险，二张兼顾润滑与手牌。 |
| 6 | **拾荒盾卫** | 中立 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **88.1** | **A** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.6%</span> | 3 张 (主力配置) | 三费强化墙，死亡补牌；前期阻挡快攻，牺牲后维持资源不断。 |
| 7 | **商人** | 中立 | 2费 | MINION | DP:2 `DRAW_1` | **86.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.3%</span> | 2 张 (主力配置) | 二费过牌站场，找跳费与大哥；身材一般，二张补足资源不挤曲线。 |
| 8 | **树人** | Green专属 | 3费 | MINION | DP:2 `FORTIFY_1` | **85.8** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.2%</span> | 2 张 (主力配置) | 三费防守点，强化后吸收快攻伤害；不抢节奏，二张稳住前期。 |
| 9 | **训练假人** | 中立 | 1费 | MINION | DP:1 | **85.1** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.1%</span> | 2 张 (主力配置) | 一费一血占场，延缓快攻并填充曲线；后期抽到价值低，不宜三张。 |
| 10 | **狂暴生长** | Green专属 | 3费 | SPELL | 攻0/防3 | **82.2** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.6%</span> | 2 张 (主力配置) | 三费三点防，前期挡刀护脸；无额外收益，但能拖到跳费回合。 |
| 11 | **剧毒花** | Green专属 | 2费 | MINION | DP:2 `DEGRADE_2` | **74.4** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +0.4%</span> | 1 张 (按需携带) | 二费降攻针对大怪，拖延对手攻势；环境合适带一张，不必满编。 |
| 12 | **翠绿萌芽** | Green专属 | 2费 | SPELL | 攻0/防0 `RAMP_1` | **73.6** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +0.3%</span> | 1 张 (按需携带) | 低费跳费润滑曲线，早一回合抢出中期大哥；卡位紧，按环境挂一。 |
| 13 | **世界树恩泽** | Green专属 | 4费 | SPELL | 攻0/防3 `RAMP_1` | **72.9** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +0.1%</span> | 1 张 (按需携带) | 四费跳费带三点防，抵挡前期伤害；费用偏高，通常只带一张。 |
| 14 | **翡翠巨熊** | Green专属 | 6费 | MINION | DP:6 `FORTIFY_2` | **68.7** | **C** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ -0.5%</span> | 0~1 张 (可选备编) | 六费防御终端，强化后能挡大生物；进攻性不足，按控制环境备一。 |
| 15 | **芽苗祭司** | Green专属 | 3费 | MINION | DP:3 `RAMP_1` | **68.0** | **C** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ -0.6%</span> | 0~1 张 (可选备编) | 三费站场跳费，衔接中期高费；身材偏弱，易被解则亏节奏。 |
| 16 | **荆棘缠绕** | Green专属 | 2费 | SPELL | 攻2/防0 | **60.0** | **C** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -1.9%</span> | 0~1 张 (可选备编) | 二费两点直伤可补刀或压血，但解场效率有限；仅作备编。 |
| 17 | **森林之狼** | Green专属 | 4费 | MINION | DP:4 | **50.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0 张 (暂不推荐) | 四费四血白板，竞争不过突袭与跳费件；卡位不足时不考虑。 |
| 18 | **远古巨树** | Green专属 | 7费 | MINION | DP:8 `FORTIFY_2` | **49.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.6%</span> | 0 张 (暂不推荐) | 七费八血强化虽厚，却慢于终结需求；不如突袭大哥直接压场。 |

---

## ⚪ 四、 【中立 (Neutral) 卡牌】全阵营适配性与战术表现分析
> **机制说明**：同一张中立卡在快攻、控制、跳费等不同战术体系下具有截然不同的战术价值与契合度。

| 中立卡名称 | 费用 | 类型 | 🔴 赤红卡组评分 | 🔵 蔚蓝卡组评分 | 🟢 翠绿卡组评分 | 最佳契合卡组 | 跨阵营战术定位 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **佣兵斥候** | 2费 | MINION | **95.9** | **97.3** | **95.9** | **蔚蓝 (防守)** | 多体系通用的高质量拼图 |
| **商人** | 2费 | MINION | **94.5** | **72.2** | **86.5** | **赤红 (快攻)** | 偏向赤红 (快攻)体系的针对性组件 |
| **酒馆密账** | 1费 | SPELL | **93.8** | **87.9** | **89.3** | **赤红 (快攻)** | 多体系通用的高质量拼图 |
| **训练假人** | 1费 | MINION | **90.9** | **77.2** | **85.1** | **赤红 (快攻)** | 多体系通用的高质量拼图 |
| **雇佣兵** | 3费 | MINION | **78.0** | **88.1** | **89.5** | **翠绿 (跳费)** | 多体系通用的高质量拼图 |
| **拾荒盾卫** | 3费 | MINION | **73.6** | **89.5** | **88.1** | **蔚蓝 (防守)** | 偏向蔚蓝 (防守)体系的针对性组件 |
