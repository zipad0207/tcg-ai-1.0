# 🏆 TCG 卡牌战力评级与构筑指南（阵营分色专榜）

> **构筑规则说明**：在 TCG 标准规则中，各阵营（赤红/蔚蓝/翠绿）由**阵营专属卡 + 双色协同卡 + 中立通用卡**构筑。本指南基于 PPO 强化学习智能体在对战环境中的实战数据，综合**【卡组携带率】**与**【局势胜率贡献 (ΔWR)】**两大维度，按阵营分色独立建榜，提供客观、严谨的构筑参考与单卡解析。

---

## 🔴 一、 【赤红 (Red) 卡组】单卡战力与构筑指南
> **战术核心**：快攻压制 · 牺牲协同 · 节奏斩杀  
> **候选牌池**：17 张Red专属卡 + 2 张双色协同卡 + 6 张中立通用卡（共 25 张候选，择优遴选 30 张入套）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | 属性/数值 | **综合评分** | 梯队 | **携带率** | **胜率贡献 (ΔWR)** | 推荐配置 | 实战构筑解析 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **赤红突破手** | Red专属 | 8费 | MINION | DP:1 `RUSH,BONUS_SCORE_1` | **97.0** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.0%</span> | 3 张 (核心满编) | 八费突袭抢分，落场即交换或压血，满编提高终结手段上手率。 |
| 2 | **赤红掠袭者** | Red专属 | 9费 | MINION | DP:1 `RUSH,BONUS_SCORE_1` | **96.5** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.9%</span> | 3 张 (核心满编) | 九费突袭带额外分，终结与补分兼备，满编保证高费回合稳定上手。 |
| 3 | **红色小队长** | Red专属 | 3费 | MINION | DP:3 `SPAWN_1_1` | **90.0** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.9%</span> | 3 张 (核心满编) | 三费站场并召一杠一，横向展开极快，是牺牲与抢血核心，满编保证节奏。 |
| 4 | **赤红突击手** | Red专属 | 2费 | MINION | DP:1 `DEATH_DRAW_1` | **88.0** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.6%</span> | 2 张 (主力配置) | 两费死亡过牌，交换后不亏手牌，适合铺场链衔接；两张保证前期密度。 |
| 5 | **集结号手** | Red专属 | 2费 | MINION | DP:1 `SPAWN_1_1` | **87.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.5%</span> | 2 张 (主力配置) | 两费铺两个频率，配合牺牲与加攻很顺，两张补曲线且不抢核心位。 |
| 6 | **训练假人** | 中立通用 | 1费 | MINION | DP:1 | **87.0** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.4%</span> | 2 张 (主力配置) | 一费低身材，前期吸收伤害或当牺牲材料，两张补足低费密度。 |
| 7 | **赤红新兵** | Red专属 | 1费 | MINION | DP:1 | **86.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.3%</span> | 2 张 (主力配置) | 一费生物填曲线，可被牺牲或支援，起手留一张，两张减少后期弱抽。 |
| 8 | **牺牲祭师** | Red专属 | 3费 | MINION | DP:3 `SACRIFICE_1_KILL_1,DEATH_DRAW_1` | **86.0** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.2%</span> | 2 张 (主力配置) | 三费牺牲解场并死亡过牌，功能复合，两张支撑中段节奏与资源。 |
| 9 | **拾荒盾卫** | 中立通用 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **83.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.8%</span> | 2 张 (主力配置) | 三费防御并死亡过牌，可护住关键生物，两张增强交换与续航。 |
| 10 | **商人** | 中立通用 | 2费 | MINION | DP:2 `DRAW_1` | **83.0** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.8%</span> | 2 张 (主力配置) | 两费站场补一张，衔接曲线并续航，两张足够避免手牌溢出。 |
| 11 | **雇佣兵** | 中立通用 | 3费 | MINION | DP:5 | **81.0** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.4%</span> | 2 张 (主力配置) | 三费五血站场，交换效率高，两张提供扎实中段阻挡与牺牲素材。 |
| 12 | **佣兵斥候** | 中立通用 | 2费 | MINION | DP:1 `RUSH` | **80.0** | **A** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +1.3%</span> | 2 张 (主力配置) | 两费突袭即时解小怪，夺回先手并填补二费，两张保证上手。 |
| 13 | **裂甲掷斧手** | Red专属 | 5费 | MINION | DP:1 `RUSH,DEGRADE_1` | **79.5** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +1.2%</span> | 1 张 (按需携带) | 五费突袭削甲，能处理难缠防御单位，针对环境挂一张，不宜多带。 |
| 14 | **赤蓝交织者** | Red/Blue双色 | 4费 | MINION | DP:4 `SUPPORT_ATK_1` | **79.5** | **B** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.2%</span> | 2 张 (按需携带) | 四费为友军加攻，铺场后增伤明显，环境慢时带两张，否则让位。 |
| 15 | **切割者** | Red专属 | 2费 | MINION | DP:2 `DEGRADE_1` | **72.5** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +0.1%</span> | 1 张 (按需携带) | 两费削弱随从可磨防守，针对高攻单位，环境合适时挂一张即可。 |
| 16 | **赤红献祭** | Red专属 | 2费 | SPELL | 攻0/防0 `SACRIFICE_1_KILL_1` | **55.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.6%</span> | 0 张 (暂不推荐) | 两费单解需牺牲生物，破坏铺场联动，快攻节奏反而受损。 |
| 17 | **血祭爆燃** | Red专属 | 3费 | SPELL | 攻3/防0 `SACRIFICE_1_KILL_1,ATTACK_ONLY` | **55.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.7%</span> | 0 张 (暂不推荐) | 三费牺牲打三，解场尚可但拖节奏，快攻更愿保留生物持续输出。 |
| 18 | **射线** | Red专属 | 2费 | SPELL | 攻2/防0 | **54.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.8%</span> | 0 张 (暂不推荐) | 两费打二效率平庸，缺乏解场或斩杀增益，直伤卡位更需功能性。 |
| 19 | **自爆** | Red专属 | 2费 | SPELL | 攻0/防0 `SACRIFICE_1_KILL_1,ATTACK_ONLY` | **54.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.9%</span> | 0 张 (暂不推荐) | 两费牺牲换单解，节奏亏损明显，快攻不愿让出场面，故不进构筑。 |
| 20 | **大块头** | Red专属 | 6费 | MINION | DP:10 `DISCARD_2` | **51.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.3%</span> | 0 张 (暂不推荐) | 六费大身材却需弃两张，快攻手牌珍贵，弃牌代价压过站场价值。 |
| 21 | **掠夺者** | Red专属 | 6费 | MINION | DP:4 `BONUS_SCORE_1` | **50.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0 张 (暂不推荐) | 六费仅四点身材，额外得分太慢，快攻六费应已终结或转伤。 |
| 22 | **赤红战意** | Red专属 | 5费 | SPELL | 攻0/防0 `TEMP_MANA_2` | **50.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 五费换临时两费，费用倒挂难滚雪球，快攻不需此类法术。 |
| 23 | **红绿共生体** | Red/Green双色 | 6费 | MINION | DP:6 `SUPPORT_ATK_2,FORTIFY_2` | **49.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.7%</span> | 0 张 (暂不推荐) | 六费辅助与防御虽扎实，但快攻六费要伤害或终结，此卡太慢。 |
| 24 | **酒馆密账** | 中立通用 | 4费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **48.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 四费抽二弃一，节奏太慢且弃牌伤资源，快攻不需此过牌方式。 |
| 25 | **破阵狂徒** | Red专属 | 8费 | MINION | DP:2 `DEGRADE_1` | **48.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 八费只提供削弱与低攻，费用过重，快攻无法容忍此等后期空档。 |

---

## 🔵 二、 【蔚蓝 (Blue) 卡组】单卡战力与构筑指南
> **战术核心**：防守反击 · 固守护盾 · 资源消耗  
> **候选牌池**：17 张Blue专属卡 + 2 张双色协同卡 + 6 张中立通用卡（共 25 张候选，择优遴选 30 张入套）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | 属性/数值 | **综合评分** | 梯队 | **携带率** | **胜率贡献 (ΔWR)** | 推荐配置 | 实战构筑解析 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **佣兵斥候** | 中立通用 | 2费 | MINION | DP:1 `RUSH` | **90.0** | **S** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.9%</span> | 3 张 (核心满编) | 两费突袭，解场夺先手；弥补防守被动，满编抢节奏。 |
| 2 | **霜盾见习官** | Blue专属 | 2费 | MINION | DP:1 `FORTIFY_1,SUPPORT_ATK_1` | **89.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 2 张 (主力配置) | 两费双向功能，固守护场同时加攻；身材偏脆，两张即可。 |
| 3 | **冰封禁制** | Blue专属 | 2费 | SPELL | 攻4/防2 | **87.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.5%</span> | 2 张 (主力配置) | 两费打四附防二，解场兼保脸；中期抢节奏，两张稳。 |
| 4 | **盾兵** | Blue专属 | 1费 | MINION | DP:1 `FORTIFY_1` | **87.0** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.4%</span> | 2 张 (主力配置) | 一费固守吸收首轮伤害，护住后续展开；中期抽到偏弱，带两张。 |
| 5 | **蔚蓝要塞** | Blue专属 | 6费 | MINION | DP:6 `FORTIFY_3,SUPPORT_ATK_1` | **86.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.3%</span> | 3 张 (主力配置) | 六费终端，固守三加攻一，落地即压制；满编保后期质量。 |
| 6 | **蔚蓝智慧** | Blue专属 | 3费 | SPELL | 攻0/防0 `DRAW_2,TEMP_MANA_1` | **85.0** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.1%</span> | 2 张 (主力配置) | 三费抽二返一费，滤牌提速；中期找要塞，两张稳定。 |
| 7 | **弓箭手** | Blue专属 | 3费 | MINION | DP:3 `SUPPORT_ATK_1` | **84.0** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.9%</span> | 2 张 (主力配置) | 三费支援加攻，后排稳定输出；身材普通，两张开路足够。 |
| 8 | **商人** | 中立通用 | 2费 | MINION | DP:2 `DRAW_1` | **83.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.8%</span> | 2 张 (主力配置) | 两费二防过一，站场兼润滑；前期不亏手牌，两张主力。 |
| 9 | **拾荒盾卫** | 中立通用 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **83.0** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.8%</span> | 2 张 (主力配置) | 三费固守一亡语抽一，换牌不亏；稳住前期，两张合适。 |
| 10 | **蓝绿潮汐者** | Blue/Green双色 | 5费 | MINION | DP:5 `DEATH_MANA_2,FORTIFY_2` | **82.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.7%</span> | 2 张 (主力配置) | 五费固守二，亡语回两费；衔接高费终端，两张保证续航。 |
| 11 | **蔚蓝守卫** | Blue专属 | 2费 | MINION | DP:2 `FORTIFY_2` | **82.0** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.6%</span> | 2 张 (主力配置) | 两费固守二，前期最稳防线；保护后排并过渡，两张主力。 |
| 12 | **雇佣兵** | 中立通用 | 3费 | MINION | DP:5 | **81.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.5%</span> | 2 张 (主力配置) | 三费五防裸墙，吸收伤害扎实；无词条但替要塞拖回合，两张。 |
| 13 | **训练假人** | 中立通用 | 1费 | MINION | DP:1 | **79.0** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +1.1%</span> | 1 张 (按需携带) | 一费一防，垫刀拖延；功能有限，仅作低费曲线补位。 |
| 14 | **防御！** | Blue专属 | 1费 | SPELL | 攻0/防2 | **78.5** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +1.0%</span> | 1 张 (按需携带) | 一费防两伤，关键回合保随从或脸；非主计划，按需一张。 |
| 15 | **蔚蓝卫士** | Blue专属 | 2费 | MINION | DP:2 `FORTIFY_1` | **76.5** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +0.7%</span> | 1 张 (按需携带) | 低费固守墙，可补前期曲线；同费蔚蓝守卫更硬，仅按需一张。 |
| 16 | **壁垒工匠** | Blue专属 | 3费 | MINION | DP:3 `FORTIFY_2` | **75.5** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +0.6%</span> | 1 张 (按需携带) | 三费固守二，防守及格；同费竞争多，仅作针对性一张。 |
| 17 | **蔚蓝破甲师** | Blue专属 | 3费 | MINION | DP:3 `DEGRADE_1` | **71.0** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ -0.2%</span> | 1 张 (按需携带) | 三费减益一，针对高防大哥；环境对策，一张足够。 |
| 18 | **蔚蓝冲击** | Blue专属 | 2费 | SPELL | 攻3/防0 | **58.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.2%</span> | 0 张 (暂不推荐) | 两费打三，直伤效率普通；防守卡组不缺解，不带。 |
| 19 | **寒晶护壁** | Blue专属 | 2费 | SPELL | 攻0/防2 `DRAW_1` | **56.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.6%</span> | 0 张 (暂不推荐) | 两费防二过一，效率尚可；但防法术卡位紧，暂不带。 |
| 20 | **蔚蓝盾卫** | Blue专属 | 4费 | MINION | DP:4 `FORTIFY_3,SUPPORT_ATK_1` | **54.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.8%</span> | 0 张 (暂不推荐) | 四费固守三加攻一，单卡不弱；四费曲线拥挤，优先雇佣兵。 |
| 21 | **酒馆密账** | 中立通用 | 4费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **50.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0 张 (暂不推荐) | 四费抽二弃一，节奏亏损；防守卡组要留手牌，不带。 |
| 22 | **赤蓝交织者** | Red/Blue双色 | 4费 | MINION | DP:4 `SUPPORT_ATK_1` | **50.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 四费支援加一，输出增益低；同费有雇佣兵撑场，不带。 |
| 23 | **火铳手** | Blue专属 | 3费 | MINION | DP:4 `SUPPORT_ATK_2` | **49.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.6%</span> | 0 张 (暂不推荐) | 三费支援加二攻，输出不错；但三费曲线拥挤，支援位有弓箭手，暂不投入。 |
| 24 | **藤甲兵** | Blue专属 | 4费 | MINION | DP:4 `FORTIFY_2` | **49.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.7%</span> | 0 张 (暂不推荐) | 四费四防固守二，数值平庸；四费竞争激烈，卡位不够。 |
| 25 | **石像鬼** | Blue专属 | 6费 | MINION | DP:8 | **48.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 六费八防无词条，节奏太慢；高费位要塞更强，故不带。 |

---

## 🟢 三、 【翠绿 (Green) 卡组】单卡战力与构筑指南
> **战术核心**：法力跳费 · 质量成长 · 终结大哥  
> **候选牌池**：17 张Green专属卡 + 2 张双色协同卡 + 6 张中立通用卡（共 25 张候选，择优遴选 30 张入套）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | 属性/数值 | **综合评分** | 梯队 | **携带率** | **胜率贡献 (ΔWR)** | 推荐配置 | 实战构筑解析 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **灭世翡翠巨龙** | Green专属 | 10费 | MINION | DP:8 `RUSH,BONUS_SCORE_1` | **98.0** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.2%</span> | 3 张 (核心满编) | 十费突袭终结，附加得分拉大差距；跳费后核心制胜点，满编提高上手与续航。 |
| 2 | **翡翠幼龙** | Green专属 | 5费 | MINION | DP:4 `RUSH,DEATH_MANA_1` | **97.0** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.0%</span> | 3 张 (核心满编) | 五费突袭解场，亡语返一费；兼顾节奏与跳费链，满编保证中期不断档。 |
| 3 | **芽苗祭司** | Green专属 | 4费 | MINION | DP:3 `RAMP_1` | **93.0** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.4%</span> | 3 张 (核心满编) | 四费站场并跳一费，衔接高费核心；前期必争节奏点，满编保证按时扩张法力。 |
| 4 | **佣兵斥候** | 中立通用 | 2费 | MINION | DP:1 `RUSH` | **89.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 2 张 (主力配置) | 两费突袭补刀，解掉一血随从并夺回先手；身材偏弱，两张作功能性解场。 |
| 5 | **剧毒花** | Green专属 | 2费 | MINION | DP:2 `DEGRADE_2` | **88.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.6%</span> | 2 张 (主力配置) | 两费削减对手两点攻，针对快攻前锋；可换掉小随从，主力配置两张稳前期。 |
| 6 | **商人** | 中立通用 | 2费 | MINION | DP:2 `DRAW_1` | **87.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.5%</span> | 2 张 (主力配置) | 两费站场并抽一，润滑跳费曲线；前期不亏手牌，两张维持展开与资源平衡。 |
| 7 | **训练假人** | 中立通用 | 1费 | MINION | DP:1 | **87.0** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.4%</span> | 2 张 (主力配置) | 一费垫场吸收攻击，保护跳费随从；功能单一，两张用于前期过渡即可。 |
| 8 | **翠绿哨兵** | Green专属 | 2费 | MINION | DP:2 `FORTIFY_2,DEATH_DRAW_1` | **86.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.3%</span> | 2 张 (主力配置) | 两费双层护甲，亡语补一张；前期稳血兼过牌，主力两张强化防守与资源。 |
| 9 | **翠绿幼苗** | Green专属 | 1费 | MINION | DP:1 `FORTIFY_1` | **86.0** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.2%</span> | 2 张 (主力配置) | 一费早期站场带护甲，吸收快攻伤害；跳费前低费曲线，两张够用避免卡手。 |
| 10 | **雇佣兵** | 中立通用 | 3费 | MINION | DP:5 | **84.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.0%</span> | 2 张 (主力配置) | 三费五攻身材扎实，前期交换占优；无词条但能扛线，两张补足低费站场压力。 |
| 11 | **拾荒盾卫** | 中立通用 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **83.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.8%</span> | 2 张 (主力配置) | 三费护甲加亡语抽牌，防守同时补资源；适合拖入高费，两张稳定中前期。 |
| 12 | **翠绿滋养** | Green专属 | 3费 | SPELL | 攻0/防0 `RAMP_1,DRAW_1` | **82.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.7%</span> | 2 张 (主力配置) | 三费跳一费并抽一，加速与补牌兼备；关键过渡牌，两张保证中期资源不断。 |
| 13 | **酒馆密账** | 中立通用 | 4费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **80.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.4%</span> | 2 张 (主力配置) | 四费抽二弃一，低费法术加速过牌；跳费后快速找核心，弃牌代价可接受。 |
| 14 | **翠绿萌芽** | Green专属 | 3费 | SPELL | 攻0/防0 `RAMP_1` | **71.5** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ -0.1%</span> | 1 张 (按需携带) | 三费跳一费，节奏偏慢；特定大费对局作额外加速，常规卡位只留一张。 |
| 15 | **荆棘缠绕** | Green专属 | 2费 | SPELL | 攻2/防0 | **58.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.2%</span> | 0 张 (暂不推荐) | 两费打二效率普通，仅能处理小随从；剧毒花与突袭已够解场，无需额外直伤。 |
| 16 | **树人** | Green专属 | 3费 | MINION | DP:2 `FORTIFY_1` | **55.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.6%</span> | 0 张 (暂不推荐) | 三费二攻仅带护甲，站场压制不足；跳费体系更需即时解场或抽牌，无卡位。 |
| 17 | **翠绿跳费者** | Green专属 | 4费 | MINION | DP:3 `RAMP_1` | **54.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.9%</span> | 0 张 (暂不推荐) | 四费三攻跳一费，身材亏损且当回合无压制；芽苗祭司更稳，卡位不留。 |
| 18 | **蓝绿潮汐者** | Blue/Green双色 | 5费 | MINION | DP:5 `DEATH_MANA_2,FORTIFY_2` | **53.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -3.0%</span> | 0 张 (暂不推荐) | 五费亡语返两费，但死亡时机不可控；同费翡翠幼龙能即时解场，优先级更高。 |
| 19 | **森林之狼** | Green专属 | 4费 | MINION | DP:4 | **52.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 四费四攻白板，交换不赚节奏；无法返场也抢不过突袭，卡位优先跳费。 |
| 20 | **红绿共生体** | Red/Green双色 | 6费 | MINION | DP:6 `SUPPORT_ATK_2,FORTIFY_2` | **51.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0 张 (暂不推荐) | 六费光环加攻带护甲，偏向铺场；本阵营依赖跳费突袭，配合度低不入选。 |
| 21 | **翠绿守护** | Green专属 | 3费 | SPELL | 攻0/防5 | **50.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 三费五防只护脸，无法影响场面；跳费后需要进攻压制，纯防御牌易被穿透。 |
| 22 | **远古巨树** | Green专属 | 7费 | MINION | DP:8 `FORTIFY_2` | **49.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.6%</span> | 0 张 (暂不推荐) | 七费八攻带护甲，身材达标却无突袭；后期需立即终结，它留给对手应对回合。 |
| 23 | **翡翠巨熊** | Green专属 | 6费 | MINION | DP:6 `FORTIFY_2` | **49.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.7%</span> | 0 张 (暂不推荐) | 六费六攻双护甲，防守尚可但缺突袭；同费不如跳费或预备终结，难扭转战局。 |
| 24 | **狂暴生长** | Green专属 | 3费 | SPELL | 攻0/防3 | **48.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 三费加三点防却无跳费抽牌，被动挨打；跳费嫌慢，快攻抗性不如低费守护。 |
| 25 | **世界树恩泽** | Green专属 | 5费 | SPELL | 攻0/防3 `RAMP_1` | **48.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 五费才跳一费且加防，节奏滞后；高费回合更需直接施压，此牌拖慢核心落地。 |

---

## ⚪ 四、 【中立 (Neutral) 卡牌】全阵营适配性与战术表现分析
> **机制说明**：同一张中立卡在快攻、控制、跳费等不同战术体系下具有截然不同的战术价值与契合度。

| 中立卡名称 | 费用 | 类型 | 🔴 赤红卡组评分 | 🔵 蔚蓝卡组评分 | 🟢 翠绿卡组评分 | 最佳契合卡组 | 跨阵营战术定位 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **训练假人** | 1费 | MINION | **87.0** | **79.0** | **87.0** | **赤红 (快攻)** | 多体系通用的高质量拼图 |
| **拾荒盾卫** | 3费 | MINION | **83.5** | **83.0** | **83.5** | **赤红 (快攻)** | 多体系通用的高质量拼图 |
| **商人** | 2费 | MINION | **83.0** | **83.5** | **87.5** | **翠绿 (跳费)** | 多体系通用的高质量拼图 |
| **雇佣兵** | 3费 | MINION | **81.0** | **81.5** | **84.5** | **翠绿 (跳费)** | 多体系通用的高质量拼图 |
| **佣兵斥候** | 2费 | MINION | **80.0** | **90.0** | **89.5** | **蔚蓝 (防守)** | 多体系通用的高质量拼图 |
| **酒馆密账** | 4费 | SPELL | **48.5** | **50.5** | **80.5** | **翠绿 (跳费)** | 偏向翠绿 (跳费)体系的针对性组件 |
