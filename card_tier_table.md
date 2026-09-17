# 🏆 TCG 卡牌战力评级与构筑指南（阵营分色专榜）

> **构筑规则说明**：在 TCG 标准规则中，各阵营（赤红/蔚蓝/翠绿）由**阵营专属卡 + 中立通用卡**构筑。本指南基于 PPO 强化学习智能体（`card_ppo_model_tuned.pth`）在对战环境中的实战数据，综合**【卡组携带率】**与**【局势胜率贡献 (ΔWR)】**两大维度，按阵营分色独立建榜，提供客观、严谨的构筑参考与单卡解析。

---

## 🔴 一、 【赤红 (Red) 卡组】单卡战力与构筑指南
> **战术核心**：快攻压制 · 牺牲协同 · 节奏斩杀  
> **候选牌池**：12 张赤红专属卡 + 6 张中立通用卡（共 18 张候选，择优遴选 30 张入套）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | 属性/数值 | **综合评分** | 梯队 | **携带率** | **胜率贡献 (ΔWR)** | 推荐配置 | 实战构筑解析 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **训练假人** | 中立 | 1费 | MINION | DP:1 | **95.9** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.8%</span> | 3 张 (核心满编) | 1费1防白板，满编三张，低费铺场与牺牲协同的底座。 |
| 2 | **佣兵斥候** | 中立 | 2费 | MINION | DP:1 `RUSH,DRAW_1` | **95.2** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.7%</span> | 3 张 (核心满编) | 2费1防突袭抽一，满编三张，解场过牌兼顾节奏。 |
| 3 | **破阵狂徒** | Red专属 | 4费 | MINION | DP:2 `RUSH,DEGRADE_1` | **94.5** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.6%</span> | 3 张 (核心满编) | 4费2防突袭降攻，满编三张，中期返场持续压制。 |
| 4 | **赤红掠袭者** | Red专属 | 5费 | MINION | DP:3 `RUSH,BONUS_SCORE_1` | **92.4** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.3%</span> | 3 张 (核心满编) | 5费3防突袭加分，满编三张，抢血与抢分兼备的终结者。 |
| 5 | **裂甲掷斧手** | Red专属 | 3费 | MINION | DP:1 `RUSH,DEGRADE_1` | **91.6** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.1%</span> | 3 张 (核心满编) | 3费1防突袭降攻，三张满编，解场夺先手的关键低费。 |
| 6 | **拾荒盾卫** | 中立 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **90.9** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.0%</span> | 3 张 (核心满编) | 3费3防加固亡语抽牌，满编三张，防守与续航兼备。 |
| 7 | **掠夺者** | Red专属 | 6费 | MINION | DP:4 `BONUS_SCORE_1` | **89.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 3 张 (主力配置) | 6费4防加分，满编三张，中后期补伤害兼抢分终端。 |
| 8 | **红色小队长** | Red专属 | 3费 | MINION | DP:3 `SPAWN_1_1` | **88.8** | **A** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.7%</span> | 3 张 (主力配置) | 3费3防召1/1，三张满编，铺场抢血的核心支点。 |
| 9 | **商人** | 中立 | 2费 | MINION | DP:2 `DRAW_1` | **85.8** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.2%</span> | 2 张 (主力配置) | 2费2防抽一，两张润滑手牌，保证攻势不断档。 |
| 10 | **赤红突击手** | Red专属 | 1费 | MINION | DP:1 `DEATH_DRAW_1` | **85.1** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.1%</span> | 2 张 (主力配置) | 1费1防亡语抽牌，前期换资源，两张补曲线不卡手。 |
| 11 | **集结号手** | Red专属 | 2费 | MINION | DP:1 `SPAWN_1_1` | **78.0** | **B** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.0%</span> | 2 张 (按需携带) | 2费1防召1/1，铺场可用，按环境需求挂两张。 |
| 12 | **血祭爆燃** | Red专属 | 3费 | SPELL | 攻3/防0 `SACRIFICE_1_KILL_1,ATTACK_ONLY` | **60.0** | **C** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -1.9%</span> | 0~1 张 (可选备编) | 3费打3牺牲一杀一，仅能攻击；备编一张针对大怪。 |
| 13 | **射线** | Red专属 | 2费 | SPELL | 攻2/防0 | **59.3** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.0%</span> | 0 张 (暂不推荐) | 2费打2直伤，效率低于随从交换，构筑空间不足。 |
| 14 | **自爆** | Red专属 | 2费 | SPELL | 攻0/防0 `SACRIFICE_1_KILL_1,ATTACK_ONLY` | **58.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.1%</span> | 0 张 (暂不推荐) | 2费牺牲一杀一且仅能攻击，亏牌抢节奏，当前不推荐。 |
| 15 | **切割者** | Red专属 | 2费 | MINION | DP:2 `DEGRADE_1` | **52.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 2费2防降一攻，身材平庸，降攻收益低，暂不携带。 |
| 16 | **雇佣兵** | 中立 | 3费 | MINION | DP:5 | **50.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.5%</span> | 0 张 (暂不推荐) | 3费5防白板，仅身材合格，无突袭过牌，不推荐。 |
| 17 | **酒馆密账** | 中立 | 1费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **49.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.6%</span> | 0 张 (暂不推荐) | 1费抽二弃一，弃牌风险高且节奏偏慢，暂不推荐。 |
| 18 | **大块头** | Red专属 | 6费 | MINION | DP:10 `DISCARD_2` | **48.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.7%</span> | 0 张 (暂不推荐) | 6费10防弃两张，费用过高且弃牌伤节奏，不推荐。 |

---

## 🔵 二、 【蔚蓝 (Blue) 卡组】单卡战力与构筑指南
> **战术核心**：防守反击 · 固守护盾 · 资源消耗  
> **候选牌池**：12 张蔚蓝专属卡 + 6 张中立通用卡（共 18 张候选，择优遴选 30 张入套）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | 属性/数值 | **综合评分** | 梯队 | **携带率** | **胜率贡献 (ΔWR)** | 推荐配置 | 实战构筑解析 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **霜盾见习官** | Blue专属 | 2费 | MINION | DP:1 `FORTIFY_1,SUPPORT_ATK_1` | **96.6** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.9%</span> | 3 张 (核心满编) | 二费一防，兼具固守与支援攻；低费润滑攻防，满编稳定前期。 |
| 2 | **酒馆密账** | 中立 | 1费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **95.2** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.7%</span> | 3 张 (核心满编) | 一费抽二弃一，低费过滤核心；三张保证启动，弃牌可处理冗余。 |
| 3 | **佣兵斥候** | 中立 | 2费 | MINION | DP:1 `RUSH,DRAW_1` | **94.5** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.6%</span> | 3 张 (核心满编) | 二费一防突袭抽一，解场过牌兼备；三张抢回先手并润滑曲线。 |
| 4 | **训练假人** | 中立 | 1费 | MINION | DP:1 | **89.3** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 2 张 (主力配置) | 一费一防，低费拖延节奏；两张开局护脸，后期可弃给密账。 |
| 5 | **火铳手** | Blue专属 | 4费 | MINION | DP:4 `SUPPORT_ATK_2` | **88.8** | **A** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.7%</span> | 3 张 (主力配置) | 四费四防二支援攻，后排增伤核心；三张保证中期输出与交换。 |
| 6 | **蔚蓝要塞** | Blue专属 | 6费 | MINION | DP:5 `FORTIFY_3,SUPPORT_ATK_1` | **88.1** | **A** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.6%</span> | 3 张 (主力配置) | 六费五防三固守，带支援攻；中后期攻防终端，三张保障上手。 |
| 7 | **雇佣兵** | 中立 | 3费 | MINION | DP:5 | **87.4** | **A** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.5%</span> | 3 张 (主力配置) | 三费五防，优质低费墙；三张稳定中期护脸并拖延至大哥。 |
| 8 | **拾荒盾卫** | 中立 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **86.0** | **A** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.2%</span> | 3 张 (主力配置) | 三费三防一固守，亡语抽一；交换不亏牌，三张维持中期续航。 |
| 9 | **商人** | 中立 | 2费 | MINION | DP:2 `DRAW_1` | **85.8** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.2%</span> | 2 张 (主力配置) | 二费二防过一牌，补资源兼站场；两张平衡手牌与节奏。 |
| 10 | **石像鬼** | Blue专属 | 6费 | MINION | DP:8 | **82.2** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.6%</span> | 2 张 (主力配置) | 六费八防无词条，中后期高质量肉盾；两张开场稳，避免卡手。 |
| 11 | **盾兵** | Blue专属 | 1费 | MINION | DP:1 `FORTIFY_1` | **77.9** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +0.9%</span> | 1 张 (按需携带) | 一费一防一固守，低费墙可护脸；功能单薄，按需一张补曲线。 |
| 12 | **蔚蓝卫士** | Blue专属 | 2费 | MINION | DP:2 `FORTIFY_1` | **74.4** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +0.4%</span> | 1 张 (按需携带) | 二费二防一固守，前期能吸收小伤害；卡位紧，按需补一张即可。 |
| 13 | **弓箭手** | Blue专属 | 3费 | MINION | DP:3 `SUPPORT_ATK_1` | **68.7** | **C** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ -0.5%</span> | 0~1 张 (可选备编) | 三费三防一支援攻，站场后补输出；效率普通，仅在备编考虑一张。 |
| 14 | **冰封禁制** | Blue专属 | 3费 | SPELL | 攻2/防2 | **60.0** | **C** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -1.9%</span> | 0~1 张 (可选备编) | 三费二攻二防，灵活但效率一般；特定节奏对局可备一张。 |
| 15 | **藤甲兵** | Blue专属 | 4费 | MINION | DP:4 `FORTIFY_2` | **55.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.7%</span> | 0 张 (暂不推荐) | 四费四防二固守，数值合格但定位重复；环境下不如其他四费曲线上限。 |
| 16 | **寒晶护壁** | Blue专属 | 2费 | SPELL | 攻0/防2 `DRAW_1` | **53.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.9%</span> | 0 张 (暂不推荐) | 二费加二防并过一牌，节奏偏慢；护脸需求低，卡位留给随从。 |
| 17 | **防御！** | Blue专属 | 1费 | SPELL | 攻0/防2 | **52.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.1%</span> | 0 张 (暂不推荐) | 一费加二防，防守效率低于随从站场；卡位不足，暂不进构筑。 |
| 18 | **壁垒工匠** | Blue专属 | 3费 | MINION | DP:3 `FORTIFY_2` | **51.5** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.3%</span> | 0 张 (暂不推荐) | 三费三防二固守，防守尚可却缺乏进攻；三费竞争激烈，暂不推荐。 |

---

## 🟢 三、 【翠绿 (Green) 卡组】单卡战力与构筑指南
> **战术核心**：法力跳费 · 质量成长 · 终结大哥  
> **候选牌池**：12 张翠绿专属卡 + 6 张中立通用卡（共 18 张候选，择优遴选 30 张入套）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | 属性/数值 | **综合评分** | 梯队 | **携带率** | **胜率贡献 (ΔWR)** | 推荐配置 | 实战构筑解析 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **灭世翡翠巨龙** | Green专属 | 9费 | MINION | DP:10 `RUSH,BONUS_SCORE_1` | **96.6** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.9%</span> | 3 张 (核心满编) | 九费突袭终结者，落地解场兼抢分，满编确保后期抽到制胜点。 |
| 2 | **翡翠幼龙** | Green专属 | 5费 | MINION | DP:3 `RUSH,DEATH_MANA_1` | **95.2** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.7%</span> | 3 张 (核心满编) | 五费突袭解场并返费，衔接高费终端，满编保障中期节奏。 |
| 3 | **佣兵斥候** | 中立 | 2费 | MINION | DP:1 `RUSH,DRAW_1` | **93.8** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.5%</span> | 3 张 (核心满编) | 两费突袭过牌，解小怪并补手牌，满编保证前期节奏顺畅。 |
| 4 | **芽苗祭司** | Green专属 | 3费 | MINION | DP:3 `RAMP_1` | **93.1** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.4%</span> | 3 张 (核心满编) | 三费站场加跳费，法力扩张核心，满编保证前期准时加速。 |
| 5 | **雇佣兵** | 中立 | 3费 | MINION | DP:5 | **90.9** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.0%</span> | 3 张 (核心满编) | 三费高身材白板，前期抗压交换优秀，满编稳住跳费过渡期。 |
| 6 | **酒馆密账** | 中立 | 1费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **89.3** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 2 张 (主力配置) | 一费过二弃一，快速找跳费与终端，两张保持资源不断档。 |
| 7 | **训练假人** | 中立 | 1费 | MINION | DP:1 | **87.9** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.5%</span> | 2 张 (主力配置) | 一费低身材吸收伤害，保护跳费随从，两张填补前期防守。 |
| 8 | **翡翠巨熊** | Green专属 | 6费 | MINION | DP:6 `FORTIFY_2` | **86.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.3%</span> | 2 张 (主力配置) | 六费高质量护甲随从，中后期稳住场面，两张支撑跳费曲线。 |
| 9 | **商人** | 中立 | 2费 | MINION | DP:2 `DRAW_1` | **83.6** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.9%</span> | 2 张 (主力配置) | 两费站场过牌，润滑跳费体系，两张兼顾节奏与手牌补充。 |
| 10 | **剧毒花** | Green专属 | 2费 | MINION | DP:2 `DEGRADE_2` | **82.2** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.6%</span> | 2 张 (主力配置) | 两费降攻克制大怪，配合跳费防守过渡，环境快攻多时两张稳定。 |
| 11 | **拾荒盾卫** | 中立 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **81.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.5%</span> | 2 张 (主力配置) | 三费护甲亡语过牌，换掉不亏手牌，两张增强中前期韧性。 |
| 12 | **树人** | Green专属 | 3费 | MINION | DP:2 `FORTIFY_1` | **80.1** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.3%</span> | 2 张 (主力配置) | 三费身材偏防守，前期拖住快攻，后期跳费后作用有限，两张即可。 |
| 13 | **森林之狼** | Green专属 | 4费 | MINION | DP:4 | **70.8** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ -0.2%</span> | 1 张 (按需携带) | 四费白板曲线尚可，但缺乏突袭跳费协同，按需补一张即可。 |
| 14 | **荆棘缠绕** | Green专属 | 2费 | SPELL | 攻2/防0 | **60.0** | **C** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -1.9%</span> | 0~1 张 (可选备编) | 两费打二能补刀，但跳费体系更需铺场，最多备一张解小怪。 |
| 15 | **远古巨树** | Green专属 | 7费 | MINION | DP:8 `FORTIFY_2` | **54.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.8%</span> | 0 张 (暂不推荐) | 七费大身材带护甲，登场太慢，被解后跳费优势尽失，故不带。 |
| 16 | **狂暴生长** | Green专属 | 3费 | SPELL | 攻0/防3 | **49.4** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.6%</span> | 0 张 (暂不推荐) | 三费仅给防不跳费，无法推动核心计划，卡位竞争差，选择放弃。 |
| 17 | **世界树恩泽** | Green专属 | 4费 | SPELL | 攻0/防3 `RAMP_1` | **48.7** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.7%</span> | 0 张 (暂不推荐) | 四费跳一费并给防，效率低于站场跳费，卡位紧时不考虑。 |
| 18 | **翠绿萌芽** | Green专属 | 2费 | SPELL | 攻0/防0 `RAMP_1` | **48.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 两费只跳一费，节奏亏损大，跳费链更需站场，故不入构筑。 |

---

## ⚪ 四、 【中立 (Neutral) 卡牌】全阵营适配性与战术表现分析
> **机制说明**：同一张中立卡在快攻、控制、跳费等不同战术体系下具有截然不同的战术价值与契合度。

| 中立卡名称 | 费用 | 类型 | 🔴 赤红卡组评分 | 🔵 蔚蓝卡组评分 | 🟢 翠绿卡组评分 | 最佳契合卡组 | 跨阵营战术定位 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **训练假人** | 1费 | MINION | **95.9** | **89.3** | **87.9** | **赤红 (快攻)** | 多体系通用的高质量拼图 |
| **佣兵斥候** | 2费 | MINION | **95.2** | **94.5** | **93.8** | **赤红 (快攻)** | 多体系通用的高质量拼图 |
| **拾荒盾卫** | 3费 | MINION | **90.9** | **86.0** | **81.5** | **赤红 (快攻)** | 多体系通用的高质量拼图 |
| **商人** | 2费 | MINION | **85.8** | **85.8** | **83.6** | **赤红 (快攻)** | 多体系通用的高质量拼图 |
| **雇佣兵** | 3费 | MINION | **50.1** | **87.4** | **90.9** | **翠绿 (跳费)** | 偏向翠绿 (跳费)体系的针对性组件 |
| **酒馆密账** | 1费 | SPELL | **49.4** | **95.2** | **89.3** | **蔚蓝 (防守)** | 偏向蔚蓝 (防守)体系的针对性组件 |
