# 🏆 TCG 卡牌战力评级与构筑指南（阵营分色专榜）

> **构筑规则说明**：在 TCG 标准规则中，各阵营（赤红/蔚蓝/翠绿）由**阵营专属卡 + 中立通用卡**构筑。本指南基于 PPO 强化学习智能体（`card_ppo_model_tuned.pth`）在对战环境中的实战数据，综合**【卡组携带率】**与**【局势胜率贡献 (ΔWR)】**两大维度，按阵营分色独立建榜，提供客观、严谨的构筑参考与单卡解析。

---

## 🔴 一、 【赤红 (Red) 卡组】单卡战力与构筑指南
> **战术核心**：快攻压制 · 牺牲协同 · 节奏斩杀  
> **候选牌池**：12 张赤红专属卡 + 6 张中立通用卡（共 18 张候选，择优遴选 30 张入套）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | 属性/数值 | **综合评分** | 梯队 | **携带率** | **胜率贡献 (ΔWR)** | 推荐配置 | 实战构筑解析 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **佣兵斥候** | 中立 | 2费 | MINION | DP:1 `RUSH,DRAW_1` | **94.5** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.6%</span> | 3 张 (核心满编) | 两费突袭过牌，解场与补资源兼得，起手留满编不亏节奏。 |
| 2 | **赤红掠袭者** | Red专属 | 5费 | MINION | DP:3 `RUSH,BONUS_SCORE_1` | **93.1** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.4%</span> | 3 张 (核心满编) | 五费突袭抢分，落场即影响场面与比分；核心满编保证上手。 |
| 3 | **雇佣兵** | 中立 | 3费 | MINION | DP:5 | **89.5** | **A** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 3 张 (主力配置) | 三费五血白板，能吃解或换频率，为后续输出争取空间。 |
| 4 | **掠夺者** | Red专属 | 6费 | MINION | DP:4 `BONUS_SCORE_1` | **88.1** | **A** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.6%</span> | 3 张 (主力配置) | 六费终结者，站场后持续扩大分差；快攻也需要满编收尾。 |
| 5 | **商人** | 中立 | 2费 | MINION | DP:2 `DRAW_1` | **87.4** | **A** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.5%</span> | 3 张 (主力配置) | 两费过牌站场，润滑手牌并衔接后续爆发；满编稳定资源。 |
| 6 | **酒馆密账** | 中立 | 1费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **87.2** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.4%</span> | 2 张 (主力配置) | 一费过二弃一，筛选关键牌；两张提高起手与中期流畅度。 |
| 7 | **红色小队长** | Red专属 | 3费 | MINION | DP:3 `SPAWN_1_1` | **86.7** | **A** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.4%</span> | 3 张 (主力配置) | 三费铺两个频率，配合牺牲与群体增益，满编保证前期场面不崩。 |
| 8 | **破阵狂徒** | Red专属 | 4费 | MINION | DP:2 `RUSH,DEGRADE_1` | **85.8** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.2%</span> | 2 张 (主力配置) | 四费突袭降攻，处理中型威胁；两张作为中期夺节奏的工具。 |
| 9 | **裂甲掷斧手** | Red专属 | 3费 | MINION | DP:1 `RUSH,DEGRADE_1` | **84.4** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.0%</span> | 2 张 (主力配置) | 突袭降攻解小卒，低攻但能夺回先手；两张补节奏不挤曲线。 |
| 10 | **赤红突击手** | Red专属 | 1费 | MINION | DP:1 `DEATH_DRAW_1` | **82.9** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.7%</span> | 2 张 (主力配置) | 低费送掉能补牌，适合牺牲与抢节奏体系；两张防上手过密。 |
| 11 | **射线** | Red专属 | 2费 | SPELL | 攻2/防0 | **79.3** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +1.2%</span> | 1 张 (按需携带) | 两费直伤补刀，能越墙收残血；卡位松时带一张补斩杀。 |
| 12 | **自爆** | Red专属 | 2费 | SPELL | 攻0/防0 `SACRIFICE_1_KILL_1,ATTACK_ONLY` | **78.6** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +1.1%</span> | 1 张 (按需携带) | 牺牲杂毛换单解，过墙斩或清关键随从；环境需求才留一张。 |
| 13 | **集结号手** | Red专属 | 2费 | MINION | DP:1 `SPAWN_1_1` | **78.0** | **B** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.0%</span> | 2 张 (按需携带) | 两费横向铺场，为牺牲与突袭垫频率；两张，避免中期空抽。 |
| 14 | **血祭爆燃** | Red专属 | 3费 | SPELL | 攻3/防0 `SACRIFICE_1_KILL_1,ATTACK_ONLY` | **60.0** | **C** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -1.9%</span> | 0~1 张 (可选备编) | 三费牺牲换三伤单解，效率一般；仅针对大生物环境备一张。 |
| 15 | **训练假人** | 中立 | 1费 | MINION | DP:1 | **57.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.3%</span> | 0 张 (暂不推荐) | 一费一血无效果，占场太弱，不如带能过牌或铺场的低费。 |
| 16 | **切割者** | Red专属 | 2费 | MINION | DP:2 `DEGRADE_1` | **53.6** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.9%</span> | 0 张 (暂不推荐) | 身材平庸，降攻效果偏防守，与快攻抢血思路相悖，不带。 |
| 17 | **大块头** | Red专属 | 6费 | MINION | DP:10 `DISCARD_2` | **52.2** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.2%</span> | 0 张 (暂不推荐) | 六费太高且弃两张牌，拖慢快攻节奏，卡手时无法止损。 |
| 18 | **拾荒盾卫** | 中立 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **50.8** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.4%</span> | 0 张 (暂不推荐) | 三费偏防守，虽有亡语过牌，但拖慢抢血，不适合本体系。 |

---

## 🔵 二、 【蔚蓝 (Blue) 卡组】单卡战力与构筑指南
> **战术核心**：防守反击 · 固守护盾 · 资源消耗  
> **候选牌池**：12 张蔚蓝专属卡 + 6 张中立通用卡（共 18 张候选，择优遴选 30 张入套）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | 属性/数值 | **综合评分** | 梯队 | **携带率** | **胜率贡献 (ΔWR)** | 推荐配置 | 实战构筑解析 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **佣兵斥候** | 中立 | 2费 | MINION | DP:1 `RUSH,DRAW_1` | **97.3** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +4.0%</span> | 3 张 (核心满编) | 两费突袭过牌，解场夺回先手，防守体系润滑，满编。 |
| 2 | **壁垒工匠** | Blue专属 | 3费 | MINION | DP:3 `FORTIFY_2` | **96.6** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.9%</span> | 3 张 (核心满编) | 三费固守二，中期防线核心，能挡两次攻击，满编保证上手。 |
| 3 | **盾兵** | Blue专属 | 1费 | MINION | DP:1 `FORTIFY_1` | **93.8** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.5%</span> | 3 张 (核心满编) | 一费固守墙，前期吸收伤害并保住血量，防守体系满编基石。 |
| 4 | **寒晶护壁** | Blue专属 | 2费 | SPELL | 攻0/防2 `DRAW_1` | **91.6** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.1%</span> | 3 张 (核心满编) | 两费加防并过牌，防守不亏手牌，体系核心满编无争议。 |
| 5 | **雇佣兵** | 中立 | 3费 | MINION | DP:5 | **88.8** | **A** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.7%</span> | 3 张 (主力配置) | 三费五防无词条，纯粹高血量站场，稳住节奏，主力满三。 |
| 6 | **酒馆密账** | 中立 | 1费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **87.9** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.5%</span> | 2 张 (主力配置) | 一费抽二弃一，低费找关键防守件，手牌事故少，带两张。 |
| 7 | **蔚蓝要塞** | Blue专属 | 6费 | MINION | DP:5 `FORTIFY_3,SUPPORT_ATK_1` | **87.4** | **A** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#dfe6e9;">⚪ +2.5%</span> | 3 张 (主力配置) | 六费固守三还加攻，终盘攻防核心，三张确保后期质量。 |
| 8 | **拾荒盾卫** | 中立 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **87.2** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.4%</span> | 2 张 (主力配置) | 三费固守一，亡语补牌，交换后不亏节奏，主力两张。 |
| 9 | **石像鬼** | Blue专属 | 6费 | MINION | DP:8 | **84.4** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.0%</span> | 2 张 (主力配置) | 六费八防白板，中后期可靠肉盾，缺少词条但能稳住场面，带两张。 |
| 10 | **火铳手** | Blue专属 | 4费 | MINION | DP:4 `SUPPORT_ATK_2` | **78.7** | **B** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.1%</span> | 2 张 (按需携带) | 四费后排加二攻，防守同时抬高反击伤害，主力带两张。 |
| 11 | **防御！** | Blue专属 | 1费 | SPELL | 攻0/防2 | **76.5** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +0.7%</span> | 1 张 (按需携带) | 一费加二防，低费护脸救随从，环境快攻多时带一张。 |
| 12 | **弓箭手** | Blue专属 | 3费 | MINION | DP:3 `SUPPORT_ATK_1` | **72.2** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ 0.0%</span> | 1 张 (按需携带) | 三费后排加攻，顺风补伤害，逆风防守差，仅作针对挂件。 |
| 13 | **霜盾见习官** | Blue专属 | 2费 | MINION | DP:1 `FORTIFY_1,SUPPORT_ATK_1` | **71.5** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ -0.1%</span> | 1 张 (按需携带) | 两费双功能，身材偏脆，前期润滑可带一张，不宜满编。 |
| 14 | **藤甲兵** | Blue专属 | 4费 | MINION | DP:4 `FORTIFY_2` | **70.1** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ -0.3%</span> | 1 张 (按需携带) | 四费固守二，对中速交换占优，但对法术直伤无力，按需一张。 |
| 15 | **冰封禁制** | Blue专属 | 3费 | SPELL | 攻2/防2 | **60.0** | **C** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -1.9%</span> | 0~1 张 (可选备编) | 三费攻防各二，灵活但效率普通，备编一张应对特定对局。 |
| 16 | **商人** | 中立 | 2费 | MINION | DP:2 `DRAW_1` | **55.1** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -2.7%</span> | 0 张 (暂不推荐) | 两费过一但身材无防守价值，体系不缺这点抽牌，故不带。 |
| 17 | **训练假人** | 中立 | 1费 | MINION | DP:1 | **52.9** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.1%</span> | 0 张 (暂不推荐) | 一费一防无特效，连盾兵都不如，浪费卡位，零张。 |
| 18 | **蔚蓝卫士** | Blue专属 | 2费 | MINION | DP:2 `FORTIFY_1` | **48.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 两费只带固守一，身材不赚节奏，竞争卡位失败，故零张。 |

---

## 🟢 三、 【翠绿 (Green) 卡组】单卡战力与构筑指南
> **战术核心**：法力跳费 · 质量成长 · 终结大哥  
> **候选牌池**：12 张翠绿专属卡 + 6 张中立通用卡（共 18 张候选，择优遴选 30 张入套）

| 排名 | 卡牌名称 | 归属 | 费用 | 类型 | 属性/数值 | **综合评分** | 梯队 | **携带率** | **胜率贡献 (ΔWR)** | 推荐配置 | 实战构筑解析 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **灭世翡翠巨龙** | Green专属 | 9费 | MINION | DP:10 `RUSH,BONUS_SCORE_1` | **96.6** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.9%</span> | 3 张 (核心满编) | 九费十攻突袭，额外得分终结；满编作核弹，跳费后连拍压制。 |
| 2 | **芽苗祭司** | Green专属 | 3费 | MINION | DP:3 `RAMP_1` | **95.9** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.8%</span> | 3 张 (核心满编) | 三费三攻跳一，跳费核心；满编保证前期准时，加速大哥登场。 |
| 3 | **翡翠幼龙** | Green专属 | 5费 | MINION | DP:3 `RUSH,DEATH_MANA_1` | **94.5** | **S+** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.6%</span> | 3 张 (核心满编) | 五费突袭解场，亡语回一费；满编抢节奏，连接跳费与大哥。 |
| 4 | **雇佣兵** | 中立 | 3费 | MINION | DP:5 | **93.8** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.5%</span> | 3 张 (核心满编) | 三费五攻身材压制，前期站场护跳费；满编保证压力与交换。 |
| 5 | **佣兵斥候** | 中立 | 2费 | MINION | DP:1 `RUSH,DRAW_1` | **93.1** | **S** | <span style="color:#ffeaa7; font-weight:bold;">100.0%</span> | <span style="color:#55efc4; font-weight:bold;">🟢 +3.4%</span> | 3 张 (核心满编) | 两费突袭过一，解小怪补手牌；满编润滑前期，保跳费节奏。 |
| 6 | **酒馆密账** | 中立 | 1费 | SPELL | 攻0/防0 `DRAW_2,DISCARD_1` | **89.3** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +2.8%</span> | 2 张 (主力配置) | 一费过二弃一，换手找跳费；两张提速，弃牌风险不宜满编。 |
| 7 | **商人** | 中立 | 2费 | MINION | DP:2 `DRAW_1` | **82.2** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.6%</span> | 2 张 (主力配置) | 两费二攻过一，润滑手牌；两张稳定找跳费，满编易卡场面。 |
| 8 | **拾荒盾卫** | 中立 | 3费 | MINION | DP:3 `FORTIFY_1,DEATH_DRAW_1` | **80.8** | **A** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.4%</span> | 2 张 (主力配置) | 三费三攻加固，亡语过一；两张防守续航，稳住跳费期。 |
| 9 | **树人** | Green专属 | 3费 | MINION | DP:2 `FORTIFY_1` | **78.7** | **B** | <span style="color:#ffeaa7; font-weight:bold;">66.7%</span> | <span style="color:#dfe6e9;">⚪ +1.1%</span> | 2 张 (按需携带) | 三费两攻加固，防守向跳费组件；两张稳前期，不必满编，后期抽到弱。 |
| 10 | **翠绿萌芽** | Green专属 | 2费 | SPELL | 攻0/防0 `RAMP_1` | **77.2** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +0.8%</span> | 1 张 (按需携带) | 两费跳一，前期提速但亏节奏；仅需一张补曲线，过多手牌空转。 |
| 11 | **训练假人** | 中立 | 1费 | MINION | DP:1 | **74.4** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +0.4%</span> | 1 张 (按需携带) | 一费一攻吸收伤害，拖到跳费；仅一张当肉盾，后期抽到无用。 |
| 12 | **远古巨树** | Green专属 | 7费 | MINION | DP:8 `FORTIFY_2` | **73.6** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +0.3%</span> | 1 张 (按需携带) | 七费八攻大墙，加固二站场；一张作中后期质量，满编卡手。 |
| 13 | **翡翠巨熊** | Green专属 | 6费 | MINION | DP:6 `FORTIFY_2` | **72.9** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ +0.1%</span> | 1 张 (按需携带) | 六费六攻加固二，扎实中坚；费用偏高，一张撑质量，卡位留给核心。 |
| 14 | **森林之狼** | Green专属 | 4费 | MINION | DP:4 | **71.5** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ -0.1%</span> | 1 张 (按需携带) | 四费四攻白板，曲线填充；无突袭难返场，一张补位，不优先。 |
| 15 | **剧毒花** | Green专属 | 2费 | MINION | DP:2 `DEGRADE_2` | **70.1** | **B** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ -0.3%</span> | 1 张 (按需携带) | 两费毒花削减敌方两攻，针对大怪；环境对策一张足够，常规卡位紧。 |
| 16 | **狂暴生长** | Green专属 | 3费 | SPELL | 攻0/防3 | **69.4** | **C** | <span style="color:#81ecec;">33.3%</span> | <span style="color:#dfe6e9;">⚪ -0.4%</span> | 0~1 张 (可选备编) | 三费加三防，保跳费随从或大哥；但被动且亏卡，仅备编一张。 |
| 17 | **荆棘缠绕** | Green专属 | 2费 | SPELL | 攻2/防0 | **60.0** | **C** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#dfe6e9;">⚪ -1.9%</span> | 0~1 张 (可选备编) | 两费打二，低效解牌；仅对快攻备编一张，主牌不配。 |
| 18 | **世界树恩泽** | Green专属 | 4费 | SPELL | 攻0/防3 `RAMP_1` | **48.0** | **D** | <span style="color:#b2bec3;">0.0%</span> | <span style="color:#e17055; font-weight:bold;">🟠 -3.8%</span> | 0 张 (暂不推荐) | 四费跳一太慢，三防也无节奏；拖累扩张，当前构筑不取。 |

---

## ⚪ 四、 【中立 (Neutral) 卡牌】全阵营适配性与战术表现分析
> **机制说明**：同一张中立卡在快攻、控制、跳费等不同战术体系下具有截然不同的战术价值与契合度。

| 中立卡名称 | 费用 | 类型 | 🔴 赤红卡组评分 | 🔵 蔚蓝卡组评分 | 🟢 翠绿卡组评分 | 最佳契合卡组 | 跨阵营战术定位 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **佣兵斥候** | 2费 | MINION | **94.5** | **97.3** | **93.1** | **蔚蓝 (防守)** | 多体系通用的高质量拼图 |
| **雇佣兵** | 3费 | MINION | **89.5** | **88.8** | **93.8** | **翠绿 (跳费)** | 多体系通用的高质量拼图 |
| **商人** | 2费 | MINION | **87.4** | **55.1** | **82.2** | **赤红 (快攻)** | 偏向赤红 (快攻)体系的针对性组件 |
| **酒馆密账** | 1费 | SPELL | **87.2** | **87.9** | **89.3** | **翠绿 (跳费)** | 多体系通用的高质量拼图 |
| **训练假人** | 1费 | MINION | **57.9** | **52.9** | **74.4** | **翠绿 (跳费)** | 特定战局下的可选备编卡 |
| **拾荒盾卫** | 3费 | MINION | **50.8** | **87.2** | **80.8** | **蔚蓝 (防守)** | 偏向蔚蓝 (防守)体系的针对性组件 |
