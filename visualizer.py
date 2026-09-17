import json
import os

def get_faction_meta(faction_val, player_id):
    f_str = str(faction_val).lower()
    if "green" in f_str or "绿" in f_str:
        return {
            "name": f"🟢 翠绿 (P{player_id} 林野跳费流)",
            "short": "🟢 绿方",
            "def_label": "🛡️ 绿防",
            "atk_label": "⚔️ 绿冲",
            "team_class": "green",
            "color": "#2ed573"
        }
    elif "blue" in f_str or "蓝" in f_str:
        return {
            "name": f"🔵 蔚蓝 (P{player_id} 守卫控制流)",
            "short": "🔵 蓝方",
            "def_label": "🛡️ 蓝防",
            "atk_label": "⚔️ 蓝冲",
            "team_class": "blue",
            "color": "#1e90ff"
        }
    else:
        return {
            "name": f"🔴 赤红 (P{player_id} 快攻突破流)",
            "short": "🔴 红方",
            "def_label": "🛡️ 红防",
            "atk_label": "⚔️ 红冲",
            "team_class": "red",
            "color": "#ff4757"
        }

def format_terminal_board(turn_count, acting_player, p0, p1, lanes, action_desc, result_log=None):
    """
    生成双路战场控制台看板 (支持红/蓝/绿自适应阵营)
    """
    lines = []
    w = 78
    
    m0 = get_faction_meta(p0.get("faction", "Red"), 0)
    m1 = get_faction_meta(p1.get("faction", "Blue"), 1)

    # 双方比分与法力指示器
    p0_score_bar = "■" * p0["score"] + "□" * (7 - p0["score"])
    p1_score_bar = "■" * p1["score"] + "□" * (7 - p1["score"])
    
    lines.append("╔" + "═" * (w - 2) + "╗")
    
    # P1 信息头
    p1_header = f" {m1['name']}  得分: [{p1_score_bar}] {p1['score']}/7  法力: 💎 {p1['mana']}/{p1['max_mana']}  手牌: {len(p1['hand'])}张"
    lines.append(f"║{p1_header:<{w-2}}║")
    lines.append("╠" + "═" * 38 + "╦" + "═" * 37 + "╣")
    lines.append("║                【左路战场】          ║               【右路战场】          ║")
    
    # P1 防守怪
    def fmt_units(units, is_atk=False):
        if not units:
            return "空"
        res = []
        for u in units:
            if is_atk:
                tag = "⚡就绪" if u.get("ready", False) else "⏳蓄势"
                res.append(f"{u['name']}({u['dp']})[{tag}]")
            else:
                res.append(f"{u['name']}(DP:{u['dp']})")
        return "、".join(res)

    l_1_def = fmt_units(lanes[0]["p1_defenders"])
    r_1_def = fmt_units(lanes[1]["p1_defenders"])
    lines.append(f"║ {m1['def_label']}: {l_1_def:<27} ║ {m1['def_label']}: {r_1_def:<26} ║")

    l_1_atk = fmt_units(lanes[0]["p1_attackers"], is_atk=True)
    r_1_atk = fmt_units(lanes[1]["p1_attackers"], is_atk=True)
    lines.append(f"║ {m1['atk_label']}: {l_1_atk:<27} ║ {m1['atk_label']}: {r_1_atk:<26} ║")

    # 对撞分界线
    lines.append("║ ┄┄┄┄┄┄┄┄ ⚡ 攻防对撞线 ┄┄┄┄┄┄┄┄ ╫ ┄┄┄┄┄┄┄┄ ⚡ 攻防对撞线 ┄┄┄┄┄┄┄┄ ║")

    # P0 进攻怪与防守怪
    l_0_atk = fmt_units(lanes[0]["p0_attackers"], is_atk=True)
    r_0_atk = fmt_units(lanes[1]["p0_attackers"], is_atk=True)
    lines.append(f"║ {m0['atk_label']}: {l_0_atk:<27} ║ {m0['atk_label']}: {r_0_atk:<26} ║")

    l_0_def = fmt_units(lanes[0]["p0_defenders"])
    r_0_def = fmt_units(lanes[1]["p0_defenders"])
    lines.append(f"║ {m0['def_label']}: {l_0_def:<27} ║ {m0['def_label']}: {r_0_def:<26} ║")

    lines.append("╠" + "═" * 38 + "╩" + "═" * 37 + "╣")
    # P0 信息头
    p0_header = f" {m0['name']}  得分: [{p0_score_bar}] {p0['score']}/7  法力: 💎 {p0['mana']}/{p0['max_mana']}  手牌: {len(p0['hand'])}张"
    lines.append(f"║{p0_header:<{w-2}}║")
    lines.append("╚" + "═" * (w - 2) + "╝")

    # 当前动作与战况反馈
    act_p_str = m0['short'] if acting_player == 0 else m1['short']
    lines.append(f"👉 [第 {turn_count:02d} 回合] {act_p_str} 决策: {action_desc}")
    if result_log:
        lines.append(f"💥 {result_log}")

    return "\n".join(lines)


def export_html_replay(snapshots, winner_info, output_path="battle_replay.html"):
    """
    导出双路对战 HTML5 回放网页 (紧凑单屏自适应版)
    """
    json_data = json.dumps(snapshots, ensure_ascii=False)
    
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TCG-AI 对局全景回放看板</title>
  <style>
    :root {{
      --bg-dark: #0b0f19;
      --card-bg: rgba(22, 30, 46, 0.88);
      --panel-border: rgba(255, 255, 255, 0.08);
      --red-team: #ff4757;
      --red-glow: rgba(255, 71, 87, 0.4);
      --blue-team: #1e90ff;
      --blue-glow: rgba(30, 144, 255, 0.4);
      --green-team: #2ed573;
      --green-glow: rgba(46, 213, 115, 0.4);
      --gold: #ffa502;
      --text-main: #f1f2f6;
      --text-dim: #a4b0be;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html, body {{
      height: 100vh;
      max-height: 100vh;
      background: radial-gradient(circle at 50% 20%, #151e33 0%, var(--bg-dark) 100%);
      color: var(--text-main);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Microsoft YaHei", sans-serif;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      padding: 6px 12px;
    }}

    /* 顶部导航与比分板 (紧凑条) */
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--card-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--panel-border);
      border-radius: 8px;
      padding: 4px 12px;
      margin-bottom: 5px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      flex-shrink: 0;
    }}
    .title-area h1 {{
      font-size: 0.92rem;
      font-weight: 700;
      letter-spacing: 0.5px;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .title-area p {{
      font-size: 0.68rem;
      color: var(--text-dim);
      margin-top: 1px;
    }}
    .match-status {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .turn-badge {{
      background: rgba(255,255,255,0.08);
      padding: 2px 10px;
      border-radius: 12px;
      font-size: 0.78rem;
      font-weight: bold;
      color: var(--gold);
    }}
    #winnerBanner {{
      font-size: 0.78rem;
      font-weight: bold;
    }}

    /* 主战场网格 (100vh 约束) */
    .arena {{
      display: grid;
      grid-template-columns: 1fr 290px;
      gap: 8px;
      flex: 1;
      min-height: 0;
      overflow: hidden;
    }}

    /* 战局主视窗 */
    .battle-field-container {{
      display: flex;
      flex-direction: column;
      gap: 5px;
      flex: 1;
      min-height: 0;
      overflow: hidden;
    }}

    /* 玩家状态条 */
    .player-strip {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--card-bg);
      border: 1px solid var(--panel-border);
      border-radius: 6px;
      padding: 3px 10px;
      min-height: 28px;
      flex-shrink: 0;
      transition: all 0.2s;
    }}
    .player-strip.blue {{ border-left: 4px solid var(--blue-team); }}
    .player-strip.red {{ border-left: 4px solid var(--red-team); }}
    .player-strip.green {{ border-left: 4px solid var(--green-team); }}

    .player-name {{
      font-weight: bold;
      font-size: 0.8rem;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .player-name.blue {{ color: var(--blue-team); text-shadow: 0 0 8px var(--blue-glow); }}
    .player-name.red {{ color: var(--red-team); text-shadow: 0 0 8px var(--red-glow); }}
    .player-name.green {{ color: var(--green-team); text-shadow: 0 0 8px var(--green-glow); }}

    .resource-group {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .mana-container {{
      display: flex;
      align-items: center;
      gap: 3px;
    }}
    .mana-crystal {{
      width: 9px;
      height: 12px;
      background: rgba(255,255,255,0.15);
      clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
    }}
    .mana-crystal.active {{
      background: #00d2d3;
      box-shadow: 0 0 6px #00d2d3;
    }}

    .score-stars {{
      display: flex;
      gap: 2px;
      align-items: center;
    }}
    .star {{
      font-size: 0.82rem;
      color: rgba(255,255,255,0.2);
    }}
    .star.filled {{
      color: var(--gold);
      text-shadow: 0 0 6px rgba(255, 165, 2, 0.8);
    }}
    .res-text {{
      font-size: 0.76rem;
    }}

    /* 双路战场 */
    .lanes-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      flex: 1;
      min-height: 0;
      overflow: hidden;
    }}
    .lane-box {{
      background: rgba(15, 23, 42, 0.7);
      border: 1px solid var(--panel-border);
      border-radius: 8px;
      padding: 5px 8px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      position: relative;
      overflow: hidden;
      flex: 1;
      min-height: 0;
    }}
    .lane-title {{
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      font-size: 1.15rem;
      font-weight: 900;
      color: rgba(255,255,255,0.03);
      pointer-events: none;
      letter-spacing: 3px;
    }}

    .lane-half {{
      display: flex;
      flex-direction: column;
      gap: 2px;
      flex: 1;
      min-height: 0;
      justify-content: space-around;
    }}

    .unit-lane-zone {{
      min-height: 34px;
      max-height: 65px;
      display: flex;
      gap: 4px;
      flex-wrap: wrap;
      align-items: center;
      padding: 2px 4px;
      border-radius: 4px;
      background: rgba(255,255,255,0.02);
      overflow-y: auto;
    }}
    .unit-lane-zone.defense {{
      border-style: dashed;
      border-width: 1px;
      border-color: rgba(255,255,255,0.08);
    }}
    .zone-label {{
      width: 100%;
      font-size: 0.62rem;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      display: flex;
      justify-content: space-between;
      margin-bottom: 1px;
    }}

    /* 对撞中线 */
    .clash-divider {{
      margin: 1px 0;
      border-top: 1px dashed rgba(255, 255, 255, 0.15);
      position: relative;
      text-align: center;
      height: 8px;
      flex-shrink: 0;
    }}
    .clash-divider span {{
      position: relative;
      top: -8px;
      background: #1e293b;
      padding: 1px 8px;
      border-radius: 6px;
      font-size: 0.58rem;
      color: var(--text-dim);
      border: 1px solid rgba(255,255,255,0.1);
    }}

    /* 卡牌随从视效 */
    .minion-card {{
      background: linear-gradient(145deg, #1e293b, #0f172a);
      border: 1px solid rgba(255,255,255,0.15);
      border-radius: 4px;
      padding: 2px 5px;
      min-width: 62px;
      max-width: 100px;
      display: flex;
      flex-direction: column;
      gap: 1px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.4);
      position: relative;
      animation: popIn 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }}
    @keyframes popIn {{
      0% {{ transform: scale(0.8); opacity: 0; }}
      100% {{ transform: scale(1); opacity: 1; }}
    }}
    .minion-card.red {{ border-top: 2px solid var(--red-team); }}
    .minion-card.blue {{ border-top: 2px solid var(--blue-team); }}
    .minion-card.green {{ border-top: 2px solid var(--green-team); }}
    .minion-card.ready {{
      box-shadow: 0 0 6px rgba(46, 213, 115, 0.4);
      border-color: var(--green-team);
    }}
    .minion-name {{
      font-size: 0.66rem;
      font-weight: 600;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .minion-stats {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.6rem;
    }}
    .dp-badge {{
      background: #374151;
      padding: 0 3px;
      border-radius: 2px;
      font-weight: bold;
      color: #38bdf8;
      font-size: 0.6rem;
    }}
    .state-tag {{
      font-size: 0.55rem;
      padding: 0 2px;
      border-radius: 2px;
    }}
    .state-tag.ready {{ background: rgba(46, 213, 115, 0.2); color: #2ed573; }}
    .state-tag.charging {{ background: rgba(255, 165, 2, 0.2); color: #ffa502; }}

    /* 手牌展示区 */
    .hand-toggle {{
      background: rgba(255,255,255,0.06);
      border: none;
      color: var(--text-dim);
      font-size: 0.68rem;
      cursor: pointer;
      padding: 1px 6px;
      border-radius: 3px;
      margin-left: 6px;
    }}
    .hand-toggle:hover {{ background: rgba(255,255,255,0.12); color: var(--text-main); }}
    .hand-panel {{
      display: none;
      flex-wrap: wrap;
      gap: 3px;
      padding: 3px 8px;
      background: rgba(15, 23, 42, 0.6);
      border-radius: 4px;
      border: 1px solid var(--panel-border);
      max-height: 46px;
      overflow-y: auto;
      flex-shrink: 0;
    }}
    .hand-panel.open {{ display: flex; }}
    .hand-card-chip {{
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 3px;
      padding: 1px 5px;
      font-size: 0.58rem;
      white-space: nowrap;
      display: flex;
      align-items: center;
      gap: 3px;
    }}
    .hand-card-chip .hc-cost {{
      background: #2563eb;
      color: white;
      border-radius: 2px;
      padding: 0 3px;
      font-weight: bold;
      font-size: 0.55rem;
    }}
    .hand-card-chip .hc-dp {{
      color: #38bdf8;
      font-weight: bold;
    }}
    .hand-card-chip .hc-spell {{
      color: #c084fc;
      font-style: italic;
    }}

    /* 右侧战报面板 */
    .log-panel {{
      background: var(--card-bg);
      border: 1px solid var(--panel-border);
      border-radius: 8px;
      padding: 6px 8px;
      display: flex;
      flex-direction: column;
      height: 100%;
      min-height: 0;
      overflow: hidden;
    }}
    .log-panel h3 {{
      font-size: 0.78rem;
      margin-bottom: 4px;
      padding-bottom: 4px;
      border-bottom: 1px solid var(--panel-border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-shrink: 0;
    }}
    .log-list {{
      flex: 1;
      min-height: 0;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 4px;
      padding-right: 2px;
    }}
    .log-item {{
      padding: 3px 6px;
      border-radius: 4px;
      background: rgba(255,255,255,0.03);
      font-size: 0.68rem;
      line-height: 1.3;
      border-left: 3px solid transparent;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .log-item:hover {{
      background: rgba(255,255,255,0.07);
    }}
    .log-item.active {{
      background: rgba(255,255,255,0.12);
      border-left-color: var(--gold);
    }}
    .log-item.red {{ border-left-color: var(--red-team); }}
    .log-item.blue {{ border-left-color: var(--blue-team); }}
    .log-item.green {{ border-left-color: var(--green-team); }}
    .log-item.score-event {{
      background: rgba(255, 165, 2, 0.1);
      border-left-color: var(--gold);
    }}

    /* 底部播控条 */
    .controls-dock {{
      background: var(--card-bg);
      border: 1px solid var(--panel-border);
      border-radius: 8px;
      padding: 4px 12px;
      margin-top: 5px;
      display: flex;
      align-items: center;
      gap: 10px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.3);
      flex-shrink: 0;
      min-height: 34px;
    }}
    .btn-group {{
      display: flex;
      gap: 4px;
    }}
    button {{
      background: #1e293b;
      border: 1px solid rgba(255,255,255,0.15);
      color: var(--text-main);
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 0.74rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 4px;
      transition: all 0.15s;
    }}
    button:hover {{
      background: #334155;
      border-color: rgba(255,255,255,0.3);
    }}
    button.primary {{
      background: #2563eb;
      border-color: #3b82f6;
    }}
    button.primary:hover {{
      background: #1d4ed8;
    }}
    .timeline-slider {{
      flex: 1;
      accent-color: #38bdf8;
      cursor: pointer;
      height: 4px;
    }}
    .control-select {{
      background: #1e293b;
      border: 1px solid rgba(255,255,255,0.15);
      color: var(--text-main);
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 0.72rem;
      cursor: pointer;
    }}
  </style>
</head>
<body>

  <!-- 顶部状态栏 -->
  <header>
    <div class="title-area">
      <h1>🃏 TCG-AI 对局全景交互式回放器</h1>
      <p>双路卡牌对战引擎 (DuelEnv) | PPO 强化学习自博弈</p>
    </div>
    <div class="match-status">
      <span class="turn-badge" id="turnDisplay">回合 0 / 0</span>
      <div id="winnerBanner"></div>
    </div>
  </header>

  <!-- 主战场网格 -->
  <div class="arena">
    <div class="battle-field-container">
      
      <!-- 上方 P1 信息栏 (默认后手) -->
      <div class="player-strip" id="p1Strip">
        <div class="player-name" id="p1Name">P1 玩家</div>
        <div class="resource-group">
          <div class="mana-container" id="blueManaContainer"></div>
          <span class="res-text" id="blueManaText">法力: 0/0</span>
          <div class="score-stars" id="blueStars"></div>
          <span class="res-text" style="font-weight:bold;" id="blueScoreText">0/7 分</span>
          <span class="res-text" id="p1HandCount">🃏0</span>
          <button class="hand-toggle" onclick="toggleHand('p1')" title="展开/收起手牌">👁</button>
        </div>
      </div>
      <div class="hand-panel open" id="p1HandPanel"></div>

      <!-- 双路核心战场 -->
      <div class="lanes-grid">
        <!-- 左路 -->
        <div class="lane-box" id="lane0">
          <div class="lane-title">LEFT LANE</div>
          
          <div class="lane-half">
            <div class="zone-label"><span id="l0_p1_def_lbl">🛡️ P1 防守区</span></div>
            <div class="unit-lane-zone defense" id="l0_blue_def"></div>
            <div class="zone-label" style="margin-top:2px;"><span id="l0_p1_atk_lbl">⚔️ P1 冲锋区</span></div>
            <div class="unit-lane-zone" id="l0_blue_atk"></div>
          </div>

          <div class="clash-divider">
            <span>⚡ 左路对撞判定线 ⚡</span>
          </div>

          <div class="lane-half">
            <div class="zone-label"><span id="l0_p0_atk_lbl">⚔️ P0 冲锋区</span></div>
            <div class="unit-lane-zone" id="l0_red_atk"></div>
            <div class="zone-label" style="margin-top:2px;"><span id="l0_p0_def_lbl">🛡️ P0 防守区</span></div>
            <div class="unit-lane-zone defense" id="l0_red_def"></div>
          </div>
        </div>

        <!-- 右路 -->
        <div class="lane-box" id="lane1">
          <div class="lane-title">RIGHT LANE</div>
          
          <div class="lane-half">
            <div class="zone-label"><span id="l1_p1_def_lbl">🛡️ P1 防守区</span></div>
            <div class="unit-lane-zone defense" id="l1_blue_def"></div>
            <div class="zone-label" style="margin-top:2px;"><span id="l1_p1_atk_lbl">⚔️ P1 冲锋区</span></div>
            <div class="unit-lane-zone" id="l1_blue_atk"></div>
          </div>

          <div class="clash-divider">
            <span>⚡ 右路对撞判定线 ⚡</span>
          </div>

          <div class="lane-half">
            <div class="zone-label"><span id="l1_p0_atk_lbl">⚔️ P0 冲锋区</span></div>
            <div class="unit-lane-zone" id="l1_red_atk"></div>
            <div class="zone-label" style="margin-top:2px;"><span id="l1_p0_def_lbl">🛡️ P0 防守区</span></div>
            <div class="unit-lane-zone defense" id="l1_red_def"></div>
          </div>
        </div>
      </div>

      <div class="hand-panel open" id="p0HandPanel"></div>
      <!-- 下方 P0 信息栏 (先手) -->
      <div class="player-strip" id="p0Strip">
        <div class="player-name" id="p0Name">P0 玩家</div>
        <div class="resource-group">
          <div class="mana-container" id="redManaContainer"></div>
          <span class="res-text" id="redManaText">法力: 0/0</span>
          <div class="score-stars" id="redStars"></div>
          <span class="res-text" style="font-weight:bold;" id="redScoreText">0/7 分</span>
          <span class="res-text" id="p0HandCount">🃏0</span>
          <button class="hand-toggle" onclick="toggleHand('p0')" title="展开/收起手牌">👁</button>
        </div>
      </div>

    </div>

    <!-- 右侧战报日志流 -->
    <div class="log-panel">
      <h3>
        <span>📜 实时决策与攻防战报</span>
        <span id="logStepCount" style="font-size:0.7rem; color:var(--text-dim);">0 步记录</span>
      </h3>
      <div class="log-list" id="logList"></div>
    </div>
  </div>

  <!-- 底部控制坞 -->
  <div class="controls-dock">
    <div class="btn-group">
      <button onclick="goTo(0)" title="首回合">|◀</button>
      <button onclick="prevTurn()" title="上一回合">◀</button>
      <button class="primary" id="playBtn" onclick="togglePlay()">▶ 自动播放</button>
      <button onclick="nextTurn()" title="下一回合">▶</button>
      <button onclick="goTo(snapshots.length - 1)" title="末回合">▶|</button>
    </div>
    
    <input type="range" class="timeline-slider" id="scrubber" min="0" max="0" value="0" oninput="onScrub(this.value)">
    
    <select class="control-select" id="speedSelect" onchange="changeSpeed(this.value)" title="播放速度">
      <option value="1200">0.5x 慢速</option>
      <option value="600" selected>1.0x 标准</option>
      <option value="300">2.0x 快速</option>
      <option value="150">4.0x 极速</option>
    </select>

    <select class="control-select" id="zoomSelect" onchange="changeZoom(this.value)" title="页面视图缩放">
      <option value="0.75">缩放 75%</option>
      <option value="0.85">缩放 85%</option>
      <option value="0.95">缩放 95%</option>
      <option value="1.0" selected>缩放 100%</option>
      <option value="1.1">缩放 110%</option>
    </select>
  </div>

  <script>
    const snapshots = {json_data};
    const winnerInfo = {json.dumps(winner_info, ensure_ascii=False)};
    let currentIndex = 0;
    let isPlaying = false;
    let playTimer = null;
    let playSpeed = 600;

    // 动态阵营识别
    const fMeta = {{
      "Red": {{ name: "红方", colorClass: "red", icon: "🔴", style: "快攻流" }},
      "Blue": {{ name: "蓝方", colorClass: "blue", icon: "🔵", style: "控制流" }},
      "Green": {{ name: "绿方", colorClass: "green", icon: "🟢", style: "跳费流" }}
    }};
    const p0F = (snapshots[0] && snapshots[0].p0 && snapshots[0].p0.faction) || "Red";
    const p1F = (snapshots[0] && snapshots[0].p1 && snapshots[0].p1.faction) || "Blue";
    const p0Meta = fMeta[p0F] || fMeta["Red"];
    const p1Meta = fMeta[p1F] || fMeta["Blue"];

    // 初始化阵营样式与标签
    document.getElementById("p0Strip").className = `player-strip ${{p0Meta.colorClass}}`;
    document.getElementById("p0Name").className = `player-name ${{p0Meta.colorClass}}`;
    document.getElementById("p0Name").textContent = `${{p0Meta.icon}} ${{p0Meta.name}} (P0 - ${{p0Meta.style}})`;

    document.getElementById("p1Strip").className = `player-strip ${{p1Meta.colorClass}}`;
    document.getElementById("p1Name").className = `player-name ${{p1Meta.colorClass}}`;
    document.getElementById("p1Name").textContent = `${{p1Meta.icon}} ${{p1Meta.name}} (P1 - ${{p1Meta.style}})`;

    document.getElementById("l0_p1_def_lbl").textContent = `🛡️ ${{p1Meta.name}}防守区`;
    document.getElementById("l0_p1_atk_lbl").textContent = `⚔️ ${{p1Meta.name}}冲锋区`;
    document.getElementById("l0_p0_atk_lbl").textContent = `⚔️ ${{p0Meta.name}}冲锋区`;
    document.getElementById("l0_p0_def_lbl").textContent = `🛡️ ${{p0Meta.name}}防守区`;

    document.getElementById("l1_p1_def_lbl").textContent = `🛡️ ${{p1Meta.name}}防守区`;
    document.getElementById("l1_p1_atk_lbl").textContent = `⚔️ ${{p1Meta.name}}冲锋区`;
    document.getElementById("l1_p0_atk_lbl").textContent = `⚔️ ${{p0Meta.name}}冲锋区`;
    document.getElementById("l1_p0_def_lbl").textContent = `🛡️ ${{p0Meta.name}}防守区`;

    document.getElementById("scrubber").max = Math.max(0, snapshots.length - 1);

    function changeZoom(val) {{
      document.body.style.zoom = val;
    }}

    function toggleHand(who) {{
      const panel = document.getElementById(who + 'HandPanel');
      panel.classList.toggle('open');
    }}

    function renderHand(panelId, handArr, countId) {{
      const panel = document.getElementById(panelId);
      const countEl = document.getElementById(countId);
      countEl.textContent = `🃏${{(handArr || []).length}}`;
      if (!handArr || handArr.length === 0) {{
        panel.innerHTML = '<span style="color:var(--text-dim); font-size:0.6rem;">无手牌</span>';
        return;
      }}
      panel.innerHTML = handArr.map(c => {{
        const isSpell = c.dp === 0 && c.cost >= 0;
        const dpPart = c.dp > 0 ? `<span class="hc-dp">DP${{c.dp}}</span>` : `<span class="hc-spell">法术</span>`;
        const tagStr = (c.tags && c.tags.length) ? `<span style="color:var(--text-dim);">${{c.tags.join(',')}}</span>` : '';
        return `<div class="hand-card-chip"><span class="hc-cost">${{c.cost}}</span>${{c.name}} ${{dpPart}} ${{tagStr}}</div>`;
      }}).join('');
    }}

    function renderUnit(u, playerIdx, isAtk) {{
      const tagClass = u.ready ? 'ready' : 'charging';
      const tagText = u.ready ? '⚡ 就绪' : '⏳ 蓄势';
      const colorClass = playerIdx === 0 ? p0Meta.colorClass : p1Meta.colorClass;
      const readyClass = u.ready ? 'ready' : '';
      return `
        <div class="minion-card ${{colorClass}} ${{readyClass}}">
          <div class="minion-name" title="${{u.name}}">${{u.name}}</div>
          <div class="minion-stats">
            <span class="dp-badge">DP ${{u.dp}}</span>
            ${{isAtk ? `<span class="state-tag ${{tagClass}}">${{tagText}}</span>` : ''}}
          </div>
        </div>
      `;
    }}

    function renderStars(containerId, score) {{
      const el = document.getElementById(containerId);
      el.innerHTML = "";
      for (let i = 0; i < 7; i++) {{
        const s = document.createElement("span");
        s.className = "star" + (i < score ? " filled" : "");
        s.textContent = "★";
        el.appendChild(s);
      }}
    }}

    function renderMana(containerId, mana, maxMana) {{
      const el = document.getElementById(containerId);
      el.innerHTML = "";
      for (let i = 0; i < 10; i++) {{
        if (i < maxMana) {{
          const c = document.createElement("div");
          c.className = "mana-crystal" + (i < mana ? " active" : "");
          el.appendChild(c);
        }}
      }}
    }}

    function renderState(idx) {{
      if (idx < 0 || idx >= snapshots.length) return;
      currentIndex = idx;
      const snap = snapshots[idx];

      // 回合与比分
      document.getElementById("turnDisplay").textContent = `第 ${{snap.turn}} 回合 (步骤 ${{idx + 1}}/${{snapshots.length}})`;
      document.getElementById("scrubber").value = idx;

      // P1 状态
      document.getElementById("blueScoreText").textContent = `${{snap.p1.score}}/7 分`;
      document.getElementById("blueManaText").textContent = `法力: ${{snap.p1.mana}}/${{snap.p1.max_mana}}`;
      renderStars("blueStars", snap.p1.score);
      renderMana("blueManaContainer", snap.p1.mana, snap.p1.max_mana);

      // P0 状态
      document.getElementById("redScoreText").textContent = `${{snap.p0.score}}/7 分`;
      document.getElementById("redManaText").textContent = `法力: ${{snap.p0.mana}}/${{snap.p0.max_mana}}`;
      renderStars("redStars", snap.p0.score);
      renderMana("redManaContainer", snap.p0.mana, snap.p0.max_mana);

      // 手牌渲染
      renderHand("p1HandPanel", snap.p1.hand, "p1HandCount");
      renderHand("p0HandPanel", snap.p0.hand, "p0HandCount");

      // 左路随从渲染
      const l0 = snap.lanes[0];
      document.getElementById("l0_blue_def").innerHTML = (l0.p1_defenders || []).map(u => renderUnit(u, 1, false)).join("") || '<span style="color:rgba(255,255,255,0.2); font-size:0.65rem;">无驻防</span>';
      document.getElementById("l0_blue_atk").innerHTML = (l0.p1_attackers || []).map(u => renderUnit(u, 1, true)).join("") || '<span style="color:rgba(255,255,255,0.2); font-size:0.65rem;">无冲锋</span>';
      document.getElementById("l0_red_atk").innerHTML = (l0.p0_attackers || []).map(u => renderUnit(u, 0, true)).join("") || '<span style="color:rgba(255,255,255,0.2); font-size:0.65rem;">无冲锋</span>';
      document.getElementById("l0_red_def").innerHTML = (l0.p0_defenders || []).map(u => renderUnit(u, 0, false)).join("") || '<span style="color:rgba(255,255,255,0.2); font-size:0.65rem;">无驻防</span>';

      // 右路随从渲染
      const l1 = snap.lanes[1];
      document.getElementById("l1_blue_def").innerHTML = (l1.p1_defenders || []).map(u => renderUnit(u, 1, false)).join("") || '<span style="color:rgba(255,255,255,0.2); font-size:0.65rem;">无驻防</span>';
      document.getElementById("l1_blue_atk").innerHTML = (l1.p1_attackers || []).map(u => renderUnit(u, 1, true)).join("") || '<span style="color:rgba(255,255,255,0.2); font-size:0.65rem;">无冲锋</span>';
      document.getElementById("l1_red_atk").innerHTML = (l1.p0_attackers || []).map(u => renderUnit(u, 0, true)).join("") || '<span style="color:rgba(255,255,255,0.2); font-size:0.65rem;">无冲锋</span>';
      document.getElementById("l1_red_def").innerHTML = (l1.p0_defenders || []).map(u => renderUnit(u, 0, false)).join("") || '<span style="color:rgba(255,255,255,0.2); font-size:0.65rem;">无驻防</span>';

      // 日志高亮与自动滚动
      const items = document.querySelectorAll(".log-item");
      items.forEach((item, i) => {{
        item.classList.toggle("active", i === idx);
        if (i === idx) {{
          item.scrollIntoView({{ behavior: "smooth", block: "nearest" }});
        }}
      }});

      if (idx === snapshots.length - 1) {{
        document.getElementById("winnerBanner").textContent = `🏆 获胜方: 【${{winnerInfo}}】`;
        document.getElementById("winnerBanner").style.color = winnerInfo.includes(p0Meta.name) ? `var(--${{p0Meta.colorClass}}-team)` : `var(--${{p1Meta.colorClass}}-team)`;
      }} else {{
        document.getElementById("winnerBanner").textContent = "";
      }}
    }}

    function buildLogList() {{
      const list = document.getElementById("logList");
      list.innerHTML = "";
      snapshots.forEach((snap, idx) => {{
        const d = document.createElement("div");
        const isP0 = snap.acting_player === 0;
        const meta = isP0 ? p0Meta : p1Meta;
        const actColor = meta.colorClass;
        const scoreEvent = snap.result_log && snap.result_log.includes("斩获") ? "score-event" : "";
        d.className = `log-item ${{actColor}} ${{scoreEvent}}`;
        d.onclick = () => goTo(idx);
        d.innerHTML = `
          <div style="display:flex; justify-content:space-between; margin-bottom:1px; font-weight:bold;">
            <span>[第 ${{snap.turn}} 回合] ${{meta.icon}} ${{meta.name}}</span>
            <span style="font-size:0.62rem; color:var(--text-dim);">#${{idx + 1}}</span>
          </div>
          <div>${{snap.action_desc}}</div>
          ${{snap.result_log ? `<div style="color:var(--gold); margin-top:2px;">${{snap.result_log}}</div>` : ''}}
        `;
        list.appendChild(d);
      }});
      document.getElementById("logStepCount").textContent = `${{snapshots.length}} 步记录`;
    }}

    function nextTurn() {{
      if (currentIndex < snapshots.length - 1) {{
        renderState(currentIndex + 1);
      }} else {{
        pause();
      }}
    }}

    function prevTurn() {{
      if (currentIndex > 0) {{
        renderState(currentIndex - 1);
      }}
    }}

    function goTo(idx) {{
      pause();
      renderState(idx);
    }}

    function onScrub(val) {{
      goTo(parseInt(val, 10));
    }}

    function togglePlay() {{
      if (isPlaying) {{
        pause();
      }} else {{
        play();
      }}
    }}

    function play() {{
      if (currentIndex >= snapshots.length - 1) {{
        currentIndex = 0;
      }}
      isPlaying = true;
      document.getElementById("playBtn").textContent = "⏸ 暂停";
      playTimer = setInterval(nextTurn, playSpeed);
    }}

    function pause() {{
      isPlaying = false;
      document.getElementById("playBtn").textContent = "▶ 播放";
      if (playTimer) {{
        clearInterval(playTimer);
        playTimer = null;
      }}
    }}

    function changeSpeed(val) {{
      playSpeed = parseInt(val, 10);
      if (isPlaying) {{
        pause();
        play();
      }}
    }}

    // 快捷键支持
    window.addEventListener("keydown", (e) => {{
      if (e.code === "Space") {{
        e.preventDefault();
        togglePlay();
      }} else if (e.code === "ArrowRight") {{
        nextTurn();
      }} else if (e.code === "ArrowLeft") {{
        prevTurn();
      }}
    }});

    // 初始化加载
    buildLogList();
    renderState(0);
  </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # 自动同步更新两处路径 (根目录与子目录)，确保用户在任意位置打开都能看到最新版
    try:
        base_name = os.path.basename(output_path)
        if base_name == "battle_replay.html":
            parent = os.path.abspath(os.path.dirname(os.path.abspath(output_path))).lower()
            cur_dir = os.path.abspath(os.path.dirname(__file__))
            root_dir = os.path.abspath(os.path.join(cur_dir, ".."))
            if parent == cur_dir.lower():
                root_target = os.path.join(root_dir, "battle_replay.html")
                with open(root_target, "w", encoding="utf-8") as rf:
                    rf.write(html_content)
            elif parent == root_dir.lower():
                sub_target = os.path.join(cur_dir, "battle_replay.html")
                with open(sub_target, "w", encoding="utf-8") as sf:
                    sf.write(html_content)
    except Exception:
        pass

    return output_path

