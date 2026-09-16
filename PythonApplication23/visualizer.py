import os
import json

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
    生成高信息密度、对齐工整的双路战场控制台可视化看板 (支持红/蓝/绿自适应阵营)
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
    导出独立可交互的双路对战 HTML5 回放器（现代暗黑拟态风格）
    """
    json_data = json.dumps(snapshots, ensure_ascii=False)
    
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>🃏 TCG-AI 对局全景回放与战术看板</title>
  <style>
    :root {{
      --bg-dark: #0b0f19;
      --card-bg: rgba(22, 30, 46, 0.85);
      --panel-border: rgba(255, 255, 255, 0.08);
      --red-team: #ff4757;
      --red-glow: rgba(255, 71, 87, 0.4);
      --blue-team: #1e90ff;
      --blue-glow: rgba(30, 144, 255, 0.4);
      --gold: #ffa502;
      --green: #2ed573;
      --text-main: #f1f2f6;
      --text-dim: #a4b0be;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: radial-gradient(circle at 50% 20%, #151e33 0%, var(--bg-dark) 100%);
      color: var(--text-main);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Microsoft YaHei", sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      padding: 16px;
      overflow-x: hidden;
    }}

    /* 顶部导航与比分板 */
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--card-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--panel-border);
      border-radius: 12px;
      padding: 12px 24px;
      margin-bottom: 16px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }}
    .title-area h1 {{
      font-size: 1.25rem;
      font-weight: 700;
      letter-spacing: 0.5px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .title-area p {{
      font-size: 0.8rem;
      color: var(--text-dim);
      margin-top: 2px;
    }}
    .match-status {{
      text-align: center;
    }}
    .turn-badge {{
      background: rgba(255,255,255,0.1);
      padding: 4px 16px;
      border-radius: 20px;
      font-size: 0.9rem;
      font-weight: bold;
      color: var(--gold);
    }}

    /* 主战场网格 */
    .arena {{
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 16px;
      flex: 1;
    }}

    /* 战局主视窗 */
    .battle-field-container {{
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}

    /* 玩家状态条 */
    .player-strip {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--card-bg);
      border: 1px solid var(--panel-border);
      border-radius: 10px;
      padding: 10px 18px;
      transition: all 0.3s;
    }}
    .player-strip.blue {{ border-left: 5px solid var(--blue-team); }}
    .player-strip.red {{ border-left: 5px solid var(--red-team); }}
    .player-name {{ font-weight: bold; font-size: 1rem; display: flex; align-items: center; gap: 8px; }}
    .player-name.blue {{ color: var(--blue-team); text-shadow: 0 0 10px var(--blue-glow); }}
    .player-name.red {{ color: var(--red-team); text-shadow: 0 0 10px var(--red-glow); }}

    .resource-group {{
      display: flex;
      align-items: center;
      gap: 20px;
    }}
    .mana-container {{
      display: flex;
      align-items: center;
      gap: 4px;
    }}
    .mana-crystal {{
      width: 14px;
      height: 18px;
      background: rgba(255,255,255,0.15);
      clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
      transition: all 0.2s;
    }}
    .mana-crystal.active {{
      background: #00d2d3;
      box-shadow: 0 0 8px #00d2d3;
    }}

    .score-stars {{
      display: flex;
      gap: 3px;
      align-items: center;
    }}
    .star {{
      font-size: 1.1rem;
      color: rgba(255,255,255,0.2);
    }}
    .star.filled {{
      color: var(--gold);
      text-shadow: 0 0 8px rgba(255, 165, 2, 0.8);
    }}

    /* 双路战场 */
    .lanes-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      flex: 1;
    }}
    .lane-box {{
      background: rgba(15, 23, 42, 0.7);
      border: 1px solid var(--panel-border);
      border-radius: 12px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      position: relative;
      overflow: hidden;
    }}
    .lane-title {{
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      font-size: 1.6rem;
      font-weight: 900;
      color: rgba(255,255,255,0.03);
      pointer-events: none;
      letter-spacing: 4px;
    }}

    .unit-lane-zone {{
      min-height: 85px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
      padding: 6px 8px;
      border-radius: 8px;
      background: rgba(255,255,255,0.02);
    }}
    .unit-lane-zone.defense {{ border-style: dashed; border-width: 1px; border-color: rgba(255,255,255,0.08); }}
    .zone-label {{
      width: 100%;
      font-size: 0.7rem;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 1px;
      display: flex;
      justify-content: space-between;
    }}

    /* 对撞中线 */
    .clash-divider {{
      margin: 10px 0;
      border-top: 1px dashed rgba(255, 255, 255, 0.15);
      position: relative;
      text-align: center;
    }}
    .clash-divider span {{
      position: relative;
      top: -10px;
      background: #1e293b;
      padding: 2px 10px;
      border-radius: 10px;
      font-size: 0.7rem;
      color: var(--text-dim);
      border: 1px solid rgba(255,255,255,0.1);
    }}

    /* 卡牌随从视效 */
    .minion-card {{
      background: linear-gradient(145deg, #1e293b, #0f172a);
      border: 1px solid rgba(255,255,255,0.15);
      border-radius: 8px;
      padding: 6px 10px;
      min-width: 90px;
      display: flex;
      flex-direction: column;
      gap: 4px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.5);
      position: relative;
      animation: popIn 0.25s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }}
    @keyframes popIn {{
      0% {{ transform: scale(0.8); opacity: 0; }}
      100% {{ transform: scale(1); opacity: 1; }}
    }}
    .minion-card.red {{ border-top: 3px solid var(--red-team); }}
    .minion-card.blue {{ border-top: 3px solid var(--blue-team); }}
    .minion-card.ready {{
      box-shadow: 0 0 10px rgba(46, 213, 115, 0.4);
      border-color: var(--green);
    }}
    .minion-name {{
      font-size: 0.8rem;
      font-weight: bold;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .minion-stats {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.75rem;
    }}
    .dp-badge {{
      background: #374151;
      padding: 1px 6px;
      border-radius: 4px;
      font-weight: bold;
      color: #38bdf8;
    }}
    .state-tag {{
      font-size: 0.65rem;
      padding: 1px 4px;
      border-radius: 3px;
    }}
    .state-tag.ready {{ background: rgba(46, 213, 115, 0.2); color: #2ed573; }}
    .state-tag.charging {{ background: rgba(255, 165, 2, 0.2); color: #ffa502; }}

    /* 右侧事件日志面板 */
    .log-panel {{
      background: var(--card-bg);
      border: 1px solid var(--panel-border);
      border-radius: 12px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      height: 100%;
    }}
    .log-panel h3 {{
      font-size: 0.95rem;
      margin-bottom: 12px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--panel-border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .log-list {{
      flex: 1;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 8px;
      padding-right: 4px;
    }}
    .log-item {{
      padding: 8px 10px;
      border-radius: 6px;
      background: rgba(255,255,255,0.03);
      font-size: 0.8rem;
      line-height: 1.4;
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
    .log-item.score-event {{
      background: rgba(255, 165, 2, 0.1);
      border-left-color: var(--gold);
    }}

    /* 底部播控条 */
    .controls-dock {{
      background: var(--card-bg);
      border: 1px solid var(--panel-border);
      border-radius: 12px;
      padding: 12px 24px;
      margin-top: 16px;
      display: flex;
      align-items: center;
      gap: 20px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }}
    .btn-group {{
      display: flex;
      gap: 8px;
    }}
    button {{
      background: #1e293b;
      border: 1px solid rgba(255,255,255,0.15);
      color: var(--text-main);
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 0.85rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
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
    }}
    .speed-select {{
      background: #1e293b;
      border: 1px solid rgba(255,255,255,0.15);
      color: var(--text-main);
      padding: 6px 10px;
      border-radius: 6px;
      font-size: 0.8rem;
    }}
  </style>
</head>
<body>

  <!-- 顶部状态栏 -->
  <header>
    <div class="title-area">
      <h1>🃏 TCG-AI 对局全景交互式回放器</h1>
      <p>双路集换式卡牌对战引擎 (DuelEnv) | PPO 强化学习自博弈</p>
    </div>
    <div class="match-status">
      <span class="turn-badge" id="turnDisplay">回合 0 / 0</span>
      <div id="winnerBanner" style="font-size:0.85rem; margin-top:4px; font-weight:bold;"></div>
    </div>
  </header>

  <!-- 主战场网格 -->
  <div class="arena">
    <div class="battle-field-container">
      
      <!-- 蓝方信息栏 -->
      <div class="player-strip blue">
        <div class="player-name blue">🔵 蓝方 (P1 - 控制防守流)</div>
        <div class="resource-group">
          <div class="mana-container" id="blueManaContainer"></div>
          <span style="font-size:0.85rem;" id="blueManaText">法力: 0/0</span>
          <div class="score-stars" id="blueStars"></div>
          <span style="font-size:0.85rem; font-weight:bold;" id="blueScoreText">0/7 分</span>
        </div>
      </div>

      <!-- 双路核心战场 -->
      <div class="lanes-grid">
        <!-- 左路 -->
        <div class="lane-box" id="lane0">
          <div class="lane-title">LEFT LANE</div>
          <div>
            <div class="zone-label"><span>🛡️ 蓝方防守区</span></div>
            <div class="unit-lane-zone defense" id="l0_blue_def"></div>
            <div class="zone-label" style="margin-top:6px;"><span>⚔️ 蓝方冲锋区</span></div>
            <div class="unit-lane-zone" id="l0_blue_atk"></div>
          </div>

          <div class="clash-divider">
            <span>⚡ 左路对撞判定线 ⚡</span>
          </div>

          <div>
            <div class="zone-label"><span>⚔️ 红方冲锋区</span></div>
            <div class="unit-lane-zone" id="l0_red_atk"></div>
            <div class="zone-label" style="margin-top:6px;"><span>🛡️ 红方防守区</span></div>
            <div class="unit-lane-zone defense" id="l0_red_def"></div>
          </div>
        </div>

        <!-- 右路 -->
        <div class="lane-box" id="lane1">
          <div class="lane-title">RIGHT LANE</div>
          <div>
            <div class="zone-label"><span>🛡️ 蓝方防守区</span></div>
            <div class="unit-lane-zone defense" id="l1_blue_def"></div>
            <div class="zone-label" style="margin-top:6px;"><span>⚔️ 蓝方冲锋区</span></div>
            <div class="unit-lane-zone" id="l1_blue_atk"></div>
          </div>

          <div class="clash-divider">
            <span>⚡ 右路对撞判定线 ⚡</span>
          </div>

          <div>
            <div class="zone-label"><span>⚔️ 红方冲锋区</span></div>
            <div class="unit-lane-zone" id="l1_red_atk"></div>
            <div class="zone-label" style="margin-top:6px;"><span>🛡️ 红方防守区</span></div>
            <div class="unit-lane-zone defense" id="l1_red_def"></div>
          </div>
        </div>
      </div>

      <!-- 红方信息栏 -->
      <div class="player-strip red">
        <div class="player-name red">🔴 红方 (P0 - 快攻突破流)</div>
        <div class="resource-group">
          <div class="mana-container" id="redManaContainer"></div>
          <span style="font-size:0.85rem;" id="redManaText">法力: 0/0</span>
          <div class="score-stars" id="redStars"></div>
          <span style="font-size:0.85rem; font-weight:bold;" id="redScoreText">0/7 分</span>
        </div>
      </div>

    </div>

    <!-- 右侧战报日志流 -->
    <div class="log-panel">
      <h3>
        <span>📜 实时决策与攻防战报</span>
        <span id="logStepCount" style="font-size:0.75rem; color:var(--text-dim);">0 步记录</span>
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
    
    <select class="speed-select" id="speedSelect" onchange="changeSpeed(this.value)">
      <option value="1200">0.5x 慢速</option>
      <option value="600" selected>1.0x 标准</option>
      <option value="300">2.0x 快速</option>
      <option value="150">4.0x 极速</option>
    </select>
  </div>

  <script>
    const snapshots = {json_data};
    const winnerInfo = {json.dumps(winner_info, ensure_ascii=False)};
    let currentIndex = 0;
    let isPlaying = false;
    let playTimer = null;
    let playSpeed = 600;

    document.getElementById("scrubber").max = Math.max(0, snapshots.length - 1);

    function renderUnit(u, isRed, isAtk) {{
      const tagClass = u.ready ? 'ready' : 'charging';
      const tagText = u.ready ? '⚡ 就绪' : '⏳ 蓄势';
      const colorClass = isRed ? 'red' : 'blue';
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

      // 蓝方状态
      document.getElementById("blueScoreText").textContent = `${{snap.p1.score}}/7 分`;
      document.getElementById("blueManaText").textContent = `法力: ${{snap.p1.mana}}/${{snap.p1.max_mana}}`;
      renderStars("blueStars", snap.p1.score);
      renderMana("blueManaContainer", snap.p1.mana, snap.p1.max_mana);

      // 红方状态
      document.getElementById("redScoreText").textContent = `${{snap.p0.score}}/7 分`;
      document.getElementById("redManaText").textContent = `法力: ${{snap.p0.mana}}/${{snap.p0.max_mana}}`;
      renderStars("redStars", snap.p0.score);
      renderMana("redManaContainer", snap.p0.mana, snap.p0.max_mana);

      // 左路随从渲染
      const l0 = snap.lanes[0];
      document.getElementById("l0_blue_def").innerHTML = (l0.p1_defenders || []).map(u => renderUnit(u, false, false)).join("") || '<span style="color:rgba(255,255,255,0.2); font-size:0.75rem;">无驻防单位</span>';
      document.getElementById("l0_blue_atk").innerHTML = (l0.p1_attackers || []).map(u => renderUnit(u, false, true)).join("") || '<span style="color:rgba(255,255,255,0.2); font-size:0.75rem;">无冲锋单位</span>';
      document.getElementById("l0_red_atk").innerHTML = (l0.p0_attackers || []).map(u => renderUnit(u, true, true)).join("") || '<span style="color:rgba(255,255,255,0.2); font-size:0.75rem;">无冲锋单位</span>';
      document.getElementById("l0_red_def").innerHTML = (l0.p0_defenders || []).map(u => renderUnit(u, true, false)).join("") || '<span style="color:rgba(255,255,255,0.2); font-size:0.75rem;">无驻防单位</span>';

      // 右路随从渲染
      const l1 = snap.lanes[1];
      document.getElementById("l1_blue_def").innerHTML = (l1.p1_defenders || []).map(u => renderUnit(u, false, false)).join("") || '<span style="color:rgba(255,255,255,0.2); font-size:0.75rem;">无驻防单位</span>';
      document.getElementById("l1_blue_atk").innerHTML = (l1.p1_attackers || []).map(u => renderUnit(u, false, true)).join("") || '<span style="color:rgba(255,255,255,0.2); font-size:0.75rem;">无冲锋单位</span>';
      document.getElementById("l1_red_atk").innerHTML = (l1.p0_attackers || []).map(u => renderUnit(u, true, true)).join("") || '<span style="color:rgba(255,255,255,0.2); font-size:0.75rem;">无冲锋单位</span>';
      document.getElementById("l1_red_def").innerHTML = (l1.p0_defenders || []).map(u => renderUnit(u, true, false)).join("") || '<span style="color:rgba(255,255,255,0.2); font-size:0.75rem;">无驻防单位</span>';

      // 日志高亮与自动滚动
      const items = document.querySelectorAll(".log-item");
      items.forEach((item, i) => {{
        item.classList.toggle("active", i === idx);
        if (i === idx) {{
          item.scrollIntoView({{ behavior: "smooth", block: "nearest" }});
        }}
      }});

      if (idx === snapshots.length - 1) {{
        document.getElementById("winnerBanner").textContent = `🏆 对局结束！获胜方: 【${{winnerInfo}}】`;
        document.getElementById("winnerBanner").style.color = winnerInfo.includes("红方") ? "var(--red-team)" : "var(--blue-team)";
      }} else {{
        document.getElementById("winnerBanner").textContent = "";
      }}
    }}

    function buildLogList() {{
      const list = document.getElementById("logList");
      list.innerHTML = "";
      snapshots.forEach((snap, idx) => {{
        const d = document.createElement("div");
        const actColor = snap.acting_player === 0 ? "red" : "blue";
        const scoreEvent = snap.result_log && snap.result_log.includes("斩获") ? "score-event" : "";
        d.className = `log-item ${{actColor}} ${{scoreEvent}}`;
        d.onclick = () => goTo(idx);
        d.innerHTML = `
          <div style="display:flex; justify-content:space-between; margin-bottom:2px; font-weight:bold;">
            <span>[第 ${{snap.turn}} 回合] ${{snap.acting_player === 0 ? '🔴 红方' : '🔵 蓝方'}}</span>
            <span style="font-size:0.7rem; color:var(--text-dim);">#${{idx + 1}}</span>
          </div>
          <div>${{snap.action_desc}}</div>
          ${{snap.result_log ? `<div style="color:var(--gold); margin-top:3px;">${{snap.result_log}}</div>` : ''}}
        `;
        list.appendChild(d);
      }});
      document.getElementById("logStepCount").textContent = `${{snapshots.length}} 步动作`;
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
      document.getElementById("playBtn").textContent = "⏸ 暂停播放";
      playTimer = setInterval(nextTurn, playSpeed);
    }}

    function pause() {{
      isPlaying = false;
      document.getElementById("playBtn").textContent = "▶ 自动播放";
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
    return output_path
