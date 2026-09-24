let arenaSocket = null;
let currentGameState = null;
let selectedCardIndex = -1;
let currentMode = "pve"; // "pve" or "aivai"
let isAiAutoplaying = false;

function initArenaSocket() {
    if (arenaSocket && (arenaSocket.readyState === WebSocket.OPEN || arenaSocket.readyState === WebSocket.CONNECTING)) {
        return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    arenaSocket = new WebSocket(`${protocol}//${window.location.host}/ws/arena`);

    arenaSocket.onopen = () => {
        console.log("[Arena] Connected to duel server.");
    };

    arenaSocket.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            if (msg.type === 'state') {
                currentGameState = msg.data;
                renderArena(msg.data);
            }
        } catch (e) {
            console.error("[Arena] Error parsing socket message:", e);
        }
    };

    arenaSocket.onclose = () => {
        console.log("[Arena] Connection closed. Auto-reconnecting in 2s...");
        setTimeout(initArenaSocket, 2000);
    };

    arenaSocket.onerror = (err) => {
        console.error("[Arena] WebSocket error:", err);
    };
}

// Auto-connect immediately
document.addEventListener('DOMContentLoaded', initArenaSocket);
if (document.readyState === 'complete' || document.readyState === 'interactive') {
    initArenaSocket();
}

// ---------------- Mode Switching ----------------
const modePveBtn = document.getElementById('mode-pve-btn');
const modeAivaiBtn = document.getElementById('mode-aivai-btn');
const pveControls = document.getElementById('pve-controls-group');
const aivaiControls = document.getElementById('aivai-controls-group');

if (modePveBtn && modeAivaiBtn) {
    modePveBtn.onclick = () => {
        currentMode = "pve";
        modePveBtn.classList.add('active');
        modeAivaiBtn.classList.remove('active');
        pveControls.style.display = 'flex';
        aivaiControls.style.display = 'none';
        
        // Stop any running autoplay if switching to PvE
        if (arenaSocket && arenaSocket.readyState === WebSocket.OPEN && isAiAutoplaying) {
            isAiAutoplaying = false;
            arenaSocket.send(JSON.stringify({ type: 'toggle_autoplay', enabled: false }));
        }
        startMatchWithSettings();
    };

    modeAivaiBtn.onclick = () => {
        currentMode = "aivai";
        modeAivaiBtn.classList.add('active');
        modePveBtn.classList.remove('active');
        aivaiControls.style.display = 'flex';
        pveControls.style.display = 'none';
        startMatchWithSettings();
    };
}

// ---------------- Match Reset / Start ----------------
window.updateArenaDeckSelectOptions = function() {
    const p0Select = document.getElementById('pve-p0-deck');
    if (!p0Select) return;
    const curVal = p0Select.value;

    const decks = window.userDecks || (window.userDeck ? [window.userDeck] : []);
    let customOptsHtml = '';
    decks.forEach((d, idx) => {
        const count = d.card_ids ? d.card_ids.length : 0;
        const fIcons = { 'Red': '🔴', 'Blue': '🔵', 'Green': '🟢' };
        const icon = fIcons[d.faction] || '⚪';
        customOptsHtml += `<option value="custom_${idx}">卡组${idx + 1}: ${icon} ${d.name || `自定义卡组${idx+1}`} (${count}/30张)</option>`;
    });

    p0Select.innerHTML = `
        <optgroup label="官方预设">
            <option value="preset_Red">🔴 赤红 · 快攻爆发流 (30张)</option>
            <option value="preset_Blue">🔵 蔚蓝 · 坚守控制流 (30张)</option>
            <option value="preset_Green">🟢 翠绿 · 成长跳费流 (30张)</option>
        </optgroup>
        <optgroup label="自定义卡组 (最多6套)">
            ${customOptsHtml || '<option disabled>暂无自定义卡组</option>'}
        </optgroup>
    `;

    if (curVal && p0Select.querySelector(`option[value="${curVal}"]`)) {
        p0Select.value = curVal;
    } else {
        p0Select.value = "preset_Red";
    }
};

window.startMatchWithSettings = async function(autoStartAi = false) {
    if (!arenaSocket || arenaSocket.readyState !== WebSocket.OPEN) {
        initArenaSocket();
        setTimeout(() => startMatchWithSettings(autoStartAi), 300);
        return;
    }

    let p0Faction = "Red";
    let p1Faction = "Blue";
    let p0Decklist = null;
    let p1Decklist = null;

    if (currentMode === "pve") {
        const p0DeckChoice = document.getElementById('pve-p0-deck')?.value || "preset_Red";
        const p1DeckChoice = document.getElementById('pve-p1-deck')?.value || "preset_Blue";

        // 1. 解析玩家 P0 出战构筑
        if (p0DeckChoice.startsWith("custom_")) {
            const slotIdx = parseInt(p0DeckChoice.replace("custom_", ""), 10);
            const ud = (window.userDecks && window.userDecks[slotIdx]) ? window.userDecks[slotIdx] : window.userDeck;
            if (!ud || !ud.card_ids || ud.card_ids.length !== 30) {
                const curCount = ud?.card_ids?.length || 0;
                alert(`所选卡组【${ud?.name || '卡组 ' + (slotIdx + 1)}】尚未满 30 张（当前 ${curCount}/30 张）！\n请先前往【卡组工坊】构筑或补满卡牌后再出战。`);
                return;
            }
            p0Faction = ud.faction || "Red";
            p0Decklist = ud.card_ids;
        } else if (p0DeckChoice === "custom") {
            const ud = window.userDeck;
            if (!ud || !ud.card_ids || ud.card_ids.length !== 30) {
                const curCount = ud?.card_ids?.length || 0;
                alert(`您的自定义卡组尚未满 30 张（当前 ${curCount}/30 张）！\n请先前往【卡组工坊】构筑或补满卡牌后再出战。`);
                return;
            }
            p0Faction = ud.faction || "Red";
            p0Decklist = ud.card_ids;
        } else {
            const f0 = p0DeckChoice.replace("preset_", "");
            p0Faction = f0;
            if (window.tcgData && window.tcgData.decks && window.tcgData.decks[f0]) {
                p0Decklist = window.tcgData.decks[f0].decklist;
            }
        }

        // 2. 解析对手 P1 AI 出战构筑
        const f1 = p1DeckChoice.replace("preset_", "");
        p1Faction = f1;
        if (window.tcgData && window.tcgData.decks && window.tcgData.decks[f1]) {
            p1Decklist = window.tcgData.decks[f1].decklist;
        }
    } else {
        p0Faction = document.getElementById('aivai-p0-faction').value;
        p1Faction = document.getElementById('aivai-p1-faction').value;
        if (window.tcgData && window.tcgData.decks) {
            p0Decklist = window.tcgData.decks[p0Faction]?.decklist || null;
            p1Decklist = window.tcgData.decks[p1Faction]?.decklist || null;
        }
    }

    selectedCardIndex = -1;
    clearHighlights();

    arenaSocket.send(JSON.stringify({
        type: 'reset',
        mode: currentMode,
        p0_faction: p0Faction,
        p1_faction: p1Faction,
        p0_decklist: p0Decklist,
        p1_decklist: p1Decklist,
        auto_start: autoStartAi
    }));
};

const startGameBtn = document.getElementById('btn-start-game');
if (startGameBtn) {
    startGameBtn.addEventListener('click', () => {
        // Prevent click from resetting match when game is actively in progress
        if (currentGameState && currentGameState.is_started && !currentGameState.done) {
            showGameToast("当前对局正在进行中。如需重新开局请点击【重新开始】", "info");
            return;
        }
        startMatchWithSettings();
    });
}
const restartGameBtn = document.getElementById('btn-restart-game');
if (restartGameBtn) {
    restartGameBtn.addEventListener('click', () => {
        if (currentGameState && currentGameState.is_started && !currentGameState.done) {
            if (!confirm("确定要放弃当前对局并重新开始吗？")) return;
        }
        startMatchWithSettings();
    });
}

// AI vs AI Controls
const aivaiToggleBtn = document.getElementById('btn-aivai-toggle');
if (aivaiToggleBtn) {
    aivaiToggleBtn.onclick = () => {
        if (!arenaSocket || arenaSocket.readyState !== WebSocket.OPEN) return;
        isAiAutoplaying = !isAiAutoplaying;
        arenaSocket.send(JSON.stringify({ type: 'toggle_autoplay', enabled: isAiAutoplaying }));
        aivaiToggleBtn.innerText = isAiAutoplaying ? "暂停推演" : "开始推演";
        aivaiToggleBtn.className = isAiAutoplaying ? "btn gold" : "btn primary";
    };
}

const aivaiStepBtn = document.getElementById('btn-aivai-step');
if (aivaiStepBtn) {
    aivaiStepBtn.onclick = () => {
        if (arenaSocket && arenaSocket.readyState === WebSocket.OPEN) {
            arenaSocket.send(JSON.stringify({ type: 'ai_step' }));
        }
    };
}

const aivaiResetBtn = document.getElementById('btn-aivai-reset');
if (aivaiResetBtn) {
    aivaiResetBtn.onclick = () => {
        isAiAutoplaying = false;
        if (aivaiToggleBtn) {
            aivaiToggleBtn.innerText = "开始推演";
            aivaiToggleBtn.className = "btn primary";
        }
        startMatchWithSettings();
    };
}

const aivaiSpeedSelect = document.getElementById('aivai-speed');
if (aivaiSpeedSelect) {
    aivaiSpeedSelect.onchange = (e) => {
        if (arenaSocket && arenaSocket.readyState === WebSocket.OPEN) {
            arenaSocket.send(JSON.stringify({ type: 'set_speed', speed: parseFloat(e.target.value) }));
        }
    };
}


// ---------------- Game UI Feedback & Notifications ----------------
let lastAnnouncedTurn = 0;
let lastAnnouncedLog = "";

function showGameToast(msg, type = "info") {
    const container = document.getElementById('arena-toast-container');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `arena-toast ${type}`;
    const icon = type === 'warning' ? '⚠️' : (type === 'error' ? '❌' : (type === 'spell' ? '✨' : (type === 'success' ? '✓' : 'ℹ️')));
    toast.innerHTML = `<span>${icon}</span><span>${msg}</span>`;
    container.appendChild(toast);
    setTimeout(() => {
        if (toast.parentNode) toast.parentNode.removeChild(toast);
    }, 2400);
}

function showTurnBanner(msg, type = "turn") {
    const banner = document.getElementById('arena-banner');
    if (!banner) return;
    banner.className = `arena-banner ${type}`;
    banner.innerHTML = msg;
    banner.classList.remove('hidden');
    setTimeout(() => {
        banner.classList.add('hidden');
    }, 1800);
}

function updateActionPrompt(text, active = false) {
    const promptText = document.getElementById('prompt-text');
    const promptBar = document.getElementById('action-prompt-bar');
    if (promptText) promptText.textContent = text;
    if (promptBar) {
        if (active) promptBar.classList.add('active');
        else promptBar.classList.remove('active');
    }
}

function getCardEffectDescription(card) {
    const lines = [];
    const isSpell = card.type === 'SPELL';
    if (!isSpell) {
        lines.push(`战力: <strong>${card.dp}</strong> 点`);
    }

    const atkVal = card.atk_spell_val || 0;
    const defVal = card.def_spell_val || 0;
    if (atkVal > 0) lines.push(`攻击增益: 为本路进攻随从提供 <strong>+${atkVal}</strong> 攻击`);
    if (defVal > 0) lines.push(`护盾抵挡: 为本路战线提供 <strong>+${defVal}</strong> 点防御护盾`);

    (card.tags || []).forEach(t => {
        const tag = String(t).toUpperCase().trim();
        if (tag === 'RUSH') lines.push('<strong>突袭</strong>: 进场回合即可发起攻击');
        else if (tag === 'ATTACK_ONLY') lines.push('<strong>仅限进攻</strong>: 无法部署在防守区');
        else if (tag === 'SACRIFICE_1_KILL_1') lines.push('<strong>献祭</strong>: 牺牲本路战力最低友军，消灭敌方战力最高随从');
        else if (tag.startsWith('FORTIFY_')) lines.push(`<strong>坚守</strong>: 部署在防守区时战力 <strong>+${tag.split('_')[1]}</strong>`);
        else if (tag.startsWith('DEGRADE_')) lines.push(`<strong>破甲</strong>: 攻击结算时削减阻挡者 <strong>${tag.split('_')[1]}</strong> 点战力`);
        else if (tag.startsWith('SUPPORT_ATK_')) lines.push(`<strong>支援</strong>: 驻守防守区时为本路所有进攻者提供 <strong>+${tag.split('_')[2]}</strong> 攻击`);
        else if (tag.startsWith('BONUS_SCORE_')) lines.push(`<strong>突破</strong>: 突破防线时额外获得 <strong>+${tag.split('_')[2]}</strong> 点胜点`);
        else if (tag.startsWith('SPAWN_')) {
            const parts = tag.split('_');
            lines.push(`<strong>战吼</strong>: 召唤 ${parts[2]} 个 ${parts[1]} 战力衍生随从`);
        }
        else if (tag.startsWith('DEATH_DRAW_')) lines.push(`<strong>亡语</strong>: 阵亡时抽取 ${tag.split('_')[2]} 张牌`);
        else if (tag.startsWith('DEATH_MANA_')) lines.push(`<strong>亡语</strong>: 阵亡时法力上限 <strong>+${tag.split('_')[2]}</strong>`);
        else if (tag.startsWith('DRAW_')) lines.push(`<strong>战吼</strong>: 抽取 ${tag.split('_')[1]} 张牌`);
        else if (tag.startsWith('RAMP_')) lines.push(`<strong>战吼</strong>: 法力上限 <strong>+${tag.split('_')[1]}</strong>`);
        else if (tag.startsWith('TEMP_MANA_')) lines.push(`<strong>战吼</strong>: 获得 ${tag.split('_')[2]} 点本回合临时法力`);
        else if (tag.startsWith('DISCARD_')) lines.push(`<strong>战吼</strong>: 弃置 ${tag.split('_')[1]} 张手牌`);
    });

    if (lines.length === 0) {
        lines.push('基础随从：标准对撞单位。');
    }
    return lines;
}

function setupCardTooltip(cardEl, card) {
    cardEl.addEventListener('mouseenter', (e) => {
        const tooltip = document.getElementById('arena-card-tooltip');
        if (!tooltip) return;
        const isSpell = card.type === 'SPELL';
        const typeStr = isSpell ? '战术法术' : '战斗随从';
        const dpStr = isSpell ? '' : `<div class="tooltip-dp">战力: ${card.dp}</div>`;
        const descList = getCardEffectDescription(card);

        tooltip.innerHTML = `
            <div class="tooltip-header">
                <span class="tooltip-title">${card.name}</span>
                <span class="tooltip-cost">${card.cost}💎</span>
            </div>
            <div class="tooltip-type">${typeStr}</div>
            ${dpStr}
            <div class="tooltip-desc">${descList.join('<br>')}</div>
        `;
        tooltip.classList.remove('hidden');
        positionTooltip(e, tooltip);
    });

    cardEl.addEventListener('mousemove', (e) => {
        const tooltip = document.getElementById('arena-card-tooltip');
        if (tooltip && !tooltip.classList.contains('hidden')) {
            positionTooltip(e, tooltip);
        }
    });

    cardEl.addEventListener('mouseleave', () => {
        const tooltip = document.getElementById('arena-card-tooltip');
        if (tooltip) tooltip.classList.add('hidden');
    });
}

function positionTooltip(e, tooltip) {
    const pad = 15;
    let x = e.clientX + pad;
    let y = e.clientY - tooltip.offsetHeight - 5;
    if (x + tooltip.offsetWidth > window.innerWidth) {
        x = e.clientX - tooltip.offsetWidth - pad;
    }
    if (y < 10) {
        y = e.clientY + pad;
    }
    tooltip.style.left = `${x}px`;
    tooltip.style.top = `${y}px`;
}

function clearHighlights() {
    document.querySelectorAll('.slot.targetable').forEach(el => {
        el.classList.remove('highlight', 'spell-target', 'atk-target', 'def-target');
        el.onclick = null;
        const labelEl = el.querySelector('.slot-label');
        const code = parseInt(el.getAttribute('data-code'));
        if (labelEl) {
            if (code === 0 || code === 2) labelEl.textContent = '我方进攻区';
            else labelEl.textContent = '我方防守区';
        }
    });
    updateActionPrompt("你的回合：请选择手牌并部署至战场", false);
}

function handleCardClick(idx) {
    if (!currentGameState || currentGameState.done) return;
    const p0 = currentGameState.p0;
    const card = p0.hand[idx];
    if (!card) return;

    const handCardEls = document.querySelectorAll('#p0-hand .hand-card');
    const cardEl = handCardEls[idx];

    // 1. 拦截非玩家回合
    if (currentMode === "pve" && currentGameState.current_player !== 0) {
        showGameToast("当前为敌方回合，请等待对方行动", "info");
        return;
    }

    // 2. 拦截法力值不足
    if (card.cost > p0.mana) {
        if (cardEl) {
            cardEl.classList.add('shake');
            setTimeout(() => cardEl.classList.remove('shake'), 400);
        }
        showGameToast(`法力不足！【${card.name}】需要 ${card.cost} 费，当前仅有 ${p0.mana} 费`, "warning");
        return;
    }

    // 3. 点击同一张牌取消选择
    if (selectedCardIndex === idx) {
        selectedCardIndex = -1;
        document.querySelectorAll('.hand-card').forEach(c => c.classList.remove('selected'));
        clearHighlights();
        return;
    }

    // 4. 正常选中卡牌
    selectedCardIndex = idx;
    document.querySelectorAll('.hand-card').forEach(c => c.classList.remove('selected'));
    if (cardEl) cardEl.classList.add('selected');
    clearHighlights();

    const mask = currentGameState.action_mask;
    const isSpell = card.type === 'SPELL';

    if (isSpell) {
        updateActionPrompt(`已选法术【${card.name}】(${card.cost}费)：请点击战线施法`, true);
    } else {
        const rushNote = (card.tags && card.tags.includes('RUSH')) ? ' [突袭]' : '';
        updateActionPrompt(`已选随从【${card.name}】(${card.cost}费/战力${card.dp})${rushNote}：请点击高亮区域部署`, true);
    }

    const targets = [
        { code: 0, el: document.getElementById('target-0-0'), lane: 0, pos: 'atk' },
        { code: 1, el: document.getElementById('target-0-1'), lane: 0, pos: 'def' },
        { code: 2, el: document.getElementById('target-1-2'), lane: 1, pos: 'atk' },
        { code: 3, el: document.getElementById('target-1-3'), lane: 1, pos: 'def' },
    ];

    let anyTargetAvailable = false;

    targets.forEach(t => {
        if (!t.el) return;
        const actionId = idx * 4 + t.code;
        const labelEl = t.el.querySelector('.slot-label');

        if (mask[actionId] > 0) {
            anyTargetAvailable = true;
            t.el.classList.add('highlight');

            const isSac = (card.tags || []).includes('SACRIFICE_1_KILL_1');
            if (isSpell) {
                t.el.classList.add('spell-target');
                const spellAtk = card.atk_spell_val || 0;
                const spellDef = card.def_spell_val || 0;
                let spellDesc = '施放法术';
                if (isSac) spellDesc = '献祭消灭';
                else if (spellAtk > 0) spellDesc = `攻击 +${spellAtk}`;
                else if (spellDef > 0) spellDesc = `护盾 +${spellDef}`;
                if (labelEl) labelEl.textContent = `${spellDesc} (点击施放)`;
            } else {
                if (t.pos === 'atk') {
                    t.el.classList.add('atk-target');
                    let rushStr = (card.tags && card.tags.includes('RUSH')) ? '突袭进场' : '进攻部署';
                    if (isSac) rushStr = '战吼献祭';
                    if (labelEl) labelEl.textContent = `${rushStr} (点击部署)`;
                } else {
                    t.el.classList.add('def-target');
                    const fortifyTag = (card.tags || []).find(x => x.startsWith('FORTIFY_'));
                    let fortStr = fortifyTag ? `坚守+${fortifyTag.split('_')[1]}` : '防守部署';
                    if (isSac) fortStr = '战吼献祭';
                    if (labelEl) labelEl.textContent = `${fortStr} (点击部署)`;
                }
            }

            t.el.onclick = () => {
                if (arenaSocket && arenaSocket.readyState === WebSocket.OPEN) {
                    let actionDesc = isSpell 
                        ? `施放法术【${card.name}】至【${t.lane === 0 ? '左路' : '右路'}】` 
                        : `部署【${card.name}】至【${t.lane === 0 ? '左路' : '右路'} ${t.pos === 'atk' ? '进攻区' : '防守区'}】`;
                    if (isSac) {
                        actionDesc = `触发献祭！【${card.name}】对准【${t.lane === 0 ? '左路' : '右路'}】发动消灭`;
                    }
                    showGameToast(actionDesc, 'success');
                    arenaSocket.send(JSON.stringify({ type: 'action', action_id: actionId }));
                }
                selectedCardIndex = -1;
                clearHighlights();
            };
        } else {
            const laneObj = currentGameState.lanes[t.lane];
            const currentUnits = t.pos === 'atk' ? laneObj.p0_attackers.length : laneObj.p0_defenders.length;
            if (!isSpell && currentUnits >= 3) {
                t.el.onclick = () => {
                    showGameToast("该战线区域随从已达上限 (3/3)，无法继续部署", "warning");
                };
            }
        }
    });

    if (!anyTargetAvailable) {
        showGameToast("战场区域已满或无可用施法目标", "warning");
    }
}

document.getElementById('btn-pass').onclick = () => {
    if (!currentGameState || currentGameState.done) return;
    if (currentMode === "pve" && currentGameState.current_player !== 0) {
        showGameToast("当前非玩家回合，无法结束回合", "info");
        return;
    }
    const passAction = currentGameState.action_mask.length - 1;
    if (arenaSocket && arenaSocket.readyState === WebSocket.OPEN) {
        showTurnBanner("交锋结算", "clash");
        showGameToast("结束回合，进入交锋结算", "info");
        arenaSocket.send(JSON.stringify({ type: 'action', action_id: passAction }));
    }
    selectedCardIndex = -1;
    clearHighlights();
};

function renderDeckTracker(state) {
    if (!state) return;

    // 1. Deck & Hand Counts
    const p0DeckCount = state.p0?.deck_count !== undefined ? state.p0.deck_count : (state.p0?.deck?.length ?? 30);
    const p1DeckCount = state.p1?.deck_count !== undefined ? state.p1.deck_count : 30;
    const p0HandCount = state.p0?.hand?.length ?? 0;
    const p0GraveCount = state.p0?.graveyard_count !== undefined ? state.p0.graveyard_count : (state.p0?.graveyard?.length ?? 0);

    const p0DeckEl = document.getElementById('p0-deck-count');
    if (p0DeckEl) p0DeckEl.innerText = p0DeckCount;
    const p1DeckEl = document.getElementById('p1-deck-count');
    if (p1DeckEl) p1DeckEl.innerText = p1DeckCount;

    const trackerP0DeckNum = document.getElementById('tracker-p0-deck-num');
    if (trackerP0DeckNum) trackerP0DeckNum.innerText = p0DeckCount;
    const trackerP0HandNum = document.getElementById('tracker-p0-hand-num');
    if (trackerP0HandNum) trackerP0HandNum.innerText = p0HandCount;
    const trackerP0GraveNum = document.getElementById('tracker-p0-grave-num');
    if (trackerP0GraveNum) trackerP0GraveNum.innerText = p0GraveCount;

    const trackerRatio = document.getElementById('tracker-ratio');
    if (trackerRatio) trackerRatio.innerText = `${p0DeckCount}/30`;
    const trackerEnemyDeckStat = document.getElementById('tracker-enemy-deck-stat');
    if (trackerEnemyDeckStat) trackerEnemyDeckStat.innerText = `敌方剩余: ${p1DeckCount}张`;

    // 2. Render Player Remaining Cards List
    const p0CardsList = document.getElementById('tracker-cards-list');
    if (p0CardsList) {
        const deckCards = state.p0?.deck || [];
        if (deckCards.length === 0) {
            p0CardsList.innerHTML = `<div class="tracker-empty">${state.is_started ? '牌库已抽空' : '对局开始后显示详细记牌清单'}</div>`;
        } else {
            const countsMap = new Map();
            deckCards.forEach(c => {
                const key = c.name;
                if (!countsMap.has(key)) {
                    countsMap.set(key, { card: c, count: 0 });
                }
                countsMap.get(key).count++;
            });

            const sorted = Array.from(countsMap.values()).sort((a, b) => {
                if (a.card.cost !== b.card.cost) return a.card.cost - b.card.cost;
                return a.card.name.localeCompare(b.card.name, 'zh-Hans-CN');
            });

            p0CardsList.innerHTML = '';
            sorted.forEach(({ card, count }) => {
                const row = document.createElement('div');
                row.className = 'tracker-card-row';
                row.innerHTML = `
                    <div class="tracker-card-left">
                        <span class="tracker-cost-gem">${card.cost}</span>
                        <span class="tracker-card-name">${card.name}</span>
                    </div>
                    <span class="tracker-card-qty">x${count}</span>
                `;
                setupCardTooltip(row, card);
                p0CardsList.appendChild(row);
            });
        }
    }

    // 3. Render Enemy Revealed Cards
    const p1RevealedList = document.getElementById('tracker-enemy-revealed-list');
    if (p1RevealedList) {
        const enemyRevealedMap = new Map();
        const addCard = (c) => {
            if (!c || !c.name) return;
            const key = c.name;
            if (!enemyRevealedMap.has(key)) {
                enemyRevealedMap.set(key, { card: c, count: 0 });
            }
            enemyRevealedMap.get(key).count++;
        };

        // Gather from enemy graveyard
        (state.p1?.graveyard || []).forEach(addCard);
        // Gather from active battlefield
        (state.lanes || []).forEach(lane => {
            (lane.p1_attackers || []).forEach(addCard);
            (lane.p1_defenders || []).forEach(addCard);
        });

        if (enemyRevealedMap.size === 0) {
            p1RevealedList.innerHTML = '<div class="tracker-empty">敌方尚未打出卡牌</div>';
        } else {
            const sortedEnemy = Array.from(enemyRevealedMap.values()).sort((a, b) => {
                if (a.card.cost !== b.card.cost) return a.card.cost - b.card.cost;
                return a.card.name.localeCompare(b.card.name, 'zh-Hans-CN');
            });

            p1RevealedList.innerHTML = '';
            sortedEnemy.forEach(({ card, count }) => {
                const row = document.createElement('div');
                row.className = 'tracker-card-row';
                row.innerHTML = `
                    <div class="tracker-card-left">
                        <span class="tracker-cost-gem">${card.cost}</span>
                        <span class="tracker-card-name">${card.name}</span>
                    </div>
                    <span class="tracker-card-qty">x${count}</span>
                `;
                setupCardTooltip(row, card);
                p1RevealedList.appendChild(row);
            });
        }
    }
}

function initCombatSidePanelTabs() {
    const tabBtnTracker = document.getElementById('tab-btn-tracker');
    const tabBtnLog = document.getElementById('tab-btn-log');
    const trackerContainer = document.getElementById('deck-tracker-container');
    const logView = document.getElementById('battle-log-view');

    if (tabBtnTracker && tabBtnLog && trackerContainer && logView) {
        tabBtnTracker.onclick = () => {
            tabBtnTracker.classList.add('active');
            tabBtnLog.classList.remove('active');
            trackerContainer.classList.add('active');
            trackerContainer.style.display = 'flex';
            logView.classList.remove('active');
            logView.style.display = 'none';
        };
        tabBtnLog.onclick = () => {
            tabBtnLog.classList.add('active');
            tabBtnTracker.classList.remove('active');
            logView.classList.add('active');
            logView.style.display = 'flex';
            trackerContainer.classList.remove('active');
            trackerContainer.style.display = 'none';
        };
    }
}

function renderArena(state) {
    if (!state) return;

    // 演武场待命 / 未开始状态处理（避免刷新即自动开局）
    if (state.is_started === false) {
        lastAnnouncedTurn = 0;
        const roundEl = document.getElementById('round-indicator');
        if (roundEl) roundEl.innerText = `回合准备`;
        const turnBadge = document.getElementById('turn-badge');
        if (turnBadge) {
            turnBadge.innerText = `演武场待命中`;
            turnBadge.className = 'badge';
        }
        const passBtn = document.getElementById('btn-pass');
        if (passBtn) {
            passBtn.disabled = true;
            passBtn.innerText = "未开始对局";
        }
        const startBtn = document.getElementById('btn-start-game');
        if (startBtn) {
            startBtn.innerText = "开始对局";
            startBtn.disabled = false;
            startBtn.className = "btn primary";
        }
        const restartBtn = document.getElementById('btn-restart-game');
        if (restartBtn) {
            restartBtn.className = "btn secondary";
        }
        updateActionPrompt("请在上方选择卡组后，点击【开始对局】进入战斗", false);
        renderLanes(state.lanes || [{p0_attackers:[],p1_attackers:[],p0_defenders:[],p1_defenders:[]},{p0_attackers:[],p1_attackers:[],p0_defenders:[],p1_defenders:[]}]);
        renderHand([], 0, state.p0_faction || "Red");
        renderP1Hand(0);
        renderPlayerStats(state.p0, state.p1);
        renderBattleLog(state.logs);
        renderDeckTracker(state);
        return;
    }

    const startBtn = document.getElementById('btn-start-game');
    if (startBtn) {
        if (state.is_started && !state.done) {
            startBtn.innerText = "对局进行中";
            startBtn.disabled = true;
            startBtn.className = "btn primary disabled";
        } else {
            startBtn.innerText = state.done ? "再战一局" : "开始对局";
            startBtn.disabled = false;
            startBtn.className = "btn primary";
        }
    }

    // Faction labels
    const fNames = { "Red": "🔴 赤红", "Blue": "🔵 蔚蓝", "Green": "🟢 翠绿" };
    const p0Badge = document.getElementById('p0-faction-badge');
    const p1Badge = document.getElementById('p1-faction-badge');

    const p0Role = state.mode === 'pve' ? '玩家' : 'AI (P0)';
    if (p0Badge) {
        const deckTag = state.is_custom_deck ? ' [自构筑]' : '';
        p0Badge.innerText = `${fNames[state.p0_faction] || state.p0_faction} ${p0Role}${deckTag}`;
    }
    if (p1Badge) {
        p1Badge.innerText = `${fNames[state.p1_faction] || state.p1_faction} AI (P1)`;
    }

    // Turn & Round Indicators
    const roundEl = document.getElementById('round-indicator');
    if (roundEl) roundEl.innerText = `回合 ${state.turn || 1}`;

    if (state.turn && state.turn !== lastAnnouncedTurn && !state.done) {
        lastAnnouncedTurn = state.turn;
        showTurnBanner(`回合 ${state.turn}`, "turn");
    }

    const turnBadge = document.getElementById('turn-badge');
    const passBtn = document.getElementById('btn-pass');

    if (state.done) {
        const winnerStr = state.winner === 0 ? "红方获胜" : (state.winner === 1 ? "蓝方获胜" : "平局");
        if (turnBadge) {
            turnBadge.innerText = `对局结束: ${winnerStr}`;
            turnBadge.className = 'badge';
        }
        if (passBtn) passBtn.disabled = true;
        updateActionPrompt(`对局已结束 (${winnerStr})。可点击上方【重新开始】再战一局`, false);
    } else if (state.mode === 'aivai') {
        if (turnBadge) {
            const actingP = state.current_player === 0 ? "P0" : "P1";
            turnBadge.innerText = `AI 推演中 (${actingP})`;
            turnBadge.className = 'badge active-turn';
        }
        if (passBtn) {
            passBtn.disabled = true;
            passBtn.innerText = "推演进行中";
        }
        updateActionPrompt(`AI 推演中，点击上方控制按钮可暂停或单步推演`, false);
    } else if (state.current_player === 0) {
        if (turnBadge) {
            turnBadge.innerText = "你的回合";
            turnBadge.className = 'badge active-turn';
        }
        if (passBtn) {
            passBtn.disabled = false;
            passBtn.innerText = "结束回合";
        }
        if (selectedCardIndex === -1) {
            updateActionPrompt("你的回合：请选择手牌并部署至战场", false);
        }
    } else {
        if (turnBadge) {
            turnBadge.innerText = "敌方行动中...";
            turnBadge.className = 'badge waiting-turn';
        }
        if (passBtn) {
            passBtn.disabled = true;
            passBtn.innerText = "敌方决策中...";
        }
        updateActionPrompt("敌方行动中，请稍候...", false);
    }

    // Update Scores & Mana Numbers
    const p1ScoreEl = document.getElementById('p1-score');
    const p1ManaEl = document.getElementById('p1-mana');
    const p1MaxManaEl = document.getElementById('p1-max-mana');
    if (p1ScoreEl) p1ScoreEl.innerText = state.p1.score;
    if (p1ManaEl) p1ManaEl.innerText = state.p1.mana;
    if (p1MaxManaEl) p1MaxManaEl.innerText = state.p1.max_mana;

    const p0ScoreEl = document.getElementById('p0-score');
    const p0ManaEl = document.getElementById('p0-mana');
    const p0MaxManaEl = document.getElementById('p0-max-mana');
    if (p0ScoreEl) p0ScoreEl.innerText = state.p0.score;
    if (p0ManaEl) p0ManaEl.innerText = state.p0.mana;
    if (p0MaxManaEl) p0MaxManaEl.innerText = state.p0.max_mana;

    // Render Mana Crystals
    const renderCrystals = (containerId, currentMana, maxMana) => {
        const container = document.getElementById(containerId);
        if (!container) return;
        let html = '';
        for (let i = 0; i < maxMana; i++) {
            const isFull = i < currentMana;
            html += `<span class="mana-crystal ${isFull ? '' : 'empty'}" title="${isFull ? '可用法力' : '已消耗法力'}"></span>`;
        }
        container.innerHTML = html;
    };
    renderCrystals('p0-mana-crystals', state.p0.mana, state.p0.max_mana);
    renderCrystals('p1-mana-crystals', state.p1.mana, state.p1.max_mana);

    // Render Score Gems
    const renderGems = (containerId, score) => {
        const container = document.getElementById(containerId);
        if (!container) return;
        let html = '';
        for (let i = 0; i < 7; i++) {
            const hasGem = i < score;
            html += `<span class="score-gem ${hasGem ? '' : 'empty'}" title="${hasGem ? '已获得胜点' : '未占领胜点'}"></span>`;
        }
        container.innerHTML = html;
    };
    renderGems('p0-score-gems', state.p0.score);
    renderGems('p1-score-gems', state.p1.score);

    // Helper for Battlefield Units
    const renderUnits = (units, isAtkZone) => units.map(u => {
        const bg = u.image_url ? `background-image: url('${u.image_url}'); background-size: cover; background-position: center;` : '';
        const readyBadge = isAtkZone ? (u.ready ? '<span class="status-badge ready">就绪</span>' : '<span class="status-badge wait">蓄势</span>') : '';
        return `
            <div class="mini-card" style="${bg}" title="${u.name} (战力:${u.dp})">
                <div class="mini-card-overlay">
                    <span class="mini-card-name">${u.name}</span>
                    <span class="mini-card-dp">⚔️ ${u.dp}</span>
                    ${readyBadge}
                </div>
            </div>
        `;
    }).join('');

    // Render Lanes
    for (let i = 0; i < 2; i++) {
        const lane = state.lanes[i];
        
        // Enemy Slots
        const p1DefEl = document.getElementById(`slot-${i}-def`);
        if (p1DefEl) {
            p1DefEl.innerHTML = lane.p1_defenders.length 
                ? renderUnits(lane.p1_defenders, false) 
                : '<span class="slot-label">敌方防守位</span>';
        }

        const p1AtkEl = document.getElementById(`slot-${i}-atk`);
        if (p1AtkEl) {
            p1AtkEl.innerHTML = lane.p1_attackers.length 
                ? renderUnits(lane.p1_attackers, true) 
                : '<span class="slot-label">敌方进攻位</span>';
        }
        
        // Allied Slots (targetable)
        const p0AtkEl = document.getElementById(`target-${i}-${i === 0 ? 0 : 2}`);
        if (p0AtkEl) {
            p0AtkEl.innerHTML = lane.p0_attackers.length 
                ? renderUnits(lane.p0_attackers, true) 
                : '<span class="slot-label">我方进攻区</span>';
        }
        
        const p0DefEl = document.getElementById(`target-${i}-${i === 0 ? 1 : 3}`);
        if (p0DefEl) {
            p0DefEl.innerHTML = lane.p0_defenders.length 
                ? renderUnits(lane.p0_defenders, false) 
                : '<span class="slot-label">我方防守区</span>';
        }
    }

    // Render Enemy Hand (Card backs with count)
    const enemyHandZone = document.getElementById('p1-hand');
    if (enemyHandZone) {
        enemyHandZone.innerHTML = '';
        const count = state.p1.hand_count || (state.p1.hand ? state.p1.hand.length : 0);
        for (let i = 0; i < count; i++) {
            const backEl = document.createElement('div');
            backEl.className = 'card-back';
            backEl.title = `敌方手牌 (${count} 张)`;
            enemyHandZone.appendChild(backEl);
        }
    }

    // Render Player Hand
    const handZone = document.getElementById('p0-hand');
    handZone.innerHTML = '';
    
    state.p0.hand.forEach((card, idx) => {
        const cardEl = document.createElement('div');
        cardEl.className = 'hand-card';
        
        const canAfford = (card.cost <= state.p0.mana) && (state.current_player === 0 || state.mode !== 'pve') && !state.done;
        if (canAfford) {
            cardEl.classList.add('playable');
        } else {
            cardEl.classList.add('unplayable');
        }

        if (card.type === 'SPELL') {
            cardEl.classList.add('spell-card');
        }

        if (selectedCardIndex === idx) {
            cardEl.classList.add('selected');
        }
        
        if (card.image_url) {
            cardEl.style.backgroundImage = `url('${card.image_url}')`;
            cardEl.style.backgroundSize = 'cover';
            cardEl.style.backgroundPosition = 'center';
        }
        
        const isSpell = card.type === 'SPELL';
        const typeIcon = isSpell ? '🪄' : '🛡️';
        const tagsTranslated = (card.tags || []).slice(0, 2).map(window.translateTag).join(' · ');
        
        const spellBonusText = isSpell 
            ? ((card.atk_spell_val ? `+${card.atk_spell_val}攻 ` : '') + (card.def_spell_val ? `+${card.def_spell_val}盾` : '战术效果'))
            : `⚔️ ${card.dp} 战力`;

        cardEl.innerHTML = `
            <div class="hand-card-cost" title="${card.cost} 费">${card.cost}</div>
            <div class="hand-card-type" title="${isSpell ? '法术牌' : '随从牌'}">${typeIcon}</div>
            <div class="hand-card-info">
                <div class="hand-card-title">${card.name}</div>
                <div class="hand-card-dp">${spellBonusText}</div>
                <div class="hand-card-tags">${tagsTranslated || (isSpell ? '法术' : '随从')}</div>
            </div>
        `;
        
        cardEl.addEventListener('click', () => handleCardClick(idx));
        setupCardTooltip(cardEl, card);
        handZone.appendChild(cardEl);
    });

    // Render Battle Logs
    const logContainer = document.getElementById('log-container');
    if (logContainer && state.logs) {
        logContainer.innerHTML = '';
        state.logs.forEach(log => {
            const div = document.createElement('div');
            let logType = 'neutral';
            if (log.includes('红方') || log.includes('玩家')) logType = 'player';
            else if (log.includes('蓝方') || log.includes('AI')) logType = 'ai';
            else if (log.includes('冲锋') || log.includes('交战') || log.includes('💥')) logType = 'clash';
            else if (log.includes('获胜') || log.includes('🏆')) logType = 'victory';

            div.className = `log-entry ${logType}`;
            div.innerText = log;
            logContainer.appendChild(div);
        });
        logContainer.scrollTop = logContainer.scrollHeight;

        const latestLog = state.logs[state.logs.length - 1];
        if (latestLog && latestLog !== lastAnnouncedLog) {
            lastAnnouncedLog = latestLog;
            if (latestLog.includes('💥 冲锋交战结算')) {
                showTurnBanner("交锋结算", "clash");
            } else if (latestLog.includes('🏆') && state.done) {
                const isP0Win = state.winner === 0;
                showTurnBanner(isP0Win ? "对局胜利" : "对局失败", "victory");
            }
        }
    }

    // Render Deck Tracker
    renderDeckTracker(state);
}

// Auto-initialize side panel tabs
document.addEventListener('DOMContentLoaded', initCombatSidePanelTabs);
if (document.readyState === 'complete' || document.readyState === 'interactive') {
    initCombatSidePanelTabs();
}
