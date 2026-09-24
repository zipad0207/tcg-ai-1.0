// Studio, Deck Builder, and AI Card Printer Logic

let userDecks = [];
let activeDeckIndex = 0;
let userDeck = {
    name: "我的自构筑卡组",
    faction: "Red",
    card_ids: []
};

let allCardsMap = {};

document.addEventListener('tcgDataLoaded', () => {
    initStudio();
});

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        if (window.tcgData) initStudio();
    });
} else {
    if (window.tcgData) initStudio();
}

async function initStudio() {
    const data = window.tcgData;
    if (!data || !data.cards) return;

    allCardsMap = {};
    let allCards = [];
    for (const faction in data.cards) {
        data.cards[faction].forEach(c => {
            allCardsMap[c.id] = c;
            allCards.push(c);
        });
    }

    // Load saved custom decks (up to 6 slots)
    try {
        const res = await fetch('/api/custom_deck');
        if (res.ok) {
            const saved = await res.json();
            if (saved && Array.isArray(saved.decks) && saved.decks.length > 0) {
                userDecks = saved.decks;
                activeDeckIndex = (typeof saved.active_deck_index === 'number' && saved.active_deck_index >= 0 && saved.active_deck_index < userDecks.length)
                    ? saved.active_deck_index : 0;
                userDeck = userDecks[activeDeckIndex];
            } else if (saved && Array.isArray(saved.card_ids)) {
                userDeck = saved;
                userDecks = [userDeck];
            }
        }
    } catch (e) {
        console.warn("Could not load custom decks:", e);
    }

    // Ensure 6 slots exist
    while (userDecks.length < 6) {
        const idx = userDecks.length + 1;
        const factions = ["Red", "Blue", "Green"];
        const f = factions[(idx - 1) % 3];
        const fCn = { "Red": "赤红", "Blue": "蔚蓝", "Green": "翠绿" }[f];
        userDecks.push({
            id: `deck_${idx}`,
            name: `${fCn}·备选槽位${idx}`,
            faction: f,
            card_ids: []
        });
    }
    userDeck = userDecks[activeDeckIndex];
    window.userDeck = userDeck;
    window.userDecks = userDecks;
    window.activeDeckIndex = activeDeckIndex;

    renderCardsGrid(allCards);
    renderDeckDrawer();
    if (window.updateArenaDeckSelectOptions) {
        window.updateArenaDeckSelectOptions();
    }

    // Filters
    const filterSelect = document.getElementById('faction-filter');
    filterSelect.onchange = (e) => {
        const val = e.target.value;
        if (val === 'All') {
            renderCardsGrid(allCards);
        } else {
            renderCardsGrid(allCards.filter(c => c.factions && c.factions.includes(val)));
        }
    };
}

let cardImageObserver = null;
function getCardImageObserver() {
    if (!cardImageObserver && 'IntersectionObserver' in window) {
        cardImageObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const el = entry.target;
                    const bg = el.dataset.bg;
                    if (bg) {
                        el.style.backgroundImage = `url('${bg}')`;
                        el.removeAttribute('data-bg');
                    }
                    observer.unobserve(el);
                }
            });
        }, { rootMargin: '300px 0px' });
    }
    return cardImageObserver;
}

function renderCardsGrid(cards) {
    const cardsGrid = document.getElementById('cards-container');
    cardsGrid.innerHTML = '';

    const obs = getCardImageObserver();

    cards.forEach(card => {
        const cardEl = document.createElement('div');
        const faction = card.factions && card.factions[0] ? card.factions[0] : 'Neutral';
        cardEl.className = `tcg-card ${faction}`;
        cardEl.setAttribute('data-card-id', card.id);
        
        const dp = card.base_dp || 0;
        const cost = card.cost || 0;
        const rawImg = card.image_url || (card.raw_data && card.raw_data.image_url ? card.raw_data.image_url : null);
        const imgUrl = rawImg ? (rawImg.includes('?') ? rawImg : `${rawImg}?v=${window.assetVersion || 1}`) : null;
        if (imgUrl) {
            if (obs) {
                cardEl.dataset.bg = imgUrl;
                obs.observe(cardEl);
            } else {
                cardEl.style.backgroundImage = `url('${imgUrl}')`;
            }
        }
        
        const isSpell = card.card_type === 'SPELL';
        const typeStr = isSpell ? '🪄 法术' : '🛡️ 随从';
        const tagsStr = (card.tags || []).map(window.translateTag).join(' · ');

        // Clicking anywhere on card opens the Lightbox Viewer
        cardEl.onclick = (e) => {
            if (e.target.closest('.card-add-to-deck-btn')) return;
            openCardViewer(card.id);
        };

        cardEl.innerHTML = `
            <div class="card-cost">${cost}</div>
            <button class="card-add-to-deck-btn" title="添加到自构筑卡组" onclick="event.stopPropagation(); addCardToDeck(${card.id})">+</button>
            <div class="card-img-placeholder">
                ${imgUrl ? '' : '🎨 暂无原画<br>(点击生成)'}
            </div>
            <div class="card-view-hint"><span>🔍 点击查看大图/详情</span></div>
            <div class="card-stats">
                <span>${isSpell ? '✨ 效果卡' : '⚔️ ' + dp + ' 战力'}</span>
                <span>${typeStr}</span>
            </div>
            <div class="card-name">${card.name}</div>
            <div class="card-tags">${tagsStr || '普通兵种'}</div>
        `;
        cardsGrid.appendChild(cardEl);
    });
}

// ----------------- Deck Builder Logic -----------------
window.isCardAllowedForFaction = function(card, deckFaction) {
    if (!card) return false;
    const targetCode = deckFaction === 'Red' ? 1 : (deckFaction === 'Blue' ? 2 : 3);
    if (card.factions) {
        if (card.factions.includes(deckFaction) || card.factions.includes("Neutral")) return true;
    }
    const cid = Number(card.id);
    const cf = Math.floor(cid / 100);
    if (cf === targetCode || cf === 9) return true;
    // 双色卡合法性
    if (cf === 4 && (targetCode === 1 || targetCode === 2)) return true; // 赤蓝
    if (cf === 5 && (targetCode === 2 || targetCode === 3)) return true; // 蓝绿
    if (cf === 6 && (targetCode === 1 || targetCode === 3)) return true; // 红绿
    return false;
};

window.addCardToDeck = function(cardId) {
    const card = allCardsMap[cardId];
    if (!card) return;

    const curFaction = userDeck.faction || 'Red';
    const fNames = { 'Red': '🔴 赤红', 'Blue': '🔵 蔚蓝', 'Green': '🟢 翠绿' };

    // 1. 严格阵营合规判定（阻断跨阵营混卡，如蓝卡加进红卡组）
    if (!window.isCardAllowedForFaction(card, curFaction)) {
        alert(`❌ 阵营不合法！\n【${card.name}】不属于【${fNames[curFaction]}】合法卡池，无法加入该卡组！\n\n规则：只能加入对应阵营卡牌或中立通用卡。如需组建该卡牌套牌，请先在抽屉上方切换当前卡组阵营。`);
        return;
    }

    // 2. 严格 30 张卡组上限
    if (userDeck.card_ids.length >= 30) {
        alert("❌ 卡组已达 30 张携带上限！无法继续添加。");
        return;
    }

    // 3. 严格同名卡 3 张上限
    const count = userDeck.card_ids.filter(id => id === cardId).length;
    if (count >= 3) {
        alert(`❌ 同名卡上限！\n卡牌【${card.name}】已达到 3 张最大携带上限！`);
        return;
    }

    userDeck.card_ids.push(cardId);
    renderDeckDrawer();
    saveUserDeckSilently();
};

window.removeCardFromDeck = function(cardId) {
    const idx = userDeck.card_ids.indexOf(cardId);
    if (idx !== -1) {
        userDeck.card_ids.splice(idx, 1);
        renderDeckDrawer();
        saveUserDeckSilently();
    }
};

window.switchActiveDeck = function(idx) {
    if (idx < 0 || idx >= userDecks.length) return;
    activeDeckIndex = idx;
    userDeck = userDecks[activeDeckIndex];
    window.userDeck = userDeck;
    window.activeDeckIndex = activeDeckIndex;

    const nameInput = document.getElementById('deck-name-input');
    if (nameInput) nameInput.value = userDeck.name || `卡组 ${idx + 1}`;

    const factionSelect = document.getElementById('deck-faction-select');
    if (factionSelect) factionSelect.value = userDeck.faction || 'Red';

    const filterSelect = document.getElementById('faction-filter');
    if (filterSelect && window.tcgData && window.tcgData.cards) {
        filterSelect.value = userDeck.faction || 'Red';
        const allCards = Object.values(allCardsMap);
        renderCardsGrid(allCards.filter(c => window.isCardAllowedForFaction(c, userDeck.faction || 'Red')));
    }

    renderDeckDrawer();
    saveUserDeckSilently();
};

function renderDeckSlotTabs() {
    const tabsContainer = document.getElementById('deck-slot-tabs');
    const badgeEl = document.getElementById('deck-slot-badge');
    const nameInput = document.getElementById('deck-name-input');
    const toggleDrawerBtn = document.getElementById('btn-toggle-deck-drawer');

    if (badgeEl) badgeEl.innerText = `槽位 ${activeDeckIndex + 1} / 6`;
    if (nameInput && document.activeElement !== nameInput) {
        nameInput.value = userDeck.name || `卡组 ${activeDeckIndex + 1}`;
    }

    if (toggleDrawerBtn) {
        const fIcons = { 'Red': '🔴', 'Blue': '🔵', 'Green': '🟢' };
        const icon = fIcons[userDeck.faction] || '⚪';
        toggleDrawerBtn.innerHTML = `🛠️ 我的构筑 · ${icon} 槽位${activeDeckIndex + 1} (${userDeck.card_ids.length}/30)`;
    }

    if (!tabsContainer) return;
    tabsContainer.innerHTML = '';

    userDecks.forEach((deck, i) => {
        const tab = document.createElement('div');
        tab.className = `deck-slot-tab${i === activeDeckIndex ? ' active' : ''}`;
        const fIcons = { 'Red': '🔴', 'Blue': '🔵', 'Green': '🟢' };
        const icon = fIcons[deck.faction] || '⚪';
        const count = deck.card_ids ? deck.card_ids.length : 0;
        const countColor = count === 30 ? 'color:#10b981;font-weight:bold;' : 'color:var(--text-muted);';

        tab.innerHTML = `
            <span class="slot-num">槽位 ${i + 1}</span>
            <span class="slot-name">${icon} ${deck.name || `卡组${i+1}`}</span>
            <span class="slot-count" style="${countColor}">${count}/30 张</span>
        `;
        tab.onclick = () => window.switchActiveDeck(i);
        tabsContainer.appendChild(tab);
    });
}

window.loadAiPresetDeck = function(faction) {
    let deckObj = null;
    if (window.tcgData && window.tcgData.decks && window.tcgData.decks[faction]) {
        deckObj = window.tcgData.decks[faction];
    }
    if (!deckObj || !deckObj.decklist || deckObj.decklist.length === 0) {
        alert("正在获取 AI 精调构筑数据，请稍候...");
        return;
    }

    const fNames = { 'Red': '🔴 赤红·突击快攻', 'Blue': '🔵 蔚蓝·防御控制', 'Green': '🟢 翠绿·成长跳费' };
    userDeck.faction = faction;
    userDeck.name = deckObj.deck_name || fNames[faction];
    userDeck.card_ids = [...deckObj.decklist];

    const factionSelect = document.getElementById('deck-faction-select');
    if (factionSelect) factionSelect.value = faction;

    const nameInput = document.getElementById('deck-name-input');
    if (nameInput) nameInput.value = userDeck.name;

    const filterSelect = document.getElementById('faction-filter');
    if (filterSelect) {
        filterSelect.value = faction;
        const allCards = Object.values(allCardsMap);
        renderCardsGrid(allCards.filter(c => window.isCardAllowedForFaction(c, faction)));
    }

    renderDeckDrawer();
    saveUserDeckSilently();
    alert(`✅ 已成功将【${userDeck.name}】官方精调构筑 (30/30 张) 导入至槽位 ${activeDeckIndex + 1}！\n可直接带去演武场出战，也可在此自由替换微调。`);
};

function renderDeckDrawer() {
    renderDeckSlotTabs();

    const countEl = document.getElementById('deck-card-count');
    const countTextEl = document.getElementById('deck-count-text');
    const manaAvgEl = document.getElementById('deck-mana-avg');
    const curveBar = document.getElementById('mana-curve-bar');
    const listContainer = document.getElementById('deck-items-list');
    const arenaCustomCount = document.getElementById('arena-custom-deck-count');
    const factionSelect = document.getElementById('deck-faction-select');

    if (factionSelect && userDeck.faction) {
        factionSelect.value = userDeck.faction;
    }

    const totalCount = userDeck.card_ids.length;
    if (countEl) countEl.innerText = `${totalCount}/30`;
    if (countTextEl) countTextEl.innerText = `当前卡牌: ${totalCount} / 30 张`;
    if (arenaCustomCount) arenaCustomCount.innerText = `${totalCount}`;

    // Calculate curve & average mana
    let totalMana = 0;
    const curve = { "0-1": 0, "2": 0, "3": 0, "4": 0, "5+": 0 };
    const cardCounts = {};

    userDeck.card_ids.forEach(id => {
        const c = allCardsMap[id];
        if (c) {
            const cost = c.cost || 0;
            totalMana += cost;
            if (cost <= 1) curve["0-1"]++;
            else if (cost === 2) curve["2"]++;
            else if (cost === 3) curve["3"]++;
            else if (cost === 4) curve["4"]++;
            else curve["5+"]++;
        }
        cardCounts[id] = (cardCounts[id] || 0) + 1;
    });

    const avg = totalCount > 0 ? (totalMana / totalCount).toFixed(1) : "0.0";
    if (manaAvgEl) manaAvgEl.innerText = `均费: ${avg}`;

    // Render Curve Bars
    if (curveBar) {
        curveBar.innerHTML = '';
        const maxVal = Math.max(1, ...Object.values(curve));
        for (const [key, val] of Object.entries(curve)) {
            const pct = Math.round((val / maxVal) * 100);
            const col = document.createElement('div');
            col.className = 'curve-bar-col';
            col.innerHTML = `
                <div class="curve-bar-fill" style="height: ${Math.max(4, pct)}%;"></div>
                <span>${key} (${val})</span>
            `;
            curveBar.appendChild(col);
        }
    }

    // Render Items
    if (listContainer) {
        if (totalCount === 0) {
            listContainer.innerHTML = '<div class="empty-deck-hint">点击卡池卡牌右上角的「+」即可添加至卡组（上限 30 张，单卡最多 3 张，不可跨阵营混卡）</div>';
        } else {
            listContainer.innerHTML = '';
            // Sort by cost then name
            const uniqueIds = Object.keys(cardCounts).map(Number).sort((a, b) => {
                const ca = allCardsMap[a] ? allCardsMap[a].cost : 0;
                const cb = allCardsMap[b] ? allCardsMap[b].cost : 0;
                return ca - cb;
            });

            uniqueIds.forEach(id => {
                const c = allCardsMap[id] || { name: `ID:${id}`, cost: 0, factions: ['Neutral'] };
                const faction = c.factions && c.factions[0] ? c.factions[0] : 'Neutral';
                const count = cardCounts[id];
                const isOverLimit = count > 3;

                const item = document.createElement('div');
                item.className = `deck-card-item ${faction}`;
                item.innerHTML = `
                    <div class="deck-card-item-left">
                        <span class="deck-card-cost">${c.cost}</span>
                        <span class="deck-card-name">${c.name}</span>
                        <span class="deck-card-count" style="${isOverLimit ? 'color:#ef4444;font-weight:900;' : ''}">x${count}${isOverLimit ? ' (超标!)' : ''}</span>
                    </div>
                    <div class="deck-card-item-controls">
                        <button class="deck-qty-btn" onclick="removeCardFromDeck(${id})">-</button>
                        <button class="deck-qty-btn" onclick="addCardToDeck(${id})">+</button>
                    </div>
                `;
                listContainer.appendChild(item);
            });
        }
    }
}

// Drawer Toggle
const drawer = document.getElementById('deck-builder-drawer');
const toggleDrawerBtn = document.getElementById('btn-toggle-deck-drawer');
const closeDrawerBtn = document.getElementById('btn-close-deck-drawer');

if (toggleDrawerBtn && drawer) {
    toggleDrawerBtn.onclick = () => drawer.classList.toggle('collapsed');
}
if (closeDrawerBtn && drawer) {
    closeDrawerBtn.onclick = () => drawer.classList.add('collapsed');
}

// Deck Name Input Listener
const deckNameInput = document.getElementById('deck-name-input');
if (deckNameInput) {
    deckNameInput.oninput = (e) => {
        const val = e.target.value.trim();
        userDeck.name = val || `卡组 ${activeDeckIndex + 1}`;
        renderDeckSlotTabs();
        if (window.updateArenaDeckSelectOptions) {
            window.updateArenaDeckSelectOptions();
        }
    };
    deckNameInput.onchange = () => {
        saveUserDeckSilently();
    };
}

// Deck Faction Select Listener
const deckFactionSelect = document.getElementById('deck-faction-select');
if (deckFactionSelect) {
    deckFactionSelect.onchange = (e) => {
        const newFaction = e.target.value;
        if (newFaction === userDeck.faction) return;

        const illegalIds = userDeck.card_ids.filter(id => {
            const c = allCardsMap[id];
            return !window.isCardAllowedForFaction(c, newFaction);
        });

        if (illegalIds.length > 0) {
            const fNames = { 'Red': '🔴 赤红', 'Blue': '🔵 蔚蓝', 'Green': '🟢 翠绿' };
            const ok = confirm(`切换至【${fNames[newFaction]}】将自动移除 ${illegalIds.length} 张不符合该阵营规则的卡牌，是否继续切换？`);
            if (!ok) {
                deckFactionSelect.value = userDeck.faction;
                return;
            }
            userDeck.card_ids = userDeck.card_ids.filter(id => {
                const c = allCardsMap[id];
                return window.isCardAllowedForFaction(c, newFaction);
            });
        }

        userDeck.faction = newFaction;
        const fNames = { 'Red': '赤红', 'Blue': '蔚蓝', 'Green': '翠绿' };
        userDeck.name = `${fNames[newFaction]}·槽位${activeDeckIndex + 1}`;
        if (deckNameInput) deckNameInput.value = userDeck.name;

        const filterSelect = document.getElementById('faction-filter');
        if (filterSelect) {
            filterSelect.value = newFaction;
            const allCards = Object.values(allCardsMap);
            renderCardsGrid(allCards.filter(c => window.isCardAllowedForFaction(c, newFaction)));
        }

        renderDeckDrawer();
        saveUserDeckSilently();
    };
}

// Deck Actions
document.getElementById('btn-deck-clear').onclick = () => {
    userDeck.card_ids = [];
    renderDeckDrawer();
    saveUserDeckSilently();
};

document.getElementById('btn-deck-fill-random').onclick = () => {
    const curFaction = userDeck.faction || 'Red';
    const legalIds = Object.keys(allCardsMap).map(Number).filter(id => window.isCardAllowedForFaction(allCardsMap[id], curFaction));
    if (legalIds.length === 0) return;

    while (userDeck.card_ids.length < 30) {
        const randId = legalIds[Math.floor(Math.random() * legalIds.length)];
        const count = userDeck.card_ids.filter(id => id === randId).length;
        if (count < 3) {
            userDeck.card_ids.push(randId);
        }
    }
    renderDeckDrawer();
    saveUserDeckSilently();
};

async function saveUserDeckSilently() {
    try {
        await fetch('/api/custom_deck', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                deck_index: activeDeckIndex,
                name: userDeck.name,
                faction: userDeck.faction,
                card_ids: userDeck.card_ids
            })
        });
        window.userDeck = userDeck;
        window.userDecks = userDecks;
        window.activeDeckIndex = activeDeckIndex;
        if (window.updateArenaDeckSelectOptions) {
            window.updateArenaDeckSelectOptions();
        }
    } catch (e) {
        console.warn("Silent deck save:", e);
    }
}

document.getElementById('btn-deck-save').onclick = async () => {
    try {
        const res = await fetch('/api/custom_deck', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                deck_index: activeDeckIndex,
                name: userDeck.name,
                faction: userDeck.faction,
                card_ids: userDeck.card_ids
            })
        });
        const data = await res.json();
        if (res.ok && data.success) {
            renderDeckSlotTabs();
            if (window.updateArenaDeckSelectOptions) window.updateArenaDeckSelectOptions();
            alert(`✅ 槽位 ${activeDeckIndex + 1}【${userDeck.name}】已成功保存！当前卡牌共 ${userDeck.card_ids.length} 张（${userDeck.faction}阵营）。`);
        } else {
            alert(`❌ 保存失败: ${data.message || '卡组不合规'}`);
        }
    } catch (e) {
        alert("保存失败: " + e.message);
    }
};

document.getElementById('btn-deck-battle').onclick = async () => {
    if (userDeck.card_ids.length !== 30) {
        alert(`⚠️ 当前卡组【${userDeck.name}】尚未满 30 张（当前 ${userDeck.card_ids.length}/30 张）！\n请点击「🎲 随机合规补满」或添加卡牌凑齐 30 张后再出战。`);
        return;
    }
    await saveUserDeckSilently();

    // Switch view to arena
    document.querySelectorAll('.nav-links li').forEach(l => l.classList.remove('active'));
    document.querySelectorAll('.view-section').forEach(v => v.classList.remove('active'));
    
    const arenaNav = document.querySelector('[data-target="arena-view"]');
    if (arenaNav) arenaNav.classList.add('active');
    const arenaView = document.getElementById('arena-view');
    if (arenaView) arenaView.classList.add('active');

    // Close drawer
    if (drawer) drawer.classList.add('collapsed');

    // Select this specific custom deck in the arena dropdown
    if (window.updateArenaDeckSelectOptions) {
        window.updateArenaDeckSelectOptions();
    }
    const p0DeckSelect = document.getElementById('pve-p0-deck');
    if (p0DeckSelect) p0DeckSelect.value = `custom_${activeDeckIndex}`;

    // Trigger start match
    if (window.startMatchWithSettings) {
        window.startMatchWithSettings();
    }
};

// ----------------- Batch Art Modal & Background Queue -----------------
const batchModal = document.getElementById('batch-art-modal');
const btnOpenBatch = document.getElementById('btn-open-batch-art');
let batchPollTimer = null;

function stopBatchArtPolling() {
    if (batchPollTimer) {
        clearInterval(batchPollTimer);
        batchPollTimer = null;
    }
}

function startBatchArtPolling() {
    stopBatchArtPolling();
    const descEl = document.getElementById('batch-art-desc');
    const statusText = document.getElementById('batch-status-text');
    const bar = document.getElementById('batch-progress-bar');
    const startBtn = document.getElementById('btn-start-batch-art');

    const check = async () => {
        try {
            const res = await fetch('/api/batch_generate_art/status');
            const data = await res.json();
            const q = data.queue_status || {};
            const isRunning = q.is_running;

            if (startBtn) startBtn.disabled = isRunning;

            if (isRunning) {
                const currentName = q.current_card ? (q.current_card.name || `新卡_${q.current_card.id}`) : '新卡';
                const pct = q.percentage || 0;
                if (bar) bar.style.width = `${pct}%`;
                if (statusText) {
                    statusText.innerHTML = `🎨 [${q.current}/${q.total}] 正在生成 <strong>${currentName}</strong> (${pct}%)... ⏳`;
                }
                if (descEl) {
                    descEl.innerText = `后台队列运行中：已成功 ${q.success} 张，失败 ${q.failed} 张。`;
                }
            } else {
                if (q.done) {
                    if (bar) bar.style.width = '100%';
                    if (statusText) {
                        statusText.innerHTML = `🎉 后台生图完成！共成功绘制 <strong>${q.success}</strong> 张卡图！`;
                    }
                    if (startBtn) startBtn.disabled = false;
                    stopBatchArtPolling();
                    // Refresh data to show new art immediately
                    const dataRes = await fetch('/api/data');
                    if (dataRes.ok) {
                        window.tcgData = await dataRes.json();
                        document.dispatchEvent(new Event('tcgDataLoaded'));
                    }
                } else {
                    const missingCount = data.missing_count || 0;
                    if (bar) bar.style.width = '0%';
                    if (descEl) descEl.innerText = `卡池中共有 ${missingCount} 张卡牌缺少原画插图。`;
                    if (statusText) statusText.innerText = missingCount > 0 ? "点击下方按钮开始后台排队绘制" : "太棒了！所有卡牌的原画已全部绘制完毕！";
                    if (startBtn) startBtn.disabled = (missingCount === 0);
                    stopBatchArtPolling();
                }
            }
        } catch (e) {
            console.warn("Poll batch art failed:", e);
        }
    };

    check();
    batchPollTimer = setInterval(check, 1500);
}

if (btnOpenBatch) {
    btnOpenBatch.onclick = () => {
        if (!batchModal) return;
        batchModal.classList.remove('hidden');
        startBatchArtPolling();
    };
}

const btnCloseBatch = document.getElementById('btn-close-batch-art');
if (btnCloseBatch) {
    btnCloseBatch.onclick = () => {
        if (batchModal) batchModal.classList.add('hidden');
        stopBatchArtPolling();
    };
}

const btnStartBatch = document.getElementById('btn-start-batch-art');
if (btnStartBatch) {
    btnStartBatch.onclick = async () => {
        const statusText = document.getElementById('batch-status-text');
        const bar = document.getElementById('batch-progress-bar');

        btnStartBatch.disabled = true;
        if (bar) bar.style.width = '5%';
        if (statusText) statusText.innerHTML = "🚀 正在向后台提交排队生图请求... ⏳";

        try {
            const res = await fetch('/api/batch_generate_art', { method: 'POST' });
            const data = await res.json();
            if (data.success) {
                startBatchArtPolling();
            } else {
                if (statusText) statusText.innerHTML = `<span style="color:red">启动失败: ${data.detail || data.message}</span>`;
                btnStartBatch.disabled = false;
            }
        } catch (e) {
            if (statusText) statusText.innerHTML = `<span style="color:red">请求失败: ${e.message}</span>`;
            btnStartBatch.disabled = false;
        }
    };
}

// ----------------- Card Viewer & Art Regeneration Modal -----------------
let currentViewerCard = null;

window.openCardViewer = function(cardId) {
    const card = allCardsMap[cardId];
    if (!card) return;
    currentViewerCard = card;

    const modal = document.getElementById('card-viewer-modal');
    if (!modal) return;

    const imgEl = document.getElementById('viewer-art-img');
    const emptyBox = document.getElementById('viewer-art-empty');
    const nameEl = document.getElementById('viewer-card-name');
    const fBadge = document.getElementById('viewer-faction-badge');
    const tBadge = document.getElementById('viewer-type-badge');
    const costEl = document.getElementById('viewer-cost');
    const dpEl = document.getElementById('viewer-dp');
    const tagsContainer = document.getElementById('viewer-tags-list');
    const promptBox = document.getElementById('viewer-prompt-preview');
    const statusText = document.getElementById('viewer-status-text');

    const faction = card.factions && card.factions[0] ? card.factions[0] : 'Neutral';
    const isSpell = card.card_type === 'SPELL';
    const imgUrl = card.image_url || (card.raw_data && card.raw_data.image_url ? card.raw_data.image_url : null);

    nameEl.innerText = card.name;
    const fNames = { "Red": "🔴 赤红", "Blue": "🔵 蔚蓝", "Green": "🟢 翠绿", "Neutral": "⚪ 中立", "Dual": "🟣 混合" };
    fBadge.innerText = fNames[faction] || faction;
    fBadge.className = `badge ${faction}`;
    tBadge.innerText = isSpell ? "🪄 法术卡" : "🛡️ 随从卡";
    tBadge.className = `badge ${isSpell ? 'cyan' : 'gold'}`;

    costEl.innerText = card.cost || 0;
    dpEl.innerText = isSpell ? '—' : (card.base_dp || 0);

    // Tags
    tagsContainer.innerHTML = '';
    const tags = card.tags || [];
    if (tags.length === 0) {
        tagsContainer.innerHTML = '<span class="viewer-tag-pill">无特殊词条 (常规兵种)</span>';
    } else {
        tags.forEach(t => {
            const pill = document.createElement('span');
            pill.className = 'viewer-tag-pill';
            pill.innerText = window.translateTag ? window.translateTag(t) : t;
            tagsContainer.appendChild(pill);
        });
    }

    // Image display
    promptBox.style.display = 'none';
    statusText.innerText = '';
    if (imgUrl) {
        imgEl.src = imgUrl;
        imgEl.classList.remove('hidden');
        emptyBox.classList.add('hidden');
    } else {
        imgEl.src = '';
        imgEl.classList.add('hidden');
        emptyBox.classList.remove('hidden');
    }

    modal.classList.remove('hidden');
};

async function executeRegenerateCurrentCard() {
    if (!currentViewerCard) return;
    const card = currentViewerCard;
    const faction = card.factions && card.factions[0] ? card.factions[0] : 'Neutral';
    const dp = card.base_dp || 0;
    
    const statusText = document.getElementById('viewer-status-text');
    const promptBox = document.getElementById('viewer-prompt-preview');
    const imgEl = document.getElementById('viewer-art-img');
    const emptyBox = document.getElementById('viewer-art-empty');
    const regenBtn = document.getElementById('viewer-btn-regen');
    const genNowBtn = document.getElementById('viewer-btn-gen-now');

    statusText.innerHTML = "🎨 正在调用快手可图（Kwai-Kolors）与纯净提示词绘制中，请稍候约 5-8 秒... ⏳";
    if (regenBtn) regenBtn.disabled = true;
    if (genNowBtn) genNowBtn.disabled = true;

    try {
        const res = await fetch('/api/generate_art', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                card_id: card.id,
                card_name: card.name,
                faction: faction,
                tags: card.tags || [],
                dp: dp
            })
        });
        const data = await res.json();
        if (data.success) {
            statusText.innerHTML = "✨ 原画绘制成功并已本地永久覆盖！";
            const freshUrl = data.data.url.includes('?') ? `${data.data.url}&t=${Date.now()}` : `${data.data.url}?t=${Date.now()}`;
            imgEl.src = freshUrl;
            imgEl.classList.remove('hidden');
            emptyBox.classList.add('hidden');
            card.image_url = freshUrl;

            // Instantly update the card in the background grid
            const gridCard = document.querySelector(`.tcg-card[data-card-id="${card.id}"]`);
            if (gridCard) {
                gridCard.style.backgroundImage = `url('${freshUrl}')`;
                const placeholder = gridCard.querySelector('.card-img-placeholder');
                if (placeholder) placeholder.innerHTML = '';
            }
            
            if (data.data.prompt) {
                promptBox.style.display = 'block';
                promptBox.innerText = `🎨 DeepSeek 构想提示词:\n${data.data.prompt}`;
            }

            window.assetVersion = Date.now();
        } else {
            const errStr = data.detail || JSON.stringify(data);
            if (errStr.includes("429") || errStr.includes("IPM limit") || errStr.includes("限频")) {
                statusText.innerHTML = `<span style="color:#f59e0b">⏳ 提示：硅基流动平台每分钟生成频次已达上限（IPM Limit，每分钟约1~2张）。请等待约 20~30 秒后再次点击即可！</span>`;
            } else {
                statusText.innerHTML = `<span style="color:red">生成失败: ${errStr}</span>`;
            }
        }
    } catch (e) {
        const errStr = e.message || String(e);
        if (errStr.includes("429") || errStr.includes("IPM limit") || errStr.includes("限频")) {
            statusText.innerHTML = `<span style="color:#f59e0b">⏳ 提示：硅基流动平台每分钟生成频次已达上限（IPM Limit，每分钟约1~2张）。请等待约 20~30 秒后再次点击即可！</span>`;
        } else {
            statusText.innerHTML = `<span style="color:red">生成出错: ${errStr}</span>`;
        }
    } finally {
        if (regenBtn) regenBtn.disabled = false;
        if (genNowBtn) genNowBtn.disabled = false;
    }
}

// Bind Viewer Modal Controls
const viewerBtnRegen = document.getElementById('viewer-btn-regen');
if (viewerBtnRegen) {
    viewerBtnRegen.addEventListener('click', executeRegenerateCurrentCard);
}
const viewerBtnGenNow = document.getElementById('viewer-btn-gen-now');
if (viewerBtnGenNow) {
    viewerBtnGenNow.addEventListener('click', executeRegenerateCurrentCard);
}

document.getElementById('viewer-btn-close').addEventListener('click', () => {
    document.getElementById('card-viewer-modal').classList.add('hidden');
});

document.getElementById('card-viewer-modal').addEventListener('click', (e) => {
    if (e.target.id === 'card-viewer-modal') {
        document.getElementById('card-viewer-modal').classList.add('hidden');
    }
});

document.getElementById('viewer-btn-add-deck').addEventListener('click', () => {
    if (currentViewerCard) {
        addCardToDeck(currentViewerCard.id);
    }
});

document.getElementById('viewer-btn-download').addEventListener('click', () => {
    if (currentViewerCard && currentViewerCard.image_url) {
        window.open(currentViewerCard.image_url, '_blank');
    } else {
        alert("该卡牌尚未生成原画图片！");
    }
});
