// TCG-AI Analytics & Balance Orchestrator Console
let radarChartInstance = null;
let pipelineWs = null;
let statusPollInterval = null;

// Expose global functions for direct HTML onclick invocation
window.startPipeline = startPipeline;
window.stopPipeline = stopPipeline;
window.refreshAnalyticsData = refreshAnalyticsData;
window.clearPipelineConsole = clearPipelineConsole;

// Initialize Analytics Module
function initAnalytics() {
    console.log("[Analytics] Initializing Balance Console...");
    initPipelineControls();
    initPipelineSocket();
    pollPipelineStatus();

    if (!statusPollInterval) {
        statusPollInterval = setInterval(pollPipelineStatus, 3000);
    }

    // Load initial stats
    if (window.tcgData && window.tcgData.faction_stats) {
        renderRadarChart(window.tcgData);
        renderMatchupMatrix(window.tcgData);
        updateMetricCards(window.tcgData);
    } else {
        // Fallback fetch
        fetch('/api/data')
            .then(res => res.json())
            .then(data => {
                window.tcgData = data;
                renderRadarChart(data);
                renderMatchupMatrix(data);
                updateMetricCards(data);
            })
            .catch(err => {
                console.error("[Analytics] Fetch /api/data failed:", err);
            });
    }
}

// Update Metric Cards
function updateMetricCards(data) {
    if (!data || !data.faction_stats) return;
    const stats = data.faction_stats;
    const red = (stats.Red && stats.Red.winrate != null) ? stats.Red.winrate : 50;
    const blue = (stats.Blue && stats.Blue.winrate != null) ? stats.Blue.winrate : 50;
    const green = (stats.Green && stats.Green.winrate != null) ? stats.Green.winrate : 50;

    const redDev = (red - 50);
    const blueDev = (blue - 50);
    const greenDev = (green - 50);

    const allDevs = Object.values(stats).map(s => Math.abs((s.winrate != null ? s.winrate : 50) - 50));
    const maxDev = allDevs.length ? Math.max(...allDevs) : Math.max(Math.abs(redDev), Math.abs(blueDev), Math.abs(greenDev));

    const redEl = document.getElementById('stat-red-wr');
    const redSub = document.getElementById('stat-red-sub');
    if (redEl) redEl.textContent = `${red.toFixed(1)}%`;
    if (redSub) redSub.textContent = `偏离度: ${redDev >= 0 ? '+' : ''}${redDev.toFixed(1)}%`;

    const blueEl = document.getElementById('stat-blue-wr');
    const blueSub = document.getElementById('stat-blue-sub');
    if (blueEl) blueEl.textContent = `${blue.toFixed(1)}%`;
    if (blueSub) blueSub.textContent = `偏离度: ${blueDev >= 0 ? '+' : ''}${blueDev.toFixed(1)}%`;

    const greenEl = document.getElementById('stat-green-wr');
    const greenSub = document.getElementById('stat-green-sub');
    if (greenEl) greenEl.textContent = `${green.toFixed(1)}%`;
    if (greenSub) greenSub.textContent = `偏离度: ${greenDev >= 0 ? '+' : ''}${greenDev.toFixed(1)}%`;

    const maxDevEl = document.getElementById('stat-max-dev');
    const badgeEl = document.getElementById('stat-balance-badge');
    if (maxDevEl) maxDevEl.textContent = `±${maxDev.toFixed(1)}%`;

    // 动态获取当前配置的目标容差与警戒线
    const targetTol = parseFloat(document.getElementById('pipe-target')?.value) || 5.0;
    const warnTol = parseFloat((targetTol * 1.6).toFixed(1));

    if (badgeEl) {
        badgeEl.className = 'metric-sub status-pill';
        if (maxDev <= targetTol) {
            badgeEl.classList.add('healthy');
            badgeEl.textContent = `✅ 总偏离 ≤${targetTol}%`;
        } else if (maxDev <= warnTol) {
            badgeEl.classList.add('warning');
            badgeEl.textContent = `⚠️ 轻微偏离 (${targetTol}~${warnTol}%)`;
        } else {
            badgeEl.classList.add('danger');
            badgeEl.textContent = `🚨 显著失衡 (>${warnTol}%)`;
        }
    }

    // Pairwise Head-to-Head Deviation
    let maxPairwiseDev = data.max_pairwise_dev;
    if (maxPairwiseDev == null && data.pairwise_matchups) {
        const devs = Object.values(data.pairwise_matchups).filter(p => !p.is_mirror).map(p => p.dev || 0);
        maxPairwiseDev = devs.length ? Math.max(...devs) : 0.0;
    }
    if (maxPairwiseDev == null) maxPairwiseDev = 0.0;

    const targetPairTol = parseFloat(document.getElementById('pipe-pairwise-target')?.value) || 5.0;
    const warnPairTol = parseFloat((targetPairTol * 1.6).toFixed(1));

    const pairDevEl = document.getElementById('stat-pairwise-dev');
    const pairBadgeEl = document.getElementById('stat-pairwise-badge');
    if (pairDevEl) pairDevEl.textContent = `±${maxPairwiseDev.toFixed(1)}%`;
    if (pairBadgeEl) {
        pairBadgeEl.className = 'metric-sub status-pill';
        if (maxPairwiseDev <= targetPairTol) {
            pairBadgeEl.classList.add('healthy');
            pairBadgeEl.textContent = `✅ 两两均 ≤${targetPairTol}%`;
        } else if (maxPairwiseDev <= warnPairTol) {
            pairBadgeEl.classList.add('warning');
            pairBadgeEl.textContent = `⚠️ 轻度克制 (±${maxPairwiseDev.toFixed(1)}%)`;
        } else {
            pairBadgeEl.classList.add('danger');
            pairBadgeEl.textContent = `🚨 克制偏离 (>${warnPairTol}%)`;
        }
    }
}

// Render Winrate Radar Chart
function renderRadarChart(data) {
    const ctx = document.getElementById('winrate-chart');
    if (!ctx || !data.faction_stats) return;

    const labels = Object.keys(data.faction_stats);
    const winrates = labels.map(f => Number((data.faction_stats[f].winrate || 50).toFixed(1)));

    if (typeof Chart === 'undefined') {
        console.warn("[Analytics] Chart.js is not loaded yet.");
        return;
    }

    // Reuse or resolve existing chart attached to canvas to avoid duplicate observer loops
    const existingChart = Chart.getChart(ctx);
    if (existingChart) {
        radarChartInstance = existingChart;
    }

    if (radarChartInstance) {
        radarChartInstance.data.labels = labels;
        if (radarChartInstance.data.datasets.length > 1) {
            radarChartInstance.data.datasets[0].data = labels.map(() => 50);
            radarChartInstance.data.datasets[1].data = winrates;
        } else {
            radarChartInstance.data.datasets[0].data = winrates;
        }
        radarChartInstance.update('none'); // Update without animation/resize reflow
        return;
    }

    try {
        radarChartInstance = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: '50% 完美平衡基准',
                        data: labels.map(() => 50),
                        borderColor: 'rgba(255, 255, 255, 0.35)',
                        borderWidth: 1.5,
                        borderDash: [5, 5],
                        pointRadius: 0,
                        fill: false
                    },
                    {
                        label: '阵营实际胜率',
                        data: winrates,
                        backgroundColor: 'rgba(56, 189, 248, 0.22)',
                        borderColor: '#38bdf8',
                        borderWidth: 2,
                        pointBackgroundColor: labels.map(f => {
                            if (f === 'Red') return '#ef4444';
                            if (f === 'Blue') return '#3b82f6';
                            if (f === 'Green') return '#10b981';
                            return '#f59e0b';
                        }),
                        pointBorderColor: '#ffffff',
                        pointBorderWidth: 1.5,
                        pointRadius: 4,
                        pointHoverRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                aspectRatio: 1.25,
                animation: {
                    duration: 300
                },
                scales: {
                    r: {
                        min: 0,
                        max: 100,
                        suggestedMin: 0,
                        suggestedMax: 100,
                        angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                        grid: { color: 'rgba(255, 255, 255, 0.12)' },
                        pointLabels: {
                            color: '#e2e8f0',
                            font: { size: 13, weight: '600' }
                        },
                        ticks: {
                            stepSize: 25,
                            color: 'rgba(255, 255, 255, 0.55)',
                            backdropColor: 'transparent',
                            font: { size: 10 },
                            callback: function(val) {
                                return val + '%';
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'bottom',
                        labels: {
                            color: '#94a3b8',
                            boxWidth: 12,
                            padding: 10,
                            font: { size: 11 }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return ` ${context.dataset.label}: ${context.raw}%`;
                            }
                        }
                    }
                }
            }
        });
    } catch (e) {
        console.error("[Analytics] Chart render error:", e);
    }
}

// Render Matchup Matrix (Symmetrical Pairwise Head-to-Head)
function renderMatchupMatrix(data) {
    const matrixContainer = document.getElementById('matchup-matrix');
    if (!matrixContainer) return;

    // 动态发现战报阵营
    const defaultColors = { Red: '#ff4757', Blue: '#1e90ff', Green: '#2ed573' };
    const palette = ['#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#10b981'];
    let factions = [];
    if (data.faction_stats) {
        const keys = Object.keys(data.faction_stats);
        ['Red', 'Blue', 'Green'].forEach(f => { if (keys.includes(f)) factions.push(f); });
        keys.forEach(f => { if (!factions.includes(f)) factions.push(f); });
    }
    if (!factions.length) factions = ['Red', 'Blue', 'Green'];
    const factionColors = { ...defaultColors };
    factions.forEach((f, idx) => {
        if (!factionColors[f]) factionColors[f] = palette[idx % palette.length];
    });

    const targetPairTol = parseFloat(document.getElementById('pipe-pairwise-target')?.value) || 5.0;
    const severePairTol = parseFloat((targetPairTol * 1.6).toFixed(1));

    let maxPairwiseDev = 0;
    const matrix = {};

    factions.forEach(f1 => {
        matrix[f1] = {};
        factions.forEach(f2 => {
            if (f1 === f2) {
                matrix[f1][f2] = { winrate: 50.0, total: 0, wins: 0, dev: 0, isMirror: true };
            } else {
                const key = `${f1}_vs_${f2}`;
                if (data.pairwise_matchups && data.pairwise_matchups[key]) {
                    const p = data.pairwise_matchups[key];
                    matrix[f1][f2] = {
                        winrate: p.winrate,
                        total: p.total,
                        wins: p.wins,
                        dev: p.dev,
                        isMirror: false
                    };
                    if (p.dev > maxPairwiseDev) maxPairwiseDev = p.dev;
                } else if (data.matchups) {
                    const k1 = `${f1}_vs_${f2}`;
                    const k2 = `${f2}_vs_${f1}`;
                    const r1 = data.matchups[k1] || {};
                    const r2 = data.matchups[k2] || {};
                    const tot = (r1.total || 0) + (r2.total || 0);
                    const wins = (r1[`${f1}_wins`] || 0) + (r2[`${f1}_wins`] || 0);
                    const wr = tot > 0 ? (wins / tot * 100) : 50.0;
                    const dev = Math.abs(wr - 50.0);
                    matrix[f1][f2] = {
                        winrate: Number(wr.toFixed(1)),
                        total: tot,
                        wins: wins,
                        dev: Number(dev.toFixed(1)),
                        isMirror: false
                    };
                    if (dev > maxPairwiseDev) maxPairwiseDev = dev;
                } else {
                    matrix[f1][f2] = { winrate: 50.0, total: 0, wins: 0, dev: 0, isMirror: false };
                }
            }
        });
    });

    let html = '<div style="overflow-x:auto;">';
    html += '<table style="width:100%; border-collapse: separate; border-spacing: 6px; text-align: center;">';
    html += '<tr><th style="padding: 10px; color:#94a3b8; font-size:0.85rem;">本方 \\ 对手</th>';
    factions.forEach(f => {
        html += `<th style="color:${factionColors[f]}; padding: 10px; font-size:0.9rem;">${f}</th>`;
    });
    html += '</tr>';

    factions.forEach(f1 => {
        html += `<tr><th style="text-align:left; padding: 10px; color:${factionColors[f1]}; font-size:0.9rem;">${f1}</th>`;
        factions.forEach(f2 => {
            const cell = matrix[f1][f2];
            if (cell.isMirror) {
                html += `<td style="background: rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 10px 6px;">
                    <div style="color:#64748b; font-size:0.95rem; font-weight:700;">50.0%</div>
                    <div style="color:#475569; font-size:0.72rem; margin-top:2px;">(内战对等)</div>
                </td>`;
            } else {
                const wr = cell.winrate;
                const dev = cell.dev;
                let bg = 'rgba(255,255,255,0.03)';
                let color = '#34d399'; // green within target
                let badgeText = '✅ 均势';
                let badgeBg = 'rgba(16, 185, 129, 0.15)';
                let badgeColor = '#10b981';

                if (dev > severePairTol) {
                    bg = wr > 50 ? 'rgba(16, 185, 129, 0.12)' : 'rgba(239, 68, 68, 0.12)';
                    color = wr > 50 ? '#34d399' : '#f87171';
                    badgeText = wr > 50 ? `优势 +${dev.toFixed(1)}%` : `劣势 -${dev.toFixed(1)}%`;
                    badgeBg = 'rgba(239, 68, 68, 0.2)';
                    badgeColor = wr > 50 ? '#34d399' : '#f87171';
                } else if (dev > targetPairTol) {
                    bg = 'rgba(245, 158, 11, 0.08)';
                    color = '#fbbf24';
                    badgeText = `偏离 ±${dev.toFixed(1)}%`;
                    badgeBg = 'rgba(245, 158, 11, 0.2)';
                    badgeColor = '#f59e0b';
                }

                html += `<td style="background: ${bg}; border:1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 8px 6px; transition: all 0.2s;">
                    <div style="color:${color}; font-size:1.15rem; font-weight:800; letter-spacing:0.5px;">${wr}%</div>
                    <div style="color:#94a3b8; font-size:0.72rem; margin: 2px 0;">${cell.wins}胜 / ${cell.total}局</div>
                    <span style="font-size:0.68rem; padding: 1px 6px; border-radius: 4px; background:${badgeBg}; color:${badgeColor}; font-weight:600;">${badgeText}</span>
                </td>`;
            }
        });
        html += '</tr>';
    });
    html += '</table></div>';

    const statusColor = maxPairwiseDev <= targetPairTol ? '#10b981' : (maxPairwiseDev <= severePairTol ? '#f59e0b' : '#ef4444');
    const statusIcon = maxPairwiseDev <= targetPairTol ? '✅ 达标' : '⚠️ 需调优';
    html += `<div style="margin-top:10px; display:flex; justify-content:space-between; align-items:center; font-size:0.78rem; color:#94a3b8; border-top:1px solid rgba(255,255,255,0.06); padding-top:8px;">
        <span>双向实机合并战报 (对局数已合并先手与后手)</span>
        <span>两两最大偏离: <strong style="color:${statusColor}; font-size:0.85rem;">±${maxPairwiseDev.toFixed(1)}%</strong> (${statusIcon} · 目标约束 ≤ ±${targetPairTol.toFixed(1)}%)</span>
    </div>`;

    matrixContainer.innerHTML = html;
}

// WebSocket for live terminal stream
function initPipelineSocket() {
    if (pipelineWs && (pipelineWs.readyState === WebSocket.OPEN || pipelineWs.readyState === WebSocket.CONNECTING)) {
        return;
    }

    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${location.host}/ws/pipeline`;

    try {
        pipelineWs = new WebSocket(wsUrl);

        pipelineWs.onopen = () => {
            console.log("[PipelineWS] Connected to live pipeline stream");
        };

        pipelineWs.onmessage = (event) => {
            const line = event.data;
            if (!line) return;
            handleConsoleLog(line);
        };

        pipelineWs.onclose = () => {
            setTimeout(initPipelineSocket, 3000);
        };

        pipelineWs.onerror = (e) => {
            console.warn("[PipelineWS] Error:", e);
        };
    } catch (e) {
        console.error("[PipelineWS] Connection error:", e);
    }
}

// Handle Console Log formatting and line appending
function handleConsoleLog(line) {
    const consoleBox = document.getElementById('pipe-console');
    if (!consoleBox) return;

    let type = '';
    if (line.startsWith('[Console]')) type = 'system';
    else if (line.includes('达成平衡') || line.includes('收敛') || line.includes('[完成]') || line.includes('🎉')) type = 'success';
    else if (line.includes('严重失衡') || line.includes('熔断') || line.includes('⚠️')) type = 'warning';
    else if (line.includes('异常') || line.includes('失败') || line.includes('错误') || line.includes('[ERROR]')) type = 'error';
    else if (line.includes('【轮次') || line.includes('[判定]') || line.includes('微调') || line.includes('DeepSeek')) type = 'highlight';

    appendConsoleLine(line, type);

    if (line.includes('流水线执行结束') || line.includes('阶段六：导出成果与报表') || line.includes('已达成平衡收敛')) {
        setTimeout(refreshAnalyticsData, 1200);
    }
}

function appendConsoleLine(text, type = '') {
    const consoleBox = document.getElementById('pipe-console');
    if (!consoleBox) return;

    const div = document.createElement('div');
    div.className = `console-line ${type}`.trim();
    div.textContent = text;
    consoleBox.appendChild(div);

    if (consoleBox.children.length > 1200) {
        consoleBox.removeChild(consoleBox.firstChild);
    }

    const autoScroll = document.getElementById('chk-auto-scroll');
    if (!autoScroll || autoScroll.checked) {
        consoleBox.scrollTop = consoleBox.scrollHeight;
    }
}

// Initialize Controls and Buttons
function initPipelineControls() {
    const btnStart = document.getElementById('btn-pipe-start');
    const btnStop = document.getElementById('btn-pipe-stop');
    const btnRefresh = document.getElementById('btn-pipe-refresh');
    const btnClear = document.getElementById('btn-clear-console');

    if (btnStart) {
        btnStart.onclick = startPipeline;
    }
    if (btnStop) {
        btnStop.onclick = stopPipeline;
    }
    if (btnRefresh) {
        btnRefresh.onclick = () => {
            appendConsoleLine('[Console] 正在刷新图表与阵营数值...', 'system');
            refreshAnalyticsData();
        };
    }
    if (btnClear) {
        btnClear.onclick = clearPipelineConsole;
    }

    const targetEl = document.getElementById('pipe-target');
    const pairwiseEl = document.getElementById('pipe-pairwise-target');
    const onToleranceChange = () => {
        if (window.tcgData) {
            updateMetricCards(window.tcgData);
            renderMatchupMatrix(window.tcgData);
        }
    };
    if (targetEl) targetEl.addEventListener('change', onToleranceChange);
    if (pairwiseEl) pairwiseEl.addEventListener('change', onToleranceChange);
}

function clearPipelineConsole() {
    const consoleBox = document.getElementById('pipe-console');
    if (consoleBox) {
        consoleBox.innerHTML = '<div class="console-line system">[Console] 控制台已清屏。点击【▶️ 启动调优流水线】即可开始。</div>';
    }
}

// Start Pipeline Execution
async function startPipeline() {
    // 1. 启动前严格检查 DeepSeek API Key 是否已配置
    try {
        const cfgRes = await fetch('/api/config/llm');
        if (cfgRes.ok) {
            const cfg = await cfgRes.json();
            if (!cfg.has_deepseek_key) {
                const dsInput = document.getElementById('cfg-deepseek-key');
                if (dsInput) {
                    dsInput.focus();
                    dsInput.style.outline = '2px solid #ff7675';
                    setTimeout(() => { dsInput.style.outline = ''; }, 4000);
                }
                alert('【未配置 DeepSeek Key 密钥】\n\n无论选择哪种调优模式，AI 平衡调优流水线的核心均需要调用 DeepSeek 大模型对失衡卡牌进行诊断、身材重构与参数微调。\n\n请先在左侧【AI 服务配置】面板中填入您的 DeepSeek API Key (sk-...) 并点击【保存配置】后再启动！');
                appendConsoleLine('❌ [Console] 启动被拦截：未检测到有效 DeepSeek Key。请在左侧面板配置后重试。', 'error');
                return;
            }
        }
    } catch (e) {
        console.warn('API config check warning:', e);
    }

    const modeEl = document.getElementById('pipe-mode');
    const epEl = document.getElementById('pipe-episodes');
    const targetEl = document.getElementById('pipe-target');

    const mode = modeEl ? modeEl.value : 'tune_only';
    const episodes = epEl ? parseInt(epEl.value, 10) : 3000;
    const targetBalance = targetEl ? parseFloat(targetEl.value) : 5.0;
    const pairwiseEl = document.getElementById('pipe-pairwise-target');
    const targetPairwiseBalance = pairwiseEl ? parseFloat(pairwiseEl.value) : 5.0;

    const btnStart = document.getElementById('btn-pipe-start');
    const btnStop = document.getElementById('btn-pipe-stop');
    const statusVal = document.getElementById('stat-runner-status');

    if (btnStart) btnStart.disabled = true;
    if (btnStop) btnStop.disabled = false;
    if (statusVal) {
        statusVal.textContent = '⚙️ 启动中...';
        statusVal.style.color = '#00cec9';
    }

    appendConsoleLine(`[Console] 🚀 正在向调度器下发启动指令: 模式 [${mode}], 单轮 [${episodes} 局], 阵营总容差 [±${targetBalance}%], 两两对抗容差 [±${targetPairwiseBalance}%]...`, 'system');

    try {
        const res = await fetch('/api/pipeline/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                mode,
                episodes,
                target_balance: targetBalance,
                target_pairwise_balance: targetPairwiseBalance
            })
        });
        const data = await res.json();
        if (data.success) {
            appendConsoleLine(`[Console] ✅ 流水线启动成功！已启动后台自博弈进程。`, 'success');
            pollPipelineStatus();
        } else {
            alert(`【流水线启动失败】\n\n${data.message}`);
            appendConsoleLine(`[Console] ❌ 启动失败: ${data.message}`, 'error');
            if (btnStart) btnStart.disabled = false;
            if (btnStop) btnStop.disabled = true;
            pollPipelineStatus();
        }
    } catch (err) {
        appendConsoleLine(`[Console] ❌ 网络或请求异常: ${err}`, 'error');
        if (btnStart) btnStart.disabled = false;
        pollPipelineStatus();
    }
}

// Stop Pipeline Execution
async function stopPipeline() {
    const btnStop = document.getElementById('btn-pipe-stop');
    if (btnStop) btnStop.disabled = true;

    appendConsoleLine('[Console] 🛑 正在请求停止流水线...', 'system');

    try {
        const res = await fetch('/api/pipeline/stop', { method: 'POST' });
        const data = await res.json();
        appendConsoleLine(`[Console] ${data.message}`, data.success ? 'warning' : 'error');
        pollPipelineStatus();
    } catch (err) {
        appendConsoleLine(`[Console] ❌ 停止流水线异常: ${err}`, 'error');
        pollPipelineStatus();
    }
}

// Poll Pipeline Status
async function pollPipelineStatus() {
    try {
        const res = await fetch('/api/pipeline/status');
        if (!res.ok) return;
        const status = await res.json();

        const btnStart = document.getElementById('btn-pipe-start');
        const btnStop = document.getElementById('btn-pipe-stop');
        const statusVal = document.getElementById('stat-runner-status');
        const statusTime = document.getElementById('stat-runner-time');

        if (status.running) {
            if (btnStart) btnStart.disabled = true;
            if (btnStop) btnStop.disabled = false;
            if (statusVal) {
                statusVal.textContent = '⚙️ 调优中...';
                statusVal.style.color = '#00cec9';
            }
            if (statusTime) {
                statusTime.textContent = `已运行: ${status.elapsed_seconds || 0}s`;
            }
        } else {
            if (btnStart) btnStart.disabled = false;
            if (btnStop) btnStop.disabled = true;
            if (statusVal) {
                statusVal.textContent = '🟢 待机就绪';
                statusVal.style.color = '#10b981';
            }
            if (statusTime) {
                statusTime.textContent = status.elapsed_seconds ? `上次耗时: ${status.elapsed_seconds}s` : '耗时: 0s';
            }
        }
    } catch (e) {
        // Silent poll error
    }
}

// Refresh Data from backend
async function refreshAnalyticsData() {
    const btnRefresh = document.getElementById('btn-pipe-refresh');
    if (btnRefresh) {
        btnRefresh.disabled = true;
        btnRefresh.innerHTML = '<span class="spin">🔄</span> 正在获取最新战报...';
    }

    try {
        const res = await fetch(`/api/data?t=${Date.now()}`);
        if (res.ok) {
            const data = await res.json();
            window.tcgData = data;
            renderRadarChart(data);
            renderMatchupMatrix(data);
            updateMetricCards(data);

            if (btnRefresh) {
                btnRefresh.innerHTML = '✅ 刷新成功';
                setTimeout(() => {
                    btnRefresh.innerHTML = '🔄 刷新数据';
                    btnRefresh.disabled = false;
                }, 700);
            }

            const ep = data.total_episodes || 0;
            const maxPDev = (data.max_pairwise_dev != null) ? data.max_pairwise_dev.toFixed(1) : '--';
            appendConsoleLine(`[Console] 🔄 数据已更新 (混战累计: ${ep} 局 | 两两对抗最大偏离: ±${maxPDev}%)。`, 'success');
        } else {
            if (btnRefresh) {
                btnRefresh.innerHTML = '❌ 刷新失败';
                setTimeout(() => {
                    btnRefresh.innerHTML = '🔄 刷新数据';
                    btnRefresh.disabled = false;
                }, 1000);
            }
        }
    } catch (e) {
        console.error("[Analytics] Failed to refresh data:", e);
        if (btnRefresh) {
            btnRefresh.innerHTML = '❌ 网络异常';
            setTimeout(() => {
                btnRefresh.innerHTML = '🔄 刷新数据';
                btnRefresh.disabled = false;
            }, 1000);
        }
    }
}

// Event Listeners for reliable execution
document.addEventListener('tcgDataLoaded', () => {
    if (window.tcgData) {
        renderRadarChart(window.tcgData);
        renderMatchupMatrix(window.tcgData);
        updateMetricCards(window.tcgData);
    }
});

// Run directly or on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAnalytics);
} else {
    initAnalytics();
}
