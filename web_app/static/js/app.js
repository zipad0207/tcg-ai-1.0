// Global state
window.tcgData = null;

// Global Tag Translator
window.translateTag = function(tag) {
    if (!tag) return '';
    const t = String(tag).toUpperCase().trim();
    if (t === 'RUSH') return '⚡ 突袭';
    if (t === 'ATTACK_ONLY') return '⚔️ 仅限进攻';
    if (t === 'SACRIFICE_1_KILL_1') return '🩸 献祭消灭';
    
    if (t.startsWith('FORTIFY_')) return `🛡️ 坚守+${t.split('_')[1]}`;
    if (t.startsWith('DEGRADE_')) return `⚔️ 破甲-${t.split('_')[1]}`;
    if (t.startsWith('SUPPORT_ATK_')) return `🌟 支援+${t.split('_')[2]}`;
    if (t.startsWith('BONUS_SCORE_')) return `🏆 突破得分+${t.split('_')[2]}`;
    if (t.startsWith('SPAWN_')) {
        const parts = t.split('_');
        return `👥 召唤${parts[2]}只(${parts[1]}攻)`;
    }
    if (t.startsWith('DEATH_DRAW_')) return `💀 亡语:抽${t.split('_')[2]}张`;
    if (t.startsWith('DEATH_MANA_')) return `💀 亡语:法力+${t.split('_')[2]}`;
    if (t.startsWith('DRAW_')) return `🎴 抽${t.split('_')[1]}张`;
    if (t.startsWith('RAMP_')) return `💎 法力上限+${t.split('_')[1]}`;
    if (t.startsWith('TEMP_MANA_')) return `✨ 临时法力+${t.split('_')[2]}`;
    if (t.startsWith('DISCARD_')) return `🗑️ 弃${t.split('_')[1]}张`;

    return tag;
};

document.addEventListener('DOMContentLoaded', async () => {
    // Handle Navigation
    const navLinks = document.querySelectorAll('.nav-links li');
    const viewSections = document.querySelectorAll('.view-section');

    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            // Remove active classes
            navLinks.forEach(l => l.classList.remove('active'));
            viewSections.forEach(v => v.classList.remove('active'));
            
            // Add active class to clicked link and target section
            link.classList.add('active');
            const targetId = link.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');
        });
    });

    // Setup AI API Configuration
    initApiConfig();

    // Fetch initial data
    try {
        const response = await fetch('/api/data');
        if (response.ok) {
            window.tcgData = await response.json();
            console.log("TCG Data loaded:", window.tcgData);
            
            // Dispatch event to notify other modules
            document.dispatchEvent(new Event('tcgDataLoaded'));
        } else {
            console.error("Failed to load TCG data", response.statusText);
        }
    } catch (e) {
        console.error("Error fetching data:", e);
    }
});

async function initApiConfig() {
    const dsKeyInput = document.getElementById('cfg-deepseek-key');
    const imgKeyInput = document.getElementById('cfg-image-key');
    const imgModelSelect = document.getElementById('cfg-image-model');
    const badge = document.getElementById('api-status-badge');
    const dsVisBtn = document.getElementById('btn-toggle-deepseek-vis');
    const imgVisBtn = document.getElementById('btn-toggle-image-vis');
    const saveBtn = document.getElementById('btn-save-api-cfg');

    if (dsVisBtn && dsKeyInput) {
        dsVisBtn.onclick = () => {
            dsKeyInput.type = dsKeyInput.type === 'password' ? 'text' : 'password';
            dsVisBtn.innerText = dsKeyInput.type === 'password' ? '👁' : '🔒';
        };
    }

    if (imgVisBtn && imgKeyInput) {
        imgVisBtn.onclick = () => {
            imgKeyInput.type = imgKeyInput.type === 'password' ? 'text' : 'password';
            imgVisBtn.innerText = imgKeyInput.type === 'password' ? '👁' : '🔒';
        };
    }

    const updateBadge = (hasDs, hasImg) => {
        if (!badge) return;
        if (hasDs && hasImg) {
            badge.innerText = "已就绪";
            badge.className = "api-status-badge connected";
        } else if (hasDs || hasImg) {
            badge.innerText = "部分就绪";
            badge.className = "api-status-badge connected";
        } else {
            badge.innerText = "未配置";
            badge.className = "api-status-badge";
        }
    };

    try {
        const res = await fetch('/api/config/llm');
        if (res.ok) {
            const data = await res.json();
            if (dsKeyInput && data.deepseek_key_masked) dsKeyInput.placeholder = data.deepseek_key_masked;
            if (imgKeyInput && data.image_key_masked) imgKeyInput.placeholder = data.image_key_masked;
            if (imgModelSelect && data.image_model) imgModelSelect.value = data.image_model;
            updateBadge(data.has_deepseek_key, data.has_image_key);
        }
    } catch (e) {
        console.warn("Failed to fetch API config:", e);
    }

    if (saveBtn) {
        saveBtn.onclick = async () => {
            const dsKey = dsKeyInput?.value.trim() || "";
            const imgKey = imgKeyInput?.value.trim() || "";
            const imgModel = imgModelSelect?.value || "Tongyi-MAI/Z-Image-Turbo";

            saveBtn.disabled = true;
            saveBtn.innerText = "保存中...";
            try {
                const res = await fetch('/api/config/llm', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        deepseek_api_key: dsKey, 
                        image_api_key: imgKey, 
                        image_model: imgModel 
                    })
                });
                const data = await res.json();
                if (res.ok && data.success) {
                    updateBadge(data.has_deepseek_key, data.has_image_key);
                    if (dsKey && dsKeyInput) {
                        dsKeyInput.value = "";
                        dsKeyInput.placeholder = dsKey.slice(0, 4) + "••••••••" + dsKey.slice(-4);
                    }
                    if (imgKey && imgKeyInput) {
                        imgKeyInput.value = "";
                        imgKeyInput.placeholder = imgKey.slice(0, 4) + "••••••••" + imgKey.slice(-4);
                    }
                    if (window.showGameToast) {
                        window.showGameToast("AI 语言模型与生图配置已保存！", "success");
                    } else {
                        alert("AI 配置已成功保存！");
                    }
                } else {
                    alert("保存失败: " + (data.message || '未知错误'));
                }
            } catch (e) {
                alert("保存失败: " + e.message);
            } finally {
                saveBtn.disabled = false;
                saveBtn.innerText = "保存配置";
            }
        };
    }
}

