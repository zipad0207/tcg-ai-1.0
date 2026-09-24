# TCG-AI: 双路集换式卡牌强化学习自博弈与 LLM 自动平衡系统

TCG-AI 是一个机制完备的双路集换式卡牌对战环境（DuelEnv），结合**强化学习自博弈（PPO）**与**大语言模型（LLM）**，实现卡牌机制策划、卡组构筑、高并发对战模拟与数值闭环调优。

系统内置现代 Web 可视化控制台，支持实机人机对弈、实时记牌器、卡组工坊、后台异步排队卡图生成与平衡遥测仪表盘。

---

## ⚡ 快速启动 (Quick Start)

### 方式 1：Windows 一键启动（推荐）
在项目根目录**直接双击运行 `run_web.bat`**：
- 自动检测并配置 Python 环境；
- 自动使用国内镜像源静默补齐依赖库；
- 启动本地 Web 服务并在浏览器自动打开 **`http://127.0.0.1:8000`**。

### 方式 2：命令行手动启动
```bash
# 1. 克隆仓库
git clone https://github.com/zipad0207/tcg-ai-1.0.git
cd tcg-ai-1.0

# 2. 安装依赖 (已支持纯 CPU 运行，有显卡自动启用 CUDA)
pip install -r requirements.txt

# 3. 启动 Web 可视化控制台
py -m uvicorn web_app.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## ⚙️ 模型与 API 配置

复制配置模板并填写 Key：
```bash
cp .env.example .env
```
配置项说明（亦可在 Web 页面左侧导航栏直接填写保存）：
- `DEEPSEEK_API_KEY`: DeepSeek API 密钥，用于数值调优与新卡设计。
- `DEEPSEEK_BASE_URL`: API 地址（官方 `https://api.deepseek.com` 或硅基流动 `https://api.siliconflow.cn/v1`）。
- `SILICONFLOW_API_KEY`: 用于调用文生图模型（Z-Image-Turbo / 可图 Kolors）生成卡牌原画。

---

## 🎮 核心系统与功能

### 1. 双路对战引擎 (DuelEnv)
- **左右双路攻防**：分为进攻区、防守区与对撞线；随从进场蓄势冲锋，支持 `RUSH` 突袭即时就绪；
- **阵营机制特色**：
  - 🔴 **赤红 (Red)**：快攻铺场、突袭冲锋、牺牲斩杀；
  - 🔵 **蔚蓝 (Blue)**：坚守驻防、光环支援、护盾控制；
  - 🟢 **翠绿 (Green)**：跳费成长、高费大核随从压制；
  - ⚪ **中立 (Neutral)**：过牌检索与通用攻防辅助；
  - 🟣 **双色卡 (Dual)**：跨阵营机制融合。
- **先后手对称平衡**：先手 3 张牌 vs 后手 4 张牌+幸运币，900 局实机测试先后手胜率达到 **50.03% vs 49.97%**，排除先后手系统偏差。

### 2. PPO 强化学习自博弈
- **Actor-Critic 架构**：状态评估与动作策略联合网络（CardNet）；
- **动态卡组演化**：基于网络价值头进行单卡胜率与状态评估，自主优化 30 张卡组搭配。

### 3. LLM 辅助数值调优与新卡生成
- 根据对局实测遥测数据（胜率偏离、单卡出牌频次、费用分布），调用大模型针对性设计新机制或微调卡牌攻防费用；
- 设有数值防废卡拦截机制（如避免费用高于临时法力增益的反向设计）。

### 4. 后台自动排队出图 (Background Art Queue)
- 新卡设计入库后，后台独立线程自动排队调用生图模型（Tongyi-MAI/Z-Image-Turbo 等）绘制插图；
- 异步非阻塞执行，完全不影响主流水线的对弈模拟与强化学习微调；
- 内置平滑控频与 429 频控重试，Web 端支持实时进度轮询。

---

## 📊 调优效果实测

### 1. 红蓝对决调优 (1,000 局 PPO 自博弈)
| 阶段 | 卡池配置 | 红方胜率 | 蓝方胜率 | 偏离度 (\|WR - 50%\|) | 状态 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **调优前 (Baseline)** | 初始蓝方坚守过强 | **18.4%** | **81.6%** | 31.6% | 严重失衡 |
| **调优后 (Tuned)** | 针对身材、费用、突袭微调 | **48.7%** | **51.3%** | **1.3%** | 达成平衡 |

### 2. 三大阵营全自动流水线 (3,000 局混战)
在 60 张全量卡池下经过多轮闭环微调，三大阵营最终胜率：
- 🔴 **赤红 (Red)**: **49.68%**
- 🔵 **蔚蓝 (Blue)**: **52.09%**
- 🟢 **翠绿 (Green)**: **48.36%**
- **三大阵营最大偏离度仅 2.09%**（完全收敛于目标平衡区间 $[45.0\%, 55.0\%]$）。

---

## 🛠️ 常用命令行工具

```bash
# 1. 运行端到端全自动印卡、构筑、对战与数值微调流水线
py pipeline_orchestrator.py --pack-name "新扩展包" --episodes 3000

# 2. 快速测试流水线 (快速演练模式，跳过新卡印刷)
py pipeline_orchestrator.py --dry-run --skip-print

# 3. 运行双阵营卡组快速对抗测试 (500 局快速评估)
py test_deck_matchup.py --episodes 500 --device cpu

# 4. 独立印卡工具 (生成新卡并自动后台排队生图)
py card_printer_deepseek.py --count 4 --auto-art

# 5. 批量生成/补全卡池缺失的卡牌原画插图
py batch_regenerate.py
```

---

## 📁 项目结构

```text
├── run_web.bat                          # Windows 一键环境检测与 Web 启动脚本
├── requirements.txt                     # 核心运行依赖清单
├── .env.example                         # 环境变量配置模板
├── pipeline_orchestrator.py             # 全流程自动扩展包印制与平衡闭环流水线
├── sandbox.py                           # 对战引擎核心 (DuelEnv 环境)
├── agent.py                             # PPO 策略价值网络模型 (CardNet)
├── train_brawl.py                       # 多阵营高并发混战对战训练器
├── train.py                             # 基础对抗训练入口
├── deck_builder_ppo.py                  # 基于 PPO 价值评估的智能卡组构筑器
├── deck_builder_deepseek.py             # 基于 LLM 的阵营套牌构筑器
├── auto_balancer_deepseek.py            # 基于对战遥测数据的卡牌数值调优工具
├── card_printer_deepseek.py             # 定向生成新机制卡牌工具
├── batch_regenerate.py                  # 批量卡图补全工具
├── test_deck_matchup.py                 # 多阵营卡组批处理对战评测
├── export_ui_data.py                    # 前端可视化数据导出接口
├── cards_config.json                    # 全量卡池数据配置 (178+ 张卡牌)
├── decks_config.json                    # 各阵营对战卡组配置
└── web_app/                             # Web 可视化控制台
    ├── main.py                          # FastAPI 后端服务
    ├── services/                        # 后台服务 (出图队列、对战会话、印卡)
    └── static/                          # 前端单页应用 (HTML/CSS/JS、卡牌原画)
```

---

## License

本项目遵循 [MIT License](LICENSE) 开源许可协议。
