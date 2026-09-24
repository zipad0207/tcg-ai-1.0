import os
import sys
import json
import mimetypes
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .services.image_gen import ZImageTurboGenerator
from .services.game_session import GameSession
from .services.card_printer import DeepSeekCardPrinter
from .services.pipeline_runner import runner as pipeline_runner
import asyncio

# Ensure parent directory is in sys.path to import export_ui_data
_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)
from export_ui_data import export_ui_data

# Ensure webp mime type is properly recognized on Windows
mimetypes.add_type("image/webp", ".webp")

app = FastAPI(title="TCG-AI Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_no_cache_header(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/static/images/"):
        # Allow aggressive browser caching for card images; updates are busted via ?v=timestamp
        response.headers["Cache-Control"] = "public, max-age=604800, immutable"
        if "Pragma" in response.headers:
            del response.headers["Pragma"]
        if "Expires" in response.headers:
            del response.headers["Expires"]
    elif path.startswith("/static/") or path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

image_generator = ZImageTurboGenerator()
card_printer = DeepSeekCardPrinter()

static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def read_root():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/api/data")
def get_ui_data():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        # Dynamically refresh and export live metrics from training_metrics_brawl.json
        return export_ui_data(root_dir)
    except Exception as e:
        print(f"[get_ui_data] Live export error: {e}")
        ui_data_path = os.path.join(root_dir, "ui_export_data.json")
        if os.path.exists(ui_data_path):
            with open(ui_data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        raise HTTPException(status_code=404, detail="ui_export_data.json not found.")

class GenerateArtRequest(BaseModel):
    card_id: int
    card_name: str
    faction: str
    tags: list
    dp: int

@app.post("/api/generate_art")
def generate_art(req: GenerateArtRequest):
    try:
        result = image_generator.generate_image(
            card_id=req.card_id,
            card_name=req.card_name,
            faction=req.faction,
            tags=req.tags,
            dp=req.dp
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class GenerateCardsRequest(BaseModel):
    faction: str = "Red"
    count: int = 2
    theme: str = ""
    auto_gen_art: bool = True

@app.post("/api/generate_new_cards")
def generate_new_cards(req: GenerateCardsRequest):
    try:
        res = card_printer.generate_new_cards(faction=req.faction, count=req.count, theme=req.theme)
        if req.auto_gen_art:
            for card in res.get("cards", []):
                try:
                    c_faction = card.get("factions", [req.faction])[0]
                    art_res = image_generator.generate_image(
                        card_id=card["id"],
                        card_name=card["name"],
                        faction=c_faction,
                        tags=card.get("tags", []),
                        dp=card.get("base_dp", 0)
                    )
                    card["image_url"] = art_res.get("url", "")
                except Exception as art_err:
                    print(f"[GenerateNewCards] Art generation error on {card['name']}: {art_err}")
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/batch_generate_art")
def batch_generate_art():
    try:
        res = image_generator.batch_generate_missing()
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/batch_generate_art/status")
def batch_generate_art_status():
    missing = image_generator.get_missing_cards()
    return {"missing_count": len(missing), "missing_cards": missing}

from typing import Optional, List

class LLMConfigRequest(BaseModel):
    deepseek_api_key: Optional[str] = ""
    image_api_key: Optional[str] = ""
    image_model: Optional[str] = "Tongyi-MAI/Z-Image-Turbo"
    # Legacy fallbacks
    api_key: Optional[str] = ""
    base_url: Optional[str] = "https://api.deepseek.com"
    model: Optional[str] = "deepseek-chat"

@app.get("/api/config/llm")
def get_llm_config():
    root_dir = os.path.dirname(os.path.dirname(__file__))
    cfg_file = os.path.join(root_dir, "llm_config.json")
    saved_cfg = {}
    if os.path.exists(cfg_file):
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                saved_cfg = json.load(f)
        except Exception:
            pass
            
    deepseek_key = saved_cfg.get("deepseek_api_key") or saved_cfg.get("api_key") or os.environ.get("DEEPSEEK_API_KEY", "")
    image_key = saved_cfg.get("image_api_key") or os.environ.get("SILICONFLOW_API_KEY", "") or os.environ.get("DASH_SCOPE_API_KEY", "")
    image_model = saved_cfg.get("image_model") or os.environ.get("IMAGE_MODEL", "Tongyi-MAI/Z-Image-Turbo")

    def mask(k):
        if not k:
            return ""
        if len(k) > 8:
            return k[:4] + "••••••••" + k[-4:]
        return "••••••••"

    return {
        "deepseek_key_masked": mask(deepseek_key),
        "has_deepseek_key": bool(deepseek_key),
        "image_key_masked": mask(image_key),
        "has_image_key": bool(image_key),
        "image_model": image_model,
        # backwards compatibility
        "has_key": bool(deepseek_key),
        "api_key_masked": mask(deepseek_key)
    }

@app.post("/api/config/llm")
def save_llm_config(req: LLMConfigRequest):
    root_dir = os.path.dirname(os.path.dirname(__file__))
    cfg_file = os.path.join(root_dir, "llm_config.json")
    
    saved_cfg = {}
    if os.path.exists(cfg_file):
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                saved_cfg = json.load(f)
        except Exception:
            pass

    # 1. DeepSeek Key
    ds_key = (req.deepseek_api_key or req.api_key or "").strip()
    if not ds_key or "••••" in ds_key:
        ds_key = saved_cfg.get("deepseek_api_key") or saved_cfg.get("api_key") or os.environ.get("DEEPSEEK_API_KEY", "")

    # 2. Image Key
    img_key = (req.image_api_key or "").strip()
    if not img_key or "••••" in img_key:
        img_key = saved_cfg.get("image_api_key") or os.environ.get("SILICONFLOW_API_KEY", "") or os.environ.get("DASH_SCOPE_API_KEY", "")

    img_model = (req.image_model or "Tongyi-MAI/Z-Image-Turbo").strip()

    if ds_key:
        os.environ["DEEPSEEK_API_KEY"] = ds_key
    if img_key:
        os.environ["SILICONFLOW_API_KEY"] = img_key
        if img_model == "dashscope":
            os.environ["DASH_SCOPE_API_KEY"] = img_key
    os.environ["IMAGE_MODEL"] = img_model

    # Sync into global image_generator instance if available
    try:
        global image_generator
        if 'image_generator' in globals() and image_generator:
            if img_key:
                image_generator.siliconflow_key = img_key
                image_generator.api_key = img_key
            image_generator.model_name = img_model
    except Exception:
        pass

    new_cfg = {
        "deepseek_api_key": ds_key,
        "image_api_key": img_key,
        "image_model": img_model,
        "api_key": ds_key
    }
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(new_cfg, f, indent=4, ensure_ascii=False)

    return {
        "success": True, 
        "has_deepseek_key": bool(ds_key),
        "has_image_key": bool(img_key),
        "image_model": img_model
    }

class CustomDeckRequest(BaseModel):
    deck_index: Optional[int] = 0
    name: str = "我的自构筑卡组"
    faction: str = "Red"
    card_ids: list

def load_all_custom_decks(root_dir: str):
    decks_path = os.path.join(root_dir, "custom_decks.json")
    old_deck_path = os.path.join(root_dir, "custom_deck.json")
    decks_cfg_path = os.path.join(root_dir, "decks_config.json")
    
    red_preset, blue_preset, green_preset = [], [], []
    if os.path.exists(decks_cfg_path):
        try:
            with open(decks_cfg_path, "r", encoding="utf-8") as f:
                d_cfg = json.load(f)
                red_preset = d_cfg.get("Red", {}).get("decklist", [])
                blue_preset = d_cfg.get("Blue", {}).get("decklist", [])
                green_preset = d_cfg.get("Green", {}).get("decklist", [])
        except Exception as e:
            print("Error loading decks_config:", e)

    data = None
    if os.path.exists(decks_path):
        try:
            with open(decks_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = None

    if not data or "decks" not in data or not isinstance(data["decks"], list) or len(data["decks"]) == 0:
        old_deck = None
        if os.path.exists(old_deck_path):
            try:
                with open(old_deck_path, "r", encoding="utf-8") as f:
                    old_deck = json.load(f)
            except Exception:
                old_deck = None
                
        user_saved_ids = old_deck.get("card_ids", []) if old_deck else []
        user_faction = old_deck.get("faction", "Blue") if old_deck else "Blue"
        user_name = old_deck.get("name", "蔚蓝·自定义构筑") if old_deck else "蔚蓝·自定义构筑"
        
        decks = [
            {"id": "deck_1", "name": "赤红·突击快攻", "faction": "Red", "card_ids": list(red_preset)},
            {"id": "deck_2", "name": user_name, "faction": user_faction, "card_ids": list(user_saved_ids if user_saved_ids else blue_preset)},
            {"id": "deck_3", "name": "翠绿·成长跳费", "faction": "Green", "card_ids": list(green_preset)},
            {"id": "deck_4", "name": "赤红·备选槽位4", "faction": "Red", "card_ids": []},
            {"id": "deck_5", "name": "蔚蓝·备选槽位5", "faction": "Blue", "card_ids": []},
            {"id": "deck_6", "name": "翠绿·备选槽位6", "faction": "Green", "card_ids": []},
        ]
        data = {"active_deck_index": 1 if user_saved_ids else 0, "decks": decks}
        with open(decks_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
    while len(data["decks"]) < 6:
        idx = len(data["decks"]) + 1
        factions = ["Red", "Blue", "Green"]
        f = factions[(idx - 1) % 3]
        f_cn = {"Red": "赤红", "Blue": "蔚蓝", "Green": "翠绿"}[f]
        data["decks"].append({"id": f"deck_{idx}", "name": f"{f_cn}·构筑槽位{idx}", "faction": f, "card_ids": []})
        
    return data

@app.post("/api/custom_deck")
def save_custom_deck(req: CustomDeckRequest):
    root_dir = os.path.dirname(os.path.dirname(__file__))
    cards_path = os.path.join(root_dir, "cards_config.json")
    
    # 1. 严格 30 张上限
    if len(req.card_ids) > 30:
        return JSONResponse(status_code=400, content={"success": False, "message": "卡组超过 30 张上限！"})
    
    # 2. 同名卡严格最多 3 张
    counts = {}
    for cid in req.card_ids:
        counts[cid] = counts.get(cid, 0) + 1
        if counts[cid] > 3:
            return JSONResponse(status_code=400, content={"success": False, "message": f"卡牌 (ID: {cid}) 超过同名卡 3 张携带上限！"})
            
    # 3. 严格阵营合法性校验（严禁将蓝卡加入红卡组等跨阵营行为）
    if os.path.exists(cards_path):
        try:
            with open(cards_path, "r", encoding="utf-8") as f:
                all_cards_data = json.load(f)
            card_db = {}
            for f_name, c_list in all_cards_data.items():
                if isinstance(c_list, list):
                    for c in c_list:
                        card_db[c["id"]] = c
                        
            f_target = req.faction
            target_code = 1 if f_target == "Red" else (2 if f_target == "Blue" else 3)
            f_names = {"Red": "🔴 赤红", "Blue": "🔵 蔚蓝", "Green": "🟢 翠绿"}

            for cid in req.card_ids:
                if cid not in card_db:
                    return JSONResponse(status_code=400, content={"success": False, "message": f"卡牌 ID {cid} 不存在！"})
                c = card_db[cid]
                c_factions = c.get("factions", [])
                cid_prefix = cid // 100
                
                allowed = False
                if f_target in c_factions or "Neutral" in c_factions:
                    allowed = True
                elif cid_prefix == target_code or cid_prefix == 9:
                    allowed = True
                elif cid_prefix == 4 and target_code in (1, 2):
                    allowed = True
                elif cid_prefix == 5 and target_code in (2, 3):
                    allowed = True
                elif cid_prefix == 6 and target_code in (1, 3):
                    allowed = True
                    
                if not allowed:
                    return JSONResponse(status_code=400, content={
                        "success": False, 
                        "message": f"卡牌【{c['name']}】属于其他阵营，无法加入【{f_names.get(f_target, f_target)}】卡组！"
                    })
        except Exception as e:
            print("Deck validation error:", e)

    data = load_all_custom_decks(root_dir)
    idx = req.deck_index if (req.deck_index is not None and 0 <= req.deck_index < len(data["decks"])) else 0
    data["decks"][idx]["name"] = req.name
    data["decks"][idx]["faction"] = req.faction
    data["decks"][idx]["card_ids"] = req.card_ids
    data["active_deck_index"] = idx

    decks_path = os.path.join(root_dir, "custom_decks.json")
    with open(decks_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    # Sync active deck to custom_deck.json
    deck_path = os.path.join(root_dir, "custom_deck.json")
    with open(deck_path, "w", encoding="utf-8") as f:
        json.dump(data["decks"][idx], f, indent=4, ensure_ascii=False)

    return {"success": True, "count": len(req.card_ids), "active_deck_index": idx, "decks": data["decks"]}

@app.get("/api/custom_decks")
def get_custom_decks():
    root_dir = os.path.dirname(os.path.dirname(__file__))
    return load_all_custom_decks(root_dir)

@app.get("/api/custom_deck")
def get_custom_deck():
    root_dir = os.path.dirname(os.path.dirname(__file__))
    data = load_all_custom_decks(root_dir)
    idx = data.get("active_deck_index", 0)
    if idx < 0 or idx >= len(data["decks"]):
        idx = 0
    active = data["decks"][idx]
    return {
        "name": active["name"],
        "faction": active["faction"],
        "card_ids": active["card_ids"],
        "active_deck_index": idx,
        "decks": data["decks"]
    }

@app.websocket("/ws/arena")
async def websocket_arena(websocket: WebSocket):
    await websocket.accept()
    session = GameSession(started=False)
    autoplay_task = None
    play_delay = 0.8
    
    async def run_autoplay():
        try:
            while not session.done:
                import asyncio
                await asyncio.sleep(play_delay)
                ai_action = session.get_ai_action()
                if ai_action is not None:
                    state = session.step(ai_action)
                    await websocket.send_json({"type": "state", "data": state})
                else:
                    break
        except asyncio.CancelledError:
            pass

    try:
        await websocket.send_json({"type": "state", "data": session.get_state_dict()})
        
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)
            
            if data["type"] == "action":
                action_id = data["action_id"]
                state = session.step(action_id)
                await websocket.send_json({"type": "state", "data": state})
                
                # In Human vs AI: if it is now AI's turn (P1), let AI play
                if session.mode == "pve":
                    while not session.done and session.env.current_player == 1 and (autoplay_task is None or autoplay_task.done()):
                        import asyncio
                        await asyncio.sleep(play_delay)
                        ai_action = session.get_ai_action()
                        if ai_action is not None:
                            state = session.step(ai_action)
                            await websocket.send_json({"type": "state", "data": state})
                        else:
                            break
                        
            elif data["type"] == "ai_step":
                if not session.done:
                    ai_action = session.get_ai_action()
                    if ai_action is not None:
                        state = session.step(ai_action)
                        await websocket.send_json({"type": "state", "data": state})

            elif data["type"] == "toggle_autoplay":
                enabled = data.get("enabled", False)
                if autoplay_task and not autoplay_task.done():
                    autoplay_task.cancel()
                    autoplay_task = None
                
                if enabled and not session.done:
                    import asyncio
                    autoplay_task = asyncio.create_task(run_autoplay())

            elif data["type"] == "set_speed":
                speed = float(data.get("speed", 1.0))
                play_delay = max(0.15, 0.8 / speed)

            elif data["type"] == "reset":
                if autoplay_task and not autoplay_task.done():
                    autoplay_task.cancel()
                    autoplay_task = None
                
                p0_f = data.get("p0_faction", "Red")
                p1_f = data.get("p1_faction", "Blue")
                p0_deck = data.get("p0_decklist")
                p1_deck = data.get("p1_decklist")
                mode = data.get("mode", "pve")
                auto_start = data.get("auto_start", False)

                state = session.reset(p0_faction=p0_f, p1_faction=p1_f, p0_decklist=p0_deck, p1_decklist=p1_deck, mode=mode)
                await websocket.send_json({"type": "state", "data": state})

                if mode == "aivai" and auto_start and not session.done:
                    import asyncio
                    autoplay_task = asyncio.create_task(run_autoplay())
                
    except WebSocketDisconnect:
        if autoplay_task and not autoplay_task.done():
            autoplay_task.cancel()
        print("Client disconnected from Arena.")

# ----------------- Pipeline Orchestrator API & WebSocket -----------------
class PipelineStartRequest(BaseModel):
    mode: str = "tune_only"
    episodes: int = 60
    target_balance: float = 5.0
    target_pairwise_balance: float = 5.0

@app.post("/api/pipeline/start")
def start_pipeline(req: PipelineStartRequest):
    return pipeline_runner.start(
        mode=req.mode,
        episodes=req.episodes,
        target_balance=req.target_balance,
        target_pairwise_balance=req.target_pairwise_balance
    )

@app.post("/api/pipeline/stop")
def stop_pipeline():
    return pipeline_runner.stop()

@app.get("/api/pipeline/status")
def get_pipeline_status():
    return pipeline_runner.get_status()

@app.websocket("/ws/pipeline")
async def websocket_pipeline(websocket: WebSocket):
    await websocket.accept()
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def listener(line):
        asyncio.run_coroutine_threadsafe(queue.put(line), loop)

    pipeline_runner.subscribe(listener)
    
    # Send recent logs upon connection
    for line in pipeline_runner.logs[-80:]:
        await websocket.send_text(line)

    try:
        while True:
            # Wait for either incoming message from client or queued log line
            msg_task = asyncio.create_task(websocket.receive_text())
            queue_task = asyncio.create_task(queue.get())
            done, pending = await asyncio.wait([msg_task, queue_task], return_when=asyncio.FIRST_COMPLETED)
            for p in pending:
                p.cancel()
            if msg_task in done:
                # Retrieve result to propagate WebSocketDisconnect when client disconnects
                _ = msg_task.result()
            if queue_task in done:
                line = queue_task.result()
                await websocket.send_text(line)
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        pipeline_runner.unsubscribe(listener)

