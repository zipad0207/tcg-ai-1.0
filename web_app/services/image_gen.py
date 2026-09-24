import os
import sys
import json
import requests
import uuid
from typing import List, Optional, Dict, Set
import threading
import queue
import time
import dashscope
from dotenv import load_dotenv

load_dotenv()

class ZImageTurboGenerator:
    """
    Interface for calling text-to-image models (e.g. Kwai-Kolors/Kolors, Tongyi-MAI/Z-Image) via SiliconFlow or DashScope.
    """
    def __init__(self):
        self.reload_config()

    def reload_config(self):
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        cfg_path = os.path.join(root_dir, "llm_config.json")
        saved_cfg = {}
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    saved_cfg = json.load(f)
            except Exception:
                pass
        self.siliconflow_key = saved_cfg.get("image_api_key") or os.getenv("SILICONFLOW_API_KEY", "")
        self.dashscope_key = os.getenv("DASH_SCOPE_API_KEY", "")
        self.api_key = self.siliconflow_key or self.dashscope_key or saved_cfg.get("api_key", "")
        self.model_name = saved_cfg.get("image_model") or os.getenv("IMAGE_MODEL", "Tongyi-MAI/Z-Image-Turbo")

    def translate_tag_cn(self, tag: str) -> str:
        t = str(tag).upper().strip()
        if t == "RUSH": return "正手持兵刃向前奔袭突击的动态姿态"
        if t == "ATTACK_ONLY": return "双手握持武器处于强攻迎战姿态"
        if t == "SACRIFICE_1_KILL_1": return "正引导释放暗影毁灭魔力的姿态"
        if t.startswith("FORTIFY_"): return "手持坚固重盾处于稳健防御姿态"
        if t.startswith("DEGRADE_"): return "兵刃泛着微弱的符文破甲光晕"
        if t.startswith("SUPPORT_ATK_"): return "周身环绕着战阵激励的微弱光辉"
        if t.startswith("SPAWN_"): return "身侧伴随着小型卫从同伴"
        if t.startswith("DEATH_DRAW_") or t.startswith("DEATH_MANA_"): return "散发着神秘深邃的微弱幽光"
        if t.startswith("DRAW_") or t.startswith("RAMP_") or t.startswith("TEMP_MANA_"): return "掌心凝聚着微弱的元素法力光芒"
        return "身形沉稳利落的战斗姿态"

    def build_prompt_template(self, card_name: str, faction: str, tags: list) -> str:
        faction_details = {
            "Red": "身着红铜轻铠或战袍，背景为火光映照的焦土石路，地面有微弱火星飘散",
            "Blue": "身着银蓝轻甲或法袍，背景为幽暗海渊与冰霜结晶，泛着微弱的幽蓝光晕",
            "Green": "身覆原木皮革与青苔毛皮，背景为晨曦穿透的原始密林，古树与藤蔓环绕",
            "Neutral": "精工打造的发条与合金质感，背景为齿轮工坊与石砖建筑",
            "Dual": "兼具双系元素特质，背景为古老神秘的元素交汇之地"
        }
        f_detail = faction_details.get(faction, faction_details["Neutral"])
        action_desc = "身形沉稳的正面战斗姿态"
        for t in tags:
            translated = self.translate_tag_cn(t)
            if translated != "身形沉稳利落的战斗姿态":
                action_desc = translated
                break
        return f"奇幻写实厚涂风格，{card_name}，{f_detail}，{action_desc}。色彩沉稳自然，构图居中，质感写实，画面纯净，无文字无水印无边框。"

    def build_prompt(self, card_name: str, faction: str, tags: list, dp: int) -> str:
        # Use LLM as AI Art Director to conceive cinematic scene prompts
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        cfg_path = os.path.join(root_dir, "llm_config.json")
        saved_cfg = {}
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    saved_cfg = json.load(f)
            except Exception:
                pass

        ds_key = saved_cfg.get("deepseek_api_key") or os.getenv("DEEPSEEK_API_KEY", "")
        if ds_key:
            llm_api_key = ds_key
            llm_base_url = saved_cfg.get("base_url") or "https://api.deepseek.com"
            llm_model = "deepseek-flash"
        elif self.siliconflow_key:
            llm_api_key = self.siliconflow_key
            llm_base_url = "https://api.siliconflow.cn/v1"
            llm_model = "deepseek-ai/DeepSeek-V3"
        else:
            llm_api_key = None

        if llm_api_key:
            try:
                from openai import OpenAI
                client = OpenAI(
                    api_key=llm_api_key,
                    base_url=llm_base_url
                )
                tags_cn = [self.translate_tag_cn(t) for t in tags]
                tags_str = "、".join(tags_cn) if tags_cn else "常规战力兵种"
                system_prompt = (
                    "你是一位集换式卡牌的美术设定师。"
                    "请为卡牌撰写一段画面描述（供绘图AI生成原画）。\n"
                    "要求：\n"
                    "1. 聚焦主体人物/生物的形体外貌、装备材质与动态姿态。\n"
                    "2. 搭配符合阵营属性的克制环境光影（Red:火光焦土; Blue:海渊冰晶; Green:原始古木; Neutral:发条机械）。\n"
                    "3. 严禁出现游戏规则与数值（不要出现点数、费用、抽牌等词）。\n"
                    "4. 严禁出现卡牌名称、书名号【】等可能诱导AI生成文字的字符。\n"
                    "5. 严禁堆砌‘大师级、杰作、超高清’等空泛形容词。\n"
                    "6. 必须是纯净中文，字数在60-100字，只返回提示词正文。"
                )
                user_prompt = f"卡牌角色概念：{card_name}，阵营：{faction}，战斗特性：{tags_str}"
                res = client.chat.completions.create(
                    model=llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=200,
                    temperature=0.7,
                    timeout=15
                )
                ds_prompt = res.choices[0].message.content.strip()
                if ds_prompt:
                    print(f"[DeepSeek Art Director] Generated prompt: {ds_prompt}")
                    return f"{ds_prompt}。构图居中，奇幻厚涂画风，自然光影，画面纯净，无文字无水印无边框。"
            except Exception as e:
                print(f"[DeepSeek Art Director] Fallback to template due to: {e}")

        return self.build_prompt_template(card_name, faction, tags)

    def save_image_from_url(self, url: str, card_id: int) -> str:
        """Downloads the image from URL and saves it locally as both WebP (high performance) and PNG."""
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        images_dir = os.path.join(root_dir, "web_app", "static", "images", "cards")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir, exist_ok=True)
            
        png_filename = f"{card_id}.png"
        png_filepath = os.path.join(images_dir, png_filename)
        webp_filename = f"{card_id}.webp"
        webp_filepath = os.path.join(images_dir, webp_filename)
        
        rsp = requests.get(url, timeout=30)
        if rsp.status_code == 200:
            with open(png_filepath, 'wb') as f:
                f.write(rsp.content)
            try:
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(rsp.content))
                img.save(webp_filepath, "WEBP", quality=82)
                serve_filename = webp_filename
            except Exception as e:
                print(f"[Warning] Failed to convert image to WebP: {e}")
                serve_filename = png_filename
            import time
            timestamp = int(time.time())
            return f"/static/images/cards/{serve_filename}?v={timestamp}"
        else:
            raise Exception(f"Failed to download image from {url}, status: {rsp.status_code}")

    def update_cards_config(self, card_id: int, image_url: str):
        """Updates cards_config.json to permanently store the image_url."""
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        config_path = os.path.join(root_dir, "cards_config.json")
        
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cards_config = json.load(f)
                
            updated = False
            for faction, cards in cards_config.items():
                if isinstance(cards, list):
                    for card in cards:
                        if card.get("id") == card_id:
                            card["image_url"] = image_url
                            updated = True
                            break
                if updated:
                    break
                    
            if updated:
                tmp_path = config_path + ".tmp"
                try:
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        json.dump(cards_config, f, indent=4, ensure_ascii=False)
                    os.replace(tmp_path, config_path)
                except Exception:
                    with open(config_path, "w", encoding="utf-8") as f:
                        json.dump(cards_config, f, indent=4, ensure_ascii=False)
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
                    
                # Also refresh ui_export_data.json
                try:
                    from export_ui_data import export_ui_data
                    export_ui_data(root_dir)
                except Exception:
                    pass

    def generate_image(self, card_id: int, card_name: str, faction: str, tags: list, dp: int, model: str = None) -> dict:
        self.reload_config()
        prompt = self.build_prompt(card_name, faction, tags, dp)
        chosen_model = model or self.model_name
        print(f"[{chosen_model}] Generating image for '{card_name}' (ID: {card_id})...")
        
        if not self.api_key:
            raise ValueError("未检测到生图 API Key，请在右上角【AI 配置】中填入您的 SiliconFlow 密钥！")

        # Prioritize SiliconFlow API
        if self.siliconflow_key:
            url = "https://api.siliconflow.cn/v1/images/generations"
            headers = {
                "Authorization": f"Bearer {self.siliconflow_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": chosen_model,
                "prompt": prompt,
                "negative_prompt": "文字, 汉字, 字母, 边框, 水印, 模糊, 扭曲, 丑陋, 畸形, 杂乱, 多余的手指, 多余的肢体, text, watermark, frame, blurry, deformed",
                "image_size": "1024x1024"
            }
            
            # Retry loop for 429 rate limiting
            import time
            max_retries = 5
            last_err = ""
            for attempt in range(max_retries + 1):
                res = requests.post(url, json=payload, headers=headers, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    result_url = data["images"][0]["url"]
                    print(f"[SiliconFlow] Success! Downloading from {result_url}")
                    local_url = self.save_image_from_url(result_url, card_id)
                    self.update_cards_config(card_id, local_url)
                    return {"url": local_url, "prompt": prompt, "status": "success"}
                elif res.status_code == 429:
                    last_err = "触发了硅基流动每分钟出图限频（IPM Limit，每分钟约1~2张）。请等待 30 秒后再试。"
                    if attempt < max_retries:
                        print(f"[SiliconFlow] 429 IPM reached, waiting 30s before retry {attempt+1}/{max_retries}...")
                        time.sleep(30)
                        continue
                else:
                    last_err = f"SiliconFlow API Error ({res.status_code}): {res.text}"
                    break
            
            raise Exception(last_err)

        # Fallback to DashScope if official key provided
        import dashscope
        dashscope.api_key = self.dashscope_key
        rsp = dashscope.ImageSynthesis.call(
            model="tongyi-mai-z-image-turbo",
            prompt=prompt,
            n=1,
            size='1024*1024'
        )
        if rsp.status_code == 200:
            result_url = rsp.output.results[0].url
            local_url = self.save_image_from_url(result_url, card_id)
            self.update_cards_config(card_id, local_url)
            return {"url": local_url, "prompt": prompt, "status": "success"}
        else:
            raise Exception(f"DashScope Error ({rsp.code}): {rsp.message}")

    def get_missing_cards(self) -> List[dict]:
        """Finds all cards that lack an image or whose image file is missing locally."""
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        config_path = os.path.join(root_dir, "cards_config.json")
        missing = []
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            cards_dir = os.path.join(root_dir, "web_app", "static", "images", "cards")
            updated = False
            for faction, cards in data.items():
                if isinstance(cards, list):
                    for c in cards:
                        cid = c.get("id")
                        img_url = c.get("image_url", "")
                        has_file = False
                        
                        # 1. Check if img_url points to a real file on disk (strip version query param ?v=)
                        if img_url:
                            clean_rel = img_url.split("?")[0].lstrip("/")
                            local_path = os.path.join(root_dir, "web_app", clean_rel)
                            if os.path.exists(local_path):
                                has_file = True

                        # 2. Check if webp or png already exists on disk by card ID
                        if not has_file and cid is not None:
                            webp_path = os.path.join(cards_dir, f"{cid}.webp")
                            png_path = os.path.join(cards_dir, f"{cid}.png")
                            if os.path.exists(webp_path):
                                c["image_url"] = f"/static/images/cards/{cid}.webp"
                                has_file = True
                                updated = True
                            elif os.path.exists(png_path):
                                c["image_url"] = f"/static/images/cards/{cid}.png"
                                has_file = True
                                updated = True

                        if not has_file:
                            c["faction"] = faction
                            missing.append(c)

            if updated:
                try:
                    with open(config_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass
        return missing

    def batch_generate_missing(self, progress_callback=None) -> dict:
        """Sequentially generates images for all cards missing art."""
        missing = self.get_missing_cards()
        total = len(missing)
        results = []
        errors = []

        for idx, card in enumerate(missing):
            c_id = card.get("id")
            c_name = card.get("name", f"卡牌_{c_id}")
            c_faction = card.get("faction") or (card.get("factions", ["Neutral"])[0] if card.get("factions") else "Neutral")
            c_tags = card.get("tags", [])
            c_dp = card.get("base_dp", 0)

            if progress_callback:
                progress_callback(idx + 1, total, c_name, "generating")

            try:
                res = self.generate_image(c_id, c_name, c_faction, c_tags, c_dp)
                results.append({"card_id": c_id, "name": c_name, "url": res["url"]})
            except Exception as e:
                print(f"[BatchGen] Error on card {c_name}: {e}")
                errors.append({"card_id": c_id, "name": c_name, "error": str(e)})

        return {
            "total": total,
            "generated": len(results),
            "errors": errors,
            "results": results
        }


class BackgroundArtQueue:
    """后台卡图排队生成队列，使用独立线程异步执行。"""
    def __init__(self, generator: Optional['ZImageTurboGenerator'] = None):
        self.generator = generator or ZImageTurboGenerator()
        self.queue = queue.Queue()
        self.queued_ids = set()
        self.lock = threading.Lock()
        self.worker_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.current_card: Optional[dict] = None
        self.current_index = 0
        self.total_count = 0
        self.success_count = 0
        self.failed_count = 0
        self.last_error = ""
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.progress_file = os.path.join(self.root_dir, "web_app", "static", "batch_progress.json")

    def _write_progress(self, status: str = "generating", done: bool = False, card_name: str = "", card_id: Optional[int] = None):
        pct = round((self.current_index / max(1, self.total_count)) * 100, 1) if self.total_count > 0 else 0.0
        data = {
            "current": self.current_index,
            "total": self.total_count,
            "card_id": card_id,
            "card_name": card_name,
            "status": status,
            "done": done,
            "success": self.success_count,
            "failed": self.failed_count,
            "percentage": pct if not done else 100.0,
            "last_error": self.last_error
        }
        try:
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception:
            pass

    def enqueue(self, cards: Optional[List[dict]] = None) -> dict:
        """Enqueues cards for background art generation and starts the worker thread if not running."""
        with self.lock:
            # If no cards specified, scan cards_config.json for truly missing cards
            if cards is None:
                cards = self.generator.get_missing_cards()

            cards_dir = os.path.join(self.root_dir, "web_app", "static", "images", "cards")
            added_count = 0
            for c in cards:
                cid = c.get("id")
                if cid is not None and cid not in self.queued_ids:
                    # Skip if already exists on disk
                    webp_path = os.path.join(cards_dir, f"{cid}.webp")
                    png_path = os.path.join(cards_dir, f"{cid}.png")
                    if os.path.exists(webp_path) or os.path.exists(png_path):
                        continue
                    self.queued_ids.add(cid)
                    self.queue.put(c)
                    added_count += 1

            if added_count > 0:
                self.total_count += added_count
                if not self.is_running:
                    self._write_progress(status="queued", done=False)

            if not self.is_running and not self.queue.empty():
                self.is_running = True
                self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
                self.worker_thread.start()

        return {
            "success": True,
            "added": added_count,
            "total_queued": self.total_count,
            "status": "running" if self.is_running else "idle"
        }

    def _worker_loop(self):
        print(f"\n[后台生图] 队列已启动，待处理卡图: {self.queue.qsize()} 张", flush=True)
        while not self.queue.empty():
            try:
                card = self.queue.get_nowait()
            except queue.Empty:
                break

            with self.lock:
                self.current_index += 1
                self.current_card = card

            c_id = card.get("id")
            c_name = card.get("name", f"新卡_{c_id}")
            c_faction = card.get("faction") or (card.get("factions", ["Neutral"])[0] if card.get("factions") else "Neutral")
            c_tags = card.get("tags", [])
            c_dp = card.get("base_dp", 0)

            self._write_progress(status="generating", done=False, card_name=c_name, card_id=c_id)
            print(f"[后台生图] [{self.current_index}/{self.total_count}] 正在生成: #{c_id} {c_name} ({c_faction})", flush=True)

            try:
                res = self.generator.generate_image(c_id, c_name, c_faction, c_tags, c_dp)
                with self.lock:
                    self.success_count += 1
                print(f"[后台生图] #{c_id} {c_name} 生成成功: {res['url']}", flush=True)
            except Exception as e:
                err_msg = str(e)
                with self.lock:
                    self.failed_count += 1
                    self.last_error = err_msg
                print(f"[后台生图] #{c_id} {c_name} 生成失败: {err_msg}", flush=True)

            self.queue.task_done()
            self._write_progress(status="generating", done=False, card_name=c_name, card_id=c_id)
            time.sleep(2.0)

        with self.lock:
            self.is_running = False
            self.current_card = None
            self._write_progress(status="finished", done=True)
            self.queued_ids.clear()

        print(f"\n[后台生图] 全部卡图生成完成 (成功: {self.success_count}, 失败: {self.failed_count})\n", flush=True)

    def get_status(self) -> dict:
        with self.lock:
            pct = round((self.current_index / max(1, self.total_count)) * 100, 1) if self.total_count > 0 else (100.0 if not self.is_running and self.total_count > 0 else 0.0)
            return {
                "is_running": self.is_running,
                "current": self.current_index,
                "total": self.total_count,
                "success": self.success_count,
                "failed": self.failed_count,
                "percentage": pct,
                "current_card": self.current_card,
                "status": "generating" if self.is_running else ("finished" if self.total_count > 0 else "idle"),
                "done": not self.is_running and self.total_count > 0,
                "queue_size": self.queue.qsize(),
                "last_error": self.last_error
            }

    def wait_until_done(self, timeout: Optional[float] = None) -> bool:
        """Blocks until current queue is empty, or timeout reached."""
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=timeout)
            return not self.worker_thread.is_alive()
        return True

# Global background queue singleton
art_queue = BackgroundArtQueue()

