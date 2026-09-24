import os
import sys
import json
import time
import subprocess
import threading
from typing import List, Dict, Optional, Callable

class PipelineRunner:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None
        self.logs: List[str] = []
        self.max_logs = 1500
        self.is_running = False
        self.mode = "tune_only"
        self.start_time: Optional[float] = None
        self.subscribers: List[Callable[[str], None]] = []
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    def subscribe(self, callback: Callable[[str], None]):
        if callback not in self.subscribers:
            self.subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[str], None]):
        if callback in self.subscribers:
            self.subscribers.remove(callback)

    def _broadcast(self, line: str):
        self.logs.append(line)
        if len(self.logs) > self.max_logs:
            self.logs.pop(0)
        for cb in list(self.subscribers):
            try:
                cb(line)
            except Exception:
                pass

    def start(self, mode: str = "tune_only", episodes: int = 3000, target_balance: float = 5.0, target_pairwise_balance: float = 8.0) -> dict:
        if self.is_running and self.process and self.process.poll() is None:
            return {"success": False, "message": "流水线当前正在运行中，请勿重复启动"}

        self.logs.clear()
        self.mode = mode
        self.start_time = time.time()
        self.is_running = True

        script_path = os.path.join(self.root_dir, "pipeline_orchestrator.py")
        cmd = [sys.executable, script_path]

        # Check if DeepSeek API Key is configured before starting
        cfg_path = os.path.join(self.root_dir, "llm_config.json")
        has_ds = False
        env_ds = (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
        if env_ds and not env_ds.startswith("sk-•••") and not env_ds.startswith("••••"):
            has_ds = True
        elif os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    c = json.load(f)
                    k = (c.get("deepseek_api_key") or c.get("api_key") or "").strip()
                    if k and not k.startswith("sk-•••") and not k.startswith("••••"):
                        has_ds = True
            except Exception:
                pass
        if not has_ds:
            self.is_running = False
            return {
                "success": False,
                "message": "【未配置 DeepSeek Key】流水线的核心是调用 DeepSeek 大模型对失衡卡牌进行诊断与参数微调。请先在左侧【AI 服务配置】面板中填入您的 DeepSeek API Key 并点击【保存配置】后再启动！"
            }

        if mode == "tune_only":
            cmd.extend([
                "--skip-print",
                "--episodes", str(episodes),
                "--target-balance", str(target_balance),
                "--target-pairwise-balance", str(target_pairwise_balance)
            ])
        elif mode == "fast_demo":
            cmd.extend([
                "--skip-print",
                "--dry-run",
                "--target-balance", str(target_balance),
                "--target-pairwise-balance", str(target_pairwise_balance)
            ])
        elif mode == "full_pack":
            cmd.extend([
                "--episodes", str(episodes),
                "--target-balance", str(target_balance),
                "--target-pairwise-balance", str(target_pairwise_balance)
            ])
        else:
            cmd.extend([
                "--skip-print",
                "--dry-run",
                "--target-pairwise-balance", str(target_pairwise_balance)
            ])

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["OMP_NUM_THREADS"] = "2"
        env["MKL_NUM_THREADS"] = "2"
        env["OPENBLAS_NUM_THREADS"] = "2"
        env["VECLIB_MAXIMUM_THREADS"] = "2"
        env["NUMEXPR_NUM_THREADS"] = "2"
        env["TORCH_NUM_THREADS"] = "2"
        env["OMP_WAIT_POLICY"] = "PASSIVE"
        env["KMP_BLOCKTIME"] = "0"
        cur_pypath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = self.root_dir if not cur_pypath else f"{self.root_dir}{os.pathsep}{cur_pypath}"

        cflags = 0
        if sys.platform == "win32":
            cflags = subprocess.BELOW_NORMAL_PRIORITY_CLASS

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=self.root_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                bufsize=1,
                creationflags=cflags
            )
        except Exception as e:
            self.is_running = False
            return {"success": False, "message": f"启动失败: {str(e)}"}

        def _decode_bytes(b: bytes) -> str:
            if not b:
                return ""
            for enc in ("utf-8", "gbk", "cp936"):
                try:
                    return b.decode(enc)
                except (UnicodeDecodeError, LookupError):
                    pass
            return b.decode("utf-8", errors="replace")

        def _sync_ui():
            try:
                if self.root_dir not in sys.path:
                    sys.path.insert(0, self.root_dir)
                from export_ui_data import export_ui_data
                export_ui_data(self.root_dir)
            except Exception as ex:
                print(f"[PipelineRunner] UI data sync error: {ex}")

        def _reader():
            self._broadcast(f"[Console] 平衡调度流水线已启动 (模式: {mode}, 参数: {' '.join(cmd[2:])})")
            try:
                for raw_line in iter(self.process.stdout.readline, b''):
                    if not raw_line or not self.is_running:
                        break
                    clean_line = _decode_bytes(raw_line).rstrip("\r\n")
                    if clean_line:
                        self._broadcast(clean_line)
            except Exception as read_err:
                if self.is_running:
                    self._broadcast(f"[Console] 管道读取异常: {read_err}")
            finally:
                if self.process:
                    try:
                        self.process.stdout.close()
                    except Exception:
                        pass
                    try:
                        rc = self.process.wait(timeout=1.5)
                    except Exception:
                        rc = self.process.poll()
                    was_running = self.is_running
                    self.is_running = False
                    _sync_ui()

                    # 确保后台生图无缝接力：流水线若提前达成平衡退出，Web 常驻服务会接管并继续在后台线程排队绘制剩余新卡
                    if was_running and mode == "full_pack":
                        try:
                            from web_app.services.image_gen import art_queue
                            art_queue.enqueue()
                            q_st = art_queue.get_status()
                            if q_st.get("is_running"):
                                rem = q_st["total"] - q_st["current"]
                                self._broadcast(f"[Console] [后台出图托管] 调优已完成，Web 常驻后台正继续自动绘制剩余 {rem} 张新卡插图...")
                        except Exception:
                            pass

                    if was_running:
                        status_msg = "已达成平衡收敛或执行完毕" if rc == 0 else f"已停止 (code: {rc})"
                        self._broadcast(f"[Console] 任务结束: {status_msg}")

        self.thread = threading.Thread(target=_reader, daemon=True)
        self.thread.start()

        return {"success": True, "message": "流水线已成功启动", "mode": mode, "cmd": cmd}

    def stop(self) -> dict:
        if not self.is_running or not self.process:
            return {"success": False, "message": "当前没有正在运行的流水线"}

        pid = self.process.pid
        self.is_running = False
        try:
            if sys.platform == "win32":
                # Forcefully kill the entire process tree (including deck_builder_ppo and train_brawl child processes)
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, text=True)
            else:
                self.process.terminate()
                time.sleep(0.2)
                if self.process.poll() is None:
                    self.process.kill()
            
            # Immediately sync latest data from disk after kill
            try:
                if self.root_dir not in sys.path:
                    sys.path.insert(0, self.root_dir)
                from export_ui_data import export_ui_data
                export_ui_data(self.root_dir)
            except Exception:
                pass

            self._broadcast("[Console] 已手动终止当前流水线及子进程。")
            return {"success": True, "message": "任务已停止"}
        except Exception as e:
            return {"success": False, "message": f"终止任务失败: {str(e)}"}

    def get_status(self) -> dict:
        running = self.is_running and (self.process is not None and self.process.poll() is None)
        return {
            "running": running,
            "mode": self.mode,
            "start_time": self.start_time,
            "elapsed_seconds": int(time.time() - self.start_time) if (running and self.start_time) else 0,
            "log_count": len(self.logs),
            "recent_logs": self.logs[-100:]
        }

runner = PipelineRunner()
