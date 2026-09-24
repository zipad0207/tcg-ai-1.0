import os
import sys
import json
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

root_dir = os.path.dirname(__file__)
sys.path.append(root_dir)

from web_app.services.image_gen import ZImageTurboGenerator

def main():
    config_path = os.path.join(root_dir, "cards_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cards_to_gen = []
    for f, l in data.items():
        if isinstance(l, list):
            for c in l:
                cards_to_gen.append({
                    "id": c["id"],
                    "name": c["name"],
                    "faction": f,
                    "tags": c.get("tags", []),
                    "dp": c.get("base_dp", 0)
                })

    cards_to_gen.sort(key=lambda x: x["id"])
    total = len(cards_to_gen)
    print(f"==================================================")
    print(f"🚀 开始全量生成/重画全部 {total} 张卡牌原画")
    print(f"🎯 使用模型: Tongyi-MAI/Z-Image-Turbo (急速超清)")
    print(f"==================================================")

    gen = ZImageTurboGenerator()
    progress_file = os.path.join(root_dir, "web_app", "static", "batch_progress.json")

    success_count = 0
    fail_count = 0

    for idx, card in enumerate(cards_to_gen):
        c_id = card["id"]
        c_name = card["name"]
        c_faction = card["faction"]
        c_tags = card["tags"]
        c_dp = card["dp"]

        # Write progress
        pct = round(((idx + 1) / total) * 100, 1)
        progress_info = {
            "current": idx + 1,
            "total": total,
            "card_id": c_id,
            "card_name": c_name,
            "status": "generating",
            "done": False,
            "success": success_count,
            "failed": fail_count,
            "percentage": pct
        }
        try:
            with open(progress_file, "w", encoding="utf-8") as pf:
                json.dump(progress_info, pf, ensure_ascii=False)
        except Exception:
            pass

        cur_time = time.strftime("%H:%M:%S")
        print(f"[{cur_time}] [{idx+1}/{total} - {pct}%] 正在生成 #{c_id} 【{c_name}】 ({c_faction})...")
        try:
            res = gen.generate_image(
                card_id=c_id,
                card_name=c_name,
                faction=c_faction,
                tags=c_tags,
                dp=c_dp,
                model="Tongyi-MAI/Z-Image-Turbo"
            )
            success_count += 1
            print(f"   -> ✅ 成功: {res['url']}")
        except Exception as e:
            fail_count += 1
            print(f"   -> ❌ 失败: {e}")

        time.sleep(0.3)

    # Final progress write
    final_info = {
        "current": total,
        "total": total,
        "status": "finished",
        "done": True,
        "success": success_count,
        "failed": fail_count,
        "percentage": 100.0
    }
    try:
        with open(progress_file, "w", encoding="utf-8") as pf:
            json.dump(final_info, pf, ensure_ascii=False)
    except Exception:
        pass

    # Synchronize UI export data
    export_script = os.path.join(root_dir, "export_ui_data.py")
    if os.path.exists(export_script):
        import subprocess
        subprocess.run([sys.executable, export_script], check=False)

    print(f"==================================================")
    print(f"🎉 全部卡图重画完成！成功: {success_count}, 失败: {fail_count}")
    print(f"==================================================")

if __name__ == "__main__":
    main()
