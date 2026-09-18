import json
import os
import sys

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    cards_path = os.path.join(root_dir, "cards_config.json")
    decks_path = os.path.join(root_dir, "decks_config.json")
    metrics_path = os.path.join(root_dir, "training_metrics_brawl.json")
    output_path = os.path.join(root_dir, "ui_export_data.json")

    export_data = {
        "cards": [],
        "decks": {},
        "faction_stats": {},
        "matchups": {}
    }

    try:
        if os.path.exists(cards_path):
            with open(cards_path, "r", encoding="utf-8") as f:
                export_data["cards"] = json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load cards config: {e}")

    try:
        if os.path.exists(decks_path):
            with open(decks_path, "r", encoding="utf-8") as f:
                export_data["decks"] = json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load decks config: {e}")

    try:
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
                export_data["faction_stats"] = metrics.get("faction_stats", {})
                export_data["matchups"] = metrics.get("matchups", {})
                export_data["total_episodes"] = metrics.get("total_episodes", 0)
    except Exception as e:
        print(f"Warning: Failed to load training metrics: {e}")

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=4, ensure_ascii=False)
        print(f"[OK] 已成功导出 UI 客户端统一数据接口文件: {output_path}")
    except Exception as e:
        print(f"Error: Failed to write output json: {e}")

if __name__ == "__main__":
    main()
