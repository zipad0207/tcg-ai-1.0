import json
import os
from tcg_common import atomic_write_json, compute_pairwise_matchups


def export_ui_data(root_dir: str = None) -> dict:
    if root_dir is None:
        root_dir = os.path.dirname(os.path.abspath(__file__))
    cards_path = os.path.join(root_dir, "cards_config.json")
    decks_path = os.path.join(root_dir, "decks_config.json")
    metrics_path = os.path.join(root_dir, "training_metrics_brawl.json")
    output_path = os.path.join(root_dir, "ui_export_data.json")

    export_data = {
        "cards": [],
        "decks": {},
        "faction_stats": {},
        "matchups": {},
        "pairwise_matchups": {},
        "max_pairwise_dev": 0.0,
        "total_episodes": 0
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

                summary = compute_pairwise_matchups(metrics)
                factions = summary["factions"] or ["Red", "Blue", "Green"]
                pairwise = {}

                for f in factions:
                    pairwise[f"{f}_vs_{f}"] = {
                        "f1": f,
                        "f2": f,
                        "total": 0,
                        "wins": 0,
                        "winrate": 50.0,
                        "dev": 0.0,
                        "is_mirror": True,
                    }

                for (f1, f2), ps in summary["pairwise_stats"].items():
                    dev = round(ps["dev"], 2)
                    pairwise[f"{f1}_vs_{f2}"] = {
                        "f1": f1,
                        "f2": f2,
                        "total": ps["total"],
                        "wins": ps["f1_wins"],
                        "winrate": round(ps["wr1"], 2),
                        "dev": dev,
                        "is_mirror": False,
                    }
                    pairwise[f"{f2}_vs_{f1}"] = {
                        "f1": f2,
                        "f2": f1,
                        "total": ps["total"],
                        "wins": ps["f2_wins"],
                        "winrate": round(ps["wr2"], 2),
                        "dev": dev,
                        "is_mirror": False,
                    }

                export_data["pairwise_matchups"] = pairwise
                export_data["max_pairwise_dev"] = round(summary["max_pairwise_dev"], 2)
    except Exception as e:
        print(f"Warning: Failed to load training metrics: {e}")

    try:
        atomic_write_json(output_path, export_data, indent=4, ensure_ascii=False)
        print(f"[OK] 已成功导出 UI 客户端统一数据接口文件: {output_path}")
    except Exception as e:
        print(f"Error: Failed to write output json: {e}")

    return export_data

def main():
    export_ui_data()

if __name__ == "__main__":
    main()
