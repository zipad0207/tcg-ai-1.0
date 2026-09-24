import json
import os
import sys

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

                # Compute symmetrical pairwise matchups (combining f1_vs_f2 and f2_vs_f1)
                f_keys = list(export_data["faction_stats"].keys())
                factions = [f for f in ["Red", "Blue", "Green"] if f in f_keys]
                for f in f_keys:
                    if f not in factions:
                        factions.append(f)
                if not factions:
                    factions = ["Red", "Blue", "Green"]
                pairwise = {}
                max_dev = 0.0

                for f1 in factions:
                    for f2 in factions:
                        key = f"{f1}_vs_{f2}"
                        if f1 == f2:
                            pairwise[key] = {
                                "f1": f1,
                                "f2": f2,
                                "total": 0,
                                "wins": 0,
                                "winrate": 50.0,
                                "dev": 0.0,
                                "is_mirror": True
                            }
                        else:
                            k1 = f"{f1}_vs_{f2}"
                            k2 = f"{f2}_vs_{f1}"
                            r1 = export_data["matchups"].get(k1, {})
                            r2 = export_data["matchups"].get(k2, {})

                            total_games = r1.get("total", 0) + r2.get("total", 0)
                            f1_wins = r1.get(f"{f1}_wins", 0) + r2.get(f"{f1}_wins", 0)
                            wr = round((f1_wins / max(1, total_games)) * 100, 2) if total_games > 0 else 50.0
                            dev = round(abs(wr - 50.0), 2)
                            if dev > max_dev:
                                max_dev = dev

                            pairwise[key] = {
                                "f1": f1,
                                "f2": f2,
                                "total": total_games,
                                "wins": f1_wins,
                                "winrate": wr,
                                "dev": dev,
                                "is_mirror": False
                            }

                export_data["pairwise_matchups"] = pairwise
                export_data["max_pairwise_dev"] = max_dev
    except Exception as e:
        print(f"Warning: Failed to load training metrics: {e}")

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=4, ensure_ascii=False)
        print(f"[OK] 已成功导出 UI 客户端统一数据接口文件: {output_path}")
    except Exception as e:
        print(f"Error: Failed to write output json: {e}")

    return export_data

def main():
    export_ui_data()

if __name__ == "__main__":
    main()
