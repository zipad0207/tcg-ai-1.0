import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import auto_balancer_deepseek as ab
import tcg_common as tc


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        content = self._responses.pop(0) if self._responses else '{"modifications": []}'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _make_sample_cards():
    return {
        "Red": [
            {
                "id": 100,
                "name": "赤红突击手",
                "card_type": "MINION",
                "cost": 3,
                "base_dp": 3,
                "atk_spell_val": 0,
                "def_spell_val": 0,
                "tags": ["RUSH", "FORTIFY_2"],
                "factions": ["Red"],
                "image_url": "/static/100.webp",
            },
            {
                "id": 103,
                "name": "射线",
                "card_type": "SPELL",
                "cost": 2,
                "base_dp": 0,
                "atk_spell_val": 3,
                "def_spell_val": 0,
                "tags": [],
                "factions": ["Red"],
            },
        ],
        "Blue": [
            {
                "id": 200,
                "name": "蔚蓝守卫",
                "card_type": "MINION",
                "cost": 7,
                "base_dp": 3,
                "atk_spell_val": 0,
                "def_spell_val": 0,
                "tags": ["FORTIFY_2"],
                "factions": ["Blue"],
            },
            {
                "id": 201,
                "name": "蔚蓝巨兽",
                "card_type": "MINION",
                "cost": 8,
                "base_dp": 4,
                "atk_spell_val": 0,
                "def_spell_val": 0,
                "tags": ["FORTIFY_3"],
                "factions": ["Blue"],
            },
        ],
        "Green": [
            {
                "id": 300,
                "name": "翠绿幼苗",
                "card_type": "MINION",
                "cost": 2,
                "base_dp": 4,
                "atk_spell_val": 0,
                "def_spell_val": 0,
                "tags": ["RAMP_1"],
                "factions": ["Green"],
            },
            {
                "id": 301,
                "name": "翠绿抽牌术",
                "card_type": "SPELL",
                "cost": 1,
                "base_dp": 0,
                "atk_spell_val": 0,
                "def_spell_val": 0,
                "tags": ["DRAW_1"],
                "factions": ["Green"],
            },
        ],
        "Neutral": [
            {
                "id": 917,
                "name": "中立能量涌动",
                "card_type": "SPELL",
                "cost": 0,
                "base_dp": 0,
                "atk_spell_val": 0,
                "def_spell_val": 0,
                "tags": ["TEMP_MANA_2"],
                "factions": ["Neutral"],
            },
            {
                "id": 920,
                "name": "通用临时润滑",
                "card_type": "SPELL",
                "cost": 0,
                "base_dp": 0,
                "atk_spell_val": 0,
                "def_spell_val": 0,
                "tags": ["TEMP_MANA_1"],
                "factions": ["Neutral"],
            },
            {
                "id": 905,
                "name": "佣兵斥候",
                "card_type": "MINION",
                "cost": 2,
                "base_dp": 3,
                "atk_spell_val": 0,
                "def_spell_val": 0,
                "tags": [],
                "factions": ["Neutral"],
            },
        ],
        "Dual": [
            {
                "id": 400,
                "name": "红蓝协作者",
                "card_type": "MINION",
                "cost": 3,
                "base_dp": 3,
                "atk_spell_val": 0,
                "def_spell_val": 0,
                "tags": ["RUSH"],
                "factions": ["Red", "Blue"],
            },
            {
                "id": 600,
                "name": "红绿共生",
                "card_type": "MINION",
                "cost": 4,
                "base_dp": 5,
                "atk_spell_val": 0,
                "def_spell_val": 0,
                "tags": [],
                "factions": ["Red", "Green"],
            },
        ],
    }


def _make_sample_metrics():
    return {
        "total_episodes": 60,
        "faction_stats": {
            "Red": {"matches": 39, "wins": 17, "winrate": 43.59},
            "Blue": {"matches": 45, "wins": 21, "winrate": 46.67},
            "Green": {"matches": 36, "wins": 22, "winrate": 61.11},
        },
        "matchups": {
            "Blue_vs_Red": {"total": 7, "Blue_wins": 5, "Red_wins": 2},
            "Red_vs_Blue": {"total": 5, "Red_wins": 3, "Blue_wins": 2},
            "Blue_vs_Green": {"total": 5, "Blue_wins": 1, "Green_wins": 4},
            "Green_vs_Blue": {"total": 6, "Green_wins": 4, "Blue_wins": 2},
            "Red_vs_Green": {"total": 7, "Red_wins": 2, "Green_wins": 5},
            "Green_vs_Red": {"total": 2, "Green_wins": 1, "Red_wins": 1},
        },
        "card_play_count": {"佣兵斥候": 190, "翠绿幼苗": 73, "幸运币": 60},
        "faction_card_plays": {
            "Red": {"佣兵斥候": 71, "赤红突击手": 50, "幸运币": 18, "法力过载": 5},
            "Green": {"佣兵斥候": 69, "翠绿幼苗": 73, "红绿共生": 52, "幸运币": 20},
        },
    }


class AutoBalancerTests(unittest.TestCase):
    def test_api_key_isolation_and_config_nulls(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "", "SILICONFLOW_API_KEY": "sf-img-key", "DEEPSEEK_BASE_URL": "https://api.deepseek.com"}, clear=False):
            with patch.object(tc, "load_llm_config", return_value={"api_key": None, "deepseek_api_key": "sk-•••1234", "base_url": "https://api.deepseek.com"}):
                with patch.object(tc.sys, "platform", "linux"):
                    self.assertEqual(ab.get_api_key(), "")

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "", "SILICONFLOW_API_KEY": "sf-llm-key", "DEEPSEEK_BASE_URL": "https://api.siliconflow.cn/v1"}, clear=False):
            with patch.object(tc, "load_llm_config", return_value={}):
                with patch.object(tc.sys, "platform", "linux"):
                    self.assertEqual(ab.get_api_key(), "sf-llm-key")

    def test_tag_validation_and_json_helpers(self):
        self.assertTrue(ab.is_legal_tag(" rush "))
        self.assertTrue(ab.is_legal_tag("Fortify_2"))
        self.assertTrue(ab.is_legal_tag("SPAWN_1_2"))
        self.assertFalse(ab.is_legal_tag("DRAW_0"))
        self.assertFalse(ab.is_legal_tag("TEMP_MANA_0"))
        self.assertFalse(ab.is_legal_tag("SPAWN_0_1"))
        self.assertFalse(ab.is_legal_tag(None))

        self.assertEqual(ab.clean_json_response(None), "{}")
        fenced = "分析{测试}\n```json\n{\"modifications\": [{\"id\": 103}]}\n```\n尾部{说明}"
        self.assertEqual(json.loads(ab.clean_json_response(fenced)), {"modifications": [{"id": 103}]})

        nested = {"factions": {"Red": [{"id": 100}], "Blue": [{"id": 200}]}}
        self.assertEqual([c["id"] for c in ab.extract_card_list(nested)], [100, 200])
        self.assertEqual(ab.extract_card_list({"id": 103, "cost": 1}), [{"id": 103, "cost": 1}])

    def test_metrics_merging_mode_selection_and_null_zero_safety(self):
        # max_dev = 3.0 <= 5.0 and merged pairwise = 6.0 <= 8.0 -> must select maintenance mode
        balanced_metrics = {
            "total_episodes": 100,
            "faction_stats": {
                "Red": {"winrate": 47.0, "wins": 47, "matches": 100},
                "Blue": {"winrate": 53.0, "wins": 53, "matches": 100},
            },
            "matchups": {
                "Red_vs_Blue": {"total": 50, "Red_wins": 22, "Blue_wins": 28},
                "Blue_vs_Red": {"total": 50, "Blue_wins": 28, "Red_wins": 22},
            },
        }
        client = FakeClient(['{"modifications": []}'])
        ab.run_deepseek_balance_and_expand(
            balanced_metrics, _make_sample_cards(), client,
            target_tolerance=5.0, target_pairwise_tolerance=8.0, severe_threshold=10.0,
        )
        prompt_sent = client.calls[0]["messages"][1]["content"]
        self.assertIn("全面平衡维持模式", prompt_sent)
        self.assertNotIn("轻量两两克制微调模式", prompt_sent)

        # Edge cases: winrate=0.0, winrate=None, total_episodes=0
        weird_metrics = {
            "total_episodes": 0,
            "faction_stats": {
                "Red": {"winrate": 0.0, "wins": 0, "matches": 10},
                "Blue": {"winrate": None, "wins": None, "matches": None},
            },
            "matchups": {"Red_vs_Blue": {"total": None, "Red_wins": None}},
        }
        summary = ab._compute_balance_metrics(weird_metrics, 8.0)
        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["deviations"]["Red"], 50.0)
        self.assertEqual(summary["deviations"]["Blue"], 0.0)

        # Target pairwise balance disabled (=0)
        sample = _make_sample_metrics()
        s_no_pair = ab._compute_balance_metrics(sample, 0.0)
        self.assertAlmostEqual(s_no_pair["effective_max_dev"], 11.11, places=2)

    def test_prompt_excludes_neutral_and_tokens_from_top5_and_strips_metadata(self):
        client = FakeClient(['{"modifications": []}'])
        ab.run_deepseek_balance_and_expand(
            _make_sample_metrics(), _make_sample_cards(), client,
            target_tolerance=5.0, target_pairwise_tolerance=8.0, severe_threshold=10.0,
            weak_factions=["Red"],
        )
        prompt = client.calls[0]["messages"][1]["content"]
        self.assertNotIn("image_url", prompt)
        top5_section = prompt.split("### 各阵营实战高频出牌遥测")[1].split("卡牌全局使用频次统计")[0]
        self.assertNotIn("佣兵斥候", top5_section)
        self.assertNotIn("幸运币", top5_section)
        self.assertNotIn("法力过载", top5_section)
        self.assertIn("红绿共生[双色]", top5_section)

    def test_e2e_main_zero_cost_weak_protection_floor_and_dedup(self):
        with tempfile.TemporaryDirectory() as tmp:
            cards_path = os.path.join(tmp, "cards.json")
            metrics_path = os.path.join(tmp, "metrics.json")
            out_path = os.path.join(tmp, "out.json")
            cards = _make_sample_cards()
            cards["meta_version"] = 1  # non-list entry safety
            with open(cards_path, "w", encoding="utf-8") as f:
                json.dump(cards, f, ensure_ascii=False)
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(_make_sample_metrics(), f, ensure_ascii=False)

            llm_reply = json.dumps({
                "modifications": [
                    # 0-cost TEMP_MANA cards kept at 0 cost -> zero phantom update
                    {"id": 917, "cost": 0, "tags": ["TEMP_MANA_2"]},
                    {"id": 920, "cost": 0, "tags": ["TEMP_MANA_1"]},
                    # Normal 1-cost DRAW_1 spell cannot drop to 0 cost; SPELL base_dp forced to 0
                    {"id": 301, "cost": 0, "base_dp": 4, "def_spell_val": 0},
                    # Weak faction (Red) tag nerf blocked + malformed cost null must not crash
                    {"id": 100, "cost": None, "tags": ["FORTIFY_2"]},
                    {"id": 103, "cost": 1, "atk_spell_val": 2},
                    # String id + duplicate entry should apply only once (4 -> 3, not 4 -> 2)
                    {"id": "300", "base_dp": 1},
                    {"id": 300, "base_dp": 1},
                    # High-cost low-DP minion: cost 7->8 with base_dp=3 unchanged stays 3
                    {"id": 200, "cost": 8, "base_dp": 3},
                    # High-cost low-DP minion: nerfing below floor stays 4 (not reversed to 5)
                    {"id": 201, "base_dp": 2},
                    # Dual card (Red+Green): Red weak, Green OP -> allowed to nerf
                    {"id": 600, "cost": 6},
                ]
            })

            fake_client = FakeClient([llm_reply])
            with patch.object(ab, "DEEPSEEK_API_KEY", "sk-test-valid"), \
                 patch.object(ab, "OpenAI", return_value=fake_client), \
                 patch.object(sys, "argv", ["auto_balancer_deepseek.py", "--cards", cards_path, "--metrics", metrics_path, "--output", out_path]):
                ab.main()

            with open(out_path, "r", encoding="utf-8") as f:
                out = json.load(f)
            by_id = {c["id"]: c for k, v in out.items() if isinstance(v, list) for c in v}

            self.assertEqual(by_id[917]["cost"], 0)
            self.assertEqual(by_id[920]["cost"], 0)
            self.assertEqual(by_id[301]["cost"], 1)
            self.assertEqual(by_id[301]["base_dp"], 0)
            self.assertEqual(by_id[100]["tags"], ["RUSH", "FORTIFY_2"])
            self.assertEqual(by_id[103]["atk_spell_val"], 3)  # weak faction Red nerf blocked
            self.assertEqual(by_id[300]["base_dp"], 3)  # deduplicated + step limited 4->3
            self.assertEqual(by_id[200]["cost"], 8)
            self.assertEqual(by_id[200]["base_dp"], 3)  # no stealth buff
            self.assertEqual(by_id[201]["base_dp"], 4)  # no direction reversal
            self.assertEqual(by_id[600]["cost"], 5)  # step limited 4->5
            self.assertEqual(out["meta_version"], 1)

            tabu_path = os.path.join(tmp, "balancer_tabu_list.json")
            with open(tabu_path, "r", encoding="utf-8") as f:
                tabu = json.load(f)
            self.assertNotIn("917", tabu)
            self.assertNotIn("920", tabu)
            self.assertNotIn("301", tabu)
            self.assertIn("300", tabu)
            self.assertIn("600", tabu)

    def test_tabu_aspiration_after_rejected_non_tabu_and_none_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            cards_path = os.path.join(tmp, "cards.json")
            metrics_path = os.path.join(tmp, "metrics.json")
            out_path = os.path.join(tmp, "out.json")
            tabu_path = os.path.join(tmp, "balancer_tabu_list.json")
            with open(cards_path, "w", encoding="utf-8") as f:
                json.dump(_make_sample_cards(), f, ensure_ascii=False)
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(_make_sample_metrics(), f, ensure_ascii=False)
            # 300 is on cooldown (wait=2 -> decremented to 1)
            with open(tabu_path, "w", encoding="utf-8") as f:
                json.dump({"300": 2}, f)

            # 2 non-tabu candidates that are both rejected/no-op, plus 1 tabu candidate 300
            llm_reply = json.dumps({
                "modifications": [
                    {"id": 100, "cost": 4},       # blocked by Red weak_factions
                    {"id": 103, "tags": ["ATTACK_ONLY"]},  # blocked by Red weak_factions (added drawback)
                    {"id": 300, "cost": 3},       # on tabu -> must be unlocked via aspiration!
                ]
            })
            fake_client = FakeClient([llm_reply])
            with patch.object(ab, "DEEPSEEK_API_KEY", "sk-test-valid"), \
                 patch.object(ab, "OpenAI", return_value=fake_client), \
                 patch.object(sys, "argv", ["auto_balancer_deepseek.py", "--cards", cards_path, "--metrics", metrics_path, "--output", out_path]):
                ab.main()

            with open(out_path, "r", encoding="utf-8") as f:
                out = json.load(f)
            green_300 = [c for c in out["Green"] if c["id"] == 300][0]
            self.assertEqual(green_300["cost"], 3)

            # Test content=None does not crash or overwrite existing tabu
            fake_none = FakeClient([None])
            with patch.object(ab, "DEEPSEEK_API_KEY", "sk-test-valid"), \
                 patch.object(ab, "OpenAI", return_value=fake_none), \
                 patch.object(sys, "argv", ["auto_balancer_deepseek.py", "--cards", cards_path, "--metrics", metrics_path, "--output", out_path]):
                ab.main()

    def test_tag_reorder_truncated_regex_nested_output_and_missing_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            cards_path = os.path.join(tmp, "cards.json")
            metrics_path = os.path.join(tmp, "metrics.json")
            missing_metrics = os.path.join(tmp, "missing_metrics.json")
            nested_out = os.path.join(tmp, "sub", "dir", "out.json")
            with open(cards_path, "w", encoding="utf-8") as f:
                json.dump(_make_sample_cards(), f, ensure_ascii=False)
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(_make_sample_metrics(), f, ensure_ascii=False)

            # Missing metrics must exit before calling OpenAI
            fake_unused = FakeClient([])
            with patch.object(ab, "DEEPSEEK_API_KEY", "sk-test-valid"), \
                 patch.object(ab, "OpenAI", return_value=fake_unused), \
                 patch.object(sys, "argv", ["auto_balancer_deepseek.py", "--cards", cards_path, "--metrics", missing_metrics, "--output", nested_out]):
                ab.main()
            self.assertEqual(len(fake_unused.calls), 0)
            self.assertFalse(os.path.exists(nested_out))

            # Truncated JSON with broken first object + valid second object + tag reorder only on 100
            truncated_raw = '{"modifications": [{"id": 300, "cost": {"broken"}, {"id": 100, "tags": ["FORTIFY_2", "RUSH"]}, {"id": "301", "cost": 2}]}'
            fake_trunc = FakeClient([truncated_raw])
            with patch.object(ab, "DEEPSEEK_API_KEY", "sk-test-valid"), \
                 patch.object(ab, "OpenAI", return_value=fake_trunc), \
                 patch.object(sys, "argv", ["auto_balancer_deepseek.py", "--cards", cards_path, "--metrics", metrics_path, "--output", nested_out]):
                ab.main()

            self.assertTrue(os.path.exists(nested_out))
            with open(nested_out, "r", encoding="utf-8") as f:
                out = json.load(f)
            by_id = {c["id"]: c for k, v in out.items() if isinstance(v, list) for c in v}
            self.assertEqual(by_id[100]["tags"], ["RUSH", "FORTIFY_2"])  # pure reorder ignored
            self.assertEqual(by_id[301]["cost"], 2)  # recovered from truncated response

            # Top-level JSON array response also parses cleanly
            self.assertEqual(json.loads(ab.clean_json_response('[{"id": 300, "cost": 3}]')), [{"id": 300, "cost": 3}])

    def test_real_repo_cards_metrics_and_sandbox_compatibility(self):
        real_cards_path = os.path.join(ROOT_DIR, "cards_config.json")
        real_metrics_path = os.path.join(ROOT_DIR, "training_metrics_brawl.json")
        if not (os.path.exists(real_cards_path) and os.path.exists(real_metrics_path)):
            self.skipTest("Repo data files not present")

        from sandbox import DuelEnv, Faction

        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "tuned_real.json")
            # Pick a real Green card (312 翠绿幼苗) and real 0-cost Neutral (917)
            llm_reply = json.dumps({
                "modifications": [
                    {"id": 917, "cost": 0, "tags": ["TEMP_MANA_2"]},
                    {"id": 312, "cost": 3, "tags": ["RUSH"]},
                ]
            })
            fake_client = FakeClient([llm_reply])
            with patch.object(ab, "DEEPSEEK_API_KEY", "sk-test-valid"), \
                 patch.object(ab, "OpenAI", return_value=fake_client), \
                 patch.object(sys, "argv", ["auto_balancer_deepseek.py", "--cards", real_cards_path, "--metrics", real_metrics_path, "--output", out_path]):
                ab.main()

            prompt = fake_client.calls[0]["messages"][1]["content"]
            self.assertIn("本阵营实战高频出牌 Top 5", prompt)
            self.assertLess(len(prompt), 80000)

            # Verify DuelEnv loads the tuned output without error
            env = DuelEnv(p0_faction=Faction.RED, p1_faction=Faction.GREEN, cards_path=out_path)
            self.assertEqual(env.card_database[917].cost, 0)
            obs = env.reset()
            self.assertEqual(obs.shape, env.obs_shape)

    def test_shared_common_helpers_and_web_services(self):
        # 1. extract_valid_tags & enforce_temp_mana_cost
        self.assertEqual(
            tc.extract_valid_tags({"tags": [" rush ", "FORTIFY_X", "DRAW_0"], "keyword_values": {"FORTIFY": 3}}),
            ["RUSH", "FORTIFY_3"],
        )
        self.assertEqual(tc.enforce_temp_mana_cost("SPELL", 2, 0, 0, ["TEMP_MANA_1"]), (0, True))
        self.assertEqual(tc.enforce_temp_mana_cost("SPELL", 2, 2, 0, ["TEMP_MANA_1"]), (2, False))

        # 2. DeepSeekCardPrinter loads llm_config.json and uses DeepSeek base_url/model
        from web_app.services.card_printer import DeepSeekCardPrinter
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "llm_config.json")
            cards_path = os.path.join(tmp, "cards_config.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump({"deepseek_api_key": "sk-ds-from-cfg", "base_url": "https://api.deepseek.com", "model": "deepseek-flash"}, f)
            with open(cards_path, "w", encoding="utf-8") as f:
                json.dump({"Red": [], "Blue": [], "Green": [], "Neutral": []}, f)

            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "", "SILICONFLOW_API_KEY": "sf-img-only", "DEEPSEEK_BASE_URL": "", "DEEPSEEK_MODEL": ""}, clear=False):
                printer = DeepSeekCardPrinter()
                printer.root_dir = tmp
                printer.cards_path = cards_path
                printer.reload_config()
                self.assertEqual(printer.api_key, "sk-ds-from-cfg")
                self.assertEqual(printer.base_url, "https://api.deepseek.com")
                self.assertEqual(printer.model, "deepseek-flash")

                # Verify generate_new_cards reloads config and enforces TEMP_MANA cost
                fake_reply = json.dumps({"cards": [{"name": "测试涌动", "card_type": "SPELL", "cost": 3, "base_dp": 0, "atk_spell_val": 0, "def_spell_val": 0, "tags": ["temp_mana_1", "draw_0"]}]})
                fake_client = FakeClient([fake_reply])
                with patch("web_app.services.card_printer.OpenAI", return_value=fake_client):
                    res = printer.generate_new_cards(faction="Red", count=1)
                self.assertTrue(res["success"])
                self.assertEqual(res["cards"][0]["cost"], 0)
                self.assertEqual(res["cards"][0]["tags"], ["TEMP_MANA_1"])

        # 3. save_llm_config preserves custom model/base_url when UI doesn't pass them
        import web_app.main as web_main
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "llm_config.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump({"deepseek_api_key": "sk-old", "base_url": "https://custom.endpoint/v1", "model": "custom-model", "deepseek_model": "custom-model"}, f)
            with patch.object(web_main, "_root_dir", tmp):
                req = web_main.LLMConfigRequest(deepseek_api_key="sk-new", image_api_key="sf-new")
                web_main.save_llm_config(req)
            with open(cfg_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved["base_url"], "https://custom.endpoint/v1")
            self.assertEqual(saved["model"], "custom-model")
            self.assertEqual(saved["deepseek_api_key"], "sk-new")

        # 4. PipelineRunner blocks start when only SILICONFLOW_API_KEY exists for api.deepseek.com
        from web_app.services.pipeline_runner import PipelineRunner
        with tempfile.TemporaryDirectory() as tmp:
            runner = PipelineRunner()
            runner.root_dir = tmp
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "", "SILICONFLOW_API_KEY": "sf-img-only", "DEEPSEEK_BASE_URL": "https://api.deepseek.com"}, clear=False):
                with patch.object(tc.sys, "platform", "linux"):
                    res = runner.start()
                    self.assertFalse(res["success"])

        # 5. export_ui_data & check_balance_status share unified bidirectional pairwise math
        import export_ui_data
        import pipeline_orchestrator as po
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "cards_config.json"), "w", encoding="utf-8") as f:
                json.dump(_make_sample_cards(), f)
            with open(os.path.join(tmp, "decks_config.json"), "w", encoding="utf-8") as f:
                json.dump({}, f)
            with open(os.path.join(tmp, "training_metrics_brawl.json"), "w", encoding="utf-8") as f:
                json.dump(_make_sample_metrics(), f)
            ui_data = export_ui_data.export_ui_data(tmp)
            self.assertAlmostEqual(ui_data["max_pairwise_dev"], 22.73, places=2)
            balanced, max_dev, spread, max_pair = po.check_balance_status(_make_sample_metrics(), 5.0, 8.0)
            self.assertFalse(balanced)
            self.assertAlmostEqual(max_pair, 22.73, places=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
