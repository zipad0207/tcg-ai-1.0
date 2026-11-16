import json
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agent import CardNet, compute_zero_sum_gae, load_cardnet_compatible
from sandbox import Card, CardType, DuelEnv, Faction, MinionInstance
import deck_builder_ppo as dbp
import auto_balancer_deepseek as ab


def _sample_cards_file(dirpath: str) -> str:
    data = {
        "Red": [
            {"id": 100, "name": "R_Rush_1", "card_type": "MINION", "cost": 1, "base_dp": 2, "atk_spell_val": 0, "def_spell_val": 0, "tags": ["RUSH"], "factions": ["Red"]},
            {"id": 101, "name": "R_Attacker_3", "card_type": "MINION", "cost": 3, "base_dp": 4, "atk_spell_val": 0, "def_spell_val": 0, "tags": [], "factions": ["Red"]},
        ],
        "Blue": [
            {"id": 203, "name": "防线护盾", "card_type": "SPELL", "cost": 1, "base_dp": 0, "atk_spell_val": 0, "def_spell_val": 6, "tags": [], "factions": ["Blue"]},
            {"id": 241, "name": "寒冰锁链", "card_type": "SPELL", "cost": 2, "base_dp": 0, "atk_spell_val": 0, "def_spell_val": 0, "tags": ["DEGRADE_2", "DRAW_1"], "factions": ["Blue"]},
        ],
        "Green": [
            {"id": 300, "name": "翠绿萌芽", "card_type": "SPELL", "cost": 2, "base_dp": 0, "atk_spell_val": 0, "def_spell_val": 0, "tags": ["RAMP_1"], "factions": ["Green"]},
            {"id": 305, "name": "远古巨树", "card_type": "MINION", "cost": 7, "base_dp": 8, "atk_spell_val": 0, "def_spell_val": 0, "tags": ["FORTIFY_2"], "factions": ["Green"]},
            {"id": 306, "name": "荆棘缠绕", "card_type": "SPELL", "cost": 3, "base_dp": 0, "atk_spell_val": 7, "def_spell_val": 0, "tags": ["DEGRADE_1"], "factions": ["Green"]},
            {"id": 308, "name": "翡翠巨熊", "card_type": "MINION", "cost": 6, "base_dp": 6, "atk_spell_val": 0, "def_spell_val": 0, "tags": ["FORTIFY_2"], "factions": ["Green"]},
            {"id": 312, "name": "翠绿幼苗", "card_type": "MINION", "cost": 1, "base_dp": 2, "atk_spell_val": 0, "def_spell_val": 0, "tags": ["FORTIFY_1"], "factions": ["Green"]},
            {"id": 321, "name": "古树庇护", "card_type": "SPELL", "cost": 3, "base_dp": 0, "atk_spell_val": 0, "def_spell_val": 5, "tags": ["FORTIFY_1"], "factions": ["Green"]},
        ],
        "Neutral": [
            {"id": 917, "name": "涌动", "card_type": "SPELL", "cost": 0, "base_dp": 0, "atk_spell_val": 0, "def_spell_val": 0, "tags": ["TEMP_MANA_2"], "factions": ["Neutral"]},
            {"id": 920, "name": "润滑", "card_type": "SPELL", "cost": 0, "base_dp": 0, "atk_spell_val": 0, "def_spell_val": 0, "tags": ["TEMP_MANA_1"], "factions": ["Neutral"]},
        ],
    }
    path = os.path.join(dirpath, "cards.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return path


class CoreLogicTests(unittest.TestCase):
    def test_sandbox_spell_mechanics_and_observation_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            cards_path = _sample_cards_file(tmp)
            env = DuelEnv(Faction.GREEN, Faction.RED, cards_path=cards_path)
            obs = env.reset()
            self.assertEqual(obs.shape, (3, 13, 20))

            p0 = env.players[0]
            p0.mana = 10
            p0.max_mana = 10

            # 1. Pure atk spell on empty board must not be playable
            thorn = env._clone_card(env.card_database[306])
            p0.hand = [thorn]
            mask = env.get_action_mask()
            self.assertEqual(mask[0], 0.0)
            self.assertEqual(mask[2], 0.0)

            # 2. Pure atk spell targeting enemy attacker clears threat without scoring face
            enemy_card = env._clone_card(env.card_database[101])
            env.lanes[0].attackers.append(MinionInstance(card=enemy_card, current_dp=4, owner=1, ready_to_attack=True))
            mask = env.get_action_mask()
            self.assertEqual(mask[0], 1.0)
            obs_after, _, _, _ = env.step(0)
            self.assertEqual(len([u for u in env.lanes[0].attackers if u.owner == 1]), 0)
            self.assertEqual(p0.score, 0)

            # 3. Sleeping attacker alone does not let pure atk spell fizzle
            env.lanes[0].attackers.append(
                MinionInstance(card=env._clone_card(env.card_database[101]), current_dp=4, owner=0, ready_to_attack=False)
            )
            p0.hand = [env._clone_card(env.card_database[306])]
            self.assertEqual(env.get_action_mask()[0], 0.0)

            # 4. Defense shield spell is playable on empty lane and FORTIFY on spell adds shield
            p0.hand = [env._clone_card(env.card_database[321])]
            mask = env.get_action_mask()
            self.assertEqual(mask[1], 1.0)
            env.step(1)
            self.assertEqual(env.lane_def_shield[0, 0], 6)

            # 5. Hand/board observation encodes spell values, ramp, and faction identity
            p0.hand = [env._clone_card(env.card_database[306]), env._clone_card(env.card_database[300])]
            obs_rich = env.get_observation()
            self.assertEqual(obs_rich[0, 0, 10], 1.0)  # Green one-hot at col 10
            self.assertEqual(obs_rich[0, 1, 8], 1.0)   # Opp Red one-hot at col 8
            self.assertAlmostEqual(obs_rich[1, 0, 8], 0.7, places=5)  # atk_spell_val = 7
            self.assertAlmostEqual(obs_rich[1, 1, 14], 1.0 / 3.0, places=5)  # RAMP_1

    def test_degrade_spell_reduces_enemy_defender_dp(self):
        with tempfile.TemporaryDirectory() as tmp:
            cards_path = _sample_cards_file(tmp)
            env = DuelEnv(Faction.BLUE, Faction.RED, cards_path=cards_path)
            env.reset()
            p0 = env.players[0]
            p0.mana = 5
            p0.max_mana = 5
            env.lanes[1].defenders.append(
                MinionInstance(card=env._clone_card(env.card_database[101]), current_dp=4, owner=1, ready_to_attack=False)
            )
            p0.hand = [env._clone_card(env.card_database[241])]  # cost 2, DEGRADE_2 + DRAW_1
            env.step(2)  # card 0 -> lane 1 atk slot (DEGRADE_2 hits lane 1 defender)
            rem_def = [u for u in env.lanes[1].defenders if u.owner == 1]
            self.assertEqual(len(rem_def), 1)
            self.assertEqual(rem_def[0].current_dp, 2)

    def test_legacy_checkpoint_upgrade_equivalence_and_zero_sum_gae(self):
        old_net = CardNet(action_dim=29, obs_shape=(3, 13, 8))
        obs_full = torch.randn(2, 3, 13, 20)
        obs_legacy = obs_full[:, :, :, :8]
        mask = torch.ones(2, 29)

        with tempfile.TemporaryDirectory() as tmp:
            old_ckpt = os.path.join(tmp, "legacy.pth")
            torch.save(old_net.state_dict(), old_ckpt)

            new_net = CardNet(action_dim=29, obs_shape=(3, 13, 20))
            mode = load_cardnet_compatible(new_net, old_ckpt, torch.device("cpu"))
            self.assertEqual(mode, "upgraded")

            old_net.eval()
            new_net.eval()
            with torch.no_grad():
                l_old, v_old = old_net(obs_legacy, mask)
                l_new, v_new = new_net(obs_full, mask)
            self.assertTrue(torch.allclose(l_old, l_new, atol=1e-5))
            self.assertTrue(torch.allclose(v_old, v_new, atol=1e-5))

        # Turn boundary zero-sum GAE: P0 sets up board -> PASS -> P1 loses -10 on turn 2
        adv, ret = compute_zero_sum_gae(
            rewards=[0.0, 0.0, -10.0],
            dones=[False, False, True],
            values=[0.5, 0.5, -2.0],
            acting_players=[0, 0, 1],
            gamma=0.99,
            gae_lambda=0.95,
        )
        self.assertLess(adv[2], 0.0)
        self.assertGreater(adv[1], 7.0)
        self.assertGreater(adv[0], 7.0)

        # Rollout truncation across player switch must invert last_val
        adv_t, _ = compute_zero_sum_gae(
            rewards=[0.0],
            dones=[False],
            values=[0.0],
            acting_players=[0],
            last_val=-4.0,
            last_player=1,
            gamma=0.99,
            gae_lambda=0.95,
        )
        self.assertAlmostEqual(adv_t[0], 0.99 * 4.0, places=5)

    def test_deck_builder_curve_repair_zero_cap_and_balancer_active_decks(self):
        cands = [
            {"id": 917, "name": "C0_A", "card_type": "SPELL", "cost": 0, "base_dp": 0, "tags": ["TEMP_MANA_2"]},
            {"id": 920, "name": "C0_B", "card_type": "SPELL", "cost": 0, "base_dp": 0, "tags": ["TEMP_MANA_1"]},
            {"id": 312, "name": "G1", "card_type": "MINION", "cost": 1, "base_dp": 2, "tags": ["FORTIFY_1"]},
            {"id": 313, "name": "G2", "card_type": "MINION", "cost": 2, "base_dp": 3, "tags": []},
            {"id": 300, "name": "G_Ramp", "card_type": "SPELL", "cost": 2, "base_dp": 0, "tags": ["RAMP_1"]},
            {"id": 320, "name": "G3_A", "card_type": "MINION", "cost": 3, "base_dp": 4, "tags": []},
            {"id": 321, "name": "G3_B", "card_type": "MINION", "cost": 3, "base_dp": 4, "tags": ["FORTIFY_1"]},
            {"id": 330, "name": "G4_A", "card_type": "MINION", "cost": 4, "base_dp": 5, "tags": []},
            {"id": 331, "name": "G4_B", "card_type": "MINION", "cost": 4, "base_dp": 5, "tags": []},
            {"id": 340, "name": "G5", "card_type": "MINION", "cost": 5, "base_dp": 6, "tags": []},
            {"id": 308, "name": "G6", "card_type": "MINION", "cost": 6, "base_dp": 6, "tags": ["FORTIFY_2"]},
            {"id": 305, "name": "G7", "card_type": "MINION", "cost": 7, "base_dp": 8, "tags": ["FORTIFY_2"]},
        ]
        by_id = {c["id"]: c for c in cands}
        prio = [917, 920, 312, 313, 320, 321, 330, 331, 340, 300, 308, 305]
        # Broken deck stuffed with 6 zero-cost spells and 0 seven-cost finishers
        broken = {917: 3, 920: 3, 312: 3, 313: 3, 320: 3, 321: 3, 330: 3, 331: 3, 340: 3, 308: 3}
        repaired = dbp._repair_deck_curve(broken, by_id, prio, "Green")
        self.assertEqual(sum(repaired.values()), 30)
        stats = dbp._bracket_counts(repaired, by_id)
        self.assertLessEqual(stats["zero"], 2)
        self.assertGreaterEqual(stats["ramp"], 3)
        self.assertGreaterEqual(stats["high7"], 2)
        self.assertGreaterEqual(stats["high6"], 4)

        # Balancer includes active deck roster annotations when decks_data is provided
        class _FakeClient:
            class _Chat:
                class _Completions:
                    def __init__(self, log):
                        self.log = log
                    def create(self, **kwargs):
                        self.log.append(kwargs["messages"][1]["content"])
                        return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": '{"modifications": []}'})()})]})()
                def __init__(self, log):
                    self.completions = self._Completions(log)
            def __init__(self):
                self.prompts = []
                self.chat = self._Chat(self.prompts)

        with tempfile.TemporaryDirectory() as tmp:
            cards_path = _sample_cards_file(tmp)
            with open(cards_path, "r", encoding="utf-8") as f:
                cards_obj = json.load(f)
            metrics_obj = {
                "total_episodes": 100,
                "faction_stats": {
                    "Red": {"matches": 66, "wins": 42, "winrate": 63.6},
                    "Blue": {"matches": 66, "wins": 33, "winrate": 50.0},
                    "Green": {"matches": 68, "wins": 25, "winrate": 36.8},
                },
                "matchups": {},
                "faction_card_plays": {"Red": {}, "Blue": {}, "Green": {}},
                "card_play_count": {},
            }
            decks_obj = {
                "Red": {"avg_cost": 2.0, "decklist": [100] * 3 + [101] * 3},
                "Blue": {"avg_cost": 1.5, "decklist": [203] * 3},
                "Green": {"avg_cost": 4.0, "decklist": [300] * 3 + [305] * 2},
            }
            fc = _FakeClient()
            ab.run_deepseek_balance_and_expand(metrics_obj, cards_obj, fc, decks_data=decks_obj)
            prompt = fc.prompts[0]
            self.assertIn("现役 30 张", prompt)
            self.assertIn("in_decks", prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
