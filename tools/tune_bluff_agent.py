#!/usr/bin/env python3
"""Tune Bluff policy parameters against the bundled standardized baseline mix.

The tuner uses paired game and private action-mixing seeds across candidate
configurations, then uses independent validation and final-test seed ranges.
Its JSONL is a game-outcome rollout set for tuning heuristic weights, not a
supervised state/action dataset for a neural model.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "local_evaluator" / "src"), str(ROOT / "agents" / "bluff-agent" / "src")]

from triad_bluff import baselines, engine, simulator  # noqa: E402
import strategy  # noqa: E402


class Candidate:
    def __init__(self, policy_seed: int) -> None:
        self.state: Any = {"v": 2, "rng_seed": policy_seed}

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        action, self.state = strategy.select_action(observation, self.state)
        return action


def evaluate(config: dict[str, Any], games: int, seed_start: int,
             policy_seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    strategy.POLICY_PARAMS.clear()
    strategy.POLICY_PARAMS.update(config)
    points = wins = seconds = thirds = fourths = draws = faults = 0
    records = []
    policy_seeds = random.Random(policy_seed)
    for index in range(games):
        seed = seed_start + index
        own_seat = index % 4
        agents: list[Any] = [None] * 4
        agents[own_seat] = Candidate(policy_seeds.getrandbits(128))
        opponents = [
            baselines.HeuristicAgent(seed * 17 + 1),
            baselines.ConservativeAgent(),
            baselines.RandomAgent(seed * 17 + 2),
        ]
        for seat in range(4):
            if seat != own_seat:
                agents[seat] = opponents.pop(0)
        result = simulator.run_game(seed, agents, engine.GameConfig())
        game_points = result.outcome.points[own_seat]
        game_faults = len(result.faults)
        points += game_points
        faults += game_faults
        placement = sorted(
            result.outcome.points,
            key=lambda seat: (-result.outcome.points[seat], seat),
        ).index(own_seat)
        if result.outcome.is_draw:
            draws += 1
            did_win = False
        elif placement == 0:
            wins += 1
            did_win = True
        elif placement == 1:
            seconds += 1
            did_win = False
        elif placement == 2:
            thirds += 1
            did_win = False
        else:
            fourths += 1
            did_win = False
        records.append({
            "game_seed": seed,
            "seat": own_seat,
            "points": game_points,
            "place": None if result.outcome.is_draw else placement + 1,
            "win": did_win,
            "draw": result.outcome.is_draw,
            "faults": game_faults,
        })
    summary = {
        "games": games,
        "seed_start": seed_start,
        "policy_seed": policy_seed,
        "wins": wins,
        "win_rate": wins / games,
        "mean_points": points / games,
        "placements": {
            "second": seconds, "third": thirds, "fourth": fourths, "draws": draws,
        },
        "faults": faults,
    }
    return summary, records


def paired_difference(candidate: list[dict[str, Any]], baseline: list[dict[str, Any]],
                      metric: str) -> dict[str, float | int]:
    base_by_key = {(row["game_seed"], row["seat"]): row for row in baseline}
    differences = [
        float(row[metric]) - float(base_by_key[(row["game_seed"], row["seat"])][metric])
        for row in candidate
    ]
    average = statistics.mean(differences)
    standard_error = (statistics.stdev(differences) / (len(differences) ** 0.5)
                      if len(differences) > 1 else 0.0)
    margin = 1.96 * standard_error
    return {
        "games": len(differences),
        "mean_candidate_minus_default": average,
        "ci95_low": average - margin,
        "ci95_high": average + margin,
    }


def sample_config(rng: random.Random) -> dict[str, Any]:
    small_pressure = rng.uniform(0.18, 0.62)
    return {
        "prior_bluff_rate": rng.uniform(0.06, 0.24),
        "prior_weight": rng.choice([2.0, 4.0, 8.0]),
        "challenge_threshold": rng.uniform(0.42, 0.64),
        "finish_relief": rng.uniform(0.12, 0.34),
        "near_finish_relief": rng.uniform(0.05, 0.17),
        "large_claim_relief": rng.uniform(0.02, 0.15),
        "pressure_small_pile_rate": small_pressure,
        "pressure_medium_pile_rate": rng.uniform(0.04, min(0.36, small_pressure)),
        "pressure_min_hand": rng.choice([6, 7, 8, 9, 10]),
        "pressure_max_pile": rng.choice([3, 4, 5, 6, 7]),
        "pressure_min_opponent_hand": rng.choice([3, 4, 5, 6, 7]),
        "adaptive_pressure": rng.choice([False, True]),
        "challenge_prior_rate": rng.uniform(0.08, 0.30),
        "challenge_prior_weight": rng.choice([2.0, 4.0, 8.0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=39,
                        help="random parameter configurations, in addition to defaults")
    parser.add_argument("--train-games", type=int, default=252)
    parser.add_argument("--validation-games", type=int, default=1500)
    parser.add_argument("--test-games", type=int, default=2000)
    parser.add_argument("--seed-start", type=int, default=160000)
    parser.add_argument("--validation-seed-start", type=int, default=170000)
    parser.add_argument("--test-seed-start", type=int, default=180000)
    parser.add_argument("--search-seed", type=int, default=730241)
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "bluff-parameter-tuning.json")
    parser.add_argument("--dataset", type=Path, default=ROOT / "reports" / "bluff-policy-training.jsonl")
    args = parser.parse_args()
    if (args.trials < 1 or args.train_games < 4 or args.validation_games < 4
            or args.test_games < 4):
        parser.error("trials must be positive and each game count must be at least 4")
    train_range = (args.seed_start, args.seed_start + args.train_games - 1)
    validation_range = (args.validation_seed_start,
                        args.validation_seed_start + args.validation_games - 1)
    test_range = (args.test_seed_start, args.test_seed_start + args.test_games - 1)
    ranges = [train_range, validation_range, test_range]
    if any(max(a[0], b[0]) <= min(a[1], b[1])
           for i, a in enumerate(ranges) for b in ranges[i + 1:]):
        parser.error("training, validation, and test seed ranges must not overlap")

    defaults = dict(strategy.POLICY_PARAMS)
    candidate_rng = random.Random(args.search_seed)
    candidates = [defaults]
    seen = {json.dumps(defaults, sort_keys=True)}
    while len(candidates) <= args.trials:
        sampled = defaults | sample_config(candidate_rng)
        key = json.dumps(sampled, sort_keys=True)
        if key not in seen:
            seen.add(key)
            candidates.append(sampled)

    training = []
    training_records = []
    for index, config in enumerate(candidates):
        result, game_records = evaluate(
            config, args.train_games, args.seed_start, args.search_seed + 1
        )
        row = {"config": config, "result": result}
        training.append(row)
        config_id = f"candidate-{index:02d}"
        training_records.extend({"config_id": config_id, "config": config, **record}
                                for record in game_records)
        print(json.dumps({
            "stage": "train", "index": index, "games": args.train_games,
            "win_rate": result["win_rate"], "mean_points": result["mean_points"],
            "faults": result["faults"],
        }), flush=True)

    training.sort(key=lambda row: (row["result"]["mean_points"], row["result"]["win_rate"]),
                  reverse=True)
    validation_seed = args.validation_seed_start
    finalists = [training[0]]
    for row in training[1:]:
        if row["config"] != finalists[0]["config"]:
            finalists.append(row)
        if len(finalists) == 3:
            break

    default_result, default_records = evaluate(
        defaults, args.validation_games, validation_seed, args.search_seed + 2
    )
    validation = []
    for row in finalists:
        result, records = evaluate(row["config"], args.validation_games, validation_seed,
                                   args.search_seed + 2)
        checked = {
            "config": row["config"],
            "train": row["result"],
            "validation": result,
            "paired_vs_default": {
                "points": paired_difference(records, default_records, "points"),
                "win_rate": paired_difference(records, default_records, "win"),
            },
        }
        validation.append(checked)
        print(json.dumps({"stage": "validation", **checked}), flush=True)

    qualifying = [row for row in validation
                  if row["paired_vs_default"]["points"]["ci95_low"] > 0.0]
    selected = (max(qualifying,
                    key=lambda row: (row["validation"]["mean_points"],
                                     row["validation"]["win_rate"]))
                if qualifying else None)
    selected_config = selected["config"] if selected else defaults

    test_seed = args.test_seed_start
    test_default, test_default_records = evaluate(
        defaults, args.test_games, test_seed, args.search_seed + 3
    )
    test_selected, test_selected_records = evaluate(
        selected_config, args.test_games, test_seed, args.search_seed + 3
    )
    test_comparison = {
        "points": paired_difference(test_selected_records, test_default_records, "points"),
        "win_rate": paired_difference(test_selected_records, test_default_records, "win"),
    }

    args.dataset.parent.mkdir(parents=True, exist_ok=True)
    with args.dataset.open("w", encoding="utf-8", newline="\n") as dataset_file:
        for record in training_records:
            dataset_file.write(json.dumps(record, separators=(",", ":")) + "\n")
    report = {
        "method": "paired random search, held-out validation and untouched final test",
        "training_data_note": "Game-level policy rollouts for heuristic parameter tuning; not supervised ML action labels.",
        "opponents": ["heuristic", "conservative", "random"],
        "seed_ranges": {
            "training_per_config": list(train_range),
            "validation": [validation_seed, validation_seed + args.validation_games - 1],
            "final_test": [test_seed, test_seed + args.test_games - 1],
        },
        "training_rollouts": len(training_records),
        "training_configurations": len(candidates),
        "unique_training_game_seeds": len({row["game_seed"] for row in training_records}),
        "paired_seed_note": "The same environment seeds are reused across configurations to reduce variance in comparisons; each config produces a separate game trajectory.",
        "training_dataset": str(args.dataset.resolve()),
        "search_seed": args.search_seed,
        "default_validation": {"config": defaults, "result": default_result},
        "training_ranked": training,
        "finalist_validation": validation,
        "selected_config": selected_config,
        "selection_reason": (
            "selected only if a finalist's paired validation improvement in mean points had a 95% confidence interval above zero; otherwise kept defaults"
        ),
        "final_test_default": test_default,
        "final_test_selected": test_selected,
        "final_test_paired_comparison": test_comparison,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "final_test", "default": test_default,
                      "selected": test_selected, "paired": test_comparison}), flush=True)
    print(json.dumps({"stage": "complete", "report": str(args.report.resolve())}), flush=True)


if __name__ == "__main__":
    main()
