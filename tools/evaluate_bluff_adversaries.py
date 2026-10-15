#!/usr/bin/env python3
"""Paired evaluation against standard and aggressive Bluff styles.

Synthetic opponents are evaluation fixtures only. The submission agent sees
the same filtered observations as in normal play.
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
    def __init__(self, seed: int) -> None:
        self.state: Any = {"v": 2, "rng_seed": seed}

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        action, self.state = strategy.select_action(observation, self.state)
        return action


class AggressiveChallengeAgent(baselines.TruthfulAgent):
    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        if observation["phase"] == "CHALLENGE":
            return {"type": "challenge"}
        return super().act(observation)


class PressureAgent:
    """Often challenges and mixes two-to-four-card claims with a lie."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.truthful = baselines.TruthfulAgent()

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        if observation["phase"] == "CHALLENGE":
            claim = observation.get("pending_claim") or {}
            if claim.get("count", 0) >= 3 or observation.get("pile_size", 0) >= 8:
                return {"type": "challenge"}
            return {"type": "challenge" if self.rng.random() < 0.30 else "accept"}
        hand = observation["hand"]
        if len(hand) < 6 or self.rng.random() >= 0.50:
            return self.truthful.act(observation)
        count = min(self.rng.choice((2, 3, 4)), len(hand))
        groups: dict[str, list[dict[str, Any]]] = {}
        for card in hand:
            groups.setdefault(card["rank"], []).append(card)
        rank = min(groups, key=lambda item: (len(groups[item]), item))
        if len(groups[rank]) >= count:
            return self.truthful.act(observation)
        cards = self.rng.sample(hand, count)
        return {"type": "place", "card_refs": [c["card_ref"] for c in cards],
                "claimed_rank": rank}


SCENARIOS = {
    "standard": ("heuristic", "conservative", "random"),
    "challenge_heavy": ("aggressive_challenge", "heuristic", "random"),
    "bluff_heavy": ("pressure", "conservative", "heuristic"),
    "combined": ("pressure", "aggressive_challenge", "heuristic"),
}


def make_opponent(style: str, seed: int) -> Any:
    if style == "heuristic":
        return baselines.HeuristicAgent(seed)
    if style == "conservative":
        return baselines.ConservativeAgent()
    if style == "random":
        return baselines.RandomAgent(seed)
    if style == "aggressive_challenge":
        return AggressiveChallengeAgent()
    if style == "pressure":
        return PressureAgent(seed)
    raise ValueError(style)


def evaluate(config: dict[str, Any], games_per_scenario: int, seed_start: int,
             policy_name: str) -> list[dict[str, Any]]:
    strategy.POLICY_PARAMS.clear()
    strategy.POLICY_PARAMS.update(config)
    rows: list[dict[str, Any]] = []
    scenario_index = 0
    for scenario, opponent_styles in SCENARIOS.items():
        for index in range(games_per_scenario):
            seed = seed_start + scenario_index * games_per_scenario + index
            our_seat = index % 4
            agents: list[Any] = [None] * 4
            agents[our_seat] = Candidate((seed * 65537 + our_seat + 1) % (2**128))
            # Keep opponent order clockwise from our seat so challenge pressure
            # is comparable and role rotation stays balanced across seats.
            for offset, style in enumerate(opponent_styles, start=1):
                seat = (our_seat + offset) % 4
                agents[seat] = make_opponent(style, seed * 17 + offset)
            result = simulator.run_game(seed, agents, engine.GameConfig())
            ranked = sorted(result.outcome.points, key=lambda s: (-result.outcome.points[s], s))
            place = ranked.index(our_seat)
            rows.append({
                "scenario": scenario, "game_seed": seed, "seat": our_seat,
                "points": result.outcome.points[our_seat],
                "win": int(not result.outcome.is_draw and place == 0),
                "place": place + 1,
                "faults": len(result.faults),
            })
        scenario_index += 1
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "games": len(rows),
        "wins": sum(row["win"] for row in rows),
        "win_rate": statistics.mean(row["win"] for row in rows),
        "mean_points": statistics.mean(row["points"] for row in rows),
        "faults": sum(row["faults"] for row in rows),
        "by_scenario": {
            scenario: {
                "games": len(part), "wins": sum(row["win"] for row in part),
                "win_rate": statistics.mean(row["win"] for row in part),
                "mean_points": statistics.mean(row["points"] for row in part),
                "faults": sum(row["faults"] for row in part),
            }
            for scenario in SCENARIOS
            if (part := [row for row in rows if row["scenario"] == scenario])
        },
    }


def paired_delta(rows: list[dict[str, Any]], baseline: list[dict[str, Any]], metric: str) -> dict[str, float]:
    base = {(row["scenario"], row["game_seed"], row["seat"]): row[metric] for row in baseline}
    delta = [row[metric] - base[(row["scenario"], row["game_seed"], row["seat"])] for row in rows]
    mean = statistics.mean(delta)
    margin = 1.96 * statistics.stdev(delta) / (len(delta) ** 0.5)
    return {"mean_candidate_minus_default": mean, "ci95_low": mean - margin,
            "ci95_high": mean + margin}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-per-scenario", type=int, default=500)
    parser.add_argument("--seed", type=int, default=220000)
    parser.add_argument("--policies", nargs="+",
                        choices=("default", "reactive", "pressure", "challenge_hunter",
                                 "aggressive_reactive"),
                        default=("default", "reactive", "pressure", "challenge_hunter",
                                 "aggressive_reactive"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.games_per_scenario < 4:
        parser.error("--games-per-scenario must be at least 4")

    default = dict(strategy.POLICY_PARAMS)
    # Preserve the pre-adaptation submission as the comparison baseline even
    # after adaptive pressure becomes the checked-in default.
    default["adaptive_pressure"] = False
    candidates = {
        "default": {},
        "reactive": {"adaptive_pressure": True},
        "pressure": {"adaptive_pressure": True, "pressure_small_pile_rate": 0.56,
                     "pressure_medium_pile_rate": 0.34, "pressure_min_hand": 6,
                     "pressure_max_pile": 8},
        "challenge_hunter": {"challenge_threshold": 0.45, "prior_bluff_rate": 0.16,
                             "adaptive_pressure": True},
        "aggressive_reactive": {"challenge_threshold": 0.45, "prior_bluff_rate": 0.16,
                                "adaptive_pressure": True,
                                "pressure_small_pile_rate": 0.56,
                                "pressure_medium_pile_rate": 0.34,
                                "pressure_min_hand": 6, "pressure_max_pile": 8},
    }
    all_rows: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    for name in args.policies:
        overrides = candidates[name]
        config = {**default, **overrides}
        rows = evaluate(config, args.games_per_scenario, args.seed, name)
        all_rows[name] = rows
        summaries[name] = summarize(rows)
        if name != "default":
            summaries[name]["paired_delta_win"] = paired_delta(rows, all_rows["default"], "win")
            summaries[name]["paired_delta_points"] = paired_delta(rows, all_rows["default"], "points")
        print(json.dumps({"policy": name, **summaries[name]}, separators=(",", ":")), flush=True)

    report = {
        "seed_start": args.seed,
        "games_per_scenario": args.games_per_scenario,
        "scenarios": SCENARIOS,
        "candidate_summaries": summaries,
        "selection_note": "Synthetic stress styles supplement the bundled standardized opponents; they are not an estimate of the organizer's roster.",
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    strategy.POLICY_PARAMS.clear()
    strategy.POLICY_PARAMS.update(default)


if __name__ == "__main__":
    main()
