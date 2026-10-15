#!/usr/bin/env python3
"""Evaluate the competitive Bluff agent against bundled baseline opponents.

Runs a balanced seat rotation: the candidate plays each of the four seats the
same number of times.  The engine supplies only filtered observations, the
same information available to a submission during a real match.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "local_evaluator" / "src"), str(ROOT / "agents" / "bluff-agent" / "src")]

from triad_bluff import baselines, engine, simulator  # noqa: E402
import strategy  # noqa: E402


PROFILES = {
    "submission": {},
    "adaptive-pressure": {
        "adaptive_pressure": True,
    },
    "static-relentless": {
        "adaptive_pressure": False,
    },
    "guarded": {
        "challenge_threshold": 0.60,
        "pressure_small_pile_rate": 0.14,
        "pressure_medium_pile_rate": 0.06,
    },
    "balanced": {
        "challenge_threshold": 0.57,
        "finish_relief": 0.17,
        "near_finish_relief": 0.07,
        "large_claim_relief": 0.05,
        "pressure_small_pile_rate": 0.28,
        "pressure_medium_pile_rate": 0.14,
        "pressure_min_hand": 8,
    },
    "pressing": {
        "challenge_threshold": 0.53,
        "finish_relief": 0.20,
        "near_finish_relief": 0.10,
        "large_claim_relief": 0.07,
        "pressure_small_pile_rate": 0.36,
        "pressure_medium_pile_rate": 0.20,
    },
    "relentless": {
        "challenge_threshold": 0.49,
        "finish_relief": 0.23,
        "near_finish_relief": 0.12,
        "large_claim_relief": 0.10,
        "pressure_small_pile_rate": 0.46,
        "pressure_medium_pile_rate": 0.28,
        "pressure_min_hand": 7,
    },
}


class Candidate:
    """Preserve the agent checkpoint across every decision in one game."""

    def __init__(self) -> None:
        self.state: Any = None

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        action, self.state = strategy.select_action(observation, self.state)
        return action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42000)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="submission")
    args = parser.parse_args()
    if args.games < 4:
        parser.error("--games must be at least 4")
    strategy.POLICY_PARAMS.update(PROFILES[args.profile])

    points = 0.0
    wins = seconds = thirds = fourths = draws = faults = 0
    per_seat: dict[int, dict[str, float]] = {
        seat: {"games": 0, "wins": 0, "points": 0} for seat in range(4)
    }
    for game in range(args.games):
        seed = args.seed + game
        our_seat = game % 4
        agents: list[Any] = [None] * 4
        agents[our_seat] = Candidate()
        opponents = [
            baselines.HeuristicAgent(seed * 17 + 1),
            baselines.ConservativeAgent(),
            baselines.RandomAgent(seed * 17 + 2),
        ]
        for seat in range(4):
            if seat != our_seat:
                agents[seat] = opponents.pop(0)
        result = simulator.run_game(seed, agents, engine.GameConfig())
        score = result.outcome.points[our_seat]
        points += score
        faults += len(result.faults)
        rank = sorted(result.outcome.points, key=lambda seat: (-result.outcome.points[seat], seat)).index(our_seat)
        if result.outcome.is_draw:
            draws += 1
        elif rank == 0:
            wins += 1
        elif rank == 1:
            seconds += 1
        elif rank == 2:
            thirds += 1
        else:
            fourths += 1
        row = per_seat[our_seat]
        row["games"] += 1
        row["wins"] += int(not result.outcome.is_draw and rank == 0)
        row["points"] += score

    report = {
        "games": args.games,
        "seed_start": args.seed,
        "profile": args.profile,
        "policy_params": strategy.POLICY_PARAMS,
        "opponents": ["heuristic", "conservative", "random"],
        "wins": wins,
        "win_rate": wins / args.games,
        "mean_points": points / args.games,
        "placements": {"second": seconds, "third": thirds, "fourth": fourths, "draws": draws},
        "faults": faults,
        "by_starting_seat": {
            str(seat): {
                **row,
                "win_rate": row["wins"] / row["games"],
                "mean_points": row["points"] / row["games"],
            }
            for seat, row in per_seat.items()
        },
    }
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
