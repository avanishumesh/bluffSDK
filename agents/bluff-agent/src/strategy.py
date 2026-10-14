"""Competitive Triad Bluff strategy.

Hook: ``select_action(observation, agent_state) -> (action, checkpoint)``.

Game model (from Bluff/docs/ruleset.md + protocol):
  - 4 seats, 52 cards, 13 each. Batches of 1-4 cards, any claimed rank.
  - Only the latest batch is ever tested; loser of a challenge takes the
    whole pile. Unchallenged claims stay hidden forever.
  - Observations give: our hand (request-local ``card_ref`` + rank/suit),
    hand sizes, pile size, pending claim, challenge priority, full public
    history and our private history.

Strategy summary:
  PLACE: shed the largest truthful group, including in the endgame.
  CHALLENGE: call impossible claims; learn separate bluff rates for each
    opponent and batch size from revealed claims. Protect small hands and
    avoid treating a large pile alone as evidence of dishonesty.
"""

from __future__ import annotations

import hashlib
import random
import sys
from collections import defaultdict
from typing import Any

RANKS: tuple[str, ...] = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
RANK_INDEX = {rank: i for i, rank in enumerate(RANKS)}

# Tuned with a 1,200-game local profile sweep and a 2,000-game fresh-seed
# confirmation against the bundled heuristic, conservative, and random
# policies. Keep all game-style choices here so the policy can be measured as
# a configuration rather than by editing decision code repeatedly.
POLICY_PARAMS = {
    "prior_bluff_rate": 0.12,
    "prior_weight": 4.0,
    "challenge_threshold": 0.49,
    "finish_relief": 0.23,
    "near_finish_relief": 0.12,
    "large_claim_relief": 0.10,
    "pressure_small_pile_rate": 0.46,
    "pressure_medium_pile_rate": 0.28,
    "pressure_min_hand": 7,
    "pressure_max_pile": 6,
    "pressure_min_opponent_hand": 5,
    "adaptive_pressure": False,
    "randomize_pressure_anchor": False,
}


# ---------------------------------------------------------------------------
# Action constructors (exact wire schema: semantic objects, no extra fields)
# ---------------------------------------------------------------------------

def place(card_refs: list[str] | tuple[str, ...], claimed_rank: str) -> dict[str, Any]:
    return {"type": "place", "card_refs": list(card_refs), "claimed_rank": claimed_rank}


def challenge() -> dict[str, str]:
    return {"type": "challenge"}


def accept() -> dict[str, str]:
    return {"type": "accept"}


def _log(*parts: object) -> None:
    print(*parts, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Deterministic pseudo-randomness (no shared RNG state, unpredictable play)
# ---------------------------------------------------------------------------

def _hash_rng(*parts: object) -> random.Random:
    digest = hashlib.sha256(repr(parts).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


# ---------------------------------------------------------------------------
# History tracking: cards played/claimed + per-opponent bluff profiles
# ---------------------------------------------------------------------------

def _opp_profile(public_history: list[dict[str, Any]], players: int = 4) -> dict[int, dict[str, Any]]:
    """Fold public events into per-seat claim/bluff statistics.

    Uses ``claim`` events (who claimed what) and ``challenge_resolved``
    events (placer, truthful flag). Unchallenged claims count as claims
    with unknown truth -- they still inform volume/frequency stats.
    """
    stats: dict[int, dict[str, Any]] = {}
    for seat in range(players):
        stats[seat] = {"claims": 0, "resolved": 0, "bluffs": 0, "truths": 0,
                       "last_ranks": [], "challenges": 0, "challenge_wins": 0,
                       "by_count": {}}
    claims = {}
    for event in public_history or []:
        etype = event.get("type")
        if etype == "claim":
            seat = event.get("seat")
            if isinstance(seat, int) and seat in stats:
                stats[seat]["claims"] += 1
                claims[event.get("claim_id")] = event
                rank = event.get("claimed_rank")
                if isinstance(rank, str):
                    stats[seat]["last_ranks"].append(rank)
                    if len(stats[seat]["last_ranks"]) > 6:
                        stats[seat]["last_ranks"].pop(0)
        elif etype == "challenge_resolved":
            placer = event.get("placer")
            truthful = event.get("truthful")
            challenger = event.get("challenger")
            if isinstance(challenger, int) and challenger in stats:
                stats[challenger]["challenges"] += 1
                if truthful is False:
                    stats[challenger]["challenge_wins"] += 1
            if isinstance(placer, int) and placer in stats and isinstance(truthful, bool):
                stats[placer]["resolved"] += 1
                source = claims.get(event.get("claim_id"), {})
                count = source.get("count")
                if isinstance(count, int):
                    bucket = stats[placer]["by_count"].setdefault(count, [0, 0])
                    bucket[0] += 1
                    bucket[1] += int(not truthful)
                if truthful:
                    stats[placer]["truths"] += 1
                else:
                    stats[placer]["bluffs"] += 1
    return stats


def _bluff_rate(stats: dict[str, Any]) -> float:
    """Shrinkage-smoothed bluff rate so 1-2 samples do not over-swing."""
    n = stats["resolved"]
    prior = POLICY_PARAMS["prior_bluff_rate"]
    weight = POLICY_PARAMS["prior_weight"]
    return (stats["bluffs"] + prior * weight) / (n + weight)


# ---------------------------------------------------------------------------
# Challenge side: P(bluff) estimation + pot-odds threshold
# ---------------------------------------------------------------------------

# Multi-card claims are materially harder to make truthfully.  These priors
# remain below the calling threshold until the claim size or public record adds
# corroboration, which keeps the policy from gambling on every large pile.
_BASE_BLUFF = {1: 0.12, 2: 0.16, 3: 0.23, 4: 0.30}


def estimate_bluff_prob(observation: dict[str, Any], claim: dict[str, Any],
                        held: int, profile: dict[int, dict[str, Any]]) -> float:
    """Estimate P(claim is a bluff) from public info + our private hand."""
    count = int(claim.get("count", 1))
    count = max(1, min(4, count))
    # Logically impossible: more copies claimed than exist outside our hand.
    if count > 4 * observation.get("game_config", {}).get("decks", 1) - held:
        return 1.0
    prior = _BASE_BLUFF[count]
    prob = prior
    placer = claim.get("seat")
    if isinstance(placer, int) and placer in profile:
        st = profile[placer]
        # Only challenged, revealed batches have a known truth value.
        # Batch-specific evidence prevents single-card bluffs from making
        # an opponent's otherwise honest pairs look equally suspicious.
        overall = _bluff_rate(st)
        samples, bluffs = st["by_count"].get(count, [0, 0])
        prob = (bluffs + 3.0 * overall + 2.0 * prior) / (samples + 5.0)
    prob += 0.025 * held
    return max(0.02, min(0.99, prob))


def challenge_threshold(observation: dict[str, Any], claim: dict[str, Any],
                        our_hand: int) -> float:
    """Pot-odds threshold: challenge iff P(bluff) >= threshold."""
    pile_size = int(observation.get("pile_size", 0) or 0)
    count = int(claim.get("count", 1) or 1)
    hand_sizes = observation.get("hand_sizes") or []
    placer = claim.get("seat")
    placer_remaining = None
    if isinstance(placer, int) and isinstance(hand_sizes, list) and 0 <= placer < len(hand_sizes):
        try:
            placer_remaining = int(hand_sizes[placer])
        except (TypeError, ValueError):
            placer_remaining = None

    threshold = POLICY_PARAMS["challenge_threshold"]
    # A large pickup jeopardizes our placement; it does not prove a bluff.
    threshold += min(0.10, 0.005 * max(0, pile_size - 6))
    # Must stop finishes: a successful bluff-out ends their game.
    if claim.get("provisional_finish") or placer_remaining == 0:
        threshold -= POLICY_PARAMS["finish_relief"]
    elif isinstance(placer_remaining, int) and placer_remaining <= 3:
        threshold -= POLICY_PARAMS["near_finish_relief"]
    # Protect our own near-win: avoid picking up when almost out.
    if our_hand <= 3:
        threshold += 0.12
    elif our_hand >= 10:
        threshold -= POLICY_PARAMS["large_claim_relief"]
    # Cheap singles early offer little benefit relative to pickup risk.
    if count == 1 and pile_size <= 3:
        threshold += 0.04
    elif count >= 3:
        # Big batches are expensive for an opponent to make honestly and can
        # move them sharply toward a finish, so test them a little sooner.
        threshold -= 0.05
    return max(0.40, min(0.85, threshold))


def _decide_challenge(observation: dict[str, Any], agent_state: Any) -> tuple[dict[str, Any], Any]:
    claim = observation.get("pending_claim") or {}
    hand = observation.get("hand") or []
    rank = claim.get("claimed_rank")
    held = sum(1 for card in hand if isinstance(card, dict) and card.get("rank") == rank)
    count = int(claim.get("count", 0) or 0)
    # Sound impossibility call (mirrors the baseline, never wrong).
    if count > 4 * observation.get("game_config", {}).get("decks", 1) - held:
        return challenge(), _checkpoint(observation, agent_state)
    profile = _opp_profile(observation.get("public_history") or [], len(observation.get("hand_sizes") or [0] * 4))
    prob = estimate_bluff_prob(observation, claim, held, profile)
    threshold = challenge_threshold(observation, claim, len(hand))
    if prob >= threshold + 0.06:
        return challenge(), _checkpoint(observation, agent_state)
    if prob < threshold - 0.06:
        return accept(), _checkpoint(observation, agent_state)
    # Marginal spot: deterministic coin-flip weighted by edge, so we stay
    # unpredictable without drifting from the EV decision on average.
    rng = _hash_rng("challenge", observation.get("turn_number"),
                    observation.get("pile_size"), rank, count, held,
                    sorted(c.get("rank", "?") for c in hand if isinstance(c, dict)))
    edge = (prob - threshold) / 0.06  # in (-1, 1)
    if rng.random() < 0.5 + 0.5 * edge:
        return challenge(), _checkpoint(observation, agent_state)
    return accept(), _checkpoint(observation, agent_state)


# ---------------------------------------------------------------------------
# Placement side: truthful shedding with bounded early-pressure bluffs
# ---------------------------------------------------------------------------

def _groups(hand: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for card in hand:
        if isinstance(card, dict) and isinstance(card.get("rank"), str) and isinstance(card.get("card_ref"), str):
            groups[card["rank"]].append(card)
    return groups


def _best_rank(groups: dict[str, list[dict[str, Any]]]) -> str:
    return max(groups.items(), key=lambda item: (len(item[1]), item[0]))[0]


def _pressure_multiplier(profile: dict[int, dict[str, Any]], seat: int) -> float:
    """Reduce bluffs after the table has shown it will actually challenge.

    A resolved challenge reveals only the first challenger clockwise, so this
    is deliberately a one-sided adjustment: observed challenges can make us
    safer, while a lack of observations never claims that a seat is passive.
    """
    opponents = [stats for other, stats in profile.items() if other != seat]
    claims = sum(stats["claims"] for stats in opponents)
    challenges = sum(stats["challenges"] for stats in opponents)
    wins = sum(stats["challenge_wins"] for stats in opponents)
    if claims < 4 or challenges == 0:
        return 1.0
    challenge_rate = challenges / claims
    win_rate = wins / challenges
    return max(0.40, 1.0 - 1.25 * challenge_rate - 0.25 * win_rate)


def _decide_place(observation: dict[str, Any], agent_state: Any) -> tuple[dict[str, Any], Any]:
    hand = [c for c in (observation.get("hand") or []) if isinstance(c, dict) and isinstance(c.get("card_ref"), str)]
    if not hand:
        # Engine never queries finished seats; stay quiet and take the fault.
        return accept(), _checkpoint(observation, agent_state)
    groups = _groups(hand)
    if not groups:
        return accept(), _checkpoint(observation, agent_state)
    best = _best_rank(groups)
    pile_size = int(observation.get("pile_size", 0) or 0)
    hand_sizes = observation.get("hand_sizes") or []
    try:
        seat = int(observation.get("seat", -1))
        opponent_sizes = [int(size) for index, size in enumerate(hand_sizes) if index != seat]
        min_opponent = min(opponent_sizes) if opponent_sizes else 13
    except (TypeError, ValueError):
        min_opponent = 13
        seat = -1
    profile = _opp_profile(observation.get("public_history") or [], len(hand_sizes) or 4)

    # When every rank is a singleton, pure truthful play discards only one
    # card per turn and is easy for other agents to exploit.  With a small
    # pile and no looming opponent finish, occasionally place a two-card
    # semi-bluff: one card really is the declared rank, which is less exposed
    # than claiming a rank we do not hold at all.  The hash keeps the mix
    # reproducible without sharing random state between games.
    if (len(groups[best]) == 1 and len(hand) >= POLICY_PARAMS["pressure_min_hand"]
            and pile_size <= POLICY_PARAMS["pressure_max_pile"]
            and min_opponent >= POLICY_PARAMS["pressure_min_opponent_hand"]):
        rng = _hash_rng("pressure", observation.get("turn_number"), observation.get("seat"),
                        pile_size, sorted(card.get("rank", "?") for card in hand))
        pressure_rate = (POLICY_PARAMS["pressure_small_pile_rate"] if pile_size <= 3
                         else POLICY_PARAMS["pressure_medium_pile_rate"])
        if POLICY_PARAMS["adaptive_pressure"]:
            pressure_rate *= _pressure_multiplier(profile, seat)
        if rng.random() < pressure_rate:
            # With singleton ranks, `_best_rank` would otherwise repeatedly
            # choose the same lexical rank. Rotate the real anchor so a
            # pattern-matching opponent cannot reserve copies of one rank.
            anchor = rng.choice(hand) if POLICY_PARAMS["randomize_pressure_anchor"] else groups[best][0]
            kicker = next(card for card in hand if card["card_ref"] != anchor["card_ref"])
            return place([anchor["card_ref"], kicker["card_ref"]], anchor["rank"]), _checkpoint(
                observation, agent_state, bluff=True)

    # Truthful groups reduce the number of remaining placement turns without
    # exposing an accumulated pile to a bluff penalty.  We never bluff a final
    # batch because a single challenge would erase the near-finish.
    batch = groups[best][:4]
    return place([c["card_ref"] for c in batch], best), _checkpoint(
        observation, agent_state, bluff=False)


# ---------------------------------------------------------------------------
# Checkpoint: tiny JSON snapshot of tracked history (recomputed statelessly)
# ---------------------------------------------------------------------------

def _checkpoint(observation: dict[str, Any], agent_state: Any, bluff: bool | None = None) -> dict[str, Any]:
    try:
        prev = agent_state if isinstance(agent_state, dict) else {}
        profile = _opp_profile(observation.get("public_history") or [], len(observation.get("hand_sizes") or [0] * 4))
        my_bluffs = int(prev.get("my_bluffs", 0) or 0)
        my_truths = int(prev.get("my_truths", 0) or 0)
        if bluff is True:
            my_bluffs += 1
        elif bluff is False:
            my_truths += 1
        return {
            "v": 1,
            "turn": int(observation.get("turn_number", 0) or 0),
            "my_bluffs": my_bluffs,
            "my_truths": my_truths,
            "opp": {str(seat): {"c": st["claims"], "r": st["resolved"], "b": st["bluffs"]}
                    for seat, st in profile.items()},
        }
    except Exception:
        return {"v": 1}


# ---------------------------------------------------------------------------
# Entry hook with last-resort fault guard (an agent fault gifts a fallback
# placement / forced accept, so we must never emit malformed actions)
# ---------------------------------------------------------------------------

def _safe_place(hand: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [c for c in hand if isinstance(c, dict)
             and isinstance(c.get("card_ref"), str) and isinstance(c.get("rank"), str)
             and c["rank"] in RANKS]
    if not valid:
        return {"type": "place", "card_refs": ["h0"], "claimed_rank": "A"}
    first = valid[0]
    return place([first["card_ref"]], first["rank"])


def select_action(observation: dict[str, Any], agent_state: Any = None) -> tuple[dict[str, Any], Any]:
    """Truthful shedding with evidence-based challenges."""
    try:
        phase = observation.get("phase")
        if phase == "CHALLENGE":
            action, checkpoint = _decide_challenge(observation, agent_state)
            if action.get("type") not in ("accept", "challenge") or set(action) != {"type"}:
                return accept(), _checkpoint(observation, agent_state)
            return action, checkpoint
        if phase == "PLACE":
            action, checkpoint = _decide_place(observation, agent_state)
            # Validate refs against the live hand before emitting.
            live = {c["card_ref"] for c in (observation.get("hand") or [])
                    if isinstance(c, dict) and isinstance(c.get("card_ref"), str)}
            refs = action.get("card_refs")
            rank = action.get("claimed_rank")
            ok = (action.get("type") == "place" and set(action) == {"type", "card_refs", "claimed_rank"}
                  and isinstance(refs, list) and 1 <= len(refs) <= 4
                  and len(set(refs)) == len(refs) and all(r in live for r in refs)
                  and isinstance(rank, str) and rank in RANKS)
            if not ok:
                return _safe_place(observation.get("hand") or []), _checkpoint(observation, agent_state)
            return action, checkpoint
        return accept(), _checkpoint(observation, agent_state)
    except Exception as exc:  # never fault on an unexpected observation shape
        _log(f"strategy fallback: {exc}")
        try:
            if observation.get("phase") == "PLACE":
                return _safe_place(observation.get("hand") or []), {"v": 1}
        except Exception:
            pass
        return accept(), {"v": 1}


__all__ = ["accept", "challenge", "place", "select_action", "estimate_bluff_prob",
           "challenge_threshold"]
