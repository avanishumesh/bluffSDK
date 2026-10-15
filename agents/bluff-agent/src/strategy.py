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
  PLACE: shed a largest truthful group; mix a bounded two-card semi-bluff
    only from singleton-heavy hands while pressure conditions are favorable.
  CHALLENGE: call impossible claims; estimate truth from selectively revealed
    claims and model clockwise challenge priority. Protect small hands and
    avoid treating a large pile alone as evidence of dishonesty.
"""

from __future__ import annotations

import hashlib
import random
import secrets
import sys
from collections import defaultdict
from typing import Any

RANKS: tuple[str, ...] = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
RANK_INDEX = {rank: i for i, rank in enumerate(RANKS)}

# Base rates were retained after broad baseline tuning. Publicly observed
# challenge behavior now adjusts pressure per opponent; fresh-seed adversarial
# evaluation supported enabling that response without raising fixed bluff rates.
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
    "adaptive_pressure": True,
    "challenge_prior_rate": 0.18,
    "challenge_prior_weight": 4.0,
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
# Private checkpointed randomness: stable across retries, opaque to opponents.
# ---------------------------------------------------------------------------

def _hash_rng(seed: int, *parts: object) -> random.Random:
    key = seed.to_bytes(16, "big", signed=False)
    digest = hashlib.blake2b(
        repr(parts).encode("utf-8"), key=key, digest_size=16
    ).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _state_seed(agent_state: Any) -> int:
    if isinstance(agent_state, dict):
        seed = agent_state.get("rng_seed")
        if type(seed) is int and 0 <= seed < 2**128:
            return seed
    return secrets.randbits(128)


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
        stats[seat] = {
            "claims": 0,
            "claims_by_count": {},
            "resolved": 0,
            "bluffs": 0,
            "truths": 0,
            "challenges": 0,
            "challenge_wins": 0,
            "challenge_known": 0,
            "challenge_accepts": 0,
            "by_count": {},
        }
    claims = {}
    claim_priority: dict[Any, list[int]] = {}
    active = set(range(players))
    for event in public_history or []:
        etype = event.get("type")
        if etype == "game_start":
            seats = event.get("active_seats")
            if isinstance(seats, list):
                active = {seat for seat in seats if type(seat) is int and seat in stats}
        elif etype == "claim":
            seat = event.get("seat")
            if isinstance(seat, int) and seat in stats:
                stats[seat]["claims"] += 1
                claim_id = event.get("claim_id")
                claims[claim_id] = event
                claim_priority[claim_id] = [
                    candidate
                    for offset in range(1, players)
                    if (candidate := (seat + offset) % players) in active
                ]
                count = event.get("count")
                if isinstance(count, int):
                    counts = stats[seat]["claims_by_count"]
                    counts[count] = counts.get(count, 0) + 1
                rank = event.get("claimed_rank")
        elif etype == "claim_accepted":
            for seat in claim_priority.get(event.get("claim_id"), []):
                stats[seat]["challenge_known"] += 1
                stats[seat]["challenge_accepts"] += 1
        elif etype == "challenge_resolved":
            placer = event.get("placer")
            truthful = event.get("truthful")
            challenger = event.get("challenger")
            order = claim_priority.get(event.get("claim_id"), [])
            if isinstance(challenger, int) and challenger in order:
                for seat in order:
                    stats[seat]["challenge_known"] += 1
                    if seat == challenger:
                        stats[seat]["challenges"] += 1
                        break
                    stats[seat]["challenge_accepts"] += 1
            elif isinstance(challenger, int) and challenger in stats:
                # Tolerate compact/synthetic histories without a matching claim.
                stats[challenger]["challenge_known"] += 1
                stats[challenger]["challenges"] += 1
            if isinstance(challenger, int) and challenger in stats:
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
        elif etype == "finish":
            seat = event.get("seat")
            if type(seat) is int:
                active.discard(seat)
    return stats


def _bluff_rate(stats: dict[str, Any]) -> float:
    """Shrinkage-smoothed bluff rate so 1-2 samples do not over-swing."""
    n = stats["resolved"]
    prior = POLICY_PARAMS["prior_bluff_rate"]
    weight = POLICY_PARAMS["prior_weight"]
    sampled_rate = (stats["bluffs"] + prior * weight) / (n + weight)
    # Challenges are selective: a revealed batch is not a random sample of
    # the opponent's claims. Trust it less when most of their claims stayed
    # hidden and therefore have unknown truth.
    coverage = n / max(stats["claims"], n, 1)
    return prior + coverage * (sampled_rate - prior)


def _challenge_rate(stats: dict[str, Any]) -> float:
    prior = POLICY_PARAMS["challenge_prior_rate"]
    weight = POLICY_PARAMS["challenge_prior_weight"]
    known = stats["challenge_known"]
    return (stats["challenges"] + prior * weight) / (known + weight)


def _challenge_selection_probs(
    observation: dict[str, Any], profile: dict[int, dict[str, Any]]
) -> tuple[float, float]:
    """Return P(no earlier challenge), P(no other challenge), respectively."""
    priority = observation.get("challenge_priority")
    seat = observation.get("seat")
    if not isinstance(priority, list) or seat not in priority:
        return 1.0, 1.0
    index = priority.index(seat)
    earlier = priority[:index]
    later = priority[index + 1 :]

    def no_challenge(seats: list[Any]) -> float:
        probability = 1.0
        for other in seats:
            if type(other) is int and other in profile:
                probability *= 1.0 - _challenge_rate(profile[other])
        return probability

    no_earlier = no_challenge(earlier)
    return no_earlier, no_earlier * no_challenge(later)


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
        bucket_rate = (bluffs + 3.0 * overall + 2.0 * prior) / (samples + 5.0)
        bucket_claims = st["claims_by_count"].get(count, samples)
        coverage = samples / max(bucket_claims, samples, 1)
        prob = prior + coverage * (bucket_rate - prior)
    prob += 0.025 * held
    return max(0.02, min(0.99, prob))


def challenge_threshold(observation: dict[str, Any], claim: dict[str, Any],
                        our_hand: int, no_earlier: float = 1.0,
                        no_other: float = 1.0) -> float:
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
    threshold = max(0.40, min(0.85, threshold))
    # A challenge only changes the result if nobody else challenges. If an
    # earlier seat calls, we cannot be selected; later calls reduce the benefit
    # of acting when the claim is false while preserving our own pickup risk.
    no_earlier = max(0.0, min(1.0, no_earlier))
    no_other = max(0.0, min(no_earlier, no_other))
    denominator = threshold * no_earlier + (1.0 - threshold) * no_other
    if denominator > 0.0:
        threshold = threshold * no_earlier / denominator
    else:
        # An earlier player is certain to challenge, so our choice cannot
        # affect the outcome; use an even threshold for the marginal mixer.
        threshold = 0.50
    return max(0.01, min(0.99, threshold))


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
    no_earlier, no_other = _challenge_selection_probs(observation, profile)
    threshold = challenge_threshold(
        observation, claim, len(hand), no_earlier=no_earlier, no_other=no_other
    )
    if prob >= threshold + 0.06:
        return challenge(), _checkpoint(observation, agent_state)
    if prob < threshold - 0.06:
        return accept(), _checkpoint(observation, agent_state)
    # Marginal spot: private seeded coin-flip weighted by edge. The seed is
    # checkpointed so retries are stable but opponents cannot predict it.
    rng = _hash_rng(agent_state["rng_seed"], "challenge", observation.get("turn_number"),
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


def _best_rank(groups: dict[str, list[dict[str, Any]]], rng: random.Random) -> str:
    largest = max(len(cards) for cards in groups.values())
    tied = [rank for rank, cards in groups.items() if len(cards) == largest]
    return rng.choice(tied)


def _pressure_multiplier(
    profile: dict[int, dict[str, Any]], seat: int,
    active_seats: set[int] | None = None,
) -> float:
    """Adjust bluff pressure to the opponents who have actually shown a style.

    Each opponent's challenge rate is smoothed toward the table prior. A
    challenge reduces pressure; repeated accepts can increase it slightly.
    Unobserved seats therefore have no effect, and finished players are
    excluded. This uses only public challenge/accept events.
    """
    prior = POLICY_PARAMS["challenge_prior_rate"]
    weight = POLICY_PARAMS["challenge_prior_weight"]
    relative_survival = 1.0
    for other, stats in profile.items():
        if other == seat or (active_seats is not None and other not in active_seats):
            continue
        known = stats["challenge_known"]
        if known <= 0:
            continue
        rate = (stats["challenges"] + prior * weight) / (known + weight)
        relative_survival *= (1.0 - rate) / (1.0 - prior)
    return max(0.30, min(1.20, relative_survival))


def _decide_place(observation: dict[str, Any], agent_state: Any) -> tuple[dict[str, Any], Any]:
    hand = [c for c in (observation.get("hand") or []) if isinstance(c, dict) and isinstance(c.get("card_ref"), str)]
    if not hand:
        # Engine never queries finished seats; stay quiet and take the fault.
        return accept(), _checkpoint(observation, agent_state)
    groups = _groups(hand)
    if not groups:
        return accept(), _checkpoint(observation, agent_state)
    pile_size = int(observation.get("pile_size", 0) or 0)
    hand_sizes = observation.get("hand_sizes") or []
    try:
        seat = int(observation.get("seat", -1))
        active_seats = observation.get("active_seats")
        if isinstance(active_seats, list):
            active = {item for item in active_seats if type(item) is int}
        else:
            active = {
                index for index, size in enumerate(hand_sizes)
                if isinstance(size, int) and size > 0
            }
        opponent_sizes = [
            int(size) for index, size in enumerate(hand_sizes)
            if index != seat and index in active
        ]
        min_opponent = min(opponent_sizes) if opponent_sizes else 13
    except (TypeError, ValueError):
        min_opponent = 13
        seat = -1
    seed = _state_seed(agent_state)
    rank_rng = _hash_rng(seed, "rank", observation.get("turn_number"), seat,
                         pile_size, sorted(card.get("rank", "?") for card in hand))
    best = _best_rank(groups, rank_rng)
    profile = _opp_profile(observation.get("public_history") or [], len(hand_sizes) or 4)

    # Limit pressure bluffs to singleton-heavy hands. The broader pair/triple
    # extension reduced win rate in paired local evaluation, so keep truthful
    # groups intact and only mix the established two-card semi-bluff.
    if (len(groups[best]) == 1 and len(hand) >= POLICY_PARAMS["pressure_min_hand"]
            and pile_size <= POLICY_PARAMS["pressure_max_pile"]
            and min_opponent >= POLICY_PARAMS["pressure_min_opponent_hand"]):
        rng = _hash_rng(seed, "pressure", observation.get("turn_number"), observation.get("seat"),
                        pile_size, sorted(card.get("rank", "?") for card in hand))
        pressure_rate = (POLICY_PARAMS["pressure_small_pile_rate"] if pile_size <= 3
                         else POLICY_PARAMS["pressure_medium_pile_rate"])
        if POLICY_PARAMS["adaptive_pressure"]:
            pressure_rate *= _pressure_multiplier(profile, seat, active)
        if rng.random() < pressure_rate:
            # One true anchor card makes the lie less exposed than a pure
            # bluff, while keeping the number of cards shed competitive.
            anchor = groups[best][0]
            kickers = [card for card in hand if card["card_ref"] != anchor["card_ref"]]
            if kickers:
                kicker = rng.choice(kickers)
                return place([anchor["card_ref"], kicker["card_ref"]], best), _checkpoint(
                    observation, agent_state)

    # Truthful groups reduce the number of remaining placement turns without
    # exposing an accumulated pile to a bluff penalty.  We never bluff a final
    # batch because a single challenge would erase the near-finish.
    batch = groups[best][:4]
    return place([c["card_ref"] for c in batch], best), _checkpoint(
        observation, agent_state)


# ---------------------------------------------------------------------------
# Checkpoint: tiny JSON snapshot of tracked history (recomputed statelessly)
# ---------------------------------------------------------------------------

def _checkpoint(observation: dict[str, Any], agent_state: Any) -> dict[str, Any]:
    try:
        return {"v": 2, "rng_seed": _state_seed(agent_state)}
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
    """Mixed shedding with priority-aware, evidence-based challenges."""
    try:
        agent_state = {"v": 2, "rng_seed": _state_seed(agent_state)}
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
