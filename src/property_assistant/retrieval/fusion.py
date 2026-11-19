"""Reciprocal Rank Fusion."""


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse multiple ranked id lists into one, without score calibration.

    Each id's fused score is ``sum(1 / (k + rank))`` over the rankings it
    appears in, with ``rank`` starting at 1. Deterministic: ties break by
    first appearance order across the rankings.

    Args:
        rankings: Ranked id lists, best first.
        k: Dampening constant; larger values flatten rank differences.

    Returns:
        ``(id, fused_score)`` pairs sorted best first.
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            if item_id not in first_seen:
                first_seen[item_id] = order
                order += 1
    return sorted(scores.items(), key=lambda pair: (-pair[1], first_seen[pair[0]]))
