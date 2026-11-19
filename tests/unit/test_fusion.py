"""Reciprocal Rank Fusion tests."""

from property_assistant.retrieval.fusion import reciprocal_rank_fusion


class TestReciprocalRankFusion:
    def test_single_ranking_preserves_order(self) -> None:
        fused = reciprocal_rank_fusion([["a", "b", "c"]])
        assert [item for item, _ in fused] == ["a", "b", "c"]

    def test_item_in_both_rankings_wins(self) -> None:
        fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
        assert fused[0][0] == "b"

    def test_scores_follow_the_rrf_formula(self) -> None:
        fused = dict(reciprocal_rank_fusion([["a"], ["a"]], k=60))
        assert fused["a"] == 2 * (1.0 / 61)

    def test_ties_break_by_first_appearance(self) -> None:
        fused = reciprocal_rank_fusion([["a"], ["b"]])
        assert [item for item, _ in fused] == ["a", "b"]

    def test_deterministic_across_calls(self) -> None:
        rankings = [["a", "b", "c"], ["c", "a"], ["b", "c", "a"]]
        assert reciprocal_rank_fusion(rankings) == reciprocal_rank_fusion(rankings)

    def test_empty_input(self) -> None:
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[]]) == []

    def test_k_dampens_rank_differences(self) -> None:
        low_k = dict(reciprocal_rank_fusion([["a", "b"]], k=1))
        high_k = dict(reciprocal_rank_fusion([["a", "b"]], k=1000))
        assert (low_k["a"] - low_k["b"]) > (high_k["a"] - high_k["b"])
