"""AnyRank-owned validation for mirrored inference modes."""

ANYRANK_INFERENCE_MODE_CHOICES = ("ranking", "ranking_no_ties")
ANYRANK_INFERENCE_MODE_SET = frozenset(ANYRANK_INFERENCE_MODE_CHOICES)


def validate_inference_mode(inference_mode):
    """Return a valid AnyRank inference mode or raise a user-facing error."""
    if inference_mode == "comparison":
        raise ValueError(
            "inference_mode='comparison' is valid for "
            "statistical_comparison_core and sequentialized-barnard-tests, "
            "but it is not applicable in AnyRank. AnyRank reports rankings "
            "and has no baseline policy. Use one of: "
            f"{', '.join(ANYRANK_INFERENCE_MODE_CHOICES)}. "
            'See README "Inference modes".'
        )
    if inference_mode not in ANYRANK_INFERENCE_MODE_SET:
        raise ValueError(
            f"Invalid AnyRank inference_mode {inference_mode!r}. "
            "Expected one of: "
            f"{', '.join(ANYRANK_INFERENCE_MODE_CHOICES)}."
        )
    return inference_mode
