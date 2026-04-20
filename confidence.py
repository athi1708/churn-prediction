"""
Prediction Confidence Interpreter
Confidence reflects:
  1. Distance from the decision boundary (0.5)
  2. SHAP directional agreement among top features
  3. Whether the customer profile is typical vs. outlier
"""

from dataclasses import dataclass


@dataclass
class ConfidenceResult:
    label: str
    score: float          # 0.0 – 1.0
    message: str
    recommendation: str
    color: str            # "red" | "orange" | "yellow" | "green" | "gray"
    icon: str             # font-awesome icon name without "fa-"


def interpret_confidence(
    probability: float,
    top_shap: list[tuple],
    churn: int,
) -> ConfidenceResult:
    """
    probability  — model output probability (0–1)
    top_shap     — [(feature, shap_value), ...]
    churn        — 0 or 1 (predicted class)
    """
    # Component 1: distance from boundary (0 at 0.5, 1 at 0.0 or 1.0)
    boundary_distance = abs(probability - 0.5) / 0.5  # 0–1

    # Component 2: SHAP directional agreement
    pred_positive = churn == 1
    if top_shap:
        agreeing = sum(
            1 for _, v in top_shap[:4]
            if (v > 0) == pred_positive
        )
        shap_agreement = agreeing / min(4, len(top_shap))
    else:
        shap_agreement = 0.5

    # Composite confidence score (weighted)
    conf_score = boundary_distance * 0.65 + shap_agreement * 0.35

    if conf_score >= 0.72:
        return ConfidenceResult(
            label="High Confidence",
            score=conf_score,
            message=(
                f"Model has strong conviction ({conf_score*100:.0f}% confidence). "
                "Multiple independent signals agree on this prediction."
            ),
            recommendation=(
                "Act on this prediction immediately — "
                "the risk assessment is reliable."
                if churn
                else "Customer is genuinely stable. Invest in upsell, not retention."
            ),
            color="red" if churn else "green",
            icon="check-circle" if not churn else "exclamation-circle",
        )

    if conf_score >= 0.45:
        return ConfidenceResult(
            label="Moderate Confidence",
            score=conf_score,
            message=(
                f"Reasonable confidence ({conf_score*100:.0f}%). "
                "Most signals agree but the customer profile has mixed characteristics."
            ),
            recommendation=(
                "Proceed with retention action but collect additional data "
                "(support ticket history, usage logs) for full picture."
            ),
            color="yellow",
            icon="exclamation-triangle",
        )

    return ConfidenceResult(
        label="Low Confidence",
        score=conf_score,
        message=(
            f"Low model confidence ({conf_score*100:.0f}%). "
            "This customer's profile is atypical — the model may be extrapolating."
        ),
        recommendation=(
            "Do not act on this prediction alone. "
            "Escalate to a human analyst for manual review."
        ),
        color="gray",
        icon="question-circle",
    )