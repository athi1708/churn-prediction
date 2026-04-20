"""
Business Insights Engine
Generates contextual, actionable insights instead of generic strings.
Each rule maps a feature + value range → a business-meaningful message.
"""

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class InsightRule:
    feature: str
    high_risk: Optional[Callable]
    medium_risk: Optional[Callable]
    messages: dict[str, str]


RULES: list[InsightRule] = [
    InsightRule(
        feature="tenure",
        high_risk=lambda v: v < 6,
        medium_risk=lambda v: 6 <= v < 18,
        messages={
            "high_risk":   "🚨 Critical window: customer is in their first 6 months — the highest-churn period. Trigger a personalised onboarding check-in call within 48 hours.",
            "medium_risk": "⚠️  Customer is in the 6–18 month zone — still vulnerable. A loyalty incentive or feature highlight campaign could lock in long-term commitment.",
            "low_risk":    "✅ Long-tenured customer (18+ months). Stable base. Best candidate for upsell or referral programme rather than retention spend.",
        },
    ),
    InsightRule(
        feature="Contract",
        high_risk=lambda v: v == 0,
        medium_risk=lambda v: v == 1,
        messages={
            "high_risk":   "🚨 Month-to-month contract — industry churn rate 42.7%. Offer a 15–20% discount for upgrading to an annual plan. Frame it as cost savings, not retention.",
            "medium_risk": "⚠️  One-year contract. Send a renewal offer 60 days before expiry. Highlight new features added since signup.",
            "low_risk":    "✅ Two-year contract holder. Lowest churn risk segment (2.8%). Great candidate for premium add-on upsell.",
        },
    ),
    InsightRule(
        feature="MonthlyCharges",
        high_risk=lambda v: v > 85,
        medium_risk=lambda v: 65 < v <= 85,
        messages={
            "high_risk":   "🚨 Monthly bill is in the top 20% ($85+). Price sensitivity is likely the primary driver. Consider a targeted bundle discount or loyalty credit before they start shopping competitors.",
            "medium_risk": "⚠️  Mid-range charges ($65–$85). Proactively communicate the value they receive — ROI messaging reduces price-driven churn by ~18%.",
            "low_risk":    "✅ Low monthly charges. Customer is on a budget plan — unlikely to churn due to price, but low LTV. Target for service upgrades.",
        },
    ),
    InsightRule(
        feature="InternetService",
        high_risk=lambda v: v == 1,
        medium_risk=lambda v: v == 0,
        messages={
            "high_risk":   "🚨 Fiber optic customer — 41% higher churn than DSL, often driven by unmet speed expectations or bill shock. Proactively send a service quality check and consider a speed upgrade at current price.",
            "medium_risk": "⚠️  DSL customer. Moderate churn rate. Highlight reliability benefits; consider an upgrade pitch to fiber with introductory pricing.",
            "low_risk":    "✅ No internet service. Low engagement risk but also low ARPU. Could be a strong upsell target if digital habits are growing.",
        },
    ),
    InsightRule(
        feature="PaymentMethod",
        high_risk=lambda v: v == 0,
        medium_risk=lambda v: v == 1,
        messages={
            "high_risk":   "🚨 Electronic check payers churn 2× more than auto-pay users. Offer a $5/month discount to switch to bank transfer or credit card autopay — this single change can reduce churn risk by ~15%.",
            "medium_risk": "⚠️  Mailed check payment — manual process creates friction. Nudge to paperless/digital payment with a one-time incentive.",
            "low_risk":    "✅ Auto-pay enrolled. Strongest payment-related retention signal — 45% lower churn than manual payers.",
        },
    ),
    InsightRule(
        feature="TechSupport",
        high_risk=lambda v: v == 0,
        medium_risk=None,
        messages={
            "high_risk":   "🚨 No tech support subscription. Customers without it are 1.5× more likely to churn after a service issue. Offer a free 3-month trial — conversion to paid is 60%+ after trial.",
            "low_risk":    "✅ Tech support enrolled. Correlates with high satisfaction. Reduces frustration-driven churn significantly.",
        },
    ),
    InsightRule(
        feature="OnlineSecurity",
        high_risk=lambda v: v == 0,
        medium_risk=None,
        messages={
            "high_risk":   "⚠️  No online security. Strong upsell opportunity — adding this service reduces churn by ~15% and increases ARPU. Frame as 'protecting what matters' not 'selling an add-on'.",
            "low_risk":    "✅ Online security active. Associated with higher satisfaction and lower churn. Consider bundling with backup for loyalty reward.",
        },
    ),
    InsightRule(
        feature="PaperlessBilling",
        high_risk=lambda v: v == 1,
        medium_risk=None,
        messages={
            "high_risk":   "⚠️  Paperless billing correlates with higher churn — likely because digital-native customers are more comparison-shopping online. Ensure they receive personalised value emails, not generic newsletters.",
            "low_risk":    "✅ Paper billing customer. More passive, less likely to shop around. Stable but hard to upsell digitally.",
        },
    ),
    InsightRule(
        feature="SeniorCitizen",
        high_risk=lambda v: v == 1,
        medium_risk=None,
        messages={
            "high_risk":   "⚠️  Senior citizen — may experience friction with digital interfaces or complex billing. Assign to a dedicated support agent and simplify plan communication.",
            "low_risk":    "✅ Non-senior customer. Standard retention playbook applies.",
        },
    ),
    InsightRule(
        feature="NoSupportServices",
        high_risk=lambda v: v == 1,
        medium_risk=None,
        messages={
            "high_risk":   "🚨 Customer has neither online security nor tech support — the highest-risk service combination. A 'protection bundle' offer addressing both could significantly reduce churn.",
            "low_risk":    "✅ At least one support service active. Reduces vulnerability to service-issue-driven churn.",
        },
    ),
    InsightRule(
        feature="M2MNoSecurity",
        high_risk=lambda v: v == 1,
        medium_risk=None,
        messages={
            "high_risk":   "🚨 Month-to-month contract WITH no online security — the two highest individual churn factors combined. This customer requires urgent dual intervention: contract upgrade offer + security trial.",
            "low_risk":    "✅ Either contracted or has security service. Risk profile is within normal range.",
        },
    ),
    InsightRule(
        feature="HighValueAtRisk",
        high_risk=lambda v: v == 1,
        medium_risk=None,
        messages={
            "high_risk":   "🚨 High-value customer (monthly charges > $70) in the critical first 12 months — the worst revenue-at-risk combination. Escalate to senior retention team immediately. Potential LTV loss: $1,400–$2,100.",
            "low_risk":    "✅ Not in the high-value + early-tenure risk zone.",
        },
    ),
    InsightRule(
        feature="AutoPay",
        high_risk=lambda v: v == 0,
        medium_risk=None,
        messages={
            "high_risk":   "⚠️  Not enrolled in auto-pay. Manual payment customers show 35% higher churn. Incentivise auto-pay enrollment with a billing credit.",
            "low_risk":    "✅ Auto-pay active — one of the strongest retention signals available.",
        },
    ),
]

_RULE_MAP = {r.feature: r for r in RULES}


def _get_risk_level(rule: InsightRule, value) -> str:
    if rule.high_risk and rule.high_risk(value):
        return "high_risk"
    if rule.medium_risk and rule.medium_risk(value):
        return "medium_risk"
    return "low_risk"


def generate_insights(data: dict, top_reasons: list[tuple]) -> list[str]:
    """
    data        — original prediction input dict (numeric encoded values)
    top_reasons — [(feature_name, shap_value), ...] from explain_prediction()
    Returns list of business-context insight strings.
    """
    insights = []

    for feature, shap_val in top_reasons:
        value = data.get(feature, 0)
        rule  = _RULE_MAP.get(feature)

        if rule:
            level   = _get_risk_level(rule, value)
            message = rule.messages.get(level)
            if message:
                insights.append(message)
                continue

        # Fallback for engineered features not in rules
        direction = "increases" if shap_val > 0 else "reduces"
        impact    = abs(shap_val)
        insights.append(
            f"{'⚠️' if shap_val > 0 else '✅'} {feature.replace('_', ' ')} "
            f"{direction} churn risk (SHAP impact: {impact:.3f})"
        )

    return insights


def get_primary_action(data: dict, churn: int, probability: float) -> str:
    """Return the single most important recommended action."""
    if churn == 0:
        return "🟢 Customer is stable. Schedule a 90-day check-in and consider an upsell offer for an add-on service."

    p = probability * 100

    if data.get("Contract", 0) == 0 and p > 60:
        return "🔴 Priority 1: Offer 15–20% discount to upgrade to an annual contract within 24 hours. Assign dedicated account manager."
    if data.get("PaymentMethod", 0) == 0:
        return "🔴 Priority 1: Incentivise switch to auto-pay (offer $5/month credit). This alone reduces churn probability by ~15%."
    if data.get("tenure", 72) < 6:
        return "🔴 Priority 1: Trigger immediate onboarding success call. Customers in month 1–6 who receive proactive outreach churn 30% less."
    if data.get("MonthlyCharges", 0) > 85:
        return "🔴 Priority 1: Proactively offer a loyalty discount or bundle credit before customer starts price comparison."
    if p > 70:
        return "🔴 High urgency: Multi-channel retention campaign required within 48 hours — call + email + in-app."

    return "🟡 Moderate risk: Add to bi-weekly monitoring list. Send personalised value recap email within 7 days."