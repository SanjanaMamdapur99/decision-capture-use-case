"""Metrics for grounded decision-provenance experiments."""


def _as_dict(record) -> dict:
    if isinstance(record, dict):
        return record
    return record.to_dict()


def compute_decision_metrics(record) -> dict:
    """Compute auditable structural and grounding metrics for one decision."""
    decision = _as_dict(record)
    retrievals = decision.get("retrievals") or []
    items = [item for retrieval in retrievals for item in retrieval.get("items", retrieval.get("retrieved", []))]
    uses = decision.get("evidence_uses") or []
    reported = {item.get("item_id") for item in uses}
    retrieved = {item.get("item_id") for item in items}
    kept = [item for item in uses if item.get("used")]
    assessments = decision.get("assessments") or []
    selected = set(decision.get("selected_candidate_ids") or [])
    selected_scores = [item.get("score") for item in assessments if item.get("candidate_id") in selected]
    rejected_scores = [item.get("score") for item in assessments if item.get("candidate_id") not in selected]
    selected_score = max((score for score in selected_scores if score is not None), default=None)
    rejected_score = max((score for score in rejected_scores if score is not None), default=None)
    explanations = uses + assessments
    return {
        "candidate_count": len(decision.get("candidates") or []),
        "retrieved_count": len(retrieved),
        "reported_evidence_count": len(reported & retrieved),
        "kept_evidence_count": len(kept),
        "grounding_coverage": len(reported & retrieved) / len(retrieved) if retrieved else None,
        "evidence_utilization": len(kept) / len(retrieved) if retrieved else None,
        "contradicting_evidence_count": sum(
            1 for item in kept if item.get("role") == "contradicting"
        ),
        "explanation_completeness": (
            sum(bool(item.get("explanation")) for item in explanations) / len(explanations)
            if explanations
            else None
        ),
        "selected_score": selected_score,
        "score_margin": (
            round(selected_score - rejected_score, 6)
            if selected_score is not None and rejected_score is not None
            else None
        ),
        "unreported_item_count": len(retrieved - reported),
    }


def evaluate_recommendation(outcome: str, recommendation: str) -> dict:
    """Score a committee recommendation against a completed-loan outcome."""
    if outcome not in {"A", "B"}:
        return {"evaluable": 0, "correct": None, "abstained": int(recommendation == "manual_review")}
    abstained = recommendation == "manual_review"
    expected = "approve" if outcome == "A" else "decline"
    return {
        "evaluable": 1,
        "correct": int(not abstained and recommendation == expected),
        "abstained": int(abstained),
        "expected_recommendation": expected,
    }
