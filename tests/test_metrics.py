import unittest

from decision_use_case.metrics import compute_decision_metrics, evaluate_recommendation


class MetricsTest(unittest.TestCase):
    def test_decision_metrics_measure_grounding_and_score_margin(self):
        decision = {
            "candidates": [
                {"candidate_id": "low_risk", "status": "selected"},
                {"candidate_id": "high_risk", "status": "rejected"},
            ],
            "assessments": [
                {"candidate_id": "low_risk", "score": 0.8, "explanation": "supported"},
                {"candidate_id": "high_risk", "score": 0.3, "explanation": "weaker"},
            ],
            "selected_candidate_ids": ["low_risk"],
            "retrievals": [{"items": [{"item_id": "e1"}, {"item_id": "e2"}]}],
            "evidence_uses": [
                {"item_id": "e1", "used": True, "role": "supporting", "explanation": "relevant"},
                {"item_id": "e2", "used": False, "role": "irrelevant", "explanation": "not relevant"},
            ],
        }

        metrics = compute_decision_metrics(decision)

        self.assertEqual(metrics["grounding_coverage"], 1.0)
        self.assertEqual(metrics["evidence_utilization"], 0.5)
        self.assertEqual(metrics["score_margin"], 0.5)
        self.assertEqual(metrics["explanation_completeness"], 1.0)

    def test_recommendation_evaluation_handles_correct_and_abstaining_predictions(self):
        self.assertEqual(evaluate_recommendation("A", "approve")["correct"], 1)
        self.assertEqual(evaluate_recommendation("B", "decline")["correct"], 1)
        self.assertEqual(evaluate_recommendation("A", "manual_review")["abstained"], 1)


if __name__ == "__main__":
    unittest.main()

