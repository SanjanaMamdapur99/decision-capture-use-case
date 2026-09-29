import unittest
from pathlib import Path

from decision_use_case.database import FinancialRepository


DATABASE = Path(__file__).parents[1] / "data" / "financial" / "financial.sqlite"


class FinancialRepositoryTest(unittest.TestCase):
    def test_target_profile_hides_outcome_and_is_grounded_in_real_data(self):
        with FinancialRepository(DATABASE) as repository:
            case = repository.get_case(4959)
            profile = repository.loan_profile(4959)

        self.assertEqual(case.outcome, "A")
        self.assertEqual(profile[0]["loan_id"], 4959)
        self.assertNotIn("status", profile[0])
        self.assertNotIn("gender", profile[0])
        self.assertEqual(profile[0]["item_id"], "loan:4959")

    def test_fairness_context_is_separated_from_risk_profile(self):
        with FinancialRepository(DATABASE) as repository:
            fairness = repository.fairness_context(5856)

        self.assertEqual(fairness[0]["item_id"], "protected-context:5856")
        self.assertIn(fairness[0]["gender"], {"M", "F"})
        self.assertTrue(all("item_id" in item for item in fairness))

    def test_transaction_summary_uses_only_information_available_before_loan(self):
        with FinancialRepository(DATABASE) as repository:
            summary = repository.transaction_summary(4959)

        self.assertEqual(summary[0]["item_id"], "transaction-summary:4959")
        self.assertLessEqual(summary[0]["latest_transaction_date"], summary[0]["loan_date"])
        self.assertGreater(summary[0]["transaction_count"], 0)

    def test_similar_loans_exclude_target_and_future_loans(self):
        with FinancialRepository(DATABASE) as repository:
            rows = repository.similar_completed_loans(4973, limit=5)
            target = repository.get_case(4973)

        self.assertTrue(rows)
        self.assertTrue(all(row["loan_id"] != 4973 for row in rows))
        self.assertTrue(all(row["loan_date"] < target.loan_date for row in rows))
        self.assertTrue(all(row["status"] in {"A", "B"} for row in rows))


if __name__ == "__main__":
    unittest.main()

