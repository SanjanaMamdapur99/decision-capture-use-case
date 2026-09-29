"""Read-only access to the financial benchmark database."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoanCase:
    """A historical loan and its hidden evaluation outcome."""

    loan_id: int
    account_id: int
    loan_date: str
    outcome: str


class FinancialRepository:
    """Query evidence without exposing target outcomes to decision agents."""

    def __init__(self, database_path: str | Path):
        path = Path(database_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        self.connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        self.connection.row_factory = sqlite3.Row

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def close(self) -> None:
        """Close the read-only database connection."""
        self.connection.close()

    def _one(self, sql: str, parameters: tuple) -> dict:
        row = self.connection.execute(sql, parameters).fetchone()
        if row is None:
            raise ValueError(f"No financial record found for parameters {parameters}")
        return dict(row)

    def _many(self, sql: str, parameters: tuple = ()) -> list[dict]:
        return [dict(row) for row in self.connection.execute(sql, parameters).fetchall()]

    def get_case(self, loan_id: int) -> LoanCase:
        """Return target metadata, including the outcome reserved for evaluation."""
        row = self._one(
            "SELECT loan_id, account_id, date AS loan_date, status FROM loan WHERE loan_id = ?",
            (loan_id,),
        )
        return LoanCase(row["loan_id"], row["account_id"], row["loan_date"], row["status"])

    def loan_profile(self, loan_id: int) -> list[dict]:
        """Return non-sensitive application and district evidence, excluding outcome."""
        row = self._one(
            """
            SELECT l.loan_id, l.account_id, l.date AS loan_date, l.amount, l.duration,
                   l.payments AS monthly_payment, a.date AS account_opened,
                   CAST((julianday(l.date) - julianday(a.date)) / 30.44 AS INTEGER) AS account_age_months,
                   a.frequency AS statement_frequency, d.A2 AS district_name, d.A3 AS region,
                   CAST(d.A4 AS INTEGER) AS district_population, d.A10 AS urban_ratio,
                   d.A11 AS average_salary, d.A12 AS unemployment_1995,
                   d.A13 AS unemployment_1996
            FROM loan l
            JOIN account a ON a.account_id = l.account_id
            JOIN district d ON d.district_id = a.district_id
            WHERE l.loan_id = ?
            """,
            (loan_id,),
        )
        row["item_id"] = f"loan:{loan_id}"
        return [row]

    def transaction_summary(self, loan_id: int) -> list[dict]:
        """Summarize the account's trailing-year behavior before the loan date."""
        row = self._one(
            """
            WITH target AS (
                SELECT loan_id, account_id, date AS loan_date FROM loan WHERE loan_id = ?
            ), eligible AS (
                SELECT t.*, target.loan_date
                FROM trans t JOIN target ON target.account_id = t.account_id
                WHERE t.date < target.loan_date
                  AND t.date >= date(target.loan_date, '-365 day')
            )
            SELECT target.loan_id, target.loan_date, COUNT(eligible.trans_id) AS transaction_count,
                   MIN(eligible.date) AS earliest_transaction_date,
                   MAX(eligible.date) AS latest_transaction_date,
                   ROUND(AVG(eligible.balance), 2) AS average_balance,
                   MIN(eligible.balance) AS minimum_balance,
                   MAX(eligible.balance) AS maximum_balance,
                   COALESCE(SUM(CASE WHEN eligible.type = 'PRIJEM' THEN eligible.amount ELSE 0 END), 0) AS inflow,
                   COALESCE(SUM(CASE WHEN eligible.type = 'VYDAJ' THEN eligible.amount ELSE 0 END), 0) AS outflow,
                   COALESCE(SUM(CASE WHEN eligible.k_symbol = 'SANKC. UROK' THEN 1 ELSE 0 END), 0) AS penalty_events
            FROM target LEFT JOIN eligible ON 1 = 1
            GROUP BY target.loan_id, target.loan_date
            """,
            (loan_id,),
        )
        row["item_id"] = f"transaction-summary:{loan_id}"
        return [row]

    def recent_transactions(self, loan_id: int, limit: int = 12) -> list[dict]:
        """Return the latest transactions known when the historical loan was issued."""
        rows = self._many(
            """
            SELECT t.trans_id, t.date, t.type, t.operation, t.amount, t.balance, t.k_symbol
            FROM trans t JOIN loan l ON l.account_id = t.account_id
            WHERE l.loan_id = ? AND t.date < l.date
            ORDER BY t.date DESC, t.trans_id DESC LIMIT ?
            """,
            (loan_id, limit),
        )
        for row in rows:
            row["item_id"] = f"transaction:{row['trans_id']}"
        return rows

    def standing_orders(self, loan_id: int) -> list[dict]:
        """Return recurring payment obligations associated with the account."""
        rows = self._many(
            """
            SELECT o.order_id, o.amount, o.k_symbol AS purpose, o.bank_to
            FROM `order` o JOIN loan l ON l.account_id = o.account_id
            WHERE l.loan_id = ? ORDER BY o.amount DESC, o.order_id
            """,
            (loan_id,),
        )
        for row in rows:
            row["item_id"] = f"standing-order:{row['order_id']}"
        return rows

    def similar_completed_loans(self, loan_id: int, limit: int = 8) -> list[dict]:
        """Return prior completed loans ranked by amount and duration similarity."""
        rows = self._many(
            """
            WITH target AS (
                SELECT loan_id, date, amount, duration FROM loan WHERE loan_id = ?
            )
            SELECT l.loan_id, l.date AS loan_date, l.amount, l.duration,
                   l.payments AS monthly_payment, l.status,
                   ROUND(ABS(l.amount - target.amount) * 1.0 / MAX(target.amount, 1)
                         + ABS(l.duration - target.duration) * 1.0 / MAX(target.duration, 1), 4) AS distance
            FROM loan l JOIN target
            WHERE l.loan_id != target.loan_id AND l.date < target.date AND l.status IN ('A', 'B')
            ORDER BY distance, l.date DESC LIMIT ?
            """,
            (loan_id, limit),
        )
        for row in rows:
            row["item_id"] = f"comparable-loan:{row['loan_id']}"
        return rows

    def fairness_context(self, loan_id: int) -> list[dict]:
        """Return protected-group context and prior outcome aggregates for auditing only."""
        target = self._one(
            """
            SELECT l.loan_id, l.date AS loan_date, c.gender, d.A3 AS region
            FROM loan l
            JOIN disp p ON p.account_id = l.account_id AND p.type = 'OWNER'
            JOIN client c ON c.client_id = p.client_id
            JOIN district d ON d.district_id = c.district_id
            WHERE l.loan_id = ?
            """,
            (loan_id,),
        )
        items = [
            {
                "item_id": f"protected-context:{loan_id}",
                "loan_id": loan_id,
                "gender": target["gender"],
                "region": target["region"],
                "note": "Protected attributes are supplied only to the fairness auditor, never the risk analyst.",
            }
        ]
        aggregates = self._many(
            """
            SELECT c.gender, d.A3 AS region, COUNT(*) AS completed_loans,
                   SUM(CASE WHEN l.status = 'A' THEN 1 ELSE 0 END) AS repaid,
                   ROUND(AVG(CASE WHEN l.status = 'A' THEN 1.0 ELSE 0.0 END), 4) AS repayment_rate
            FROM loan l
            JOIN disp p ON p.account_id = l.account_id AND p.type = 'OWNER'
            JOIN client c ON c.client_id = p.client_id
            JOIN district d ON d.district_id = c.district_id
            WHERE l.date < ? AND l.status IN ('A', 'B')
            GROUP BY c.gender, d.A3 ORDER BY c.gender, d.A3
            """,
            (target["loan_date"],),
        )
        for row in aggregates:
            row["item_id"] = f"group-history:{row['gender']}:{row['region']}"
        return items + aggregates
