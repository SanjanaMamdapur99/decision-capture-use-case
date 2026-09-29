"""Flowcept-instrumented evidence tools for the financial workflow."""

from dataclasses import dataclass
from typing import Callable

from flowcept import flowcept_tool

from decision_use_case.database import FinancialRepository


@dataclass(frozen=True)
class FinancialTools:
    """Captured tool functions and matching uncaptured baseline functions."""

    captured: dict[str, Callable]
    baseline: dict[str, Callable]


def build_financial_tools(
    repository: FinancialRepository,
    recent_transaction_limit: int,
    comparable_loan_limit: int,
) -> FinancialTools:
    """Create tool functions bound to one read-only repository."""
    baseline = {
        "loan_profile": repository.loan_profile,
        "transaction_summary": repository.transaction_summary,
        "recent_transactions": lambda loan_id: repository.recent_transactions(
            loan_id, recent_transaction_limit
        ),
        "standing_orders": repository.standing_orders,
        "similar_completed_loans": lambda loan_id: repository.similar_completed_loans(
            loan_id, comparable_loan_limit
        ),
        "fairness_context": repository.fairness_context,
    }
    captured = {
        name: flowcept_tool(
            tool_name=name,
            tool_type="database",
            query_method="loan_id",
            query_arg="loan_id",
        )(function)
        for name, function in baseline.items()
    }
    return FinancialTools(captured=captured, baseline=baseline)
