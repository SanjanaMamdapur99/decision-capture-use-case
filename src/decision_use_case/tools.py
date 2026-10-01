"""Flowcept-instrumented evidence tools for the financial workflow."""

from dataclasses import dataclass
from typing import Callable

from flowcept import flowcept_tool
from pydantic import BaseModel, Field

from decision_use_case.database import FinancialRepository


@dataclass(frozen=True)
class FinancialTools:
    """Captured tool functions and matching uncaptured baseline functions."""

    captured: dict[str, Callable]
    baseline: dict[str, Callable]


class LoanProfileTool(BaseModel):
    """Read the target loan's non-sensitive application and district profile."""

    loan_id: int = Field(description="The target historical loan ID from the agent task.")


class TransactionSummaryTool(BaseModel):
    """Read aggregate account behavior from the year before the loan application."""

    loan_id: int = Field(description="The target historical loan ID from the agent task.")


class RecentTransactionsTool(BaseModel):
    """Read recent account transactions preceding the target loan application."""

    loan_id: int = Field(description="The target historical loan ID from the agent task.")
    limit: int = Field(default=12, ge=1, description="Maximum number of recent transactions requested.")


class StandingOrdersTool(BaseModel):
    """Read recurring payment obligations associated with the target account."""

    loan_id: int = Field(description="The target historical loan ID from the agent task.")


class SimilarCompletedLoansTool(BaseModel):
    """Read earlier completed loans ranked by amount and duration similarity."""

    loan_id: int = Field(description="The target historical loan ID from the agent task.")
    limit: int = Field(default=8, ge=1, description="Maximum number of comparable loans requested.")


class FairnessContextTool(BaseModel):
    """Read protected-group context and historical aggregates for a fairness audit only."""

    loan_id: int = Field(description="The target historical loan ID from the agent task.")


TOOL_SCHEMAS = {
    "loan_profile": LoanProfileTool,
    "transaction_summary": TransactionSummaryTool,
    "recent_transactions": RecentTransactionsTool,
    "standing_orders": StandingOrdersTool,
    "similar_completed_loans": SimilarCompletedLoansTool,
    "fairness_context": FairnessContextTool,
}

SCHEMA_TO_TOOL = {schema.__name__: name for name, schema in TOOL_SCHEMAS.items()}


def _tool(capture: bool, **decorator_kwargs):
    """Apply Flowcept capture without changing the executable tool contract."""

    def decorator(function):
        return flowcept_tool(**decorator_kwargs)(function) if capture else function

    return decorator


def _make_tools(
    repository: FinancialRepository,
    target_loan_id: int,
    recent_transaction_limit: int,
    comparable_loan_limit: int,
    *,
    capture: bool,
) -> dict[str, Callable]:
    """Build role-selectable tools whose arguments are authored by the model."""

    def reject_wrong_target(loan_id: int) -> list[dict] | None:
        if loan_id == target_loan_id:
            return None
        return [
            {
                "item_id": f"tool-error:wrong-loan:{loan_id}",
                "content": (
                    f"Refused loan_id {loan_id}; this agent turn is restricted to target loan_id "
                    f"{target_loan_id}."
                ),
            }
        ]

    @_tool(
        capture,
        tool_name="loan_profile",
        tool_type="database",
        query_method="loan_id",
        query_arg="loan_id",
    )
    def loan_profile(loan_id: int) -> list[dict]:
        """Query the non-sensitive application and district profile for a loan."""
        refusal = reject_wrong_target(loan_id)
        return refusal or repository.loan_profile(loan_id)

    @_tool(
        capture,
        tool_name="transaction_summary",
        tool_type="database",
        query_method="loan_id",
        query_arg="loan_id",
    )
    def transaction_summary(loan_id: int) -> list[dict]:
        """Query trailing-year account aggregates available before the loan date."""
        refusal = reject_wrong_target(loan_id)
        return refusal or repository.transaction_summary(loan_id)

    @_tool(
        capture,
        tool_name="recent_transactions",
        tool_type="database",
        query_method="loan_id",
        query_arg="loan_id",
    )
    def recent_transactions(loan_id: int, limit: int = 12) -> list[dict]:
        """Query recent transactions available before the loan date."""
        refusal = reject_wrong_target(loan_id)
        return refusal or repository.recent_transactions(loan_id, min(limit, recent_transaction_limit))

    @_tool(
        capture,
        tool_name="standing_orders",
        tool_type="database",
        query_method="loan_id",
        query_arg="loan_id",
    )
    def standing_orders(loan_id: int) -> list[dict]:
        """Query recurring obligations for the target account."""
        refusal = reject_wrong_target(loan_id)
        return refusal or repository.standing_orders(loan_id)

    @_tool(
        capture,
        tool_name="similar_completed_loans",
        tool_type="database",
        query_method="loan_id",
        query_arg="loan_id",
    )
    def similar_completed_loans(loan_id: int, limit: int = 8) -> list[dict]:
        """Query comparable completed loans that predate the target loan."""
        refusal = reject_wrong_target(loan_id)
        return refusal or repository.similar_completed_loans(loan_id, min(limit, comparable_loan_limit))

    @_tool(
        capture,
        tool_name="fairness_context",
        tool_type="database",
        query_method="loan_id",
        query_arg="loan_id",
    )
    def fairness_context(loan_id: int) -> list[dict]:
        """Query protected context and prior group aggregates for fairness auditing."""
        refusal = reject_wrong_target(loan_id)
        return refusal or repository.fairness_context(loan_id)

    return {
        "loan_profile": loan_profile,
        "transaction_summary": transaction_summary,
        "recent_transactions": recent_transactions,
        "standing_orders": standing_orders,
        "similar_completed_loans": similar_completed_loans,
        "fairness_context": fairness_context,
    }


def build_financial_tools(
    repository: FinancialRepository,
    target_loan_id: int,
    recent_transaction_limit: int,
    comparable_loan_limit: int,
) -> FinancialTools:
    """Create identical captured and baseline tools for one target loan."""
    arguments = (repository, target_loan_id, recent_transaction_limit, comparable_loan_limit)
    return FinancialTools(
        captured=_make_tools(*arguments, capture=True),
        baseline=_make_tools(*arguments, capture=False),
    )
