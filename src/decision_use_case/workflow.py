"""Multi-agent, tool-grounded financial decision workflow."""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from flowcept import DecisionCapture, Flowcept, FlowceptTask, retrieval_scope
from langchain_openai import ChatOpenAI

from decision_use_case.config import ExperimentConfig, ModelSettings
from decision_use_case.database import FinancialRepository
from decision_use_case.metrics import compute_decision_metrics, evaluate_recommendation
from decision_use_case.tools import FinancialTools, build_financial_tools


RISK_CANDIDATES = ("low_risk", "high_risk", "manual_review")
FAIRNESS_CANDIDATES = ("acceptable", "bias_concern", "insufficient_evidence")
COMMITTEE_CANDIDATES = ("approve", "decline", "manual_review")


@dataclass(frozen=True)
class AgentResult:
    """One agent's captured decision and execution duration."""

    agent_id: str
    record: object
    duration_seconds: float


def build_model(settings: ModelSettings):
    """Build an Ollama, vLLM, or hosted OpenAI-compatible chat model."""
    api_key = os.getenv(settings.api_key_env, "not-required-for-local-endpoints")
    arguments = {
        "model": settings.name,
        "api_key": api_key,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
    }
    if settings.base_url:
        arguments["base_url"] = settings.base_url
    if settings.reasoning_effort:
        arguments["reasoning_effort"] = settings.reasoning_effort
    return ChatOpenAI(**arguments)


def _decision_context(role: str, question: str, candidate_ids: tuple[str, ...], model_name: str) -> dict:
    return {
        "research_role": role,
        "question": question,
        "model": model_name,
        "required_candidate_ids": list(candidate_ids),
        "instructions": [
            f"Return exactly these candidates and assess each one: {list(candidate_ids)}.",
            "Select exactly one candidate.",
            "Treat assessment scores as comparative confidence from 0 to 1, not calibrated probabilities.",
            "Base conclusions only on evidence marked used and explicitly acknowledge contradictory evidence.",
            "Do not infer facts absent from the supplied evidence.",
        ],
    }


def _record_summary(record) -> dict:
    selected = record.selected_candidate_ids[0]
    candidate = next(item for item in record.candidates if item.candidate_id == selected)
    assessment = next(item for item in record.assessments if item.candidate_id == selected)
    return {
        "selected_candidate_id": selected,
        "selected_candidate": candidate.content,
        "score": assessment.score,
        "explanation": assessment.explanation,
        "evidence_ids": assessment.evidence_ids,
    }


def _validate_record(record, candidate_ids: tuple[str, ...], agent_id: str) -> None:
    returned = {candidate.candidate_id for candidate in record.candidates}
    if returned != set(candidate_ids):
        raise ValueError(
            f"{agent_id} must return candidates {list(candidate_ids)}; returned {sorted(returned)}"
        )
    if len(record.selected_candidate_ids) != 1:
        raise ValueError(f"{agent_id} must select exactly one candidate")


def _invoke_agent(
    *,
    agent_id: str,
    role: str,
    question: str,
    request: str,
    candidate_ids: tuple[str, ...],
    evidence_tool_names: tuple[str, ...],
    tools: FinancialTools,
    loan_id: int,
    model,
    model_name: str,
    workflow_id: str,
    capture_mode: str,
    prompt_item_chars: int,
    input_entity_ids: list[str],
) -> AgentResult:
    """Run one controlled agent turn with or without retrieval provenance."""
    started = time.perf_counter()
    with FlowceptTask(
        activity_id=agent_id,
        agent_id=agent_id,
        workflow_id=workflow_id,
        used={"role": role, "loan_id": loan_id, "input_entity_ids": input_entity_ids},
        capture_telemetry=False,
    ) as agent_task:
        context = _decision_context(role, question, candidate_ids, model_name)
        functions = tools.captured if capture_mode == "provenance" else tools.baseline
        baseline_evidence = []

        scope = retrieval_scope(
            agent_id=agent_id,
            workflow_id=workflow_id,
            parent_task_id=agent_task.get_id(),
        )
        if capture_mode == "provenance":
            scope.__enter__()
        try:
            for tool_name in evidence_tool_names:
                results = functions[tool_name](loan_id=loan_id)
                if capture_mode == "baseline":
                    baseline_evidence.extend(results)

            complaint = None
            record = None
            for attempt in range(2):
                capture = DecisionCapture(
                    decision_type=f"financial_{agent_id.replace('-', '_')}",
                    context=context,
                    agent_id=agent_id,
                    workflow_id=workflow_id,
                    parent_task_id=agent_task.get_id(),
                    input_entity_ids=input_entity_ids,
                    output_entity_ids=[f"decision:{loan_id}:{agent_id}"],
                    llm=model,
                    retrievals=[] if capture_mode == "baseline" else None,
                    prompt_item_chars=prompt_item_chars,
                )
                message = request
                if baseline_evidence:
                    message += "\n\nUNCAPTURED BASELINE EVIDENCE:\n" + json.dumps(
                        baseline_evidence, indent=2, default=str
                    )
                if complaint:
                    message += f"\n\nYour previous response was rejected: {complaint}"
                try:
                    record = capture.invoke(message)
                    _validate_record(record, candidate_ids, agent_id)
                    break
                except ValueError as error:
                    if attempt == 1:
                        raise
                    complaint = str(error)
        finally:
            if capture_mode == "provenance":
                scope.__exit__(None, None, None)

        duration = time.perf_counter() - started
        agent_task.end(
            generated={
                "selected_candidate_id": record.selected_candidate_ids[0],
                "decision_id": record.decision_id,
                "duration_seconds": duration,
            }
        )
    return AgentResult(agent_id=agent_id, record=record, duration_seconds=duration)


def run_case(config: ExperimentConfig, loan_id: int, model=None) -> dict:
    """Run the three-agent benchmark for one historical loan."""
    model = model or build_model(config.model)
    with FinancialRepository(config.database_path) as repository:
        case = repository.get_case(loan_id)
        tools = build_financial_tools(
            repository,
            config.workflow.recent_transaction_limit,
            config.workflow.comparable_loan_limit,
        )
        with Flowcept(
            workflow_name=f"Financial Decision Provenance ({config.workflow.capture_mode})",
            workflow_args={
                "loan_id": loan_id,
                "model": config.model.name,
                "capture_mode": config.workflow.capture_mode,
            },
            start_persistence=config.workflow.start_persistence,
            check_safe_stops=False,
        ) as flowcept:
            workflow_id = flowcept.current_workflow_id
            risk = _invoke_agent(
                agent_id="risk-analyst",
                role="financial risk analyst",
                question="Which risk disposition is supported by pre-decision financial evidence?",
                request=(
                    "Assess repayment risk for this historical application without using its hidden outcome. "
                    "Weigh affordability, account behavior, obligations, and prior comparable cases."
                ),
                candidate_ids=RISK_CANDIDATES,
                evidence_tool_names=(
                    "loan_profile",
                    "transaction_summary",
                    "recent_transactions",
                    "standing_orders",
                    "similar_completed_loans",
                ),
                tools=tools,
                loan_id=loan_id,
                model=model,
                model_name=config.model.name,
                workflow_id=workflow_id,
                capture_mode=config.workflow.capture_mode,
                prompt_item_chars=config.workflow.prompt_item_chars,
                input_entity_ids=[f"loan:{loan_id}"],
            )
            fairness = _invoke_agent(
                agent_id="fairness-auditor",
                role="independent fairness and governance auditor",
                question="Does the proposed risk assessment require a fairness intervention?",
                request=(
                    "Audit the risk recommendation below. Protected attributes must not determine credit risk; "
                    "use group history only to identify representational limitations or disparate-risk concerns.\n"
                    + json.dumps(_record_summary(risk.record), indent=2, default=str)
                ),
                candidate_ids=FAIRNESS_CANDIDATES,
                evidence_tool_names=("fairness_context",),
                tools=tools,
                loan_id=loan_id,
                model=model,
                model_name=config.model.name,
                workflow_id=workflow_id,
                capture_mode=config.workflow.capture_mode,
                prompt_item_chars=config.workflow.prompt_item_chars,
                input_entity_ids=[f"decision:{loan_id}:risk-analyst"],
            )
            committee = _invoke_agent(
                agent_id="credit-committee",
                role="decision committee synthesizer",
                question="Which non-binding lending recommendation follows from the evidence and audit?",
                request=(
                    "Issue a retrospective research recommendation. Choose manual review whenever evidence is "
                    "insufficient or the fairness audit identifies a concern.\n\nRISK ASSESSMENT:\n"
                    + json.dumps(_record_summary(risk.record), indent=2, default=str)
                    + "\n\nFAIRNESS AUDIT:\n"
                    + json.dumps(_record_summary(fairness.record), indent=2, default=str)
                ),
                candidate_ids=COMMITTEE_CANDIDATES,
                evidence_tool_names=("loan_profile", "transaction_summary", "similar_completed_loans"),
                tools=tools,
                loan_id=loan_id,
                model=model,
                model_name=config.model.name,
                workflow_id=workflow_id,
                capture_mode=config.workflow.capture_mode,
                prompt_item_chars=config.workflow.prompt_item_chars,
                input_entity_ids=[
                    f"decision:{loan_id}:risk-analyst",
                    f"decision:{loan_id}:fairness-auditor",
                ],
            )

    agents = [risk, fairness, committee]
    recommendation = committee.record.selected_candidate_ids[0]
    return {
        "workflow_id": workflow_id,
        "loan_id": loan_id,
        "model": config.model.name,
        "capture_mode": config.workflow.capture_mode,
        "hidden_outcome": case.outcome,
        "recommendation": recommendation,
        "evaluation": evaluate_recommendation(case.outcome, recommendation),
        "agents": [
            {
                "agent_id": result.agent_id,
                "duration_seconds": round(result.duration_seconds, 6),
                "decision": result.record.to_dict(),
                "metrics": compute_decision_metrics(result.record),
            }
            for result in agents
        ],
        "total_agent_duration_seconds": round(sum(result.duration_seconds for result in agents), 6),
    }
