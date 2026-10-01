"""Multi-agent, tool-grounded financial decision workflow."""

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from flowcept import DecisionCapture, Flowcept, FlowceptTask, retrieval_scope
from flowcept.instrumentation.flowcept_agent_task import FlowceptLLM
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from decision_use_case.config import ExperimentConfig, ModelSettings
from decision_use_case.database import FinancialRepository
from decision_use_case.metrics import compute_decision_metrics, evaluate_recommendation
from decision_use_case.tools import SCHEMA_TO_TOOL, TOOL_SCHEMAS, FinancialTools, build_financial_tools


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


def _choose_and_run_tools(
    *,
    request: str,
    available_tool_names: tuple[str, ...],
    functions: dict,
    loan_id: int,
    model,
    agent_id: str,
    workflow_id: str,
    parent_task_id: str,
    capture_mode: str,
) -> list[dict]:
    """Let the model author and execute calls from the agent's available tools."""
    schemas = [TOOL_SCHEMAS[name] for name in available_tool_names]
    planner = model.bind_tools(schemas)
    if capture_mode == "provenance":
        planner = FlowceptLLM(
            planner,
            agent_id=agent_id,
            workflow_id=workflow_id,
            parent_task_id=parent_task_id,
            return_response_object=True,
        )

    messages = [
        {
            "role": "system",
            "content": (
                "Choose and call the financial evidence tools needed for the task. "
                f"The target loan_id is {loan_id}; use exactly that value in every tool call. "
                "Call every source needed for a defensible decision, but do not make the decision yet."
            ),
        },
        {"role": "user", "content": request},
    ]
    evidence = []
    for attempt in range(2):
        response = planner.invoke(messages)
        calls = getattr(response, "tool_calls", None) or []
        failures = []
        for call in calls:
            tool_name = SCHEMA_TO_TOOL.get(call.get("name"))
            if tool_name not in available_tool_names:
                failures.append(f"Unavailable tool requested: {call.get('name')!r}")
                continue
            try:
                arguments = TOOL_SCHEMAS[tool_name].model_validate(call.get("args") or {}).model_dump()
            except ValidationError as error:
                failures.append(f"Invalid arguments for {call.get('name')}: {error}")
                continue
            results = functions[tool_name](**arguments) or []
            evidence.extend(results)
            if not results or all(
                str(item.get("item_id", item.get("id", ""))).startswith("tool-error:")
                for item in results
                if isinstance(item, dict)
            ):
                failures.append(
                    f"{call.get('name')}({json.dumps(arguments, default=str)}) returned "
                    f"{json.dumps(results, default=str)[:500]}"
                )
        if calls and len(failures) < len(calls):
            break
        if attempt == 0:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "No usable evidence was returned. Correct the arguments and call one or more tools again.\n"
                        + "\n".join(failures or ["No tool was called."])
                    ),
                }
            )
    usable = [
        item
        for item in evidence
        if not str(item.get("item_id", item.get("id", ""))).startswith("tool-error:")
    ]
    if not usable:
        raise ValueError(f"{agent_id} obtained no evidence from its available tools")
    return evidence


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


def _validate_record(record, agent_id: str) -> None:
    if len(record.candidates) < 2:
        raise ValueError(f"{agent_id} must generate at least two alternatives")
    if len(record.selected_candidate_ids) != 1:
        raise ValueError(f"{agent_id} must select exactly one candidate")


def _recommendation_from_record(record) -> str:
    """Read the benchmark outcome label from the selected, model-generated alternative."""
    selected_id = record.selected_candidate_ids[0]
    selected = next(candidate for candidate in record.candidates if candidate.candidate_id == selected_id)
    content = selected.content if isinstance(selected.content, str) else json.dumps(selected.content)
    labels = set(re.findall(r"\b(?:approve|decline|manual[_ -]review)\b", content.lower()))
    normalized = {label.replace(" ", "_").replace("-", "_") for label in labels}
    if len(normalized) != 1:
        raise ValueError(
            "The selected committee alternative must contain exactly one outcome label: "
            "approve, decline, or manual_review"
        )
    return normalized.pop()


def _invoke_agent(
    *,
    agent_id: str,
    role: str,
    question: str,
    request: str,
    evidence_tool_names: tuple[str, ...],
    tools: FinancialTools,
    loan_id: int,
    model,
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
        used={
            "role": role,
            "loan_id": loan_id,
            "input_entity_ids": input_entity_ids,
            "available_tools": list(evidence_tool_names),
        },
        capture_telemetry=False,
    ) as agent_task:
        context = {"research_role": role, "question": question}
        functions = tools.captured if capture_mode == "provenance" else tools.baseline

        scope = retrieval_scope(
            agent_id=agent_id,
            workflow_id=workflow_id,
            parent_task_id=agent_task.get_id(),
        )
        if capture_mode == "provenance":
            scope.__enter__()
        try:
            selected_evidence = _choose_and_run_tools(
                request=request,
                available_tool_names=evidence_tool_names,
                functions=functions,
                loan_id=loan_id,
                model=model,
                agent_id=agent_id,
                workflow_id=workflow_id,
                parent_task_id=agent_task.get_id(),
                capture_mode=capture_mode,
            )
            baseline_evidence = selected_evidence if capture_mode == "baseline" else []

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
                    _validate_record(record, agent_id)
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
            loan_id,
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
                evidence_tool_names=("fairness_context",),
                tools=tools,
                loan_id=loan_id,
                model=model,
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
                    "Generate and assess distinct retrospective recommendation alternatives. Each alternative's "
                    "content must state exactly one outcome label: approve, decline, or manual_review. Select "
                    "manual_review whenever evidence is insufficient or the fairness audit identifies a concern."
                    "\n\nRISK ASSESSMENT:\n"
                    + json.dumps(_record_summary(risk.record), indent=2, default=str)
                    + "\n\nFAIRNESS AUDIT:\n"
                    + json.dumps(_record_summary(fairness.record), indent=2, default=str)
                ),
                evidence_tool_names=("loan_profile", "transaction_summary", "similar_completed_loans"),
                tools=tools,
                loan_id=loan_id,
                model=model,
                workflow_id=workflow_id,
                capture_mode=config.workflow.capture_mode,
                prompt_item_chars=config.workflow.prompt_item_chars,
                input_entity_ids=[
                    f"decision:{loan_id}:risk-analyst",
                    f"decision:{loan_id}:fairness-auditor",
                ],
            )

    agents = [risk, fairness, committee]
    recommendation = _recommendation_from_record(committee.record)
    tasks = [message for message in Flowcept.buffer if message.get("workflow_id") == workflow_id]
    return {
        "workflow_id": workflow_id,
        "loan_id": loan_id,
        "model": config.model.name,
        "capture_mode": config.workflow.capture_mode,
        "hidden_outcome": case.outcome,
        "recommendation": recommendation,
        "evaluation": evaluate_recommendation(case.outcome, recommendation),
        "tasks": tasks,
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
