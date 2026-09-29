"""Configuration for reproducible decision-capture experiments."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelSettings:
    """OpenAI-compatible model endpoint settings."""

    name: str
    base_url: str | None = None
    api_key_env: str = "DECISION_LLM_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 1800
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class WorkflowSettings:
    """Controls capture behavior and evidence budgets."""

    capture_mode: str = "provenance"
    start_persistence: bool = False
    prompt_item_chars: int = 1200
    recent_transaction_limit: int = 12
    comparable_loan_limit: int = 8


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete experiment configuration."""

    database_path: Path
    output_dir: Path
    loan_ids: list[int]
    model: ModelSettings
    workflow: WorkflowSettings = field(default_factory=WorkflowSettings)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load YAML configuration and resolve paths relative to that file."""
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base = config_path.parent
    model = ModelSettings(**payload["model"])
    workflow = WorkflowSettings(**payload.get("workflow", {}))
    if workflow.capture_mode not in {"provenance", "baseline"}:
        raise ValueError("workflow.capture_mode must be 'provenance' or 'baseline'")
    return ExperimentConfig(
        database_path=(base / payload["database_path"]).resolve(),
        output_dir=(base / payload.get("output_dir", "runs")).resolve(),
        loan_ids=[int(value) for value in payload.get("loan_ids", [])],
        model=model,
        workflow=workflow,
    )
