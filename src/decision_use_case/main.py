"""Command-line entry point for the decision-provenance study."""

import argparse
from pathlib import Path

from decision_use_case.config import load_experiment_config
from decision_use_case.reporting import save_experiment
from decision_use_case.workflow import run_case


def main() -> None:
    """Run configured historical loan cases and export traces and metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiment.yaml"))
    parser.add_argument("--loan-id", type=int, action="append", dest="loan_ids")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    loan_ids = args.loan_ids or config.loan_ids
    if not loan_ids:
        raise ValueError("Configure loan_ids or pass at least one --loan-id")

    results = []
    for loan_id in loan_ids:
        print(f"Running loan {loan_id} with {config.model.name} ({config.workflow.capture_mode})")
        result = run_case(config, loan_id)
        results.append(result)
        print(
            f"  recommendation={result['recommendation']} outcome={result['hidden_outcome']} "
            f"correct={result['evaluation'].get('correct')}"
        )

    json_path, csv_path = save_experiment(results, config.output_dir)
    print(f"Saved provenance trace: {json_path}")
    print(f"Saved research metrics: {csv_path}")


if __name__ == "__main__":
    main()
