"""Export experiment traces and analysis-ready metrics."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def save_experiment(results: list[dict], output_dir: str | Path) -> tuple[Path, Path]:
    """Write lossless JSON traces and a flat CSV metric table."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = destination / f"experiment-{stamp}.json"
    csv_path = destination / f"metrics-{stamp}.csv"
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    rows = []
    for case in results:
        for agent in case["agents"]:
            rows.append(
                {
                    "workflow_id": case["workflow_id"],
                    "loan_id": case["loan_id"],
                    "model": case["model"],
                    "capture_mode": case["capture_mode"],
                    "hidden_outcome": case["hidden_outcome"],
                    "recommendation": case["recommendation"],
                    "correct": case["evaluation"].get("correct"),
                    "abstained": case["evaluation"].get("abstained"),
                    "agent_id": agent["agent_id"],
                    "duration_seconds": agent["duration_seconds"],
                    **agent["metrics"],
                }
            )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path
