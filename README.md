# Financial Decision Provenance Benchmark

A reproducible research use case for studying tool-grounded LLM decisions with Flowcept's
Decision Capture and retrieval-provenance extensions. It uses a historical Czech banking
database and produces analysis-ready JSON traces and CSV metrics.

This is a retrospective research benchmark, not a production credit-scoring system.

## Research design

For each historical loan, three agents run sequentially:

1. **Risk analyst** examines the application, pre-loan transactions, standing orders, and
   prior comparable loans. It selects `low_risk`, `high_risk`, or `manual_review`.
2. **Fairness auditor** receives protected attributes separately and checks whether the
   risk recommendation requires an intervention. Protected attributes are never shown to
   the risk analyst.
3. **Credit committee** synthesizes the risk assessment and fairness audit into the
   non-binding recommendation `approve`, `decline`, or `manual_review`.

The target loan's repayment status is hidden from every agent. It is revealed only after
the final recommendation to measure retrospective accuracy. Comparable outcomes are
restricted to loans dated before the target, preventing temporal leakage.

In `provenance` mode, `@flowcept_tool`, `retrieval_scope`, and `DecisionCapture` record
queries, every retrieved item, evidence IDs, keep/drop verdicts, explanations, candidates,
assessments, and selections. In `baseline` mode, the same evidence is placed in the prompt
without retrieval/evidence-use capture, supporting controlled comparisons.

## Data

The included SQLite database contains 682 loans, 4,500 accounts, 5,369 clients, and more
than one million transactions. Evidence tools open it in read-only mode.

## Setup

```powershell
git clone https://github.com/SanjanaMamdapur99/decision-capture-use-case.git
cd decision-capture-use-case
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Flowcept is installed directly from the public `feature/decision-provenance` branch through
`pyproject.toml`.

Create a local `settings.yaml` from Flowcept's full sample settings. On Windows, ensure it
contains:

```yaml
sys_metadata:
  environment_id: "laptop"
  sys_name: "Windows"
  node_name: "local-machine"
```

`settings.yaml` is ignored because service addresses and runtime settings differ between
machines.

## Model configuration

Edit `experiment.yaml`. The default uses local Ollama:

```yaml
model:
  name: llama3.1:8b
  base_url: http://localhost:11434/v1
  api_key_env: DECISION_LLM_API_KEY
```

Pull and serve the model before running:

```powershell
ollama pull llama3.1:8b
```

For a GPU server running an OpenAI-compatible endpoint such as vLLM, change `name` and
`base_url`. If authentication is required, set the configured environment variable:

```powershell
$env:DECISION_LLM_API_KEY = "your-token"
```

No code changes are required when switching models or endpoints. The selected model must
support OpenAI-compatible structured JSON-schema responses.

## Run

```powershell
$env:FLOWCEPT_SETTINGS_PATH = "$PWD\settings.yaml"
python -m decision_use_case.main --config experiment.yaml
```

Run one or more specific loans:

```powershell
python -m decision_use_case.main --config experiment.yaml --loan-id 5358 --loan-id 5856
```

Set `workflow.capture_mode` to `baseline` or `provenance` and repeat with the same loan IDs,
model, and temperature for a controlled comparison.

## Outputs and metrics

Each experiment creates:

- `runs/experiment-<timestamp>.json`: lossless decisions, retrievals, evidence verdicts,
  candidates, assessments, selections, model, outcome, and workflow IDs.
- `runs/metrics-<timestamp>.csv`: one analysis-ready row per agent and case.

Metrics include grounding coverage, evidence utilization, unreported evidence, contradictory
evidence retained, explanation completeness, selected confidence, score margin, candidate
count, latency, abstention, and retrospective recommendation accuracy. Flowcept's task trace
also records model invocations and provider-reported or estimated token usage.

## Tests

Tests use the real SQLite database and require no model or service:

```powershell
python -m unittest discover -s tests -v
```

