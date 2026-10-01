# Financial Decision Provenance Benchmark

A reproducible research workflow for studying tool-grounded LLM decisions with Flowcept.
Three agents inspect a historical Czech banking case, generate their own alternatives through
`DecisionCapture.invoke()`, assess them against retrieved evidence, and produce a non-binding
retrospective recommendation.

This is a research benchmark, not a production credit-scoring system.

## What the workflow records

For every agent turn, the provenance trace separates:

1. **Retrieval ground truth** — the tools called, the model-authored arguments, and every item returned.
2. **Evidence filtering** — a keep/drop verdict and explanation for every retrieved item.
3. **Decision generation** — model-generated candidates, assessments, scores, and selection.

Application code does not define or add candidates. Flowcept's `DecisionCapture.invoke()` owns
candidate generation and structured decision validation.

The agents run sequentially:

1. The **risk analyst** chooses financial evidence tools and generates risk alternatives.
2. The **fairness auditor** receives the risk result and separately accesses protected-group context.
3. The **credit committee** synthesizes both results into an `approve`, `decline`, or
   `manual_review` research recommendation.

The target loan outcome is hidden until all three decisions finish. Prior comparable loans are
restricted to cases dated before the target loan.

## Recommended environment

- Windows 10 22H2 or newer with PowerShell.
- Python 3.11 or 3.12 and Git.
- A supported NVIDIA GPU with a current NVIDIA driver.
- Ollama for Windows running on the same server as this workflow.
- Internet access during setup to download Python dependencies, Flowcept, the model, and database.

Ollama's current platform and GPU requirements are documented in its
[Windows installation guide](https://docs.ollama.com/windows) and
[hardware support guide](https://docs.ollama.com/gpu).

## Complete Windows PowerShell setup

Run every section below in order on the Windows GPU server. Use the same PowerShell session after
activating the virtual environment and setting the environment variables.

### 1. Install the prerequisites

If Git or Python 3.11 is not already installed, install them with Windows Package Manager:

```powershell
winget install --exact --id Git.Git
winget install --exact --id Python.Python.3.11
```

Install Ollama for Windows:

```powershell
winget install --exact --id Ollama.Ollama
```

If `winget` cannot find Ollama, use the official
[Ollama Windows installer](https://ollama.com/download/windows).

Close PowerShell after installation, open a new PowerShell window, and verify the tools:

```powershell
git --version
py -3.11 --version
ollama --version
nvidia-smi
```

Update the NVIDIA driver before continuing if `nvidia-smi` cannot see the GPU. Ollama's Windows
documentation currently requires a sufficiently recent NVIDIA driver for GPU acceleration.

### 2. Verify the Ollama server

The Windows Ollama application normally starts in the background and serves
`http://localhost:11434`.

```powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

If that request fails, start Ollama in a dedicated PowerShell window and leave it running:

```powershell
ollama serve
```

### 3. Clone the repository and create a virtual environment

```powershell
git clone https://github.com/SanjanaMamdapur99/decision-capture-use-case.git
Set-Location decision-capture-use-case
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

If PowerShell blocks virtual-environment activation, allow scripts only for the current process and
retry activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The editable install automatically installs the `llm_agent` and `mongo` extras from the pinned
Flowcept `feature/decision-provenance` branch. Do not install a separate PyPI Flowcept version on
top of this environment.

Confirm the expected branch implementation is importable:

```powershell
python -c "from flowcept import DecisionCapture, flowcept_tool, retrieval_scope; print('Flowcept decision capture ready')"
```

### 4. Download the benchmark database

The 71 MB SQLite database is intentionally not stored in Git. Download the pinned BIRD copy:

```powershell
New-Item -ItemType Directory -Force data\financial | Out-Null
curl.exe --fail --location `
  "https://huggingface.co/datasets/prem-research/birdbench/resolve/03abfc646adfd2ff0ab33ef69df16579446d6572/validation/dev_databases/financial/financial.sqlite?download=true" `
  --output data\financial\financial.sqlite
Get-FileHash -Algorithm SHA256 data\financial\financial.sqlite
```

Expected SHA-256:

```text
D15D89CDB068A202B6F2B99342AF44DFFC1D52545B39CEAF62EFDC0BA570101E
```

The source dataset is the PKDD'99 Czech financial dataset. The pinned database copy is published
through the [BIRD benchmark dataset](https://huggingface.co/datasets/prem-research/birdbench).

### 5. Create an isolated Flowcept configuration

The default experiment uses Flowcept's offline profile. Redis and MongoDB are not required.

```powershell
$env:FLOWCEPT_SETTINGS_PATH = "$PWD\settings.yaml"
flowcept --init-settings --full -y
flowcept --config-profile full-offline -y
```

Keep `FLOWCEPT_SETTINGS_PATH` set in every terminal used to run the workflow. `settings.yaml` is
machine-local and ignored by Git.

### 6. Pull the recommended Ollama model

```powershell
ollama pull qwen3:8b
ollama list
```

`qwen3:8b` is the default in `experiment.yaml` because its Ollama template supports tool calls.
The workflow also requires the local OpenAI-compatible endpoint's `tools`, `response_format`, and
reasoning-control support. Ollama documents these fields in its
[OpenAI compatibility reference](https://docs.ollama.com/api/openai-compatibility).

Set the local placeholder API key expected by OpenAI-compatible clients:

```powershell
$env:DECISION_LLM_API_KEY = "ollama"
```

Ollama ignores this value for local requests.

### 7. Run the deterministic tests

```powershell
python -m unittest discover -s tests -v
```

### 8. Run one smoke-test loan

```powershell
python -m decision_use_case.main --config experiment.yaml --loan-id 5358
```

While it is running, use another terminal to verify GPU placement:

```powershell
ollama ps
nvidia-smi
```

`ollama ps` should report GPU use for the loaded model. On NVIDIA, `nvidia-smi` should show the
Ollama process and allocated VRAM.

### 9. Run the complete configured experiment

```powershell
python -m decision_use_case.main --config experiment.yaml
```

The default configuration evaluates four historical loans. Each loan runs three agents and may
make several model calls, so the full experiment takes materially longer than the smoke test.

## Configuration

The default [`experiment.yaml`](experiment.yaml) uses:

```yaml
model:
  name: qwen3:8b
  base_url: http://localhost:11434/v1
  api_key_env: DECISION_LLM_API_KEY
  temperature: 0
  max_tokens: 1800
  reasoning_effort: none

workflow:
  capture_mode: provenance
  start_persistence: false
  prompt_item_chars: 1200
```

To try another Ollama model, first pull it and then change `model.name`:

```powershell
ollama pull qwen3:14b
```

Use a model that supports both tool calling and structured output. Larger models generally need
more VRAM. If Ollama is running on another machine, change `base_url` to that server's reachable
`http://HOST:11434/v1` endpoint and configure Ollama networking appropriately.

## Provenance and baseline modes

- `capture_mode: provenance` records tool calls, complete retrievals, evidence verdicts,
  candidates, assessments, selections, and model invocations.
- `capture_mode: baseline` executes the same model-selected tools but places their results directly
  in the decision prompt without retrieval/evidence-use capture.

Run the same loan IDs, model, temperature, and evidence limits in both modes for a controlled
comparison.

## Outputs

Each execution writes:

- `runs/experiment-<timestamp>.json` — decisions, retrievals, evidence verdicts, candidates,
  assessments, selections, evaluation outcomes, and the complete captured Flowcept task trace.
- `runs/metrics-<timestamp>.csv` — one analysis-ready row per agent and loan.

Metrics include grounding coverage, evidence utilization, unreported evidence, contradictory
evidence retained, explanation completeness, selected confidence, score margin, candidate count,
latency, abstention, and retrospective recommendation accuracy.

## Troubleshooting

### `Connection refused` for port 11434

```powershell
Get-Process ollama -ErrorAction SilentlyContinue
Invoke-RestMethod http://localhost:11434/api/tags
```

If Ollama is not running, launch the Ollama Windows application or run `ollama serve` in a separate
PowerShell window.

### Model runs on CPU

Check `ollama ps` and `nvidia-smi`. Update the NVIDIA driver and verify that the GPU is supported by
Ollama. Restart Ollama after changing GPU-related environment variables or drivers.

### Database missing

```powershell
Get-Item data\financial\financial.sqlite
Get-FileHash -Algorithm SHA256 data\financial\financial.sqlite
```

Re-run the pinned download command if the file is absent or the hash differs.

### Structured response or tool-call failures

Confirm that `experiment.yaml` names a downloaded, tool-capable model:

```powershell
ollama list
ollama show qwen3:8b
```

Then retry the single-loan smoke test before launching all four cases.

### Wrong Flowcept version

Reinstall this project inside the active virtual environment. This reinstalls Flowcept from the
pinned Git branch declared in `pyproject.toml`:

```powershell
python -m pip uninstall -y flowcept decision-capture-use-case
python -m pip install -e .
```
