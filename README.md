# Decision Capture Use Case

A standalone tool-grounded decision workflow using Flowcept Decision Capture. Flowcept is installed directly from the public `feature/decision-provenance` branch, as configured in `pyproject.toml`.

## Setup

```powershell
git clone https://github.com/SanjanaMamdapur99/decision-capture-use-case.git
cd decision-capture-use-case
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Create a local `settings.yaml` from Flowcept's sample settings. Under `sys_metadata`, set values for Windows:

```yaml
sys_metadata:
  environment_id: "laptop"
  sys_name: "Windows"
  node_name: "local-machine"
```

`settings.yaml` is intentionally ignored because runtime and service configuration can differ between machines.

## Run

```powershell
$env:FLOWCEPT_SETTINGS_PATH = "$PWD\settings.yaml"
python -m decision_use_case.main
```

Expected output:

```text
Flowcept decision-capture imports are available.
```

