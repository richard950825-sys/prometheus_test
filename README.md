# Evo MVP

A CLI tool for generating, auditing, and reporting on candidate proposals using Python, Typer, and Pydantic.

## Installation

1. Ensure Python 3.11+ is installed.
2. Clone the repository.
3. Create a virtual environment and install dependencies:

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
# OR
pip install -e .
```

## Usage

ensure you are in the project root directory.

### 1. Initialize a Session
Creates a new session directory in `sessions/`.

```bash
python -m evo init-session --name demo
```

### 2. Generate Candidates
Generates mock candidates.

```bash
python -m evo generate --session sessions/demo --n 3
```

### 3. Audit Candidates
Checks candidates against `ruleset.yaml`.

```bash
python -m evo audit --session sessions/demo
```

### 4. Generate Report
Creates a Markdown report in the session directory.

```bash
python -m evo report --session sessions/demo
```

### 5. Iterative Evolution (New!)
Run automated multi-round evolution with patching and tie-breaking.

```bash
# Start a fresh iteration (clears previous data in session)
evo iterate --session sessions/demo_iter --reset

# Resume or run without clearing
evo iterate --session sessions/demo_iter
```

**Note on Flags**:
- `--reset`: Use this flag to enable reset (clear session candidates). Omit it to disable.
- `--no-reset`: Explicitly disable reset (optional).

## Configuration

Modify `ruleset.yaml` to change audit rules.
