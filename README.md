# Evo MVP

A session-based CLI for generating workflow candidates, auditing them, and driving a windtunnel + release-gate loop with an explicit Forge state machine.

This repo focuses on **explicit state routing**, **shared blackboard (ForgeState)**, and **windtunnel-driven gates** (Oracle + regression) for iterative improvement.

---

## Features

- **Forge state machine**: Explicit Node A-G pipeline with traceable routing and persisted ForgeState.
- **Windtunnel contract**: Standardized `WindTunnelSpec` and `WindTunnelReport` artifacts.
- **Release gates**: Must-not-regress, SLO, and quality checks with failure bundles.
- **Patch verification loop**: Any HARD gate failure triggers patch + re-test, even when `--rounds 1`.
- **Failure bundles**: Reproducible context + failing tasks + pointers for regression.

---

## Requirements

- Python 3.11+
- Node.js + npm (only needed if you want the GitHub MCP server)

---

## Install

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate
pip install -e .
```

---

## Quick Start

### 1) Initialize a session

```bash
.\venv\Scripts\python.exe -m evo.main init-session --name demo
```

### 2) Run the Forge state machine

```bash
.\venv\Scripts\python.exe -m evo.main forge-run \
  --session sessions/demo \
  --rounds 1 \
  --population 3 \
  --suite rag_windtunnel_v1 \
  --replications 3 \
  --seed 42 \
  --perturb \
  --budget-sweep "tool_calls=2,4;tokens=80,160" \
  --perturb-sweep "miss_prob=0.0,0.3;noise_docs=0,2" \
  --patch-mode strategy \
  --reset
```

What you get:
- `sessions/<name>/forge_state.json` (shared blackboard)
- `sessions/<name>/state_machine/trace.md` (node routing trace)
- windtunnel evidence + report under `sessions/<name>/evidence/`
- release gate results under `sessions/<name>/gates/`
- recommendation artifacts when gates pass

---

## CLI Commands

### Session basics

```bash
.\venv\Scripts\python.exe -m evo.main init-session --name demo
.\venv\Scripts\python.exe -m evo.main generate --session sessions/demo --n 3
.\venv\Scripts\python.exe -m evo.main audit --session sessions/demo
.\venv\Scripts\python.exe -m evo.main report --session sessions/demo
```

### Forge state machine (recommended)

```bash
.\venv\Scripts\python.exe -m evo.main forge-run --session sessions/demo --rounds 2 --population 3
```

### Legacy iterate (compatibility wrapper)

```bash
.\venv\Scripts\python.exe -m evo.main iterate --session sessions/demo --rounds 2 --population 3
```

### Metrics and suite runs

```bash
.\venv\Scripts\python.exe -m evo.main run --session sessions/demo --suite rag_windtunnel_v1 --replications 3
.\venv\Scripts\python.exe -m evo.main metrics --session sessions/demo --suite rag_windtunnel_v1
```

---

## Forge State Machine

Nodes follow the explicit A-G route:

- **A Ingest**: Loads requirements into `ForgeState.user_requirements`
- **B Generate**: Produces candidate workflows and sets `current_workflow`
- **C Static Audit**: Audits + writes `required_fixes` (deduped by rule_id)
- **D Windtunnel**: Runs suite for `current_workflow`, writes report + stats
- **E Synthesis**: Converts failures into actionable WDR updates + gate decisions
- **F Revise/Patch**: Applies targeted patches and prepares for re-audit
- **G Package**: Produces final recommendation once gates pass

Routing:
- C Reject -> F -> C (re-audit), with no-progress watchdog to regenerate
- C Pass -> D -> E -> (F if HARD gate fail) -> C -> ...
- Only gate pass triggers G Package

Trace file:
- `sessions/<name>/state_machine/trace.md`

---

## Windtunnel Contract

Schema:
- `evo/windtunnel/spec.py` (`WindTunnelSpec`)
- `evo/windtunnel/report.py` (`WindTunnelReport`)

Artifacts:
- `sessions/<name>/evidence/workflow/<candidate_id>/windtunnel/spec_<suite>.json`
- `sessions/<name>/evidence/workflow/<candidate_id>/windtunnel/report_<suite>.json`

---

## Release Gates

Gate rules are configured in `gate_rules.yaml`:

- **Must-not-regress**: loop rate / unauthorized tool calls / injection success
- **SLO**: latency/cost/timeouts thresholds
- **Quality**: pass mean >= baseline - epsilon

Gate outputs:
- `sessions/<name>/gates/last_result.json`
- `sessions/<name>/gates/baseline.json`

---

## Failure Bundles (Replay)

When a HARD gate fails, a failure bundle is created:

```
sessions/<name>/failure_bundles/<candidate_id>/<timestamp>/
  context.json
  failing_tasks.json
  pointers.json
```

This enables targeted regression and replay against worst-case sweep points.

---

## Project Structure

```
.
??? evo/
?   ??? forge/                # ForgeState + state machine
?   ??? windtunnel/            # Spec/report contracts
?   ??? oracle/                # Windtunnel runner
?   ??? gates.py               # Release gates
?   ??? metrics.py             # Metrics extraction
?   ??? ...
??? eval_suites/
??? sessions/                  # session outputs (not committed)
??? gate_rules.yaml
??? ruleset.yaml
??? ruleset_workflow.yaml
??? README.md
```

---

## Notes

- Use the venv Python for all CLI runs on this repo.
- Session artifacts can be large; do not commit `sessions/` or log files.
- The LLM generator requires `google-genai` and a valid `GEMINI_API_KEY`/`GOOGLE_API_KEY`.

---

## License

Internal MVP.
