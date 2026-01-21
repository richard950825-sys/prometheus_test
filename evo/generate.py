from datetime import datetime
import uuid
from pathlib import Path
from typing import Dict, Any

from .models import (
    Candidate, WorkflowIR, Agent, Step, Controls, WorkflowBudget, WorkflowAcceptance,
    Constraint, Risk
)
from .storage import (
    init_session_structure,
    load_requirements,
    save_candidate,
    rebuild_index,
    load_metadata,
    save_metadata,
    reset_session,
)


def generate_candidates(session_path: str, n: int = 3, reset: bool = False):
    if reset:
        reset_session(session_path)
        print("Resetting session: candidates and audits cleared.")

    init_session_structure(session_path)
    requirements = load_requirements(session_path)

    try:
        from .llm_gemini import generate_candidate
    except Exception as e:
        generate_candidate = None
        print(f"Warning: LLM generator unavailable ({e}). Using fallback candidates.")

    def normalize_candidate(candidate: Candidate, reqs: Dict[str, Any]):
        if not candidate.proposal:
            return
        p = candidate.proposal
        notes = []
        if not p.acceptance.hard_constraints:
            p.acceptance.hard_constraints = [
                Constraint(metric="latency.p95_ms", op="<=", threshold=p.slo.p95_latency_ms),
                Constraint(metric="errors.rate", op="<=", threshold=p.slo.error_rate),
            ]
            notes.append("Backfilled missing hard_constraints from SLO.")
        lt = p.experiments.load_test
        if lt.target_rps is None:
            default_rps = 1000
            if "traffic" in reqs and "peak_rps" in reqs["traffic"]:
                default_rps = int(reqs["traffic"]["peak_rps"])
            lt.target_rps = default_rps
            notes.append(f"Backfilled load_test.target_rps to {default_rps}")
        if not p.risks:
            p.risks.append(Risk(
                title="Unknown risk: needs review",
                mitigation="Run load/chaos tests and review dependencies",
            ))
            notes.append("Backfilled empty risks list.")
        p.normalization_notes.extend(notes)

    metadata = {
        "name": Path(session_path).name,
        "created_at": str(datetime.now()),
        "generation_stats": {"attempts": 0, "failures": 0, "retries": 0},
    }

    strategies = ["cost-optimized", "reliability-optimized", "throughput-optimized"]
    for i in range(1, n + 1):
        strategy = strategies[(i - 1) % len(strategies)]
        print(f"Generating candidate {i}/{n} (Strategy: {strategy})...")
        try:
            if not generate_candidate:
                raise RuntimeError("LLM generator unavailable")
            candidate = generate_candidate(requirements, strategy_hint=strategy)
            new_uuid = f"wf-{uuid.uuid4()}"
            candidate.id = new_uuid
            candidate.strategy = strategy
            filename = f"cand_{candidate.id}.json"
            save_candidate(session_path, candidate, update_index=False)
            print(f"Candidate {i} saved to {filename}")
        except Exception as e:
            print(f"Gemini generation failed for candidate {i}: {e}. Using fallback.")
            metadata["generation_stats"]["failures"] += 1
            candidate = Candidate(
                id=f"fallback_{i}_{int(datetime.now().timestamp())}",
                workflow_ir=WorkflowIR(
                    title=f"Fallback Workflow {i}",
                    goal="Robust RAG Workflow (Fallback)",
                    agents=[Agent(name="Orchestrator", role="coordinator", model="gemini-pro")],
                    steps=[Step(id="retrieve", agent="Orchestrator", action="retrieve", inputs=["query"], outputs=["docs"])],
                    controls=Controls(budget=WorkflowBudget(max_total_turns=10)),
                    acceptance=WorkflowAcceptance(tests=[]),
                ),
            )
            normalize_candidate(candidate, requirements)
            save_candidate(session_path, candidate, update_index=False)

    old_meta = load_metadata(session_path)
    if hasattr(old_meta, "update"):
        old_meta.update(metadata)
        metadata = old_meta
    save_metadata(session_path, metadata)
    rebuild_index(session_path)
