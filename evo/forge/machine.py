from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Iterable

from .state import ForgeState
from ..audit_engine import audit_session
from ..generate import generate_candidates
from ..metrics import (
    compute_metrics_for_candidate,
    load_metrics_for_candidate,
    validate_metrics_consistency,
    write_metrics,
)
from ..oracle.rag_mini_runner import run_suite_for_candidate
from ..patching import apply_patch
from ..recommend import pick_champion, write_recommendation
from ..gates import evaluate_release_gate
from ..storage import (
    ensure_dir,
    iter_candidates,
    load_audit_result,
    load_candidate,
    load_requirements,
    load_windtunnel_report,
    read_json,
    migrate_legacy_ids_logic,
    rebuild_index,
    reset_session,
    save_forge_state,
    load_forge_state,
    get_suite_path,
)


def _trace_path(session_path: str) -> Path:
    return ensure_dir(Path(session_path) / "state_machine") / "trace.md"


def _append_trace(session_path: str, line: str) -> None:
    trace_path = _trace_path(session_path)
    if not trace_path.exists():
        trace_path.write_text("# Forge State Machine Trace\n", encoding="utf-8")
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _persist_state(session_path: str, state: ForgeState, node_name: str, message: str) -> None:
    save_forge_state(session_path, state.model_dump(mode="json"))
    timestamp = datetime.now().isoformat()
    _append_trace(session_path, f"- [{timestamp}] {node_name}: {message}")


def _current_candidate_id(state: ForgeState) -> Optional[str]:
    if state.current_workflow and isinstance(state.current_workflow, dict):
        return state.current_workflow.get("id") or state.current_workflow.get("candidate_id")
    return None


def _unique_violations(violations: Iterable[Any]) -> list[Any]:
    unique: list[Any] = []
    seen: set[str] = set()
    for v in violations:
        rule_id = getattr(v, "rule_id", None)
        if not rule_id:
            continue
        if rule_id in seen:
            continue
        seen.add(rule_id)
        unique.append(v)
    return unique


def _metric_value(metrics_list: Iterable[Any], metric_key: str) -> Optional[float]:
    for metric in metrics_list:
        key = metric.get("metric_key") if isinstance(metric, dict) else getattr(metric, "metric_key", None)
        if key != metric_key:
            continue
        val = metric.get("value") if isinstance(metric, dict) else getattr(metric, "value", None)
        try:
            return float(val)
        except (TypeError, ValueError):
            return None
    return None


def _metric_raw(metrics_list: Iterable[Any], metric_key: str) -> Any:
    for metric in metrics_list:
        key = metric.get("metric_key") if isinstance(metric, dict) else getattr(metric, "metric_key", None)
        if key == metric_key:
            return metric.get("value") if isinstance(metric, dict) else getattr(metric, "value", None)
    return None


def _top_fail_modes(failure_cases: list[Dict[str, Any]], limit: int = 3) -> list[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for case in failure_cases:
        category = case.get("category") or "unknown"
        counts[category] = counts.get(category, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [{"category": name, "count": count} for name, count in ranked[:limit]]


def _build_top_fail_modes_from_report(summary: Dict[str, Any], limit: int = 6) -> list[Dict[str, Any]]:
    modes: list[Dict[str, Any]] = []
    for task_id, count in (summary.get("fail_by_task_id") or {}).items():
        modes.append({"dimension": "task_id", "key": task_id, "count": count})
    for assertion, count in (summary.get("fail_by_assertion") or {}).items():
        modes.append({"dimension": "assertion", "key": assertion, "count": count})
    modes.sort(key=lambda item: item.get("count", 0), reverse=True)
    return modes[:limit]


def _create_failure_bundle(
    session_path: str,
    candidate_id: str,
    suite_id: str,
    report: Dict[str, Any],
    metrics_list: list[Dict[str, Any]],
    gate_fail_rule_ids: list[str],
    run_config: Dict[str, Any],
) -> Optional[str]:
    if not report:
        return None
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    bundle_dir = ensure_dir(
        Path(session_path) / "failure_bundles" / candidate_id / timestamp
    )

    summary = report.get("summary", {})
    worst_context_raw = _metric_raw(metrics_list, "robustness.worst_case_context")
    sweep_params = None
    aggregate_ref = None
    if isinstance(worst_context_raw, str):
        try:
            worst_context = json.loads(worst_context_raw)
            sweep_params = worst_context.get("sweep_params")
            aggregate_ref = worst_context.get("aggregate_ref")
        except json.JSONDecodeError:
            sweep_params = None

    context = {
        "suite_id": suite_id,
        "seed": summary.get("seed"),
        "replications": summary.get("replications"),
        "perturb": summary.get("perturb"),
        "sweep_params": sweep_params,
        "gate_fail_rule_ids": gate_fail_rule_ids,
        "run_config": run_config,
    }
    (bundle_dir / "context.json").write_text(json.dumps(context, indent=2), encoding="utf-8")

    suite_path = get_suite_path(suite_id)
    suite = read_json(suite_path) if suite_path else {}
    suite_tasks = {t.get("task_id"): t for t in suite.get("tasks", [])}

    failing_tasks = []
    for run_info in report.get("runs", []):
        run_ref = run_info.get("run_file")
        if not run_ref:
            continue
        run_path = Path(session_path) / run_ref
        if not run_path.exists():
            continue
        run_data = read_json(run_path)
        for task in run_data.get("tasks", []):
            if task.get("passed"):
                continue
            task_id = task.get("task_id")
            suite_task = suite_tasks.get(task_id, {})
            failing_tasks.append({
                "task_id": task_id,
                "category": task.get("category"),
                "expected": suite_task.get("expected"),
                "observed": {
                    "answer": task.get("answer"),
                    "structured": task.get("structured"),
                    "assertion_failures": task.get("assertion_failures"),
                },
                "citations": (task.get("structured") or {}).get("citations") if task.get("structured") else None,
                "run_index": run_data.get("run_index"),
                "seed": run_data.get("seed"),
                "sweep_params": run_data.get("sweep_params"),
            })

    if aggregate_ref:
        agg_path = Path(session_path) / aggregate_ref
        if agg_path.exists():
            aggregate = read_json(agg_path)
            failures = aggregate.get("failures", {})
            for ref in failures.get("sample_refs", []) or []:
                if "#task=" not in ref:
                    continue
                run_ref, task_id = ref.split("#task=", 1)
                run_path = Path(session_path) / run_ref
                if not run_path.exists():
                    continue
                run_data = read_json(run_path)
                suite_task = suite_tasks.get(task_id, {})
                for task in run_data.get("tasks", []):
                    if task.get("task_id") != task_id:
                        continue
                    failing_tasks.append({
                        "task_id": task_id,
                        "category": task.get("category"),
                        "expected": suite_task.get("expected"),
                        "observed": {
                            "answer": task.get("answer"),
                            "structured": task.get("structured"),
                            "assertion_failures": task.get("assertion_failures"),
                        },
                        "citations": (task.get("structured") or {}).get("citations") if task.get("structured") else None,
                        "run_index": run_data.get("run_index"),
                        "seed": run_data.get("seed"),
                        "sweep_params": run_data.get("sweep_params"),
                    })

    (bundle_dir / "failing_tasks.json").write_text(
        json.dumps(failing_tasks, indent=2),
        encoding="utf-8",
    )

    pointers = {
        "aggregate_ref": report.get("artifacts", {}).get("run_aggregate"),
        "run_refs": [r.get("run_file") for r in report.get("runs", []) if r.get("run_file")],
        "metrics_ref": f"candidates/cand_{candidate_id}/outputs/metrics.json",
        "report_ref": report.get("artifacts", {}).get("report_path"),
    }
    (bundle_dir / "pointers.json").write_text(json.dumps(pointers, indent=2), encoding="utf-8")

    return str(bundle_dir.relative_to(Path(session_path)))


def node_a_ingest(session_path: str, state: ForgeState) -> ForgeState:
    requirements = load_requirements(session_path)
    state.user_requirements = requirements
    _persist_state(session_path, state, "Node A", "Ingested requirements")
    return state


def node_b_generate(session_path: str, state: ForgeState, population: int, reset: bool) -> ForgeState:
    generate_candidates(session_path, n=population, reset=reset)
    rebuild_index(session_path)
    migrate_legacy_ids_logic(session_path, inplace=True)

    drafts = []
    for cand in iter_candidates(session_path, include_patched=True):
        drafts.append(cand.model_dump(mode="json"))
    state.draft_workflows = drafts
    if drafts:
        state.current_workflow = drafts[0]
    _persist_state(session_path, state, "Node B", f"Generated {len(drafts)} candidates")
    return state


def node_c_audit(session_path: str, state: ForgeState, suite_id: str) -> Tuple[ForgeState, str, set[str], int, int]:
    audit_session(session_path, include_patched=True, suite_id=suite_id, migrate_ids=False)
    decision = "Reject"
    required_fixes: list[Dict[str, Any]] = []
    unique_rule_ids: set[str] = set()
    findings_count = 0
    unique_count = 0
    reason = "No current candidate."
    risks_to_validate: list[str] = []
    violations_payload: list[Dict[str, Any]] = []

    cand_id = _current_candidate_id(state)
    if cand_id:
        result = load_audit_result(session_path, cand_id)
        if result:
            decision = "Pass" if result.passed else "Reject"
            findings_count = len(result.violations)
            unique_violations = _unique_violations(result.violations)
            unique_count = len(unique_violations)
            for v in unique_violations:
                entry = {
                    "rule_id": v.rule_id,
                    "severity": v.severity.value,
                    "message": v.message,
                    "fix_suggestion": v.fix_suggestion,
                    "target_path": None,
                }
                required_fixes.append(entry)
                unique_rule_ids.add(v.rule_id)
                violations_payload.append(entry)
            risks_to_validate = [v.message for v in unique_violations if v.severity.value == "RISK"]
            reason = f"Audit {decision.lower()} ({findings_count} findings, {unique_count} unique)."

    state.audit_report = {
        "decision": decision,
        "required_fixes": required_fixes,
        "reason": reason,
        "risks_to_validate": risks_to_validate,
        "violations": violations_payload,
        "findings_count": findings_count,
        "unique_rule_ids": sorted(unique_rule_ids),
    }
    state.required_fixes = required_fixes if decision == "Reject" else []
    if unique_rule_ids:
        unique_list = ",".join(sorted(unique_rule_ids))
    else:
        unique_list = "none"
    _persist_state(
        session_path,
        state,
        "Node C",
        f"{decision}: {reason} unique_rule_ids=[{unique_list}]",
    )
    return state, decision, unique_rule_ids, findings_count, unique_count


def node_d_windtunnel(
    session_path: str,
    state: ForgeState,
    suite_id: str,
    replications: int,
    seed: Optional[int],
    perturb: bool,
    budget_sweep: Optional[str],
    perturb_sweep: Optional[str],
) -> ForgeState:
    cand_id = _current_candidate_id(state)
    if not cand_id:
        _persist_state(session_path, state, "Node D", "No current candidate to run windtunnel")
        return state

    suite_path = get_suite_path(suite_id)
    if not suite_path:
        _persist_state(session_path, state, "Node D", f"Suite {suite_id} not found")
        return state

    run_suite_for_candidate(
        session_path,
        cand_id,
        str(suite_path),
        replications=replications,
        seed=seed,
        perturb=perturb,
        budget_sweep=budget_sweep,
        perturb_sweep=perturb_sweep,
    )

    metrics_dict = compute_metrics_for_candidate(session_path, cand_id, suite_id=suite_id)
    write_metrics(session_path, cand_id, metrics_dict)

    sweeps_enabled = bool(budget_sweep or perturb_sweep)
    ok, _issues = validate_metrics_consistency(
        session_path,
        cand_id,
        require_sweeps=sweeps_enabled,
    )
    if not ok:
        run_suite_for_candidate(
            session_path,
            cand_id,
            str(suite_path),
            replications=replications,
            seed=seed,
            perturb=perturb,
            budget_sweep=budget_sweep,
            perturb_sweep=perturb_sweep,
        )
        metrics_dict = compute_metrics_for_candidate(session_path, cand_id, suite_id=suite_id)
        write_metrics(session_path, cand_id, metrics_dict)

    report = load_windtunnel_report(session_path, cand_id, suite_id)
    if report:
        state.simulation_summary = report.get("summary")
        state.failure_cases = report.get("failure_clusters") or []

    metrics_list = load_metrics_for_candidate(session_path, cand_id)
    pass_mean = _metric_value(metrics_list, "stability.pass_rate_mean") or _metric_value(metrics_list, "quality.pass_rate")
    flake = _metric_value(metrics_list, "stability.flake_rate")
    faith_mean = _metric_value(metrics_list, "stability.faithfulness_mean") or _metric_value(metrics_list, "quality.faithfulness")
    worst_pass = _metric_value(metrics_list, "robustness.worst_case_pass_mean")
    worst_flake = _metric_value(metrics_list, "robustness.worst_case_flake")
    summary = f"pass_mean={pass_mean} flake={flake} faith_mean={faith_mean} worst_pass={worst_pass} worst_flake={worst_flake}"
    _persist_state(session_path, state, "Node D", summary)
    return state


def node_e_synthesis(
    session_path: str,
    state: ForgeState,
    suite_id: str,
    run_config: Dict[str, Any],
) -> Tuple[ForgeState, bool, list[str]]:
    cand_id = _current_candidate_id(state)
    if not cand_id:
        _persist_state(session_path, state, "Node E", "No current candidate to synthesize")
        return state, False, []

    audit_session(session_path, include_patched=True, suite_id=suite_id, migrate_ids=False)
    audit_result = load_audit_result(session_path, cand_id)
    hard_violations = []
    risk_violations = []
    if audit_result:
        for v in audit_result.violations:
            if v.severity.value == "HARD":
                hard_violations.append(v)
            elif v.severity.value == "RISK":
                risk_violations.append(v)

    required_fixes = []
    hard_rule_ids = []
    recommended_patches = []
    for v in hard_violations:
        required_fixes.append({
            "rule_id": v.rule_id,
            "severity": v.severity.value,
            "message": v.message,
            "fix_suggestion": v.fix_suggestion,
            "target_path": None,
        })
        hard_rule_ids.append(v.rule_id)
        if v.fix_suggestion:
            recommended_patches.append({
                "rule_id": v.rule_id,
                "suggestion": v.fix_suggestion,
            })

    gate_result = evaluate_release_gate(session_path, cand_id, suite_id=suite_id)
    regression_fail = gate_result.get("status") == "FAIL"
    if regression_fail:
        required_fixes.append({
            "rule_id": "RGATE",
            "severity": "HARD",
            "message": "Regression gate failed",
            "fix_suggestion": "Inspect failure bundle and improve worst-case robustness.",
            "target_path": None,
        })
        recommended_patches.append({
            "rule_id": "RGATE",
            "suggestion": "Investigate failure bundle and improve robustness metrics.",
        })

    metrics_list = load_metrics_for_candidate(session_path, cand_id)
    worst_context_raw = _metric_raw(metrics_list, "robustness.worst_case_context")
    worst_case_context = None
    if isinstance(worst_context_raw, str):
        try:
            worst_case_context = json.loads(worst_context_raw)
        except Exception:
            worst_case_context = worst_context_raw

    report = load_windtunnel_report(session_path, cand_id, suite_id)
    report_summary = report.get("summary", {}) if report else {}

    worst_case_failures: Dict[str, Any] = {}
    aggregate_ref = None
    if isinstance(worst_case_context, dict):
        aggregate_ref = worst_case_context.get("aggregate_ref")
    if aggregate_ref:
        agg_path = Path(session_path) / aggregate_ref
        if agg_path.exists():
            worst_case_failures = read_json(agg_path).get("failures", {})

    top_fail_modes: list[Dict[str, Any]] = []
    if worst_case_failures:
        for task_id, count in (worst_case_failures.get("by_task_id") or {}).items():
            top_fail_modes.append({"dimension": "task_id", "key": task_id, "count": count})
        for assertion, count in (worst_case_failures.get("by_assertion") or {}).items():
            top_fail_modes.append({"dimension": "assertion", "key": assertion, "count": count})
        top_fail_modes.sort(key=lambda item: item.get("count", 0), reverse=True)
        top_fail_modes = top_fail_modes[:6]
    if not top_fail_modes:
        top_fail_modes = _build_top_fail_modes_from_report(report_summary)
    if not top_fail_modes:
        top_fail_modes = _top_fail_modes(state.failure_cases)

    only_o300 = set(hard_rule_ids) == {"O300"} and not regression_fail
    if only_o300 and worst_case_failures:
        by_assertion = worst_case_failures.get("by_assertion", {})
        by_task = worst_case_failures.get("by_task_id", {})
        targeted = []
        if by_assertion.get("citations_required", 0):
            targeted.append({
                "rule_id": "O300",
                "suggestion": "Low-budget tier: require non-empty factual citations for citations_required tasks.",
                "targets": {"dimension": "assertion", "key": "citations_required", "count": by_assertion.get("citations_required")},
            })
        if by_assertion.get("conflict_detected", 0) or by_assertion.get("must_mention_conflict", 0):
            targeted.append({
                "rule_id": "O300",
                "suggestion": "Low-budget tier: enforce conflict detection template and dual-citation for conflicting docs.",
                "targets": {"dimension": "assertion", "key": "conflict_detected", "count": by_assertion.get("conflict_detected", 0)},
            })
        if by_task.get("conflict_2", 0):
            targeted.append({
                "rule_id": "O300",
                "suggestion": "Low-budget tier: force citations of all conflict_2 doc_ids (doc_c/doc_d).",
                "targets": {"dimension": "task_id", "key": "conflict_2", "count": by_task.get("conflict_2")},
            })
        worst_flake = _metric_value(metrics_list, "robustness.worst_case_flake")
        if worst_flake is not None and worst_flake > 0.1:
            targeted.append({
                "rule_id": "O300",
                "suggestion": "Low-budget tier: enable deterministic template to reduce flake.",
                "targets": {"dimension": "metric", "key": "worst_case_flake", "value": worst_flake},
            })
        if targeted:
            recommended_patches.extend(targeted)

    bundle_ref = None
    if required_fixes:
        bundle_ref = _create_failure_bundle(
            session_path,
            cand_id,
            suite_id,
            report,
            metrics_list,
            hard_rule_ids,
            run_config,
        )

    update = {
        "timestamp": datetime.now().isoformat(),
        "source": "windtunnel",
        "worst_case_context": worst_case_context,
        "worst_case_failures": worst_case_failures,
        "top_fail_modes": top_fail_modes,
        "recommended_patches": [
            dict(patch, bundle_ref=bundle_ref) for patch in recommended_patches
        ],
        "notes": "Synthesized windtunnel signals into patch guidance.",
    }
    state.wdr_updates.append(update)

    state.required_fixes = required_fixes if required_fixes else []
    audit_report = state.audit_report or {}
    existing_risks = set(audit_report.get("risks_to_validate") or [])
    for v in risk_violations:
        existing_risks.add(v.message)
    audit_report["risks_to_validate"] = sorted(existing_risks)
    audit_report["gate_decision"] = {
        "oracle_hard_rule_ids": hard_rule_ids,
        "regression": gate_result,
        "hard_fail": bool(required_fixes),
        "bundle_ref": bundle_ref,
    }
    state.audit_report = audit_report

    hard_fail = bool(required_fixes)
    rule_list = ",".join(hard_rule_ids) if hard_rule_ids else "none"
    _persist_state(
        session_path,
        state,
        "Node E",
        f"HARD gate fail={hard_fail} rule_ids=[{rule_list}]",
    )

    if bundle_ref:
        bundle_path = Path(session_path) / bundle_ref / "failing_tasks.json"
        entries = []
        if bundle_path.exists():
            try:
                bundle_tasks = json.loads(bundle_path.read_text(encoding="utf-8"))
                entries = [
                    {
                        "task_id": entry.get("task_id"),
                        "assertion_failures": entry.get("observed", {}).get("assertion_failures"),
                        "category": entry.get("category"),
                    }
                    for entry in bundle_tasks[:20]
                ]
            except Exception:
                entries = []
        state.failure_cases.append({
            "bundle_ref": bundle_ref,
            "entries": entries,
        })

    return state, hard_fail, hard_rule_ids


def node_f_revise(
    session_path: str,
    state: ForgeState,
    patch_mode: str,
    include_advice: bool,
) -> Tuple[ForgeState, bool, list[str]]:
    cand_id = _current_candidate_id(state)
    if not cand_id:
        _persist_state(session_path, state, "Node F", "No current candidate to patch")
        return state, False, []

    attempted_rule_ids = [
        fix.get("rule_id")
        for fix in state.required_fixes
        if isinstance(fix, dict) and fix.get("rule_id")
    ]
    patch_applied = apply_patch(session_path, cand_id, apply_advice=include_advice, mode=patch_mode)
    updated_fixes: list[Dict[str, Any]] = []
    for fix in state.required_fixes:
        if not isinstance(fix, dict):
            continue
        entry = dict(fix)
        if entry.get("rule_id") in attempted_rule_ids:
            entry["attempted"] = True
        updated_fixes.append(entry)
    state.required_fixes = updated_fixes
    if patch_applied:
        patched_id = f"{cand_id}_patched"
        try:
            patched = load_candidate(session_path, patched_id)
            state.current_workflow = patched.model_dump(mode="json")
        except Exception:
            pass
        patch_msg = f"Patched {cand_id} -> {patched_id}"
    else:
        patch_msg = "Patch no-op"

    state.wdr_updates.append({
        "timestamp": datetime.now().isoformat(),
        "patch_applied": patch_applied,
        "patch_mode": patch_mode,
        "attempted_rule_ids": attempted_rule_ids,
        "source": "audit_reject_loop",
    })
    _persist_state(session_path, state, "Node F", patch_msg)
    return state, patch_applied, attempted_rule_ids


def node_g_package(session_path: str, state: ForgeState) -> ForgeState:
    rec_data, md_content = pick_champion(session_path, include_patched=True)
    if rec_data and md_content:
        write_recommendation(session_path, rec_data, md_content)
    _persist_state(session_path, state, "Node G", "Recommendation packaged")
    return state


def run_forge_machine(
    session_path: str,
    rounds: int = 2,
    population: int = 3,
    suite_id: str = "rag_mini",
    patch_mode: str = "strategy",
    include_advice: bool = False,
    reset: bool = False,
    replications: int = 1,
    seed: Optional[int] = None,
    perturb: bool = False,
    budget_sweep: Optional[str] = None,
    perturb_sweep: Optional[str] = None,
    verify_rounds: int = 1,
) -> Dict[str, Any]:
    if reset:
        reset_session(session_path)

    state_payload = load_forge_state(session_path)
    state = ForgeState(**state_payload) if state_payload else ForgeState()

    node_a_ingest(session_path, state)

    needs_generation = not state.draft_workflows
    no_progress_count = 0
    no_progress_limit = 2
    last_reject_rules: Optional[set[str]] = None

    run_config = {
        "suite_id": suite_id,
        "replications": replications,
        "seed": seed,
        "perturb": perturb,
        "budget_sweep": budget_sweep,
        "perturb_sweep": perturb_sweep,
    }

    should_exit = False
    while state.iteration_count < rounds:
        if needs_generation:
            node_b_generate(session_path, state, population, reset=False)

        state, decision, rule_ids, _, _ = node_c_audit(session_path, state, suite_id)
        if decision == "Reject":
            if not state.required_fixes:
                needs_generation = True
                last_reject_rules = rule_ids
                _persist_state(session_path, state, "Loop", "Reject with no required_fixes -> regenerate")
                continue

            state, patch_applied, _ = node_f_revise(
                session_path,
                state,
                patch_mode=patch_mode,
                include_advice=include_advice,
            )
            state, decision_after, rule_ids_after, _, _ = node_c_audit(session_path, state, suite_id)
            if decision_after == "Reject":
                if last_reject_rules is not None and rule_ids_after == last_reject_rules:
                    no_progress_count += 1
                else:
                    no_progress_count = 1
                last_reject_rules = rule_ids_after

                if not patch_applied:
                    needs_generation = True
                    _persist_state(session_path, state, "Loop", "Patch no-op -> regenerate")
                    continue
                if no_progress_count >= no_progress_limit:
                    needs_generation = True
                    _persist_state(session_path, state, "Loop", "No progress after patch -> regenerate")
                    continue
                needs_generation = False
                continue

            decision = "Pass"
            last_reject_rules = None
            no_progress_count = 0
            needs_generation = False
        else:
            last_reject_rules = None
            no_progress_count = 0
            needs_generation = False

        node_d_windtunnel(
            session_path,
            state,
            suite_id=suite_id,
            replications=replications,
            seed=seed,
            perturb=perturb,
            budget_sweep=budget_sweep,
            perturb_sweep=perturb_sweep,
        )
        state, hard_fail, _hard_rule_ids = node_e_synthesis(
            session_path,
            state,
            suite_id=suite_id,
            run_config=run_config,
        )
        state.iteration_count += 1
        _persist_state(session_path, state, "Loop", f"Iteration {state.iteration_count} complete")

        if hard_fail:
            state, patch_applied, _ = node_f_revise(
                session_path,
                state,
                patch_mode=patch_mode,
                include_advice=include_advice,
            )
            if not patch_applied:
                needs_generation = True
                _persist_state(session_path, state, "Loop", "Patch no-op after gate fail -> regenerate")
                continue

            for verify_idx in range(max(1, verify_rounds)):
                state, decision_after, rule_ids_after, _, _ = node_c_audit(session_path, state, suite_id)
                if decision_after == "Reject":
                    if last_reject_rules is not None and rule_ids_after == last_reject_rules:
                        no_progress_count += 1
                    else:
                        no_progress_count = 1
                    last_reject_rules = rule_ids_after

                    if no_progress_count >= no_progress_limit:
                        needs_generation = True
                        _persist_state(session_path, state, "Loop", "No progress after patch -> regenerate")
                    else:
                        needs_generation = False
                    break

                node_d_windtunnel(
                    session_path,
                    state,
                    suite_id=suite_id,
                    replications=replications,
                    seed=seed,
                    perturb=perturb,
                    budget_sweep=budget_sweep,
                    perturb_sweep=perturb_sweep,
                )
                state, verify_hard_fail, hard_rule_ids_after = node_e_synthesis(
                    session_path,
                    state,
                    suite_id=suite_id,
                    run_config=run_config,
                )
                if not verify_hard_fail:
                    state.stop_reason = "gate_passed"
                    _persist_state(session_path, state, "Loop", "Gate passed after verification")
                    should_exit = True
                    break

                if last_reject_rules is not None and set(hard_rule_ids_after) == last_reject_rules:
                    no_progress_count += 1
                else:
                    no_progress_count = 1
                last_reject_rules = set(hard_rule_ids_after)

                if verify_idx == max(1, verify_rounds) - 1 and no_progress_count >= no_progress_limit:
                    needs_generation = True
                    _persist_state(session_path, state, "Loop", "No progress after verification -> regenerate")
                else:
                    needs_generation = False

            if should_exit:
                break
            continue

        state.stop_reason = "gate_passed"
        _persist_state(session_path, state, "Loop", "Gate passed -> package")
        should_exit = True
        break

    if state.iteration_count >= rounds and not state.stop_reason:
        gate_decision = (state.audit_report or {}).get("gate_decision", {})
        if gate_decision.get("hard_fail"):
            state.stop_reason = "gate_fail"
            _persist_state(session_path, state, "Loop", "Gate failed at max rounds")
        else:
            state.stop_reason = "max_rounds_reached"
            _persist_state(session_path, state, "Loop", "Max rounds reached")

    if state.stop_reason == "gate_passed":
        node_g_package(session_path, state)
    return {"iterations": state.iteration_count, "stop_reason": state.stop_reason}
