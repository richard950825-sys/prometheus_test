from datetime import datetime
from typing import Dict, Any, List

from .audit_engine import audit_session
from .diffing import write_diff
from .generate import generate_candidates
from .metrics import (
    compute_and_write_for_session,
    compute_metrics_for_candidate,
    load_metrics_for_candidate,
    format_round_metrics_summary,
    validate_metrics_consistency,
    write_metrics,
)
from .oracle.rag_mini_runner import run_suite_for_candidate
from .patching import apply_patch
from .recommend import pick_champion, write_recommendation
from .storage import (
    reset_session,
    rebuild_index,
    migrate_legacy_ids_logic,
    iter_candidates,
    setup_round_dir,
    archive_round_artifacts,
    load_candidate,
    get_suite_path,
    save_iterate_manifest,
    save_trace,
)


def run_iterate(
    session_path: str,
    rounds: int = 2,
    population: int = 3,
    suite_id: str = "rag_mini",
    patch_mode: str = "strategy",
    include_patched: bool = False,
    reset: bool = False,
    include_advice: bool = False,
    replications: int = 1,
    seed: int | None = None,
    perturb: bool = False,
    budget_sweep: str | None = None,
    perturb_sweep: str | None = None,
) -> Dict[str, Any]:
    if reset:
        reset_session(session_path)

    manifest = {
        "session": session_path,
        "rounds": rounds,
        "population": population,
        "config": {
            "patch_mode": patch_mode,
            "replications": replications,
            "seed": seed,
            "perturb": perturb,
            "budget_sweep": budget_sweep,
            "perturb_sweep": perturb_sweep,
        },
        "start_time": datetime.now().isoformat(),
    }
    save_iterate_manifest(session_path, manifest)

    trace = []
    trace_md = ["# Evolution Trace", f"Started: {manifest['start_time']}", ""]
    current_champion_id = None

    print(f"Starting Evolution ({rounds} rounds)...")

    for r in range(rounds):
        print(f"\n=== Round {r} ===")
        round_dir = setup_round_dir(session_path, r)
        suite_path = None

        if r == 0:
            generate_candidates(session_path, n=population, reset=False)
        rebuild_index(session_path)
        migrate_legacy_ids_logic(session_path, inplace=True)

        inc_patch = r > 0

        has_workflow_ir = False
        for c in iter_candidates(session_path, include_patched=inc_patch):
            if c.workflow_ir:
                has_workflow_ir = True
                break

        suite_arg = suite_id if has_workflow_ir else None
        if suite_arg:
            suite_path = get_suite_path(suite_arg)
            if not suite_path:
                print(f"Suite {suite_arg} not found.")
            else:
                print(f"Running suite '{suite_arg}' on candidates...")
                count = 0
                for c in iter_candidates(session_path, include_patched=True):
                    run_suite_for_candidate(
                        session_path,
                        c.id,
                        str(suite_path),
                        replications=replications,
                        seed=seed,
                        perturb=perturb,
                        budget_sweep=budget_sweep,
                        perturb_sweep=perturb_sweep,
                    )
                    count += 1
                print(f"Executed suite on {count} candidates.")

        compute_and_write_for_session(session_path, include_patched=True, suite_id=suite_id)

        sweeps_enabled = bool(budget_sweep or perturb_sweep)
        if suite_path:
            refreshed = []
            for c in iter_candidates(session_path, include_patched=True):
                ok, issues = validate_metrics_consistency(
                    session_path,
                    c.id,
                    require_sweeps=sweeps_enabled,
                )
                if ok:
                    continue
                print(f"Metrics mismatch for {c.id}: {issues}. Re-running sweeps/metrics.")
                run_suite_for_candidate(
                    session_path,
                    c.id,
                    str(suite_path),
                    replications=replications,
                    seed=seed,
                    perturb=perturb,
                    budget_sweep=budget_sweep,
                    perturb_sweep=perturb_sweep,
                )
                metrics_dict = compute_metrics_for_candidate(session_path, c.id, suite_id=suite_id)
                write_metrics(session_path, c.id, metrics_dict)
                refreshed.append(c.id)
            if refreshed:
                print(f"Recomputed metrics for {len(refreshed)} candidates to enforce sweep consistency.")
        elif sweeps_enabled:
            print("Warning: sweeps requested but suite file not found; metrics consistency cannot be enforced.")

        audit_session(session_path, include_patched=inc_patch, suite_id=suite_id, migrate_ids=False)

        rec_data, md_content = pick_champion(session_path, include_patched=True)
        if not rec_data:
            print("Error: No recommendation found.")
            break
        write_recommendation(session_path, rec_data, md_content)

        winner = rec_data["winner"]
        win_id = winner["id"]
        win_score = winner["final_score"]
        win_risks = winner["risk_count"]
        win_strat = winner["strategy"]
        is_patched = "_patched" in win_id

        print(f"Round {r} Champion: {win_id} (Score: {win_score}, Risks: {win_risks})")

        win_metrics = []
        try:
            win_metrics = load_metrics_for_candidate(session_path, win_id)
        except Exception:
            pass

        trace_item = {
            "round": r,
            "champion": winner,
            "metrics": win_metrics,
            "timestamp": datetime.now().isoformat(),
        }
        trace.append(trace_item)

        patched_status = " (patched)" if is_patched else ""
        metrics_str = format_round_metrics_summary(win_metrics)
        md_summary = f"**Round {r}**: Champion `{win_id}`{patched_status} - Score {win_score}, Risks {win_risks}, Strategy `{win_strat}`.{metrics_str}"
        trace_md.append(md_summary)

        archive_round_artifacts(session_path, round_dir, rec_data)

        if r > 0 and win_id == current_champion_id and win_risks == 0:
            print("Stable optimal champion found (Risks=0). Stopping early.")
            trace_md.append("> **Stablized**: No further improvements needed.")
            break

        current_champion_id = win_id

        if r < rounds - 1:
            should_patch = True
            if is_patched and not include_patched:
                print(f"Champion {win_id} is already patched. Skipping re-patch (allow_multi_patch=False).")
                trace_md.append("> **Skip Patch**: Champion already patched.")
                should_patch = False

            if should_patch:
                print(f"Patching champion {win_id}...")
                try:
                    patch_applied = apply_patch(session_path, win_id, apply_advice=include_advice, mode=patch_mode)

                    if not patch_applied:
                        print(f"Patch no-op: {win_id} already compliant.")
                        trace_md.append("> **Skip Patch**: no-op (candidate already compliant).")
                        if win_risks == 0 or win_id == current_champion_id:
                            print("Champion stabilized after no-op patch.")
                            trace_md.append("> **Stabilized**: Champion stabilized after no-op.")
                            break
                        continue

                    patched_id = f"{win_id}_patched"
                    rebuild_index(session_path)
                    print(f"Generating diff for {win_id} -> {patched_id}...")
                    try:
                        write_diff(session_path, win_id, patched_id)
                        p_cand = load_candidate(session_path, patched_id)
                        patch_notes = []
                        wf = p_cand.workflow_ir
                        prop = p_cand.proposal
                        if wf:
                            patch_notes = wf.patch_notes
                        elif prop:
                            patch_notes = prop.patch_notes
                        trace_md.append(f"> **Patch Applied**: `{patched_id}` created.")
                        if patch_notes:
                            trace_md.append("> Rules fixed:")
                            for note in patch_notes[:5]:
                                trace_md.append(f"> - {note}")
                            if len(patch_notes) > 5:
                                trace_md.append(f"> - ... ({len(patch_notes)-5} more)")
                    except Exception as e:
                        print(f"Diff generation skipped/failed: {e}")
                        trace_md.append(f"> Patch applied but diff generation failed: {e}")

                except Exception as e:
                    print(f"Patch failed: {e}")
                    trace_md.append(f"> **Patch Failed**: {e}")

    trace_path = save_trace(session_path, trace, trace_md)
    print("\nEvolution completed.")
    print(f"Final Champion: {current_champion_id}")
    print(f"Trace saved to: {trace_path}")
    return {"count": len(trace)}
