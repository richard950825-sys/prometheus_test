from datetime import datetime
from pathlib import Path

from .storage import save_diff, get_candidate_path, load_candidate


def write_diff(session_path: str, base_id: str, target_id: str):
    session_dir = Path(session_path)
    compare_dir = session_dir / "compare"
    compare_dir.mkdir(exist_ok=True)

    base_file = get_candidate_path(session_path, base_id)
    target_file = get_candidate_path(session_path, target_id)

    if not base_file.exists():
        raise FileNotFoundError(f"Base candidate {base_id} not found")
    if not target_file.exists():
        raise FileNotFoundError(f"Target candidate {target_id} not found")

    base = load_candidate(session_path, base_id)
    target = load_candidate(session_path, target_id)

    changes = []

    def check_change(path, b_val, t_val, reason_guess=""):
        if b_val != t_val:
            changes.append({
                "path": path,
                "from": b_val,
                "to": t_val,
                "reason": reason_guess,
            })

    p_base = base.proposal
    p_target = target.proposal
    wf_base = base.workflow_ir
    wf_target = target.workflow_ir

    if wf_base and wf_target:
        check_change("workflow_ir.steps.count", len(wf_base.steps), len(wf_target.steps), "Step added/removed")
        if wf_base.controls and wf_target.controls:
            b_budget = wf_base.controls.budget
            t_budget = wf_target.controls.budget
            if b_budget and t_budget:
                check_change(
                    "workflow_ir.controls.budget.max_total_turns",
                    b_budget.max_total_turns,
                    t_budget.max_total_turns,
                    "W008 or manual",
                )
        if wf_base.controls and wf_target.controls:
            check_change(
                "workflow_ir.controls.fallbacks.count",
                len(wf_base.controls.fallbacks),
                len(wf_target.controls.fallbacks),
                "Fallback added",
            )
        base_has_verify = any(s.action == "verify" for s in wf_base.steps)
        target_has_verify = any(s.action == "verify" for s in wf_target.steps)
        if not base_has_verify and target_has_verify:
            changes.append({
                "path": "workflow_ir.steps.verify",
                "from": "missing",
                "to": "added",
                "reason": "W006",
            })
        diff_id = f"{base_id}_vs_{target_id}"
        json_out = {
            "base_id": base_id,
            "target_id": target_id,
            "type": "workflow_ir",
            "changes": changes,
            "patch_notes": wf_target.patch_notes,
        }
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        md_lines = [
            f"# Diff: {base_id} vs {target_id}",
            f"Date: {timestamp}",
            f"**Type**: WorkflowIR",
            "",
            "## Change Summary",
        ]
        if not changes:
            md_lines.append("No structural changes detected.")
        else:
            for c in changes:
                md_lines.append(f"- **{c['path']}**: `{c['from']}` -> `{c['to']}` ({c['reason']})")
        if wf_target.patch_notes:
            md_lines.append("")
            md_lines.append("## Patch Notes")
            for note in wf_target.patch_notes:
                md_lines.append(f"- {note}")
        save_diff(session_path, base_id, target_id, json_out, "\n".join(md_lines))
        print(f"Diff generated for {base_id} vs {target_id}")
        return

    if wf_base or wf_target:
        print("Warning: Mismatched candidate types (one WorkflowIR, one not). Diff skipped.")
        return

    if not p_base or not p_target:
        print("Warning: One or both candidates lack proposal. Diff skipped.")
        return

    json_out = {
        "base": base_id,
        "target": target_id,
        "strategies": {"base": p_base.strategy, "target": p_target.strategy},
        "changes": changes,
        "patch_notes": p_target.patch_notes,
    }
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md_lines = [
        f"# Diff: {base_id} vs {target_id}",
        f"Date: {timestamp}",
        f"**Strategies**: {p_base.strategy} -> {p_target.strategy}",
        "",
        "## Change Summary",
    ]
    if not changes:
        md_lines.append("No structural changes detected.")
    else:
        for c in changes:
            md_lines.append(f"- **{c['path']}**: `{c['from']}` -> `{c['to']}` ({c['reason']})")
    if p_target.patch_notes:
        md_lines.append("")
        md_lines.append("## Patch Notes (Target)")
        for note in p_target.patch_notes:
            md_lines.append(f"- {note}")
    save_diff(session_path, base_id, target_id, json_out, "\n".join(md_lines))
    print(f"Diff generated for {base_id} vs {target_id}")
