from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import SessionMetadata
from .storage import (
    write_json,
    init_session_structure,
    load_all_audits,
    load_candidate,
    iter_candidates,
    get_diff_summary,
    load_recommedation_md,
    migrate_legacy_ids_logic,
    rebuild_index,
    get_candidate_path,
    get_suite_path,
)


class EvoCore:
    def __init__(self, base_dir: str = "sessions"):
        self.base_dir = Path(base_dir)

    def init_session(self, name: str):
        session_dir = self.base_dir / name
        if session_dir.exists():
            raise FileExistsError(f"Session '{name}' already exists.")
        init_session_structure(str(session_dir))
        metadata = SessionMetadata(name=name)
        write_json(session_dir / "metadata.json", metadata.model_dump(mode="json"))
        print(f"Session '{name}' initialized at {session_dir}")

    def migrate_ids(self, session_path: str, inplace: bool = False):
        migrate_legacy_ids_logic(session_path, inplace)

    def _update_candidate_index(self, session_path: str):
        rebuild_index(session_path)

    def _get_candidate_path(self, session_path: str, candidate_id: str) -> Path:
        return get_candidate_path(session_path, candidate_id)

    def generate(self, session_path: str, n: int = 3, reset: bool = False):
        from .generate import generate_candidates
        generate_candidates(session_path, n=n, reset=reset)

    def patch(self, session_path: str, candidate_id: str, apply_advice: bool = False, mode: str = "quick") -> bool:
        from .patching import apply_patch
        return apply_patch(session_path, candidate_id, apply_advice=apply_advice, mode=mode)

    def recommend(self, session_path: str, include_patched: bool = False):
        from .recommend import recommend_best
        recommend_best(session_path, include_patched=include_patched)

    def diff(self, session_path: str, base_id: str, target_id: str):
        from .diffing import write_diff
        write_diff(session_path, base_id, target_id)

    def audit(self, session_path: str, include_patched: bool = False):
        from .audit_engine import audit_session
        audit_session(session_path, include_patched=include_patched)

    def report(self, session_path: str, include_patched: bool = False):
        from .metrics import format_key_metrics_lines, format_windtunnel_metrics_lines, load_metrics_for_candidate
        session_dir = Path(session_path)
        results = load_all_audits(session_path, include_patched=include_patched)
        report_lines = ["# Session Audit Report", f"Date: {datetime.now()}", "", "## Summary"]
        if not results:
            report_lines.append("No audit results found.")
        else:
            passed_count = sum(1 for r in results if r.passed)
            total = len(results)
            report_lines.append(f"Total Candidates: {total}")
            report_lines.append(f"Passed: {passed_count}")
            report_lines.append(f"Failed: {total - passed_count}")
        report_lines.append("")
        report_lines.append("## Detailed Results")
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            icon = "OK" if r.passed else "FAIL"
            patched_tag = " (PATCHED)" if "_patched" in r.candidate_id else ""
            report_lines.append(f"### Candidate {r.candidate_id} {icon} ({status}){patched_tag}")
            report_lines.append(f"- **Score**: {r.score}")
            try:
                cand = load_candidate(session_path, r.candidate_id)
                strat = cand.strategy or (cand.proposal.strategy if cand.proposal else "Unknown")
                report_lines.append(f"- **Strategy**: {strat}")
            except Exception:
                pass
            try:
                report_lines.extend(format_key_metrics_lines(session_path, r.candidate_id))
            except Exception as e:
                report_lines.append(f"  - Error loading metrics: {e}")
            try:
                cand_metrics = load_metrics_for_candidate(session_path, r.candidate_id)
                report_lines.extend(format_windtunnel_metrics_lines(cand_metrics))
            except Exception as e:
                report_lines.append(f"  - Error loading windtunnel metrics: {e}")
            if r.violations:
                report_lines.append("- **Violations**:")
                for v in r.violations:
                    report_lines.append(f"  - [{v.severity.value}] {v.rule_id}: {v.message}")
                    if v.fix_suggestion:
                        report_lines.append(f"    Fix: {v.fix_suggestion}")
            else:
                report_lines.append("- No violations found.")
            summary = list(get_diff_summary(session_path, r.candidate_id))
            if summary:
                report_lines.append("- **Comparison**:")
                compare_dir = session_dir / "compare"
                if compare_dir.exists():
                    for df in compare_dir.glob(f"diff_*_vs_{r.candidate_id}.md"):
                        report_lines.append(f"  - [View Diff]({df.name})")
            report_lines.append("")
        rec_md_content = load_recommedation_md(session_path)
        if rec_md_content:
            report_lines.append("")
            report_lines.append("---")
            report_lines.append("")
            report_lines.append(rec_md_content)
        report_file = session_dir / "report.md"
        report_file.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"Report generated at {report_file}")

    def generate_metrics(self, session_path: str, include_patched: bool = False, suite_id: str = "rag_mini"):
        from .metrics import compute_and_write_for_session
        compute_and_write_for_session(session_path, include_patched=include_patched, suite_id=suite_id)

    def run_suite(
        self,
        session_path: str,
        suite_name: str = "rag_mini",
        include_patched: bool = False,
        replications: int = 1,
        seed: Optional[int] = None,
        perturb: bool = False,
        budget_sweep: Optional[str] = None,
        perturb_sweep: Optional[str] = None,
    ):
        from .oracle.rag_mini_runner import run_suite_for_candidate
        suite_file = get_suite_path(suite_name)
        if not suite_file:
            print(f"Suite {suite_name} not found.")
            return
        print(f"Running suite '{suite_name}' on candidates...")
        count = 0
        for c in iter_candidates(session_path, include_patched=include_patched):
            run_suite_for_candidate(
                session_path,
                c.id,
                str(suite_file),
                replications=replications,
                seed=seed,
                perturb=perturb,
                budget_sweep=budget_sweep,
                perturb_sweep=perturb_sweep,
            )
            count += 1
        print(f"Executed suite on {count} candidates.")

    def iterate(
        self,
        session_path: str,
        rounds: int = 3,
        population: int = 3,
        topk: int = 1,
        patch_mode: str = "strategy",
        include_advice: bool = False,
        reset: bool = False,
        allow_multi_patch: bool = False,
        suite: Optional[str] = None,
        replications: int = 1,
        seed: Optional[int] = None,
        perturb: bool = False,
        budget_sweep: Optional[str] = None,
        perturb_sweep: Optional[str] = None,
    ):
        suite_id = suite or "rag_mini"
        return self.forge_run(
            session_path=session_path,
            rounds=rounds,
            population=population,
            patch_mode=patch_mode,
            include_advice=include_advice,
            reset=reset,
            suite_id=suite_id,
            replications=replications,
            seed=seed,
            perturb=perturb,
            budget_sweep=budget_sweep,
            perturb_sweep=perturb_sweep,
        )

    def forge_run(
        self,
        session_path: str,
        rounds: int = 2,
        population: int = 3,
        patch_mode: str = "strategy",
        include_advice: bool = False,
        reset: bool = False,
        suite_id: str = "rag_mini",
        replications: int = 1,
        seed: Optional[int] = None,
        perturb: bool = False,
        budget_sweep: Optional[str] = None,
        perturb_sweep: Optional[str] = None,
    ):
        from .forge.machine import run_forge_machine
        return run_forge_machine(
            session_path=session_path,
            rounds=rounds,
            population=population,
            suite_id=suite_id,
            patch_mode=patch_mode,
            include_advice=include_advice,
            reset=reset,
            replications=replications,
            seed=seed,
            perturb=perturb,
            budget_sweep=budget_sweep,
            perturb_sweep=perturb_sweep,
        )
