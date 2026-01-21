"""
Storage utilities for evo.
Handles all file I/O, candidate loading/saving, and index management.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Iterator, Tuple
import shutil
import uuid
import yaml
from .models import Candidate, AuditResult, MetricsOutput, Metric, Ruleset, Rule

def load_ruleset(path: str = "ruleset.yaml") -> Dict[str, Any]:
    """Load ruleset from YAML."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file and return its contents as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any], indent: int = 2) -> None:
    """Write data to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_index(session_path: str) -> Dict[str, str]:
    """Load the candidate index from a session."""
    session_dir = Path(session_path)
    index_file = session_dir / "candidates" / "index.json"
    
    if index_file.exists():
        try:
            return read_json(index_file)
        except json.JSONDecodeError:
            print("Warning: index.json corrupted.")
            return {}
    return {}


def rebuild_index(session_path: str) -> Dict[str, str]:
    """
    Scans candidates directory and builds an index mapping ID -> file path.
    Returns the rebuilt index.
    """
    session_dir = Path(session_path)
    candidates_dir = session_dir / "candidates"
    index_file = candidates_dir / "index.json"
    
    if not candidates_dir.exists():
        return {}

    index = {}
    known_ids = set()
    
    for f_path in candidates_dir.glob("*.json"):
        if f_path.name == "index.json":
            continue
        try:
            data = read_json(f_path)
            cand_id = data.get("id") or data.get("candidate_id")
            
            if cand_id:
                if cand_id in known_ids:
                    print(f"CRITICAL WARNING: Duplicate ID {cand_id} found in {f_path.name}. Index will point to last seen.")
                
                index[cand_id] = f"candidates/{f_path.name}"
                known_ids.add(cand_id)
            else:
                print(f"Warning: No 'id' found in {f_path.name}")
                
        except Exception as e:
            print(f"Warning: Failed to read candidate file {f_path}: {e}")
    
    write_json(index_file, index)
    print(f"Rebuilt candidate index: {len(index)} items.")
    return index


def get_candidate_path(session_path: str, candidate_id: str) -> Path:
    """
    Resolves candidate file path using index.json.
    Falls back to direct path or directory scan if index lookup fails.
    """
    session_dir = Path(session_path)
    candidates_dir = session_dir / "candidates"
    index_file = candidates_dir / "index.json"
    
    # Try direct path first (legacy/fallback)
    legacy_path = candidates_dir / f"cand_{candidate_id}.json"
    if legacy_path.exists():
        return legacy_path

    # Try index
    if index_file.exists():
        try:
            index = read_json(index_file)
            
            if candidate_id in index:
                rel_path = index[candidate_id]
                return session_dir / rel_path
            else:
                # Provide helpful error
                known_ids = list(index.keys())[:5]
                msg = f"Candidate ID '{candidate_id}' not found in index."
                if known_ids:
                    msg += f" Known IDs (partial): {known_ids}"
                raise FileNotFoundError(msg)
        except json.JSONDecodeError:
            print("Warning: index.json corrupted. Scanning directory manually as fallback...")
    
    # Fallback: Manual Scan
    print(f"Scanning directory for candidate {candidate_id}...")
    for f_path in candidates_dir.glob("*.json"):
        if f_path.name == "index.json":
            continue
        try:
            data = read_json(f_path)
            if data.get("id") == candidate_id:
                return f_path
        except:
            pass
    
    raise FileNotFoundError(f"Candidate file for ID '{candidate_id}' not found in {candidates_dir}")


def load_candidate(session_path: str, candidate_id: str) -> Candidate:
    """Load a candidate by ID from a session."""
    cand_path = get_candidate_path(session_path, candidate_id)
    data = read_json(cand_path)
    return Candidate(**data)


def save_candidate(session_path: str, candidate: Candidate, update_index: bool = True) -> Path:
    """
    Save a candidate to the session's candidates directory.
    Optionally updates the index after saving.
    Returns the path to the saved file.
    """
    session_dir = Path(session_path)
    candidates_dir = ensure_dir(session_dir / "candidates")
    
    cand_path = candidates_dir / f"cand_{candidate.id}.json"
    
    
    with open(cand_path, "w", encoding="utf-8") as f:
        f.write(candidate.model_dump_json(indent=2))
    
    if update_index:
        rebuild_index(session_path)
    
    return cand_path


def init_session_structure(session_path: str) -> Path:
    """Initialize session directory and default requirements."""
    session_dir = ensure_dir(Path(session_path))
    
    # Create default structure
    ensure_dir(session_dir / "candidates")
    ensure_dir(session_dir / "audits")
    ensure_dir(session_dir / "evidence")
    
    # Requirements
    req_file = Path("requirements.json")
    if not req_file.exists():
        # Default Template
        template = {
          "project_name": f"{session_dir.name}_project",
          "goal": "Explain the system goal here",
          "features": ["Feature A", "Feature B"],
          "traffic": {
            "peak_rps": 5000,
            "avg_rps": 800,
            "read_write_ratio": "70:30",
            "payload_size_kb": 2
          },
          "slo": {
            "p95_latency_ms": 200,
            "error_rate": 0.01,
            "availability": 0.999
          },
          "constraints": {
            "data_residency": "EU",
            "compliance": ["GDPR"],
            "monthly_budget_usd": 20000,
            "team_size": 2,
            "time_to_mvp_weeks": 4
          },
          "preferences": {
            "prefer_managed_services": True,
            "prefer_kubernetes": False,
            "languages": ["python", "node"],
            "cloud": "gcp|aws|azure|any"
          }
        }
        write_json(req_file, template)
        print("Created new questionnaire-style 'requirements.json'.")
    
    return session_dir


def load_requirements(session_path: str) -> Dict[str, Any]:
    """Load requirements from session or root fallback."""
    session_dir = Path(session_path)
    # Check session-specific first
    req_file = session_dir / "requirements.json"
    if req_file.exists():
        return read_json(req_file)
    
    # Fallback to root
    req_file = Path("requirements.json")
    if req_file.exists():
        return read_json(req_file)
    return {}


def load_metadata(session_path: str) -> Dict[str, Any]:
    """Load session metadata."""
    session_dir = Path(session_path)
    meta_file = session_dir / "metadata.json"
    if meta_file.exists():
        return read_json(meta_file)
    return {}


def save_metadata(session_path: str, data: Dict[str, Any]):
    """Save session metadata."""
    session_dir = Path(session_path)
    write_json(session_dir / "metadata.json", data)


def clear_session_data(session_path: str):
    """Clear candidates, audits, and evidence for a reset."""
    session_dir = Path(session_path)
    for dirname in ["candidates", "audits", "evidence", "evolution", "compare", "state_machine", "gates"]:
        d = session_dir / dirname
        if d.exists():
            shutil.rmtree(d)
    
    # Also remove report/rec files
    for f in ["recommendation.json", "recommendation.md", "audit_results.json", "report.md", "forge_state.json"]:
        fp = session_dir / f
        if fp.exists():
            fp.unlink()


def reset_session(session_path: str):
    """Clear session data and re-initialize structure."""
    clear_session_data(session_path)
    init_session_structure(session_path)


def iter_candidates(session_path: str, include_patched: bool = True) -> Iterator[Candidate]:
    """Iterate over all loaded candidates in the session."""
    session_dir = Path(session_path)
    candidates_dir = session_dir / "candidates"
    if not candidates_dir.exists():
        return

    # Use index if available
    index_file = candidates_dir / "index.json"
    files_to_read = []
    
    if index_file.exists():
        try:
            index = read_json(index_file)
            for v in index.values():
                files_to_read.append(session_dir / v)
        except:
            files_to_read = list(candidates_dir.glob("cand_*.json"))
    else:
        files_to_read = list(candidates_dir.glob("cand_*.json"))

    for f_path in files_to_read:
        if not f_path.exists(): continue
        if not include_patched and "_patched" in f_path.name: continue
        
        try:
            yield Candidate(**read_json(f_path))
        except Exception as e:
            print(f"Error loading candidate {f_path}: {e}")


def load_ruleset(path: str = "ruleset.yaml") -> Dict[str, Any]:
    """Load ruleset YAML."""
    p = Path(path)
    if not p.exists():
        return {"rules": []}
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def save_audit_result(session_path: str, result: AuditResult):
    """Save a single audit result."""
    session_dir = Path(session_path)
    audits_dir = ensure_dir(session_dir / "audits")
    path = audits_dir / f"audit_{result.candidate_id}.json"
    write_json(path, result.model_dump(mode='json'))


def clear_audits(session_path: str):
    """Clear audits directory."""
    session_dir = Path(session_path)
    audits_dir = session_dir / "audits"
    if audits_dir.exists():
        shutil.rmtree(audits_dir)
    ensure_dir(audits_dir)


def save_recommendation(session_path: str, data: Dict, md_content: str):
    """Save recommendation artifacts."""
    session_dir = Path(session_path)
    write_json(session_dir / "recommendation.json", data)
    (session_dir / "recommendation.md").write_text(md_content, encoding="utf-8")


def load_all_audits(session_path: str, include_patched: bool = False) -> List[AuditResult]:
    """Load all audit results."""
    session_dir = Path(session_path)
    audits_dir = session_dir / "audits"
    results = []
    
    if not audits_dir.exists():
        # Fallback legacy
        f = session_dir / "audit_results.json"
        if f.exists():
            data = read_json(f)
            return [AuditResult(**r) for r in data]
        return []

    for f in audits_dir.glob("audit_*.json"):
        if not include_patched and "_patched" in f.name: continue
        try:
            results.append(AuditResult(**read_json(f)))
        except: pass
    return results


def load_recommendation(session_path: str) -> Dict[str, Any]:
    """Load recommendation JSON if available."""
    session_dir = Path(session_path)
    rec_json = session_dir / "recommendation.json"
    if rec_json.exists():
        return read_json(rec_json)
    return {}


def save_metrics(session_path: str, candidate_id: str, output: MetricsOutput):
    """Save metrics for a candidate."""
    session_dir = Path(session_path)
    out_dir = ensure_dir(session_dir / "candidates" / f"cand_{candidate_id}" / "outputs")
    write_json(out_dir / "metrics.json", output.model_dump(mode='json'))


def get_suite_path(suite_name: str) -> Optional[Path]:
    """Return suite path if it exists."""
    suite_file = Path("eval_suites") / f"{suite_name}.json"
    if suite_file.exists():
        return suite_file
    return None


def load_metrics(session_path: str, candidate_id: str) -> List[Dict[str, Any]]:
    """Load metrics raw dicts for a candidate."""
    session_dir = Path(session_path)
    m_file = session_dir / "candidates" / f"cand_{candidate_id}" / "outputs" / "metrics.json"
    if m_file.exists():
        try:
            data = read_json(m_file)
            return data.get("metrics", [])
        except: pass
    return []


def save_diff(session_path: str, base_id: str, target_id: str, json_data: Dict, md_content: str):
    """Save comparison artifacts."""
    session_dir = Path(session_path)
    comp_dir = ensure_dir(session_dir / "compare")
    diff_id = f"{base_id}_vs_{target_id}"
    
    write_json(comp_dir / f"diff_{diff_id}.json", json_data)
    (comp_dir / f"diff_{diff_id}.md").write_text(md_content, encoding="utf-8")


def get_diff_summary(session_path: str, target_id: str) -> Iterator[str]:
    """Retrieve summary lines from the relevant diff file for a target candidate."""
    session_dir = Path(session_path)
    comp_dir = session_dir / "compare"
    if not comp_dir.exists(): return
    
    # Look for any diff ending in _vs_{target_id}.md
    files = list(comp_dir.glob(f"diff_*_vs_{target_id}.md"))
    if not files: return
    
    # Read first match
    content = files[0].read_text(encoding="utf-8")
    if "## Change Summary" in content:
        summary = content.split("## Change Summary")[1]
        for line in summary.split("\n"):
            line = line.strip()
            if line.startswith("##"): break
            if line.startswith("-"):
                yield line


def migrate_legacy_ids_logic(session_path: str, inplace: bool):
    """Migrate legacy IDs to UUIDs."""
    session_dir = Path(session_path)
    candidates_dir = session_dir / "candidates"
    if not candidates_dir.exists():
        return

    print(f"Migrating IDs in '{session_path}' (Inplace: {inplace})...")
    migrations = {} 

    for f_path in candidates_dir.glob("*.json"):
        if f_path.name == "index.json": continue
        
        try:
            data = read_json(f_path)
            old_id = data.get("id")
            
            # Check compliance
            is_compliant = False
            if old_id and old_id.startswith("wf-"):
                    try:
                        uuid.UUID(old_id[3:])
                        is_compliant = True
                    except ValueError: pass
            
            if is_compliant: continue

            # Generate New ID
            new_id = f"wf-{uuid.uuid4()}"
            migrations[old_id] = {"path": f_path, "new_id": new_id, "data": data}
            print(f"Planned: {old_id} -> {new_id}")
            
        except Exception as e:
            print(f"Error reading {f_path}: {e}")

    if not migrations:
        print("No legacy IDs found.")
        return

    if not inplace:
        print("Dry run complete. Use --inplace to execute.")
        return

    for old_id, info in migrations.items():
        f_path = info["path"]
        new_id = info["new_id"]
        data = info["data"]
        
        data["legacy_id"] = old_id
        # Strategy heuristic
        strategies = ["cost-optimized", "reliability-optimized", "throughput-optimized"]
        for s in strategies:
            if s in old_id:
                data["strategy"] = s
                break
        
        data["id"] = new_id
        
        new_path = candidates_dir / f"cand_{new_id}.json"
        write_json(new_path, data)
        f_path.unlink()
        print(f"Migrated: {f_path.name} -> {new_path.name}")
    
    rebuild_index(session_path)


def save_evidence_run(session_path: str, candidate_id: str, suite_name: str, data: Dict):
    """Save evidence run result."""
    session_dir = Path(session_path)
    dest = ensure_dir(session_dir / "evidence" / "workflow" / candidate_id)
    write_json(dest / f"run_{suite_name}.json", data)


def get_evidence_runs_dir(session_path: str, candidate_id: str, suite_name: str) -> Path:
    """Return directory for replicated evidence runs."""
    session_dir = Path(session_path)
    return ensure_dir(session_dir / "evidence" / "workflow" / candidate_id / "runs" / suite_name)


def save_evidence_run_replication(
    session_path: str,
    candidate_id: str,
    suite_name: str,
    run_index: int,
    data: Dict,
) -> str:
    """Save a replicated evidence run."""
    dest = get_evidence_runs_dir(session_path, candidate_id, suite_name)
    filename = f"run_{run_index:03d}.json"
    write_json(dest / filename, data)
    return f"evidence/workflow/{candidate_id}/runs/{suite_name}/{filename}"


def save_evidence_run_aggregate(
    session_path: str,
    candidate_id: str,
    suite_name: str,
    data: Dict,
    filename: str = "run_aggregate.json",
) -> str:
    """Save aggregate stats for replicated runs."""
    dest = get_evidence_runs_dir(session_path, candidate_id, suite_name)
    write_json(dest / filename, data)
    return f"evidence/workflow/{candidate_id}/runs/{suite_name}/{filename}"


def get_evidence_sweeps_dir(session_path: str, candidate_id: str, suite_name: str) -> Path:
    """Return directory for sweep aggregates."""
    session_dir = Path(session_path)
    return ensure_dir(session_dir / "evidence" / "workflow" / candidate_id / "sweeps" / suite_name)


def save_evidence_sweep_run(
    session_path: str,
    candidate_id: str,
    suite_name: str,
    sweep_name: str,
    run_index: int,
    data: Dict,
) -> str:
    """Save a sweep run artifact."""
    sweeps_dir = get_evidence_sweeps_dir(session_path, candidate_id, suite_name)
    dest = ensure_dir(sweeps_dir / sweep_name)
    filename = f"run_{run_index:03d}.json"
    write_json(dest / filename, data)
    return f"evidence/workflow/{candidate_id}/sweeps/{suite_name}/{sweep_name}/{filename}"


def save_evidence_sweep_aggregate(
    session_path: str,
    candidate_id: str,
    suite_name: str,
    sweep_name: str,
    data: Dict,
) -> str:
    """Save sweep aggregate artifact."""
    sweeps_dir = get_evidence_sweeps_dir(session_path, candidate_id, suite_name)
    dest = ensure_dir(sweeps_dir / sweep_name)
    write_json(dest / "aggregate.json", data)
    return f"evidence/workflow/{candidate_id}/sweeps/{suite_name}/{sweep_name}/aggregate.json"


def load_evidence_run_summary(session_path: str, candidate_id: str, suite_name: str) -> Dict:
    """Load evidence run summary."""
    session_dir = Path(session_path)
    f = session_dir / "evidence" / "workflow" / candidate_id / f"run_{suite_name}.json"
    if f.exists():
        return read_json(f).get("summary", {})
    return {}


def load_evidence_run_aggregate(
    session_path: str,
    candidate_id: str,
    suite_name: str,
    filename: str = "run_aggregate.json",
) -> Dict[str, Any]:
    """Load aggregated evidence stats for replicated runs."""
    session_dir = Path(session_path)
    f = session_dir / "evidence" / "workflow" / candidate_id / "runs" / suite_name / filename
    if f.exists():
        return read_json(f)
    return {}


def load_evidence_sweep_aggregates(
    session_path: str,
    candidate_id: str,
    suite_name: str,
) -> List[Dict[str, Any]]:
    """Load sweep aggregate artifacts for a candidate."""
    sweeps_dir = get_evidence_sweeps_dir(session_path, candidate_id, suite_name)
    if not sweeps_dir.exists():
        return []
    aggregates: List[Dict[str, Any]] = []
    for sweep_dir in sweeps_dir.iterdir():
        if not sweep_dir.is_dir():
            continue
        agg_file = sweep_dir / "aggregate.json"
        if not agg_file.exists():
            continue
        try:
            aggregates.append({
                "path": f"evidence/workflow/{candidate_id}/sweeps/{suite_name}/{sweep_dir.name}/aggregate.json",
                "data": read_json(agg_file),
            })
        except Exception:
            continue
    return aggregates


def load_evidence_run(session_path: str, candidate_id: str, suite_name: str) -> Dict[str, Any]:
    """Load full evidence run output."""
    session_dir = Path(session_path)
    f = session_dir / "evidence" / "workflow" / candidate_id / f"run_{suite_name}.json"
    if f.exists():
        return read_json(f)
    return {}


def load_audit_result(session_path: str, candidate_id: str) -> Optional[AuditResult]:
    """Load a single audit result for a candidate."""
    session_dir = Path(session_path)
    audit_file = session_dir / "audits" / f"audit_{candidate_id}.json"
    if audit_file.exists():
        try:
            return AuditResult(**read_json(audit_file))
        except Exception:
            return None
    return None


def setup_round_dir(session_path: str, round_num: int) -> Path:
    """Create and return round directory."""
    session_dir = Path(session_path)
    r_dir = ensure_dir(session_dir / "evolution" / "rounds" / f"round_{round_num:02d}")
    return r_dir


def archive_round_artifacts(session_path: str, round_dir: Path, rec_data: Dict):
    """Copy current artifacts to round archive."""
    session_dir = Path(session_path)
    
    # Recommendation
    rec_json = session_dir / "recommendation.json"
    if rec_json.exists(): shutil.copy2(rec_json, round_dir / "recommendation.json")
    
    rec_md = session_dir / "recommendation.md"
    if rec_md.exists(): shutil.copy2(rec_md, round_dir / "recommendation.md")
    
    audit_res = session_dir / "audit_results.json"
    if audit_res.exists(): shutil.copy2(audit_res, round_dir / "audits_index.json")
    
    all_cands = [c["id"] for c in rec_data.get("all_candidates", [])]
    write_json(round_dir / "population.json", all_cands)


def load_suite(suite_name: str) -> Optional[Dict[str, Any]]:
    """Load an evaluation suite from the eval_suites directory."""
    suite_file = Path(f"eval_suites/{suite_name}.json")
    if suite_file.exists():
        return read_json(suite_file)
    return None


def read_text_file(path: Path) -> str:
    """Read a text file and return its content."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text_file(path: Path, content: str):
    """Write a text file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def get_evolution_dir(session_path: str) -> Path:
    """Ensure and return evolution directory."""
    session_dir = Path(session_path)
    return ensure_dir(session_dir / "evolution")


def save_trace(session_path: str, trace: List[Dict[str, Any]], trace_md_lines: List[str]) -> Path:
    """Save evolution trace artifacts."""
    evo_dir = get_evolution_dir(session_path)
    write_json(evo_dir / "trace.json", trace)
    md_path = evo_dir / "trace.md"
    write_text_file(md_path, "\n".join(trace_md_lines))
    return md_path


def save_iterate_manifest(session_path: str, manifest: Dict[str, Any]):
    """Save evolution manifest."""
    evo_dir = get_evolution_dir(session_path)
    write_json(evo_dir / "manifest.json", manifest)

def load_recommedation_md(session_path: str) -> str:
    """Load recommendation markdown content."""
    session_dir = Path(session_path)
    rec_md = session_dir / "recommendation.md"
    if rec_md.exists():
        return read_text_file(rec_md)
    return ""


def get_forge_state_path(session_path: str) -> Path:
    """Return path to forge_state.json for a session."""
    session_dir = Path(session_path)
    return session_dir / "forge_state.json"


def save_forge_state(session_path: str, data: Dict[str, Any]) -> Path:
    """Persist forge state to session."""
    path = get_forge_state_path(session_path)
    write_json(path, data)
    return path


def load_forge_state(session_path: str) -> Dict[str, Any]:
    """Load forge state if it exists."""
    path = get_forge_state_path(session_path)
    if path.exists():
        return read_json(path)
    return {}


def get_windtunnel_dir(session_path: str, candidate_id: str) -> Path:
    """Ensure and return windtunnel artifacts directory for a candidate."""
    session_dir = Path(session_path)
    return ensure_dir(session_dir / "evidence" / "workflow" / candidate_id / "windtunnel")


def save_windtunnel_spec(session_path: str, candidate_id: str, suite_name: str, data: Dict[str, Any]) -> str:
    """Save windtunnel spec artifact."""
    dest = get_windtunnel_dir(session_path, candidate_id)
    filename = f"spec_{suite_name}.json"
    write_json(dest / filename, data)
    return f"evidence/workflow/{candidate_id}/windtunnel/{filename}"


def save_windtunnel_report(session_path: str, candidate_id: str, suite_name: str, data: Dict[str, Any]) -> str:
    """Save windtunnel report artifact."""
    dest = get_windtunnel_dir(session_path, candidate_id)
    filename = f"report_{suite_name}.json"
    write_json(dest / filename, data)
    return f"evidence/workflow/{candidate_id}/windtunnel/{filename}"


def load_windtunnel_report(session_path: str, candidate_id: str, suite_name: str) -> Dict[str, Any]:
    """Load windtunnel report if available."""
    session_dir = Path(session_path)
    report_path = session_dir / "evidence" / "workflow" / candidate_id / "windtunnel" / f"report_{suite_name}.json"
    if report_path.exists():
        return read_json(report_path)
    return {}

