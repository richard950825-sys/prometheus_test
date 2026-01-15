import json
import os
import yaml
import random
from datetime import datetime
from pathlib import Path
import uuid
from typing import List, Dict, Any, Optional
from .models import (
    SessionMetadata, Candidate, AuditResult, Violation, Ruleset, Severity, Rule,
    WorkflowIR, Agent, Step, Controls, WorkflowBudget, WorkflowAcceptance, ArchitectureTest
)
from .storage import (
    read_json, write_json, ensure_dir, load_index, rebuild_index, 
    get_candidate_path, load_candidate, save_candidate
)

class EvoCore:
    def __init__(self, base_dir: str = "sessions"):
        self.base_dir = Path(base_dir)

    def init_session(self, name: str):
        session_dir = self.base_dir / name
        if session_dir.exists():
            raise FileExistsError(f"Session '{name}' already exists.")
        ensure_dir(session_dir)
        
        metadata = SessionMetadata(name=name)
        write_json(session_dir / "metadata.json", metadata.model_dump(mode='json'))
        
        print(f"Session '{name}' initialized at {session_dir}")

        # Create/Update requirements.json with new template
        req_file = Path("requirements.json")
        if not req_file.exists():
            template = {
              "project_name": f"{name}_project",
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
        else:
            print("Using existing 'requirements.json'.")

    def migrate_ids(self, session_path: str, inplace: bool = False):
        """
        Migrates legacy candidate IDs to strict 'wf-<uuid4>' format.
        Updates internal JSON content and filenames.
        Optionally updates audits if possible (challenging due to filenames, but we attempt).
        """
        session_dir = Path(session_path)
        candidates_dir = session_dir / "candidates"
        if not candidates_dir.exists():
            print("No candidates directory found.")
            return

        print(f"Migrating IDs in '{session_path}' (Inplace: {inplace})...")
        migrations = {} # old_id -> new_id

        # Phase 1: Calculate migrations
        for f_path in candidates_dir.glob("*.json"):
            if f_path.name == "index.json": continue
            
            try:
                data = read_json(f_path)
                old_id = data.get("id")
                
                # Check if already Migration Compliant (wf-UUID)
                is_compliant = False
                if old_id and old_id.startswith("wf-"):
                     suffix = old_id[3:]
                     try:
                         # verify if suffix is real UUID
                         uuid.UUID(suffix)
                         is_compliant = True
                     except ValueError:
                         pass
                
                if is_compliant:
                     # print(f"Skipping compliant ID: {old_id}")
                     continue

                # Generate New ID
                new_id = f"wf-{uuid.uuid4()}"
                migrations[old_id] = {"path": f_path, "new_id": new_id, "data": data}
                print(f"Planned: {old_id} -> {new_id}")
                
            except Exception as e:
                print(f"Error reading {f_path}: {e}")

        if not migrations:
            print("No legacy IDs found to migrate.")
            return

        if not inplace:
            print("Dry run complete. Use --inplace to execute.")
            return

        # Phase 2: Execute
        for old_id, info in migrations.items():
            f_path = info["path"]
            new_id = info["new_id"]
            data = info["data"]
            
            # Update Content
            data["legacy_id"] = old_id
            
            # Extract strategy from old ID if possible and save as tag if missing
            # heuristic: "reliability-optimized" in old_id
            strategies = ["cost-optimized", "reliability-optimized", "throughput-optimized"]
            for s in strategies:
                if s in old_id:
                   data["strategy"] = s
                   break
            
            data["id"] = new_id
            
            # Write New File
            new_path = candidates_dir / f"cand_{new_id}.json"
            write_json(new_path, data)
            
            # Delete Old File
            f_path.unlink()
            print(f"Migrated: {f_path.name} -> {new_path.name}")
        
        # Rebuild Index
        self._update_candidate_index(session_path)

    def _update_candidate_index(self, session_path: str):
        """Wrapper for storage.rebuild_index - scans candidates directory and builds index."""
        rebuild_index(session_path)

    def _get_candidate_path(self, session_path: str, candidate_id: str) -> Path:
        """Wrapper for storage.get_candidate_path - resolves candidate file path using index.json."""
        return get_candidate_path(session_path, candidate_id)

    def generate(self, session_path: str, n: int = 3, reset: bool = False):
        session_dir = Path(session_path)
        candidates_dir = session_dir / "candidates"
        audits_dir = session_dir / "audits"
        metadata_file = session_dir / "metadata.json"
        
        if not session_dir.exists():
            raise FileNotFoundError(f"Session path '{session_path}' not found.")
            
        # Refreshed metadata
        metadata = {
            "name": session_dir.name, 
            "created_at": str(datetime.now()),
            "generation_stats": {"attempts": 0, "failures": 0, "retries": 0}
        }
        
        # Reset Logic
        if reset:
            import shutil
            if candidates_dir.exists():
                shutil.rmtree(candidates_dir)
            if audits_dir.exists():
                shutil.rmtree(audits_dir)
            print("Resetting session: candidates and audits cleared.")
        
        candidates_dir.mkdir(exist_ok=True)
        
        req_file = Path("requirements.json")
        requirements = {}
        if req_file.exists():
            requirements = read_json(req_file)
        
        from .llm_gemini import generate_candidate
        # Legacy imports removed for Workflow IR pivot
        # from .models import Proposal, Architecture, Components, ComponentDetails, ReliabilityConfig, SLO, Acceptance, Experiments, ExperimentConfig, Risk, ChaosTest, Constraint
        
        def normalize_candidate(candidate: Candidate, requirements: Dict[str, Any]):
            if not candidate.proposal:
                return
            
            p = candidate.proposal
            notes = []
            
            # 1. Normalize Acceptance Constraints
            if not p.acceptance.hard_constraints:
                p.acceptance.hard_constraints = [
                    Constraint(metric="latency.p95_ms", op="<=", threshold=p.slo.p95_latency_ms),
                    Constraint(metric="errors.rate", op="<=", threshold=p.slo.error_rate)
                ]
                notes.append("Backfilled missing hard_constraints from SLO.")
                
            # 2. Normalize Load Test
            lt = p.experiments.load_test
            # Note: Pydantic fields might be None if Optional, but ExperimentConfig fields defined as int/str are required. 
            # If generated by Gemini via Pydantic model, they exist. 
            # However, if target_rps is optional in model (it is), we check it.
            if lt.target_rps is None:
                # default to 1000 or from requirements
                default_rps = 1000
                if "traffic" in requirements and "peak_rps" in requirements["traffic"]:
                    default_rps = int(requirements["traffic"]["peak_rps"])
                lt.target_rps = default_rps
                notes.append(f"Backfilled load_test.target_rps to {default_rps}")
                
            # duration_s is int in model, so it must exist to pass validation.
            # But if we want to enforce specific default if it was somehow 0 or low? 
            # User said: "If experiments.load_test missing duration_s" -> Model says duration_s is int (required).
            # So Gemini wouldn't have passed validation if it was missing. 
            # But let's assume we might relax model later or it passed with 0.
            # Let's check logic: if 0, update? Or just trust model validation?
            # User instruction: "If duration_s missing... default 900". 
            # Since strict Pydantic model requires it, it won't be "missing" in key sense, but maybe value sense?
            # Let's stick to target_rps which is Optional.
            
            # 3. Normalize Risks
            if not p.risks:
                p.risks.append(Risk(
                    title="Unknown risk: needs review",
                    mitigation="Run load/chaos tests and review dependencies"
                ))
                notes.append("Backfilled empty risks list.")
                
            # 4. Normalize Cache/Queue Types
            if p.architecture and p.architecture.components:
                comps = p.architecture.components
                if comps.cache is None:
                    # If None, create one with type="none"? Model says cache is Optional[ComponentDetails]
                    # User says: "If cache.type missing: default 'none'"
                    # If component itself is missing, let's leave it None or set to type=none?
                    # Rule `is_cache_missing` checks if it is None OR type=="none".
                    # So leaving it None is fine for logic, but for "Complete Structure" maybe user wants explicit object?
                    # Let's leave it as None if None. 
                    # WAIT, `is_cache_missing` returns True if None. 
                    # If we want to "fix" it? No, user normalized logic validation triggers.
                    # Normalized only "补齐". If cache is None, it means no cache component.
                    # If user means "if cache component exists but type is missing" -> Impossible with strict Pydantic (type is str).
                    # Maybe user implies "If architecture.components.cache is None => set it to type='none'"?
                    pass
                
                # Check Queue
                # Same logic.
            
            p.normalization_notes.extend(notes)

        candidates = []
        strategies = ["cost-optimized", "reliability-optimized", "throughput-optimized"]
        
        for i in range(1, n + 1):
            strategy = strategies[(i-1) % len(strategies)]
            print(f"Generating candidate {i}/{n} (Strategy: {strategy})...")
            
            try:
                candidate = generate_candidate(requirements, strategy_hint=strategy)
                
                # Strict UUID Logic (Phase 10)
                # Ignore metadata-based ID masking. 
                # Generate stable UUID
                new_uuid = f"wf-{uuid.uuid4()}"
                
                # Preserve strategy in field if not present (although model might have it)
                if hasattr(candidate, "workflow_ir") and candidate.workflow_ir:
                     # Add strategy tag if missing? (It's top level in Candidate model now?)
                     # Candidate model has 'strategy' field? Let's check models.py
                     pass
                
                # Assign ID
                candidate.id = new_uuid
                candidate.strategy = strategy # Explicit assignment
                
                # Save Immediately (Filename also strict)
                filename = f"cand_{candidate.id}.json"
                save_candidate(session_path, candidate, update_index=False)
                
                candidates.append(candidate)
                print(f"Candidate {i} saved to {filename}")
                
            except Exception as e:
                print(f"Gemini generation failed for candidate {i}: {e}. Using fallback.")
                metadata["generation_stats"]["failures"] += 1
                
                # Fallback
                candidate = Candidate(
                    id=f"fallback_{i}_{int(datetime.now().timestamp())}",
                    workflow_ir=WorkflowIR(
                        title=f"Fallback Workflow {i}",
                        goal="Robust RAG Workflow (Fallback)",
                        agents=[Agent(name="Orchestrator", role="coordinator", model="gemini-pro")],
                        steps=[Step(id="retrieve", agent="Orchestrator", action="retrieve", inputs=["query"], outputs=["docs"])],
                        controls=Controls(budget=WorkflowBudget(max_total_turns=10)),
                        acceptance=WorkflowAcceptance(tests=[])
                    )
                )
                normalize_candidate(candidate, requirements)
                cand_file = candidates_dir / f"cand_{candidate.id}.json"
                save_candidate(session_path, candidate, update_index=False)
                candidates.append(candidate)
            
        # Save Metadata
        if metadata_file.exists():
                 old_meta = read_json(metadata_file)
                 # merge or update? let's update for now
                 if isinstance(old_meta, dict):
                    old_meta.update(metadata)
                    metadata = old_meta
        
        
        # Update index
        self._update_candidate_index(session_path)

        write_json(metadata_file, metadata)

    def patch(self, session_path: str, candidate_id: str, apply_advice: bool = False, mode: str = "quick") -> bool:
        """
        Patches a candidate to fix audit violations.
        Returns True if actual changes were made, False if no-op (candidate already compliant).
        """
        session_dir = Path(session_path)
        # kand_file = candidates_dir / f"cand_{candidate_id}.json"
        # candidates_dir = session_dir / "candidates" # Not needed for lookup if using helper, but maybe needed for other things?
        # Let's keep common vars if used later. 
        # Helper uses session_path.
        
        cand_file = self._get_candidate_path(session_path, candidate_id)
        
        # Load Requirements for Strategy Context
        req_file = session_dir / "requirements.json"
        requirements = {}
        if not req_file.exists():
             # Fallback to root requirements if session one missing
             req_file = Path("requirements.json")
        if req_file.exists():
            requirements = read_json(req_file)
        
        if not cand_file.exists():
             raise FileNotFoundError(f"Candidate file not found: {cand_file}")
             
        cand = load_candidate(session_path, candidate_id)
            
        p = cand.proposal
        wf = cand.workflow_ir
        
        if not p and not wf:
            print("Candidate has no proposal or workflow data to patch.")
            return

        notes = []
        
        # --- WORKFLOW PATCHING ---
        if wf:
             # Fix W006: Missing Verify Step
             # Logic: If no 'verify' step, insert one after 'synthesis' or at end
             has_verify = any(s.action == "verify" for s in wf.steps)
             if not has_verify:
                  # Create Verify Step using correct model fields
                  from .models import Step
                  
                  # Find a suitable agent for verification (use first agent or "Critic" if exists)
                  verifier_agent = wf.agents[0].name if wf.agents else "Verifier"
                  for ag in wf.agents:
                       if "critic" in ag.name.lower() or "validator" in ag.name.lower():
                           verifier_agent = ag.name
                           break
                  
                  s_verify = Step(
                      id="step_verify",
                      agent=verifier_agent,
                      action="verify",
                      inputs=["response"],
                      outputs=["verified_response"],
                      guards={"max_retries": 2, "timeout_s": 60}
                  )
                  wf.steps.append(s_verify)
                  notes.append("Fixed W006: Injected 'verify' step.")
                  
             # Fix W008: Budget Too Loose
             # If max_total_turns > 50 -> scale down to 30
             if wf.controls and wf.controls.budget:
                  if wf.controls.budget.max_total_turns > 50:
                       old_val = wf.controls.budget.max_total_turns
                       wf.controls.budget.max_total_turns = 30
                       notes.append(f"Fixed W008: Reduced max_total_turns from {old_val} to 30.")
             
             # Apply Advice (A002 -> maybe "Add fallback")
             if apply_advice:
                  if not wf.controls.fallbacks:
                       from .models import Fallback
                       wf.controls.fallbacks = [
                            Fallback(trigger="retrieval_empty", action="rewrite_query", max_depth=1)
                       ]
                       notes.append("Applied Advice: Added basic fallback for retrieval_empty.")

        # --- ARCHITECTURE PATCHING ---
        if p:
            # --- Architecture Helpers ---

            
            # --- Helper: Check Cache Missing ---
            def is_cache_missing(prop):
                if not prop.architecture or not prop.architecture.components: return True
                c = prop.architecture.components.cache
                if c is None: return True
                if isinstance(c, dict): return c.get("type", "none") == "none"
                return getattr(c, "type", "none") == "none"

            # --- Helper: Check Queue Missing ---
            def is_queue_missing(prop):
                if not prop.architecture or not prop.architecture.components: return True
                q = prop.architecture.components.queue
                if q is None: return True
                if isinstance(q, dict): return q.get("type", "none") == "none"
                return getattr(q, "type", "none") == "none"

            # --- Basic Fixes (Quick Mode) ---
            
            # R001 Fix: High Retries
            if p.architecture and p.architecture.reliability:
                rel = p.architecture.reliability
                retries = rel.retries
                current_attempts = 0
                if isinstance(retries, dict):
                    current_attempts = int(retries.get("max_attempts", 0))
                
                if current_attempts >= 3:
                    rel.retries = {"max_attempts": 1, "backoff": "exponential"}
                    notes.append("Fixed R001: Set retries.max_attempts to 1 and backoff to exponential.")
            
            # R002 Fix: Client Timeout
            if p.architecture and p.architecture.reliability:
                rel = p.architecture.reliability
                timeouts = rel.timeouts_ms
                client_to = timeouts.get("client", 0)
                server_to = timeouts.get("server", 0)
                
                if client_to <= server_to and server_to > 0:
                    new_client = max(server_to + 300, server_to * 2)
                    timeouts["client"] = new_client
                    notes.append(f"Fixed R002: Increased client timeout to {new_client}ms (> server timeout {server_to}ms).")
            
            # R005 Fix: Retries without Idempotency
            # Logic: if retries > 0 and idempotency == 'not_supported' -> set to 'recommended', add Risk
            r005_triggered = False
            if p.architecture and p.architecture.reliability:
                rel = p.architecture.reliability
                retries = rel.retries
                attempts = 0
                if isinstance(retries, dict):
                     attempts = int(retries.get("max_attempts", 0))
                
                idempotency = rel.idempotency
                if attempts > 0 and idempotency == "not_supported":
                    r005_triggered = True
                    rel.idempotency = "recommended"
                    from .models import Risk
                    p.risks.append(Risk(
                        title="Idempotency Risk",
                        mitigation="Retries detected. Idempotency-Key recommended to prevent double-writes."
                    ))
                    notes.append("Fixed R005: Set idempotency to 'recommended' and added Risk due to active retries.")

            # C001 Fix: Data Residency Mismatch
            residency_req = requirements.get("constraints", {}).get("data_residency")
            if residency_req and residency_req.lower() in ["eu", "europe"]:
                # Check compliance
                needs_c001 = False
                if not p.compliance: needs_c001 = True
                elif not p.compliance.data_residency_statement: needs_c001 = True
                elif residency_req.lower() not in p.compliance.data_residency_statement.lower(): needs_c001 = True
                
                if needs_c001:
                    # Initialize compliance object if missing
                    if not p.compliance:
                        # from .models import Compliance
                        # To avoid strict import issues inside method if not imported globally
                        # Rely on Dict-like assignment if model permits, but we are using Pydantic models
                        # It's better to instantiate Compliance model
                        from .models import Compliance
                        p.compliance = Compliance(gdpr_compliant=True, data_residency_statement="")
                    
                    new_stmt = f"All customer data will be stored and processed in {residency_req} regions only."
                    p.compliance.data_residency_statement = new_stmt
                    p.compliance.gdpr_compliant = True
                    notes.append(f"Fixed C001: Added explicit data residency statement for {residency_req}.")

            # C002 Fix: Missing Cost Estimate
            budget_req = requirements.get("constraints", {}).get("monthly_budget_usd")
            if budget_req:
                 # Check provided?
                 needs_c002 = False
                 if not p.estimates: needs_c002 = True
                 elif not p.estimates.estimate_band: needs_c002 = True
                 
                 if needs_c002:
                     # C002 Fix: Missing Cost Estimate
                     from .models import CostEstimate
                     band = "medium"
                     if budget_req <= 10000: band = "low"
                     elif budget_req > 30000: band = "high"
                     
                     est_val = int(budget_req * 0.9)
                     est_range = (int(budget_req * 0.8), int(budget_req * 1.0))
                     
                     if not p.estimates:
                         p.estimates = CostEstimate(
                             monthly_cost_usd=est_val,
                             estimate_range_usd_per_month=est_range,
                             estimate_band=band,
                             confidence="medium"
                         )
                     else:
                         p.estimates.estimate_band = band
                         if not p.estimates.monthly_cost_usd:
                              p.estimates.monthly_cost_usd = est_val
                         if not p.estimates.estimate_range_usd_per_month:
                              p.estimates.estimate_range_usd_per_month = est_range
                         p.estimates.confidence = "medium"
                     
                     notes.append(f"Fixed C002: Computed cost estimate band '{band}' with medium confidence based on budget.")

            # --- Cache Logic (A002 + Strategy S2) ---
            # Unified decision logic to avoid conflicting notes/risks
            
            should_inject_redis = False
            cache_reason = ""
            
            # Check S2 Strategy Trigger
            if mode == "strategy" and requirements:
                traffic = requirements.get("traffic", {})
                ratio = traffic.get("read_write_ratio", "") # e.g. "70:30"
                if ratio:
                    try:
                        read_part = int(ratio.split(":")[0])
                        if read_part >= 70:
                            should_inject_redis = True
                            cache_reason = f"Applied Strategy S2: Injected Redis cache due to high read ratio ({ratio})."
                    except:
                        pass
            
            # Check Manual Advice Trigger
            if apply_advice and not should_inject_redis:
                should_inject_redis = True
                cache_reason = "Applied Advice A002: Injected Redis cache component."

            # Apply Cache Logic
            if is_cache_missing(p):
                 if should_inject_redis:
                     # Inject Redis
                     # Use generic helper or direct assignment
                     if p.architecture.components.cache is None:
                         p.architecture.components.cache = {"type": "redis", "notes": cache_reason}
                     elif isinstance(p.architecture.components.cache, dict):
                         p.architecture.components.cache["type"] = "redis"
                         p.architecture.components.cache["notes"] = cache_reason
                     else:
                         p.architecture.components.cache.type = "redis"
                         p.architecture.components.cache.notes = cache_reason
                     
                     notes.append(cache_reason)
                     
                     # Add Risk: Complexity
                     from .models import Risk
                     p.risks.append(Risk(
                         title="Cache Complexity",
                         mitigation="Introduction of Redis increases operational complexity. Ensure eviction policies and consistency checks."
                     ))
                 else:
                     # Default A002 Behavior: Warning only
                     from .models import Risk
                     p.risks.append(Risk(
                         title="Missing Cache Layer", 
                         mitigation="Cache unenabled: may impact read performance. Evaluate Redis for high-read paths."
                     ))
                     notes.append("Fixed A002: Added Risk entry for missing cache.")
        
        if not notes:
            print("No patchable violations found or candidate already compliant.")
            return False  # No-op: skip saving patched copy
            
        # Update metadata
        if p: p.patch_notes.extend(notes)
        if wf:
             # WorkflowIR has patch_notes list
             wf.patch_notes.extend(notes)
        
        # Save Patched Candidate
        # Naming: target_id = base_id + "_patched"
        # If already patched? Add another _patched? or replace?
        # User requested: target.id = base.id + "_patched"
        
        if "_patched" in cand.id:
            # Already patched, keeping same ID? Or appending?
            # "Allow patching already patched candidates" logic is handled in iterate loop usually.
            # Here let's just ensure we don't explode length if multi-patch.
            # But specific Requirement: "target.id = base.id + '_patched'"
            # If base has patched, we append.
            pass
            
        new_id = f"{cand.id}_patched"
        cand.id = new_id
        
        # Save
        candidates_dir = session_dir / "candidates" # Ensure defined
        new_path = candidates_dir / f"cand_{new_id}.json"
        
        save_candidate(session_path, cand)
            
        print(f"Patched candidate saved to: {new_path.name}")
        print(f"Notes: {notes}")
        
        # Auto-Index
        self._update_candidate_index(session_path)
        
        # The original code had this print statement after the save block,
        # but the new snippet implies it should be removed or integrated differently.
        # I'll keep the original print format for "Changes applied:"
        print("Changes applied:")
        for n in notes:
            print(f"- {n}")
        
        return True  # Changes were made

    def recommend(self, session_path: str, include_patched: bool = False):
        session_dir = Path(session_path)
        audits_dir = session_dir / "audits"
        candidates_dir = session_dir / "candidates"
        req_file = session_dir / "requirements.json"
        
        if not audits_dir.exists():
            print("No audit results found. Run 'evo audit' first.")
            return

        # Load Requirements
        requirements = {}
        if not req_file.exists():
             req_file = Path("requirements.json")
        if req_file.exists():
            requirements = read_json(req_file)
            
        primary_goal = requirements.get("preferences", {}).get("primary_goal", "").lower()
        
        scored_candidates = []
        
        for audit_file in audits_dir.glob("audit_*.json"):
            # Check patched filter
            if not include_patched and "_patched" in audit_file.name: continue
            
            res = AuditResult(**read_json(audit_file))
                
            # 1. Filter HARD Failures
            has_hard = any(v.severity == Severity.HARD for v in res.violations)
            if has_hard: continue
            
            # 2. Calculate Final Score
            # final = score - 8 * (risk_count)
            risk_count = sum(1 for v in res.violations if v.severity == Severity.RISK)
            final_score = res.score - (8 * risk_count)
            
            # Get Strategy
            strategy = "balanced"
            cand_path = candidates_dir / f"cand_{res.candidate_id}.json"
            if cand_path.exists():
                c_data = read_json(cand_path)
                if c_data.get("strategy"):
                    strategy = c_data["strategy"]
                elif c_data.get("proposal") and c_data["proposal"].get("strategy"):
                    strategy = c_data["proposal"]["strategy"]
            
            scored_candidates.append({
                "id": res.candidate_id,
                "strategy": strategy,
                "audit_score": res.score,
                "final_score": final_score,
                "risk_count": risk_count,
                "violations": [v.model_dump() for v in res.violations]
            })
            
        if not scored_candidates:
            print("No viable candidates found (all failed HARD constraints or no audits).")
            return
            
        # 3. Sort Logic
        # 3. Sort Logic with Tie-Breaker
        # Primary Sort: Final Score (descending)
        scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Identify Top Tier (candidates within 0.5 of the top score)
        top_score = scored_candidates[0]["final_score"]
        top_tier = [c for c in scored_candidates if top_score - c["final_score"] < 0.5]
        
        winner = top_tier[0]
        tie_broken = False
        decision_reason = "Highest Score"
        
        if len(top_tier) > 1:
            tie_broken = True
            
            # Helper to calculate complexity
            def get_complexity(c_id):
                try:
                    path = candidates_dir / f"cand_{c_id}.json"
                    data = read_json(path)
                    p_data = data.get("proposal", {})
                    arch = p_data.get("architecture", {})
                    comps = arch.get("components", {})
                    
                    score = 0
                    # Cache
                    c = comps.get("cache")
                    if c and (isinstance(c, dict) and c.get("type") != "none" or hasattr(c, "type") and c.type != "none"):
                        score += 1
                    # Queue
                    q = comps.get("queue")
                    if q and (isinstance(q, dict) and q.get("type", "none") != "none" or getattr(q, "type", "none") != "none"):
                         score += 1
                    # Style
                    style = arch.get("style", "").lower()
                    if "microservices" in style or "event" in style:
                         score += 2
                    return score
                except:
                    return 0

            # Helper for patch count (Change count) based on ID diff logic or metadata?
            # Approximation: if patched, higher complexity/instability?
            # User wants: "patched changes count" less is better.
            # We can use '_patched' vs not, or check patch notes length.
            # Let's use patch notes count.
            def get_patch_count(c_id):
                 if "_patched" not in c_id: return 0
                 # Try to find diff file for true count?
                 # Or load candidate and count patch_notes
                 try:
                    path = candidates_dir / f"cand_{c_id}.json"
                    data = read_json(path)
                    return len((data.get("proposal") or {}).get("patch_notes") or [])
                 except:
                    return 0

            # Helper for Confidence
            def get_confidence_score(c_id):
                # High=3, Med=2, Low=1
                try:
                    path = candidates_dir / f"cand_{c_id}.json"
                    data = read_json(path)
                    est = (data.get("proposal") or {}).get("estimates", {})
                    conf = est.get("confidence", "low").lower() if est else "low"
                    if conf == "high": return 3
                    if conf == "medium": return 2
                    return 1
                except:
                    return 1

            # Sort Top Tier
            # 1. Goal Match (DESC) (primary_goal in strategy)
            # 2. Complexity (ASC)
            # 3. Patch Count (ASC)
            # 4. Confidence (DESC)
            
            def tie_key(c):
                goal_match = 1 if primary_goal and primary_goal in c["strategy"].lower() else 0
                comp = get_complexity(c["id"])
                p_count = get_patch_count(c["id"])
                conf = get_confidence_score(c["id"])
                # Return tuple for sort (desc items negated for asc sort if using a single direction? No, python sort stability)
                # Python sort is stable. We can sort by keys in reverse order of importance?
                # Actually, standard Tuple sort:
                # (GoalMatch DESC, Complexity ASC, PatchCheck ASC, Confidence DESC)
                # We can map ASC items to negative?
                # complexity: lower is better -> negate? No, we want ASC. 
                # Sort reverse=True implies bigger is better.
                # So: GoalMatch (1>0), Complexity (Low>High => -Comp), PCount (-Count), Confidence (3>1)
                return (goal_match, -comp, -p_count, conf)
            
            top_tier.sort(key=tie_key, reverse=True)
            winner = top_tier[0]
            decision_reason = "Tie-breaker (Goal > Complexity > Patches > Confidence)"
        
        # Load Winner Proposal
        winner_cand = load_candidate(session_path, winner['id'])
        p = winner_cand.proposal
        if not p:
             # Workflow IR fallback: Mock empty proposal structure-like object or use safe access?
             # For now, safe access via dict if needed, or just guard blocks.
             pass
        
        print(f"Recommended: {winner['id']} ({winner['strategy']}). Reason: {decision_reason}")
        
        # --- Build Sections ---
        
        # 1. Fit to Requirements
        req_traffic = requirements.get("traffic", {})
        req_slo = requirements.get("slo", {})
        budget_req = requirements.get("constraints", {}).get("monthly_budget_usd", "Not provided")
        
        fit_lines = ["## Fit to Requirements"]
        
        # Peak RPS & SLO (Same as before)
        req_rps = req_traffic.get("peak_rps", "N/A")
        prop_rps = "N/A"
        if p and p.experiments and p.experiments.load_test:
            prop_rps = p.experiments.load_test.target_rps
        fit_lines.append(f"- **Peak RPS**: {req_rps} -> Proposal load test target: {prop_rps}")
        
        req_p95 = req_slo.get("p95_latency_ms", "N/A")
        prop_p95 = "N/A"
        # Fit to Requirements with Metrics
        # Loading Metrics if available
        metrics_data = []
        metrics_file = session_dir / "candidates" / f"cand_{winner['id']}" / "outputs" / "metrics.json"
        if metrics_file.exists():
            md_obj = read_json(metrics_file)
            metrics_data = md_obj.get("metrics", [])
        
        # Helper to get metric
        def get_m(key):
            found = next((m for m in metrics_data if m["metric_key"] == key), None)
            return found
            
        # SLO (P95)
        m_p95 = get_m("latency.p95_ms")
        p95_val = m_p95["value"] if m_p95 else ((p.slo.p95_latency_ms if p.slo else "?") if p else "?")
        p95_src = f" ({m_p95['source'][0]}...)" if m_p95 else "" 
        fit_lines.append(f"- **SLO p95**: {req_p95}ms -> Proposal: {p95_val}ms{p95_src}")
        
        # Data Residency
        m_eu = get_m("compliance.eu_residency")
        req_res = requirements.get("constraints", {}).get("data_residency", "N/A")
        
        if m_eu:
            res_val = "Compliant (EU)" if m_eu["value"] else "Non-Compliant"
            res_conf = f" [{m_eu['confidence']}]"
        else:
            # Fallback
            comp = p.compliance if p else None
            stmt = comp.data_residency_statement if comp else None
            res_val = stmt if stmt else "Not explicitly stated"
            res_conf = ""
            
        fit_lines.append(f"- **Data Residency**: {req_res} -> {res_val}{res_conf}")
        
        # QUALITY (Pass Rate / Faithfulness) - Workflow Specific
        m_pass = get_m("quality.pass_rate")
        m_faith = get_m("quality.faithfulness")
        if m_pass:
             fit_lines.append(f"- **Pass Rate**: {m_pass['value']*100:.1f}%")
        if m_faith:
             fit_lines.append(f"- **Faithfulness**: {m_faith['value']*100:.1f}%")
        
        # LATENCY (P50 for Workflow, P95 for Arch)
        m_p50 = get_m("latency.p50_ms")
        if m_p50:
             fit_lines.append(f"- **P50 Latency**: {m_p50['value']}ms")

        
        # Budget
        m_cost = get_m("cost.monthly_usd")
        budget_req = requirements.get("constraints", {}).get("monthly_budget_usd", "N/A")
        
        if m_cost:
            cost_val = m_cost["value"]
            cost_conf = m_cost["confidence"]
            cost_src = m_cost["source"]
            prop_cost = f"${cost_val} ({cost_conf} conf, src: {cost_src})"
        else:
            # Fallback
            est = p.estimates if p else None
            c_val = est.monthly_cost_usd if est and est.monthly_cost_usd else "N/A"
            band = est.estimate_band if est and est.estimate_band else ""
            prop_cost = f"${c_val} {band}"

        fit_lines.append(f"- **Budget**: ${budget_req} -> {prop_cost}")

        # 2. Why this over #2
        why_lines = ["", f"## Why this over #2"]
        if len(scored_candidates) > 1:
            runner_up = top_tier[1] if len(top_tier) > 1 else scored_candidates[1] 
            # If not tie broken (i.e. top_tier was size 1), runner up is #2 in master list.
            # If tie broken, runner up is #2 in top_tier.
            
            r_strat = runner_up.get("strategy", "unknown")
            why_lines.append(f"Selected vs Runner-up (`{runner_up['id']}` - {r_strat}):")
            
            if tie_broken:
                why_lines.append(f"**Tie-breaker Decision** (Scores within 0.5):")
                why_lines.append(f"- **Criteria**: Goal Alignment > Low Complexity > Fewer Patches > High Confidence.")
                why_lines.append(f"- **Winner Strategy**: {winner['strategy']} (Goal: {primary_goal or 'None'})")
            else:
                # Score Advantage
                diff_risk = winner['risk_count'] - runner_up['risk_count']
                if diff_risk < 0:
                     why_lines.append(f"- **Lower Risk**: Has {abs(diff_risk)} fewer risks.")
                else: 
                     why_lines.append(f"- **Better Score**: {winner['final_score']} vs {runner_up['final_score']}")


        # 3. What Was Patched
        patch_lines = ["", "## What Was Patched"]
        if "_patched" in winner['id']:
            # Try to find diff file: diff_*_vs_{winner_id}.md
            compare_dir = session_dir / "compare"
            # Logic: winner['id'] is target. Base is winner['id'] without _patched.
            # But wait, patch command might generate diff_<id>_vs_<id>_patched.
            # Let's simple regular expression match or glob
            diff_files = list(compare_dir.glob(f"diff_*_vs_{winner['id']}.md"))
            
            found_summary = False
            if diff_files:
                # Prioritize correct base if multiple? usually one.
                target_diff = diff_files[0]
                # Read content and extract Change Summary
                content = target_diff.read_text(encoding="utf-8")
                if "## Change Summary" in content:
                    summary_part = content.split("## Change Summary")[1]
                    changes = []
                    for line in summary_part.split("\n"):
                        line = line.strip()
                        if line.startswith("##"): break
                        if line.startswith("-"):
                            changes.append(line)
                            if len(changes) >= 5: break # Limit to 5
                    
                    if changes:
                        patch_lines.extend(changes)
                        patch_lines.append(f"\n[View Full Diff]({target_diff.name})")
                        found_summary = True
            
            if not found_summary:
                # Fallback to proposal patch_notes
                if p and p.patch_notes:
                     patch_lines.append("> [!WARNING]")
                     patch_lines.append("> Diff file not found. Using internal patch notes.")
                     for note in p.patch_notes:
                         patch_lines.append(f"- {note}")
                else:
                     patch_lines.append("Patch applied, but no specific change summary found (No diff or notes).")
        else:
            patch_lines.append("No patch applied (Original Candidate).")

        # 4. Trade-offs / Caveats
        trade_lines = ["", "## Trade-offs / Caveats"]
        caveats = []
        
        if p:
             # Check components
             cache = p.architecture.components.cache
             if cache and (isinstance(cache, dict) and cache.get("type") == "redis" or getattr(cache, "type", "") == "redis"):
                  caveats.append("- **Redis**: Adds operational complexity (eviction policies, consistency).")
             
             queue = p.architecture.components.queue
             if queue and (isinstance(queue, dict) and queue.get("type") != "none" or getattr(queue, "type", "none") != "none"):
                  caveats.append("- **Async Queue**: Introduces eventual consistency. Consumers must be idempotent.")
                  
             # Check reliability
             retries = p.architecture.reliability.retries
             attempts = 0
             if isinstance(retries, dict): attempts = int(retries.get("max_attempts", 0))
             if attempts <= 1:
                  caveats.append("- **Low Retries**: Strict retry policy (1 attempt). May reduce success rate during blips.")
        else:
             caveats.append("- **Workflow**: See trace for execution details.")
             caveats.append("- **Low Retries**: Strict retry policy (1 attempt). May reduce success rate during blips; allow client-side handling.")
        
        # Check Timeouts
        if p:
             timeouts = p.architecture.reliability.timeouts_ms
             client_to = timeouts.get("client", 0)
             if client_to > 1000:
                  caveats.append(f"- **High Timeout**: Client timeout {client_to}ms is generous. Monitor for thread pool exhaustion.")

        if not caveats:
             caveats.append("- **Standard Architecture**: No specific high-risk trade-offs detected.")
             
        trade_lines.extend(caveats)

        # Assemble Markdown
        strategy_title = winner['strategy'].title()
        md_lines = [
            f"# Recommendation: {strategy_title}",
            f"**Candidate ID**: `{winner['id']}`",
            f"**Final Score**: {winner['final_score']} (Audit: {winner['audit_score']}, Risks: {winner['risk_count']})",
            ""
        ]
        
        md_lines.extend(fit_lines)
        md_lines.extend(why_lines)
        md_lines.extend(patch_lines)
        md_lines.extend(trade_lines)
        
        # Rationale conclusion
        md_lines.append("")
        md_lines.append("## Strategic Alignment")
        
        if primary_goal:
             if primary_goal in winner['strategy'].lower():
                 match_msg = f"**Primary Goal**: `{primary_goal}`. Candidate strategy `{winner['strategy']}` aligns perfectly."
             else:
                 match_msg = f"**Primary Goal**: `{primary_goal}`. Best scoring option (Strategy `{winner['strategy']}` selected due to score/constraints)."
        else:
             match_msg = f"**Strategy**: `{winner['strategy']}`. (No primary goal constraints specified)."
        
        md_lines.append(match_msg)
        
        # 4. Outputs (JSON detailed)
        rec_data = {
            "session": session_dir.name,
            "primary_goal": primary_goal,
            "winner": winner,
            "all_candidates": [
                {k: v for k, v in c.items() if k != "violations"} for c in scored_candidates
            ]
        }
        
        write_json(session_dir / "recommendation.json", rec_data)
             
        (session_dir / "recommendation.md").write_text("\n".join(md_lines), encoding="utf-8")

    def diff(self, session_path: str, base_id: str, target_id: str):
        session_dir = Path(session_path)
        candidates_dir = session_dir / "candidates"
        compare_dir = session_dir / "compare"
        compare_dir.mkdir(exist_ok=True)
        
        base_file = self._get_candidate_path(session_path, base_id)
        target_file = self._get_candidate_path(session_path, target_id)
        
        if not base_file.exists(): raise FileNotFoundError(f"Base candidate {base_id} not found")
        if not target_file.exists(): raise FileNotFoundError(f"Target candidate {target_id} not found")
        
        base = load_candidate(session_path, base_id)
        target = load_candidate(session_path, target_id)
        
        changes = []
        
        # Helper to compare simple paths
        def check_change(path, b_val, t_val, reason_guess=""):
            if b_val != t_val:
                changes.append({
                    "path": path,
                    "from": b_val,
                    "to": t_val,
                    "reason": reason_guess
                })

        p_base = base.proposal
        p_target = target.proposal
        wf_base = base.workflow_ir
        wf_target = target.workflow_ir
        
        # Handle WorkflowIR diff
        if wf_base and wf_target:
            # Compare steps count
            check_change("workflow_ir.steps.count", len(wf_base.steps), len(wf_target.steps), "Step added/removed")
            
            # Compare budget
            if wf_base.controls and wf_target.controls:
                b_budget = wf_base.controls.budget
                t_budget = wf_target.controls.budget
                if b_budget and t_budget:
                    check_change("workflow_ir.controls.budget.max_total_turns", 
                                 b_budget.max_total_turns, t_budget.max_total_turns, "W008 or manual")
            
            # Compare fallbacks count
            if wf_base.controls and wf_target.controls:
                check_change("workflow_ir.controls.fallbacks.count",
                             len(wf_base.controls.fallbacks), len(wf_target.controls.fallbacks), "Fallback added")
            
            # Check for new verify step (W006)
            base_has_verify = any(s.action == "verify" for s in wf_base.steps)
            target_has_verify = any(s.action == "verify" for s in wf_target.steps)
            if not base_has_verify and target_has_verify:
                changes.append({
                    "path": "workflow_ir.steps.verify",
                    "from": "missing",
                    "to": "added",
                    "reason": "W006"
                })
            
            # Generate JSON output
            diff_id = f"{base_id}_vs_{target_id}"
            json_out = {
                "base_id": base_id,
                "target_id": target_id,
                "type": "workflow_ir",
                "changes": changes,
                "patch_notes": wf_target.patch_notes
            }
            json_path = compare_dir / f"diff_{diff_id}.json"
            write_json(json_path, json_out)
            
            # Markdown
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            md_lines = [
                f"# Diff: {base_id} vs {target_id}",
                f"Date: {timestamp}",
                f"**Type**: WorkflowIR",
                "",
                "## Change Summary"
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
            
            md_path = compare_dir / f"diff_{diff_id}.md"
            (compare_dir / f"diff_{diff_id}.md").write_text("\n".join(md_lines), encoding="utf-8")
            
            print(f"Diff generated at {md_path}")
            return
        
        # If either candidate has WorkflowIR but not both, skip (mismatched types)
        if wf_base or wf_target:
            print("Warning: Mismatched candidate types (one WorkflowIR, one not). Diff skipped.")
            return
        
        # Handle Proposal diff (legacy)
        if not p_base or not p_target:
            print("Warning: One or both candidates lack proposal. Diff skipped.")
            return
        
        # Check nested attributes exist before accessing
        if not hasattr(p_base, 'architecture') or not p_base.architecture:
            print("Warning: Base proposal lacks architecture. Diff skipped.")
            return
        if not hasattr(p_target, 'architecture') or not p_target.architecture:
            print("Warning: Target proposal lacks architecture. Diff skipped.")
            return
        if not hasattr(p_base.architecture, 'reliability') or not p_base.architecture.reliability:
            print("Warning: Base architecture lacks reliability. Diff skipped.")
            return
        if not hasattr(p_target.architecture, 'reliability') or not p_target.architecture.reliability:
            print("Warning: Target architecture lacks reliability. Diff skipped.")
            return
        
        # reliability.retries.max_attempts
        b_retries = 0
        if isinstance(p_base.architecture.reliability.retries, dict):
            b_retries = int(p_base.architecture.reliability.retries.get("max_attempts", 0))
        
        t_retries = 0
        if isinstance(p_target.architecture.reliability.retries, dict):
             t_retries = int(p_target.architecture.reliability.retries.get("max_attempts", 0))
        
        if b_retries != t_retries:
            reason = "R001" if t_retries < b_retries and b_retries >= 3 else "Manual Adjustment"
            check_change("architecture.reliability.retries.max_attempts", b_retries, t_retries, reason)
            
        # reliability.timeouts_ms.client
        b_client_to = p_base.architecture.reliability.timeouts_ms.get("client", 0)
        t_client_to = p_target.architecture.reliability.timeouts_ms.get("client", 0)
        
        if b_client_to != t_client_to:
            reason = "R002" if t_client_to > b_client_to else "Manual Adjustment"
            check_change("architecture.reliability.timeouts_ms.client", b_client_to, t_client_to, reason)

        # reliability.idempotency
        b_idem = p_base.architecture.reliability.idempotency
        t_idem = p_target.architecture.reliability.idempotency
        if b_idem != t_idem:
            reason = "R005" if t_idem in ["recommended", "required"] and b_idem == "not_supported" else "Manual Adjustment"
            check_change("architecture.reliability.idempotency", b_idem, t_idem, reason)
            
        # components.cache.type
        def get_cache_type(prop):
            if not prop.architecture or not prop.architecture.components: return "none"
            c = prop.architecture.components.cache
            if c is None: return "none"
            if isinstance(c, dict): return c.get("type", "none")
            return c.type
            
        b_cache = get_cache_type(p_base)
        t_cache = get_cache_type(p_target)
        
        if b_cache != t_cache:
            reason = "A002" if t_cache != "none" and b_cache == "none" else "Manual Adjustment"
            check_change("architecture.components.cache.type", b_cache, t_cache, reason)

        # Generate outputs
        diff_id = f"{base_id}_vs_{target_id}"
        
        # JSON
        json_out = {
            "base": base_id,
            "target": target_id,
            "strategies": {"base": p_base.strategy, "target": p_target.strategy},
            "changes": changes,
            "patch_notes": p_target.patch_notes
        }
        json_path = compare_dir / f"diff_{diff_id}.json"
        write_json(json_path, json_out)
            
        # Markdown
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        md_lines = [
            f"# Diff: {base_id} vs {target_id}",
            f"Date: {timestamp}",
            f"**Strategies**: {p_base.strategy} -> {p_target.strategy}",
            "",
            "## Change Summary"
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
                
        md_path = compare_dir / f"diff_{diff_id}.md"
        (compare_dir / f"diff_{diff_id}.md").write_text("\n".join(md_lines), encoding="utf-8")
            
        print(f"Diff generated at {md_path}")


    def audit(self, session_path: str, include_patched: bool = False):
        session_dir = Path(session_path)
        candidates_dir = session_dir / "candidates"
        legacy_file = session_dir / "candidates.json"
        
        # Migration logic
        if legacy_file.exists() and not candidates_dir.exists():
            print("Detecting legacy format. Migrating to 'candidates/' directory...")
            candidates_dir.mkdir()
            legacy_data = read_json(legacy_file)
            for item in legacy_data:
                c = Candidate(**item)
                save_candidate(session_path, c, update_index=False)
            print("Migration complete.")
        
        if not candidates_dir.exists():
            raise FileNotFoundError("candidates/ directory not found in session.")
            
        ruleset_file = Path("ruleset.yaml")
        if not ruleset_file.exists():
            raise FileNotFoundError("ruleset.yaml not found in current directory.")

        rules_data = yaml.safe_load(ruleset_file.read_text(encoding="utf-8"))
        # Ensure Rule loading handles optional fields
        rules = []
        for r in rules_data['rules']:
            rules.append(Rule(**r))
        ruleset = Ruleset(rules=rules)
        
        # Load Requirements for Contextual Rules
        req_file = session_dir / "requirements.json"
        requirements = {}
        if req_file.exists():
            requirements = read_json(req_file)

        # Load candidates from files
        candidates = []
        
        # Helper to get candidate files
        index_file = candidates_dir / "index.json"
        candidate_map = {}
        if index_file.exists():
            try:
                raw_index = read_json(index_file)
                candidate_map = {k: session_dir / v for k, v in raw_index.items()}
            except: pass
        
        if not candidate_map:
             # Fallback
             candidate_map = {p.stem.replace("cand_", ""): p for p in candidates_dir.glob("cand_*.json")}

        for cid, cand_file in candidate_map.items():
            # Filter patched candidates unless requested
            if not include_patched and "_patched" in cid:
                continue
                
            if not cand_file.exists():
                print(f"Warning: Candidate file {cand_file} missing from index.")
                continue

            try:
                candidates.append(Candidate(**read_json(cand_file)))
            except Exception as e:
                print(f"Error loading candidate {cand_file}: {e}")

        # Clean audits directory to ensure report matches current candidates
        import shutil
        audits_dir = session_dir / "audits"
        if audits_dir.exists():
            shutil.rmtree(audits_dir)
        audits_dir.mkdir(exist_ok=True)

        results = []
        for cand in candidates:
            violations = []
            
            # Helper context for eval
            # We construct a secure evaluation context
            # Functions allow the ruleset to remain declarative but powerful
            
            proposal = cand.proposal
            
            def acceptance_has_metrics(required_metrics):
                if not proposal or not proposal.acceptance: return False
                existing = [c.metric for c in proposal.acceptance.hard_constraints]
                return all(m in existing for m in required_metrics)

            def get_retry_max_attempts():
                if not proposal: return 0
                
                # Check architecture.reliability exists
                if not proposal.architecture or not proposal.architecture.reliability:
                    return 0
                
                retries = proposal.architecture.reliability.retries
                 # retries can be dict or object depending on load, pydantic model enforces Dict though
                # but let's be safe
                if isinstance(retries, dict):
                    return int(retries.get("max_attempts", 0))
                return 0

            def get_timeout(typ):
                if not proposal or not proposal.architecture or not proposal.architecture.reliability: return 0
                return proposal.architecture.reliability.timeouts_ms.get(typ, 0)
            
            def queue_used_requires_risks():
                if not proposal or not proposal.architecture or not proposal.architecture.components: return False
                # Check if queue is used
                q = proposal.architecture.components.queue
                queue_active = False
                if q:
                     # Check type via attribute or dict logic if accessed dynamically, but Pydantic object access is via dot
                     # Wait, ComponentDetails is object. 
                     if isinstance(q, dict):
                         queue_active = q.get("type", "none") != "none"
                     else:
                         queue_active = q.type != "none"
                
                if queue_active:
                    return len(proposal.risks) == 0 # Violation if risks empty
                return False

            def has_meaningful_chaos_test():
                if not proposal or not proposal.experiments: return False
                return len(proposal.experiments.chaos_test) > 0

            def is_cache_missing():
                if not proposal or not proposal.architecture or not proposal.architecture.components: return True
                c = proposal.architecture.components.cache
                if not c: return True
                if isinstance(c, dict): return c.get("type", "none") == "none"
                return c.type == "none"

            # Flatten requirements for easy access or just pass dict
            # We implement property access for 'requirements.constraints.data_residency' via a simple wrapper or dict access
            # But python eval on dict works if we access keys, not dots.
            # To support "requirements.constraints.data_residency", we can wrap it in a class or ensure rule uses dict access: requirements['constraints']['data_residency']
            # OR we can wrap it in SimpleNamespace
            from types import SimpleNamespace
            
            def dict_to_obj(d):
                if not isinstance(d, dict): return d
                return SimpleNamespace(**{k: dict_to_obj(v) for k, v in d.items()})
            
            req_obj = dict_to_obj(requirements) if requirements else None
            
            # Compliance Check Helper
            # "proposal.compliance.data_residency_statement missing or mismatched"
            # It's hard to express complex logic in rules string. 
            # Simplest: ruleset uses a function `check_residency_match()`?
            
            def check_residency_mismatch():
                 # Triggered if Requirement exists but Proposal fails
                 if not req_obj or not hasattr(req_obj, "constraints"): return False
                 target = getattr(req_obj.constraints, "data_residency", None)
                 if not target or target == "none" or target == "N/A": return False
                 
                 # Check proposal
                 if not proposal.compliance: return True
                 stmt = proposal.compliance.data_residency_statement
                 if not stmt: return True
                 
                 # Basic keyword match
                 if target.lower() in ["eu", "europe"]:
                      if not any(k in stmt.lower() for k in ["eu", "europe"]): return True
                 return False

            def check_budget_missing():
                 if not req_obj or not hasattr(req_obj, "constraints"): return False
                 budget = getattr(req_obj.constraints, "monthly_budget_usd", None)
                 if not budget: return False
                 
                 if not proposal.estimates or not proposal.estimates.estimate_band: return True
                 return False

            eval_context = {
                "proposal": proposal,
                "requirements": req_obj,
                "acceptance_has_metrics": acceptance_has_metrics,
                "get_retry_max_attempts": get_retry_max_attempts,
                "get_timeout": get_timeout,
                "queue_used_requires_risks": queue_used_requires_risks,
                "has_meaningful_chaos_test": has_meaningful_chaos_test,
                "is_cache_missing": is_cache_missing,
                # New helpers reducing complex logic in YAML
                "check_residency_mismatch": check_residency_mismatch,
                "check_budget_missing": check_budget_missing
            }

            # Check Workflow Context
            if hasattr(cand, "workflow_ir") and cand.workflow_ir:
                 # Load Workflow Rules
                 wf_rules_file = Path("ruleset_workflow.yaml")
                 ruleset_to_use = ruleset
                 if wf_rules_file.exists():
                      with open(wf_rules_file, "r") as f:

                           data = yaml.safe_load(f)
                           ruleset_to_use = Ruleset(rules=[Rule(**r) for r in data["rules"]])
                 
                 # Workflow Eval Context
                 eval_context = {
                      "candidate": cand,
                      "requirements": req_obj
                 }
                 
                 if cand.workflow_ir:
                      for rule in ruleset_to_use.rules:
                          try:
                              condition_met = eval(rule.condition, {"__builtins__": {"len": len, "any": any, "all": all, "sum": sum, "max": max, "min": min}}, eval_context)
                              if condition_met: # Workflow rules usually: condition is satisfied = PASS? No, my rules are checks.
                                  # My rules logic: "len < 2" -> Violation. "not exists" -> Violation.
                                  # So if condition is True -> Violation.
                                  # Except "Acceptance tests missing..." -> True means missing -> Violation.
                                  # Wait, existing rules logic: H* condition met -> Good?
                                  # My Workflow Rules are written as VIOLATION conditions mostly (e.g. "len < 2").
                                  # Let's align logic.
                                  # The existing logic (lines 1166-1171) handles H/R/A differentiation.
                                  # If I write: "W001" (Workflow Hard?)
                                  # Let's assume W* = Conditions are VIOLATIONS if True.
                                  
                                  # Let's adjust existing logic slightly to support 'W' series.
                                  is_violation = True # Default for W series if condition met
                                  
                                  violations.append(Violation(
                                      rule_id=rule.id,
                                      severity=rule.severity,
                                      message=rule.description,
                                      fix_suggestion=rule.fix_suggestion
                                  ))
                          except Exception as e:
                              print(f"Error evaluating rule {rule.id}: {e}")
                 
            elif proposal:
                for rule in ruleset.rules:

                                try:
                                    # Safe eval might be needed in production, for MVP eval is acceptable for local tools
                                    condition_met = eval(rule.condition, {"__builtins__": {}}, eval_context)
                                    
                                    # Rule semantics: condition True means "Compliance" or "Violation"?
                                    # Usually condition describes the "Requirement". So True = Good.
                                    # BUT, looking at ruleset:
                                    # H001: "proposal... is not None" -> True is Good.
                                    # R001: "retries >= 3" -> True is BAD (Risk).
                                    
                                    # Let's check IDs.
                                    # H-series: Essential Requirements (True = Pass)
                                    # R-series: Risks (True = Fail/Warning)
                                    # A-series: Advice (True = Trigger Suggestion? Or True = Good state?)
                                    
                                    # Let's standardize:
                                    # H*: Condition is success criteria. If False -> Violation.
                                    # R*: Condition is risk presence. If True -> Violation.
                                    # A*: Condition is negative state? 
                                    #     A001: "has_meaningful...() == False" -> True means missing.
                                    #     A002: "is_cache_missing()" -> True means missing.
                                     
                                    is_violation = False
                                    
                                    if rule.id.startswith("H"):
                                        if not condition_met: is_violation = True
                                    elif rule.id.startswith("R"):
                                        if condition_met: is_violation = True
                                    elif rule.id.startswith("A"):
                                        if condition_met: is_violation = True
                                    
                                    if is_violation:
                                        violations.append(Violation(
                                            rule_id=rule.id,
                                            severity=rule.severity,
                                            message=rule.description,
                                            fix_suggestion=rule.fix_suggestion
                                        ))
                                        
                                except Exception as e:
                                    print(f"Error evaluating rule {rule.id}: {e}")
            else:
                pass

            passed = not any(v.severity == Severity.HARD for v in violations)
            score = 100 - (len(violations) * 10) 
            
            result = AuditResult(
                candidate_id=cand.id,
                passed=passed,
                violations=violations,
                score=max(0, score)
            )
            results.append(result)
            
            # Save individual audit result
            audit_path = audits_dir / f"audit_{result.candidate_id}.json"
            write_json(audit_path, result.model_dump(mode='json'))
        
        # Save summary for compatibility
        audit_file = session_dir / "audit_results.json"
        write_json(audit_file, [r.model_dump(mode='json') for r in results])
        
        print(f"Audit completed. Results saved to {audits_dir}")

    def report(self, session_path: str, include_patched: bool = False):
        session_dir = Path(session_path)
        audits_dir = session_dir / "audits"
        
        results = []
        if audits_dir.exists():
            for audit_file in audits_dir.glob("audit_*.json"):
                try:
                    res = AuditResult(**read_json(audit_file))
                    # Filter patched candidates logic (audit results might exist even if not valid for this report view)
                    if not include_patched and "_patched" in res.candidate_id:
                        continue
                    results.append(res)
                except Exception as e:
                    print(f"Error loading audit result {audit_file}: {e}")
        else:
             # Fallback to summary file if audits dir doesn't exist (legacy)
             audit_file = session_dir / "audit_results.json"
             if not audit_file.exists():
                 raise FileNotFoundError("No audit results found.")
                 data = read_json(audit_file)
                 results = [AuditResult(**r) for r in data]
            
        report_lines = ["# Session Audit Report", f"Date: {datetime.now()}", "", "## Summary"]
        
        passed_count = sum(1 for r in results if r.passed)
        total = len(results)
        report_lines.append(f"Total Candidates: {total}")
        report_lines.append(f"Passed: {passed_count}")
        report_lines.append(f"Failed: {total - passed_count}")
        report_lines.append("")
        
        report_lines.append("## Detailed Results")
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            icon = "✅" if r.passed else "❌"
            patched_tag = " (PATCHED)" if "_patched" in r.candidate_id else ""
            report_lines.append(f"### Candidate {r.candidate_id} {icon} ({status}){patched_tag}")
            report_lines.append(f"- **Score**: {r.score}")
            
            # Attempt to read strategy from candidate file
            try:
                cand_path = self._get_candidate_path(session_path, r.candidate_id)
                if cand_path.exists():
                    strat = read_json(cand_path).get("proposal", {}).get("strategy", "Unknown")
                    report_lines.append(f"- **Strategy**: {strat}")
            except:
                pass
            
            # Key Metrics
            metrics_file = session_dir / "candidates" / f"cand_{r.candidate_id}" / "outputs" / "metrics.json"
            if metrics_file.exists():
                report_lines.append("- **Key Metrics**:")
                try:
                    md_obj = read_json(metrics_file)
                    m_list = md_obj.get("metrics", [])
                        
                    m_p95 = next((m for m in m_list if m["metric_key"] == "latency.p95_ms"), None)
                    if m_p95: report_lines.append(f"  - P95 Latency: {m_p95['value']}ms ({m_p95['source']})")
                    
                    m_rps = next((m for m in m_list if m["metric_key"] == "throughput.rps"), None)
                    if m_rps: report_lines.append(f"  - Throughput: {m_rps['value']} RPS")
                        
                    m_cost = next((m for m in m_list if m["metric_key"] == "cost.monthly_usd"), None)
                    if m_cost: report_lines.append(f"  - Cost: ${m_cost['value']} ({m_cost['confidence']})")
                    
                    m_comp = next((m for m in m_list if m["metric_key"] == "complexity.score"), None)
                    if m_comp: report_lines.append(f"  - Complexity Score: {m_comp['value']}")
                    
                    m_eu = next((m for m in m_list if m["metric_key"] == "compliance.eu_residency"), None)
                    if m_eu: report_lines.append(f"  - EU Compliant: {m_eu['value']} ({m_eu['confidence']})")
                except Exception as e:
                    report_lines.append(f"  - Error loading metrics: {e}")
            
            if r.violations:
                report_lines.append("- **Violations**:")
                for v in r.violations:
                    report_lines.append(f"  - [{v.severity.value}] {v.rule_id}: {v.message}")
                    if v.fix_suggestion:
                        report_lines.append(f"    Fix: {v.fix_suggestion}")
            else:
                report_lines.append("- No violations found.")
                
            # Check for Diffs
            compare_dir = session_dir / "compare"
            if compare_dir.exists():
                diff_files = list(compare_dir.glob(f"diff_*_vs_{r.candidate_id}.md"))
                if diff_files:
                    report_lines.append("- **Comparison**:")
                    for df in diff_files:
                        # Extract base ID
                        # Filename: diff_<base>_vs_<target>.md
                        # base = filename.replace("diff_", "").replace(f"_vs_{r.candidate_id}.md", "")
                        # Let's just link it
                        report_lines.append(f"  - [View Diff]({df.name})")
            
            report_lines.append("")

        # Append Recommendation if available
        rec_md = session_dir / "recommendation.md"
        if rec_md.exists():
            report_lines.append("")
            report_lines.append("---")
            report_lines.append("")
            report_lines.append(rec_md.read_text(encoding="utf-8"))

        report_file = session_dir / "report.md"
        report_file.write_text("\n".join(report_lines), encoding="utf-8")
            
        print(f"Report generated at {report_file}")

    
    def generate_metrics(self, session_path: str, include_patched: bool = False):
        from .models import MetricsOutput, Metric
        session_dir = Path(session_path)
        candidates_dir = session_dir / "candidates"
        evidence_dir = session_dir / "evidence"
        evidence_dir.mkdir(exist_ok=True)
        
        if not candidates_dir.exists():
            print("No candidates found.")
            return

        requirements = {}
        req_file = Path("requirements.json")
        if req_file.exists():
             with open(req_file, "r") as f:
                 requirements = json.load(f)
        
        count = 0
        count = 0
        
        # Helper to get candidate files
        index_file = candidates_dir / "index.json"
        candidate_map = {}
        if index_file.exists():
            try:
                raw_index = read_json(index_file)
                candidate_map = {k: session_dir / v for k, v in raw_index.items()}
            except: pass
        
        if not candidate_map:
             candidate_map = {p.stem.replace("cand_", ""): p for p in candidates_dir.glob("cand_*.json")}

        for cid, cand_file in candidate_map.items():
            if not include_patched and "_patched" in cid:
                continue
            
            if not cand_file.exists(): continue
            
            c_data = read_json(cand_file)
            
            # Helper to access proposal regardless of v1/v2
            wf = c_data.get("workflow_ir")
            prop = c_data.get("proposal") or c_data.get("content") or {}
            
            if not prop and not wf: continue
            
            cand_id = c_data["id"]
            outputs_dir = candidates_dir / cand_file.stem / "outputs"
            outputs_dir.mkdir(parents=True, exist_ok=True)
            
            metrics_list = []
            
            # 1. Latency P95
            p_slo = prop.get("slo") or {}
            p95 = p_slo.get("p95_latency_ms")
            if p95:
                metrics_list.append(Metric(
                    metric_key="latency.p95_ms", value=p95, unit="ms", 
                    source="proposal", confidence="medium"
                ))

            # 2. Error Rate
            err = p_slo.get("error_rate")
            if err is not None:
                metrics_list.append(Metric(
                    metric_key="errors.rate", value=err, unit="", 
                    source="proposal", confidence="medium"
                ))
            
            # 3. Throughput
            exp = prop.get("experiments") or {}
            load = exp.get("load_test") or {}
            rps = load.get("target_rps")
            if rps:
                 metrics_list.append(Metric(
                    metric_key="throughput.rps", value=rps, unit="rps", 
                    source="proposal", confidence="medium"
                ))
            
            # 4. Complexity Score
            # Cache(+1) + Queue(+1) + Style(Micro/Event=+2, Modular=+1)
            arch = prop.get("architecture") or {}
            style = arch.get("style", "monolith")
            comps = arch.get("components") or {}
            
            score = 0
            if comps.get("cache"): score += 1
            if comps.get("queue"): score += 1
            if style in ["microservices", "event-driven"]: score += 2
            elif style == "modular-monolith": score += 1
            
            metrics_list.append(Metric(
                metric_key="complexity.score", value=score, unit="",
                source="static_estimate", confidence="high"
            ))
            
            # 5. Cost
            est = prop.get("estimates") or {}
            cost = est.get("monthly_cost_usd")
            cost_range = est.get("estimate_range_usd_per_month")
            confidence = est.get("confidence", "low")
            
            final_cost = cost
            if final_cost is None and cost_range:
                final_cost = int((cost_range[0] + cost_range[1]) / 2) # median
            
            if final_cost is not None:
                metrics_list.append(Metric(
                    metric_key="cost.monthly_usd", value=final_cost, unit="usd",
                    source="static_estimate", confidence=confidence
                ))
            
            # 6. Compliance
            # Check requirements for EU
            eu_req = requirements.get("constraints", {}).get("data_residency") == "EU"
            comp = prop.get("compliance") or {}
            stmt = comp.get("data_residency_statement", "")
            is_eu = "EU" in stmt if stmt else False
            
            if eu_req:
                metrics_list.append(Metric(
                    metric_key="compliance.eu_residency", value=is_eu, unit="bool",
                    source="evidence" if eu_req else "proposal", confidence="medium" if stmt else "low"
                ))
            
            # 7. Workflow Metrics (Evidence)
            # Check for run_rag_mini.json
            evidence_dir = session_dir / "evidence" / "workflow" / cand_id
            for suite_file in evidence_dir.glob("run_*.json"):
                run_data = read_json(suite_file)
                summary = run_data.get("summary") or {}
                    
                # Pass Rate
                metrics_list.append(Metric(
                    metric_key="quality.pass_rate", value=summary.get("pass_rate", 0.0), unit="",
                    source="evidence", confidence="high", evidence_refs=[f"evidence/workflow/{cand_id}/{suite_file.name}"]
                ))
                # Faithfulness
                metrics_list.append(Metric(
                    metric_key="quality.faithfulness", value=summary.get("faithfulness", 0.0), unit="",
                    source="evidence", confidence="high"
                ))
                # Tokens
                metrics_list.append(Metric(
                    metric_key="cost.token_estimate", value=summary.get("total_tokens", 0), unit="tokens",
                    source="evidence", confidence="high"
                ))
                # Tool Calls
                metrics_list.append(Metric(
                    metric_key="cost.tool_calls", value=summary.get("total_tool_calls", 0), unit="",
                    source="evidence", confidence="high"
                ))
                # Latency P50
                metrics_list.append(Metric(
                    metric_key="latency.p50_ms", value=summary.get("p50_latency_ms", 0), unit="ms",
                    source="evidence", confidence="medium"
                ))
                # Fail Rate
                metrics_list.append(Metric(
                     metric_key="stability.fail_rate", value=summary.get("fail_rate", 0.0), unit="",
                     source="evidence", confidence="high"
                ))
            
            # Save metrics
            output = MetricsOutput(candidate_id=cand_id, metrics=metrics_list)
            write_json(outputs_dir / "metrics.json", output.model_dump(mode='json'))
            count += 1
            
        print(f"Generated metrics for {count} candidates.")

    def run_suite(self, session_path: str, suite_name: str = "rag_mini", include_patched: bool = False):
        session_dir = Path(session_path)
        candidates_dir = session_dir / "candidates"
        evidence_base = session_dir / "evidence" / "workflow"
        
        suite_file = Path(f"eval_suites/{suite_name}.json")
        if not suite_file.exists():
            print(f"Suite {suite_name} not found.")
            return
            
        with open(suite_file, "r") as f:
            suite = read_json(suite_file)
            
        print(f"Running suite '{suite_name}' on candidates...")
        count = 0
        count = 0
        
        # Helper to get candidate files
        index_file = candidates_dir / "index.json"
        candidate_map = {}
        if index_file.exists():
            try:
                raw_index = read_json(index_file)
                candidate_map = {k: session_dir / v for k, v in raw_index.items()}
            except: pass
        
        if not candidate_map:
             candidate_map = {p.stem.replace("cand_", ""): p for p in candidates_dir.glob("cand_*.json")}

        for cid, cand_file in candidate_map.items():
            if not include_patched and "_patched" in cid:
                continue
            
            if not cand_file.exists(): 
                print(f"Skipping missing file for {cid}")
                continue
            
            # Simulate Run
            run_result = self._sim_run_rag(suite)
            
            # Save Evidence
            c_evidence_dir = evidence_base / cid
            c_evidence_dir.mkdir(parents=True, exist_ok=True)
            
            write_json(c_evidence_dir / f"run_{suite_name}.json", run_result)
            
            count += 1
        print(f"Executed suite on {count} candidates.")

    def _sim_run_rag(self, suite: Dict[str, Any]) -> Dict[str, Any]:
        tasks = suite.get("tasks", [])
        results = []
        
        total_tokens = 0
        total_tool_calls = 0
        latencies = []
        passes = 0
        fails = 0
        faithful_count = 0
        
        for t in tasks:
            # 1. Retrieval (Mock)
            # Simple keyword match
            q_tokens = set(t["question"].lower().split())
            best_doc = None
            max_overlap = -1
            
            docs = t.get("docs", [])
            for d in docs:
                d_tokens = set(d["text"].lower().split())
                overlap = len(q_tokens.intersection(d_tokens))
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_doc = d
            
            retrieved_doc = best_doc if best_doc else (docs[0] if docs else None)
            tool_calls = 1 # retrieval
            
            # 2. Synthesize (Mock)
            # Generate perfect answer if doc found
            expected = t.get("expected", {})
            must_contain = expected.get("must_contain", [])
            any_contain = expected.get("must_contain_any", [])
            citations_req = expected.get("citations_required", False)
            
            answer = ""
            if must_contain:
                answer = f"The answer is {must_contain[0]}."
            elif any_contain:
                 answer = f"One such number is {any_contain[0]}."
            else:
                answer = "Here is the answer."
                
            if citations_req and retrieved_doc:
                answer += f" (cite:{retrieved_doc['doc_id']})"
                
            # Tokens Estimate (chars / 4)
            tokens = int(len(answer) / 4) + int(len(t["question"]) / 4)
            if retrieved_doc:
                tokens += int(len(retrieved_doc["text"]) / 4)
            
            # 3. Verify
            passed = True
            # check content
            for m in must_contain:
                if m not in answer: passed = False
            
            if any_contain:
                found_any = False
                for m in any_contain:
                    if m in answer: found_any = True
                if not found_any: passed = False
            
            # check citation
            is_faithful = True
            if citations_req:
                if "(cite:" not in answer: 
                    passed = False
                    is_faithful = False
                elif retrieved_doc and retrieved_doc["doc_id"] not in answer:
                    is_faithful = False # Cited wrong doc?
            
            # Stats
            if passed: passes += 1
            else: fails += 1
            if is_faithful: faithful_count += 1
            
            total_tokens += tokens
            total_tool_calls += tool_calls
            # Latency estimate: 500ms + tokens * 0.1ms
            lat = 500 + int(tokens * 0.1)
            latencies.append(lat)
            
            results.append({
                "task_id": t["task_id"],
                "passed": passed,
                "answer": answer,
                "latency_ms": lat,
                "tokens": tokens
            })
            
        # Summary
        p50 = sorted(latencies)[len(latencies)//2] if latencies else 0
        pass_rate = passes / len(tasks) if tasks else 0
        faithfulness = faithful_count / len(tasks) if tasks else 0
        
        return {
            "summary": {
                "pass_rate": pass_rate,
                "faithfulness": faithfulness, 
                "total_tokens": total_tokens,
                "total_tool_calls": total_tool_calls,
                "p50_latency_ms": p50,
                "fail_rate": 1.0 - pass_rate
            },
            "tasks": results
        }

    def iterate(self, session_path: str, rounds: int = 3, population: int = 3, topk: int = 1, patch_mode: str = "strategy", include_advice: bool = False, reset: bool = False, allow_multi_patch: bool = False, suite: Optional[str] = None):
        import shutil
        import time
        
        
        from pathlib import Path
        session_dir = Path(session_path)
        evo_dir = session_dir / "evolution"
        
        if reset and evo_dir.exists():
            shutil.rmtree(evo_dir)
        
        # Also clear stale recommendation and evidence on reset
        if reset:
            rec_file = session_dir / "recommendation.json"
            if rec_file.exists(): rec_file.unlink()
            
            evidence_dir = session_dir / "evidence"
            if evidence_dir.exists(): shutil.rmtree(evidence_dir)
            
        ensure_dir(evo_dir)
        ensure_dir(evo_dir / "rounds")
        
        # Manifest
        manifest = {
            "session": session_path,
            "rounds": rounds,
            "population": population,
            "config": {
                "patch_mode": patch_mode,
                "include_advice": include_advice,
                "allow_multi_patch": allow_multi_patch
            },
            "start_time": datetime.now().isoformat()
        }
        write_json(evo_dir / "manifest.json", manifest)
            
        trace = []
        trace_md = ["# Evolution Trace", f"Started: {manifest['start_time']}", ""]
        
        current_champion_id = None
        
        print(f"Starting Evolution ({rounds} rounds)...")
        
        for r in range(rounds):
            print(f"\n=== Round {r} ===")
            round_dir = evo_dir / "rounds" / f"round_{r:02d}"
            round_dir.mkdir(exist_ok=True)
            
            # 1. Generate (or Reuse)
            if r == 0:
                self.generate(session_path, n=population, reset=reset)
            else:
                pass

            # 2. Audit & Recommend
            inc_patch = (r > 0)
            self.audit(session_path, include_patched=inc_patch)
            
            # If workflow, run suite
            # Check if any candidate has workflow_ir
            has_workflow_ir = False
            for cand_file in (session_dir / "candidates").glob("cand_*.json"):
                if not inc_patch and "_patched" in cand_file.name:
                    continue
                c_data = read_json(cand_file)
                if c_data.get("workflow_ir"):
                    has_workflow_ir = True
                    break
            
            suite_arg = suite if suite else ("rag_mini" if has_workflow_ir else None)
            if suite_arg:
                 self.run_suite(session_path, suite_name=suite_arg, include_patched=True)
            
            self.generate_metrics(session_path, include_patched=True)
            self.recommend(session_path, include_patched=True)
            
            # 3. Capture Champion
            rec_json = session_dir / "recommendation.json"
            if not rec_json.exists():
                print("Error: No recommendation found.")
                break
                
            rec_data = read_json(rec_json)
            
            winner = rec_data["winner"]
            win_id = winner["id"]
            win_score = winner["final_score"]
            win_risks = winner["risk_count"]
            win_strat = winner["strategy"]
            is_patched = "_patched" in win_id
            
            print(f"Round {r} Champion: {win_id} (Score: {win_score}, Risks: {win_risks})")
            
            # Load metrics for champion
            win_metrics = []
            metrics_file = session_dir / "candidates" / f"cand_{win_id}" / "outputs" / "metrics.json"
            if metrics_file.exists():
                md = read_json(metrics_file)
                win_metrics = md.get("metrics", [])
            
            # Trace Entry
            trace_item = {
                "round": r,
                "champion": winner,
                "metrics": win_metrics,
                "timestamp": datetime.now().isoformat()
            }
            trace.append(trace_item)
            
            patched_status = " (patched)" if is_patched else ""
            metrics_str = ""
            if win_metrics:
                # Extract key metrics
                p95 = next((m["value"] for m in win_metrics if m["metric_key"] == "latency.p95_ms"), None)
                p50 = next((m["value"] for m in win_metrics if m["metric_key"] == "latency.p50_ms"), None)
                cost = next((m["value"] for m in win_metrics if m["metric_key"] == "cost.monthly_usd"), None)
                pass_rate = next((m["value"] for m in win_metrics if m["metric_key"] == "quality.pass_rate"), None)

                parts = []
                if p95 is not None: parts.append(f"P95: {p95}ms")
                elif p50 is not None: parts.append(f"P50: {p50}ms")
                
                if pass_rate is not None: parts.append(f"Pass: {pass_rate*100:.0f}%")

                if cost is not None: parts.append(f"Cost: ${cost}")
                else: parts.append("Cost: N/A")
                
                metrics_str = " [" + ", ".join(parts) + "]"

            md_summary = f"**Round {r}**: Champion `{win_id}`{patched_status} - Score {win_score}, Risks {win_risks}, Strategy `{win_strat}`.{metrics_str}"
            trace_md.append(md_summary)

            # Copy Artifacts
            shutil.copy2(rec_json, round_dir / "recommendation.json")
            shutil.copy2(session_dir / "recommendation.md", round_dir / "recommendation.md")
            if (session_dir / "audit_results.json").exists():
                 shutil.copy2(session_dir / "audit_results.json", round_dir / "audits_index.json")
            
            all_cands = [c["id"] for c in rec_data["all_candidates"]]
            write_json(round_dir / "population.json", all_cands)

            # 4. Stop Condition / Patching
            if r > 0 and win_id == current_champion_id and win_risks == 0:
                print("Stable optimal champion found (Risks=0). Stopping early.")
                trace_md.append("> **Stablized**: No further improvements needed.")
                break
            
            current_champion_id = win_id
            
            # Prepare Next Round: Patch Champion
            if r < rounds - 1:
                # Logic: Check if we should patch
                should_patch = True
                
                # Check 1: If risks == 0 and we are happy? User said "if risks=0 can stop", handled above.
                # If risks > 0 we patch.
                # Check 2: Double patching
                if is_patched and not allow_multi_patch:
                    print(f"Champion {win_id} is already patched. Skipping re-patch (allow_multi_patch=False).")
                    trace_md.append("> **Skip Patch**: Champion already patched.")
                    should_patch = False
                
                if should_patch:
                    # Patching
                    print(f"Patching champion {win_id}...")
                    try:
                        patch_applied = self.patch(session_path, win_id, apply_advice=include_advice, mode=patch_mode)
                        
                        if not patch_applied:
                            # No-op: candidate already compliant
                            print(f"Patch no-op: {win_id} already compliant.")
                            trace_md.append("> **Skip Patch**: no-op (candidate already compliant).")
                            # Trigger early stabilization check
                            if win_risks == 0:
                                print("Champion is compliant with 0 risks. Stabilized.")
                                trace_md.append("> **Stabilized**: Champion compliant, no further patches needed.")
                                break
                            continue  # Skip diff generation, go to next round
                        
                        # Identify new ID.
                        # If base is raw, new is raw_patched.
                        # If base is raw_patched (and allow_multi_patch=True), new is raw_patched_patched (depending on patch logic impl).
                        # Let's verify patch logic. Assuming it appends _patched.
                        patched_id = f"{win_id}_patched"
                        
                        # Ensure index is updated before diff (patched file now exists)
                        self._update_candidate_index(session_path)
                        
                        # Generate Diff
                        # Handle Diff Target naming:
                        # Base: win_id
                        # Target: patched_id
                        print(f"Generating diff for {win_id} -> {patched_id}...")
                        try:
                            self.diff(session_path, win_id, patched_id)
                            
                            diff_json = session_dir / "compare" / f"diff_{win_id}_vs_{patched_id}.json"
                            if diff_json.exists():
                                 shutil.copy2(diff_json, round_dir / f"diff_{win_id}_vs_{patched_id}.json")
                            md_file = session_dir / "compare" / f"diff_{win_id}_vs_{patched_id}.md"
                            if md_file.exists(): 
                                shutil.copy2(md_file, round_dir / f"diff_report.md")
                            
                            # Update Trace with Patch Details
                            # Read patched candidate for notes
                            p_cand_path = session_dir / "candidates" / f"cand_{patched_id}.json"
                            patch_notes = []
                            if p_cand_path.exists():
                                    cd = read_json(p_cand_path)
                                    # Check workflow_ir first, then proposal
                                    wf = cd.get("workflow_ir") or {}
                                    prop = cd.get("proposal") or {}
                                    patch_notes = wf.get("patch_notes") or prop.get("patch_notes") or []
                            
                            trace_md.append(f"> **Patch Applied**: `{patched_id}` created.")
                            if patch_notes:
                                trace_md.append("> Rules fixed:")
                                for note in patch_notes[:5]: # Limit 5
                                    trace_md.append(f"> - {note}")
                                if len(patch_notes) > 5: trace_md.append(f"> - ... ({len(patch_notes)-5} more)")
                            
                        except Exception as e:
                            print(f"Diff generation skipped/failed: {e}")
                            trace_md.append(f"> Patch applied but diff generation failed: {e}")

                    except Exception as e:
                        print(f"Patch failed: {e}")
                        trace_md.append(f"> **Patch Failed**: {e}")
        
        # Finalize
        write_json(evo_dir / "trace.json", trace)
        (evo_dir / "trace.md").write_text("\n".join(trace_md), encoding="utf-8")
            
        print("\nEvolution completed.")
        print(f"Final Champion: {current_champion_id}")
        print(f"Trace saved to: {evo_dir / 'trace.md'}")
