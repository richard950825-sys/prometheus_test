from pathlib import Path
import json

from .models import Step, Risk, Compliance, CostEstimate, Fallback, Agent, Severity
from .storage import load_candidate, load_requirements, save_candidate, rebuild_index, load_audit_result, load_metrics, read_json


def apply_patch(
    session_path: str,
    candidate_id: str,
    apply_advice: bool = False,
    mode: str = "quick",
) -> bool:
    session_dir = Path(session_path)
    requirements = load_requirements(session_path)
    try:
        cand = load_candidate(session_path, candidate_id)
    except Exception:
        raise FileNotFoundError(f"Candidate file for {candidate_id} not found.")

    p = cand.proposal
    wf = cand.workflow_ir
    if not p and not wf:
        print("Candidate has no proposal or workflow data to patch.")
        return False

    notes = []

    if wf:
        audit = load_audit_result(session_path, candidate_id)
        rule_ids = {v.rule_id for v in audit.violations} if audit else set()
        hard_rule_ids = {v.rule_id for v in audit.violations if v.severity == Severity.HARD} if audit else set()
        metrics_list = load_metrics(session_path, candidate_id)

        def metric_raw(metric_key: str):
            for m in metrics_list:
                if isinstance(m, dict) and m.get("metric_key") == metric_key:
                    return m.get("value")
            return None

        def has_critic_agent(agents):
            for ag in agents:
                name = (ag.name or "").lower()
                role = (ag.role or "").lower()
                if "critic" in name or "review" in name or "critic" in role or "review" in role:
                    return True
            return False

        def critic_agent_name(agents):
            for ag in agents:
                name = (ag.name or "").lower()
                role = (ag.role or "").lower()
                if "critic" in name or "review" in name or "critic" in role or "review" in role:
                    return ag.name
            if agents:
                return agents[0].name
            return "critic"

        def unique_step_id(base_id: str) -> str:
            existing = {s.id for s in wf.steps}
            if base_id not in existing:
                return base_id
            idx = 2
            while f"{base_id}_{idx}" in existing:
                idx += 1
            return f"{base_id}_{idx}"

        def ensure_verify_step(note_id: str) -> Step:
            verify_step = next((s for s in wf.steps if s.action == "verify"), None)
            if verify_step:
                return verify_step
            verifier_agent = critic_agent_name(wf.agents)
            verify_step = Step(
                id=unique_step_id("step_verify"),
                agent=verifier_agent,
                action="verify",
                inputs=["response", "citations"],
                outputs=["verified_response"],
                guards={
                    "max_retries": 2,
                    "timeout_s": 60,
                    "conflict_detection_required": True,
                    "conflict_citations_required": True,
                    "injection_refusal_required": True,
                    "require_factual_citations": True,
                },
            )
            wf.steps.append(verify_step)
            notes.append(f"Fixed {note_id}: Injected 'verify' step.")
            return verify_step

        def ensure_critic_agent(note_id: str):
            if not wf.agents:
                planner = Agent(name="planner", role="planner", model="gemini-pro")
                critic = Agent(name="critic", role="critic/reviewer", model="gemini-pro")
                wf.agents.extend([planner, critic])
                notes.append(f"Fixed {note_id}: Added planner and critic agents.")
                return
            if has_critic_agent(wf.agents):
                if len(wf.agents) < 2:
                    wf.agents.append(Agent(
                        name="critic" if "critic" not in [a.name for a in wf.agents] else "critic_2",
                        role="critic/reviewer",
                        model=wf.agents[0].model,
                    ))
                    notes.append(f"Fixed {note_id}: Added critic agent to reach minimum agents.")
                return

            reviewer = next(
                (a for a in wf.agents if "review" in (a.name or "").lower() or "review" in (a.role or "").lower()),
                None,
            )
            if reviewer:
                reviewer.role = "critic/reviewer"
                notes.append(f"Fixed {note_id}: Marked existing reviewer as critic.")
                if len(wf.agents) < 2:
                    wf.agents.append(Agent(
                        name="critic" if "critic" not in [a.name for a in wf.agents] else "critic_2",
                        role="critic/reviewer",
                        model=wf.agents[0].model,
                    ))
                    notes.append(f"Fixed {note_id}: Added critic agent to reach minimum agents.")
            else:
                wf.agents.append(Agent(
                    name="critic" if "critic" not in [a.name for a in wf.agents] else "critic_2",
                    role="critic/reviewer",
                    model=wf.agents[0].model,
                ))
                notes.append(f"Fixed {note_id}: Added critic agent.")
                if len(wf.agents) < 2:
                    wf.agents.append(Agent(
                        name="planner" if "planner" not in [a.name for a in wf.agents] else "planner_2",
                        role="planner",
                        model=wf.agents[0].model,
                    ))
                    notes.append(f"Fixed {note_id}: Added planner agent to reach minimum agents.")

        def ensure_min_steps(note_id: str):
            needs_steps = len(wf.steps) < 3
            if not needs_steps:
                return
            actions = {s.action for s in wf.steps}
            if "plan" not in actions:
                wf.steps.insert(0, Step(
                    id=unique_step_id("step_plan"),
                    agent=wf.agents[0].name if wf.agents else "planner",
                    action="plan",
                    inputs=["query"],
                    outputs=["plan"],
                ))
                notes.append(f"Fixed {note_id}: Added plan step.")
            actions = {s.action for s in wf.steps}
            if "retrieve" not in actions and "synthesize" not in actions:
                insert_idx = 1 if wf.steps else 0
                wf.steps.insert(insert_idx, Step(
                    id=unique_step_id("step_retrieve"),
                    agent=wf.agents[0].name if wf.agents else "planner",
                    action="retrieve",
                    inputs=["query"],
                    outputs=["docs"],
                ))
                notes.append(f"Fixed {note_id}: Added retrieve step.")
            if not any(s.action == "verify" for s in wf.steps):
                ensure_verify_step(note_id)

        def ensure_stop_conditions(note_id: str):
            if not wf.controls:
                return
            if wf.controls.stop_conditions is None:
                wf.controls.stop_conditions = []
            if "budget_exceeded" not in wf.controls.stop_conditions:
                wf.controls.stop_conditions.append("budget_exceeded")
                notes.append(f"Fixed {note_id}: Added stop_conditions 'budget_exceeded'.")

        def ensure_budget_policy(note_id: str) -> bool:
            if not wf.controls:
                return False
            policy = wf.controls.budget_policy or {}
            changed = False
            if policy.get("mode") != "degrade_gracefully":
                policy["mode"] = "degrade_gracefully"
                changed = True
            if policy.get("on_budget_exceeded") != "return_best_effort_with_citations":
                policy["on_budget_exceeded"] = "return_best_effort_with_citations"
                changed = True
            if changed:
                wf.controls.budget_policy = policy
                notes.append(f"Fixed {note_id}: Added budget_policy degrade mode.")
            return changed

        def ensure_budget_guards(note_id: str) -> bool:
            verify_step = ensure_verify_step(note_id)
            guards = verify_step.guards or {}
            changed = False
            if not guards.get("budget_mode"):
                guards["budget_mode"] = "tight"
                changed = True
            if not bool(guards.get("disable_self_correction")):
                guards["disable_self_correction"] = True
                changed = True
            if not bool(guards.get("require_factual_citations")):
                guards["require_factual_citations"] = True
                changed = True
            if guards.get("response_style") != "short_with_citations":
                guards["response_style"] = "short_with_citations"
                changed = True
            if changed:
                verify_step.guards = guards
                notes.append(f"Fixed {note_id}: Added budget-aware verify guards.")
            return changed

        def ensure_budget_steps(note_id: str) -> bool:
            changed = False
            retrieve_step = next((s for s in wf.steps if s.action == "retrieve"), None)
            if retrieve_step:
                guards = retrieve_step.guards or {}
                if guards.get("top_k") != 1:
                    guards["top_k"] = 1
                    retrieve_step.guards = guards
                    notes.append(f"Fixed {note_id}: Set retrieve top_k=1 for budget mode.")
                    changed = True
            synth_step = next((s for s in wf.steps if s.action in ["synthesize", "compose", "respond", "decide"]), None)
            if not synth_step:
                synth_step = next((s for s in wf.steps if s.action == "verify"), None)
            if synth_step:
                guards = synth_step.guards or {}
                synth_changed = False
                if not bool(guards.get("structured_output_only")):
                    guards["structured_output_only"] = True
                    synth_changed = True
                if guards.get("reason_max_tokens") is None:
                    guards["reason_max_tokens"] = 20
                    synth_changed = True
                if synth_changed:
                    synth_step.guards = guards
                    notes.append(f"Fixed {note_id}: Enforced structured minimal synthesis output.")
                    changed = True
            return changed

        # Structural HARD fixes first.
        if "W001" in rule_ids or len(wf.agents) < 2 or not has_critic_agent(wf.agents):
            ensure_critic_agent("W001")

        if "W002" in rule_ids or len(wf.steps) < 3:
            ensure_min_steps("W002")

        if "W004" in rule_ids or ("budget_exceeded" not in (wf.controls.stop_conditions or [])):
            ensure_stop_conditions("W004")

        if "W011" in rule_ids:
            ensure_verify_step("W011")

        # Keep existing verify safety.
        if "W006" in rule_ids or not any(s.action == "verify" for s in wf.steps):
            ensure_verify_step("W006")

        if wf.controls and wf.controls.budget:
            if wf.controls.budget.max_total_turns > 50:
                old_val = wf.controls.budget.max_total_turns
                wf.controls.budget.max_total_turns = 30
                notes.append(f"Fixed W008: Reduced max_total_turns from {old_val} to 30.")

        # Oracle fixes after structural hardening.
        needs_worst_case_fix = "O300" in rule_ids
        needs_injection_fallback = "O100" in rule_ids or needs_worst_case_fix
        needs_conflict_guard = "O200" in rule_ids

        if needs_worst_case_fix:
            changed = False
            changed = ensure_budget_policy("O300") or changed
            changed = ensure_budget_guards("O300") or changed
            changed = ensure_budget_steps("O300") or changed
            if changed:
                worst_context_raw = metric_raw("robustness.worst_case_context")
                if isinstance(worst_context_raw, str):
                    notes.append("Fixed O300: Applied budget-aware degradation based on worst-case sweep.")

        o300_hard = "O300" in hard_rule_ids
        if o300_hard:
            worst_context_raw = metric_raw("robustness.worst_case_context")
            failures = {}
            if isinstance(worst_context_raw, str):
                try:
                    worst_context = json.loads(worst_context_raw)
                    aggregate_ref = worst_context.get("aggregate_ref")
                    if aggregate_ref:
                        agg_path = Path(session_path) / aggregate_ref
                        if agg_path.exists():
                            failures = read_json(agg_path).get("failures", {})
                except json.JSONDecodeError:
                    failures = {}

            by_assertion = failures.get("by_assertion", {}) if isinstance(failures, dict) else {}
            by_task = failures.get("by_task_id", {}) if isinstance(failures, dict) else {}
            strict_citations = bool(by_assertion.get("citations_required", 0))
            conflict_fail = bool(by_assertion.get("conflict_detected", 0) or by_assertion.get("must_mention_conflict", 0))
            injection_fail = bool(by_assertion.get("injection_refused", 0))
            conflict_task_hot = bool(by_task.get("conflict_2", 0))

            if not wf.controls.budget_tiers:
                wf.controls.budget_tiers = []
            tier1 = {
                "name": "tier1",
                "max_tool_calls": 2,
                "max_tokens": 160,
                "strict_templates": True,
                "deterministic_mode": True,
                "strict_citations": strict_citations,
                "strict_injection_refusal": injection_fail,
                "force_all_conflict_citations": conflict_fail or conflict_task_hot,
            }
            wf.controls.budget_tiers = [tier1]
            notes.append("Fixed O300: Added budget_tiers for low-budget strict mode.")

            verify_step = ensure_verify_step("O300")
            guards = verify_step.guards or {}
            if strict_citations:
                guards["require_factual_citations"] = True
                guards["budget_citations_required"] = True
                notes.append("Fixed O300: Enforced citations under low budget.")
            if conflict_fail or conflict_task_hot:
                guards["conflict_template"] = "require_conflict_with_dual_citations"
                guards["conflict_citations_required"] = True
                notes.append("Fixed O300: Added conflict verification template for low budget.")
            if injection_fail:
                guards["injection_refusal_required"] = True
                notes.append("Fixed O300: Enforced injection refusal under low budget.")
            verify_step.guards = guards

            worst_flake = metric_raw("robustness.worst_case_flake")
            try:
                if worst_flake is not None and float(worst_flake) > 0.1:
                    tier1["deterministic_mode"] = True
                    notes.append("Fixed O300: Enabled deterministic mode for low budget to reduce flake.")
            except (TypeError, ValueError):
                pass

        if needs_injection_fallback:
            if wf.controls and wf.controls.fallbacks is None:
                wf.controls.fallbacks = []
            existing = False
            if wf.controls and wf.controls.fallbacks:
                for fb in wf.controls.fallbacks:
                    when = fb.when if hasattr(fb, "when") else fb.get("when")
                    action = fb.do if hasattr(fb, "do") else fb.get("do")
                    if when == "injection_detected" and action == "refuse_and_use_factual_only":
                        existing = True
                        break
            if not existing:
                if wf.controls and wf.controls.fallbacks is not None:
                    wf.controls.fallbacks.append(Fallback(
                        when="injection_detected",
                        do="refuse_and_use_factual_only",
                    ))
                    notes.append("Fixed O100: Added injection_detected fallback to refuse and cite factual sources.")

            verify_step = ensure_verify_step("O100")
            guards = verify_step.guards or {}
            if not guards.get("injection_refusal_required"):
                guards["injection_refusal_required"] = True
                verify_step.guards = guards
                notes.append("Fixed O100: Required structured injection refusal in verify step.")

        if needs_conflict_guard:
            verify_step = ensure_verify_step("O200")
            guards = verify_step.guards or {}
            changed = False
            if not guards.get("conflict_detection_required"):
                guards["conflict_detection_required"] = True
                changed = True
            if not guards.get("conflict_citations_required"):
                guards["conflict_citations_required"] = True
                changed = True
            if changed:
                verify_step.guards = guards
                notes.append("Fixed O200: Strengthened verify step for structured conflict checks.")
        if apply_advice:
            if not wf.controls.fallbacks:
                wf.controls.fallbacks = [
                    Fallback(when="retrieval_empty", do="rewrite_query")
                ]
                notes.append("Applied Advice: Added basic fallback for retrieval_empty.")

    if p:
        def is_cache_missing(prop):
            if not prop.architecture or not prop.architecture.components:
                return True
            c = prop.architecture.components.cache
            if c is None:
                return True
            if isinstance(c, dict):
                return c.get("type", "none") == "none"
            return getattr(c, "type", "none") == "none"

        def is_queue_missing(prop):
            if not prop.architecture or not prop.architecture.components:
                return True
            q = prop.architecture.components.queue
            if q is None:
                return True
            if isinstance(q, dict):
                return q.get("type", "none") == "none"
            return getattr(q, "type", "none") == "none"

        if p.architecture and p.architecture.reliability:
            rel = p.architecture.reliability
            retries = rel.retries
            current_attempts = 0
            if isinstance(retries, dict):
                current_attempts = int(retries.get("max_attempts", 0))
            if current_attempts >= 3:
                rel.retries = {"max_attempts": 1, "backoff": "exponential"}
                notes.append("Fixed R001: Set retries.max_attempts to 1 and backoff to exponential.")

        if p.architecture and p.architecture.reliability:
            rel = p.architecture.reliability
            timeouts = rel.timeouts_ms
            client_to = timeouts.get("client", 0)
            server_to = timeouts.get("server", 0)
            if client_to <= server_to and server_to > 0:
                new_client = max(server_to + 300, server_to * 2)
                timeouts["client"] = new_client
                notes.append(f"Fixed R002: Increased client timeout to {new_client}ms (> server timeout {server_to}ms).")

        if p.architecture and p.architecture.reliability:
            rel = p.architecture.reliability
            retries = rel.retries
            attempts = 0
            if isinstance(retries, dict):
                attempts = int(retries.get("max_attempts", 0))
            idempotency = rel.idempotency
            if attempts > 0 and idempotency == "not_supported":
                rel.idempotency = "recommended"
                p.risks.append(Risk(
                    title="Idempotency Risk",
                    mitigation="Retries detected. Idempotency-Key recommended to prevent double-writes.",
                ))
                notes.append("Fixed R005: Set idempotency to 'recommended' and added Risk due to active retries.")

        residency_req = requirements.get("constraints", {}).get("data_residency")
        if residency_req and residency_req.lower() in ["eu", "europe"]:
            needs_c001 = False
            if not p.compliance:
                needs_c001 = True
            elif not p.compliance.data_residency_statement:
                needs_c001 = True
            elif residency_req.lower() not in p.compliance.data_residency_statement.lower():
                needs_c001 = True
            if needs_c001:
                if not p.compliance:
                    p.compliance = Compliance(gdpr_compliant=True, data_residency_statement="")
                new_stmt = f"All customer data will be stored and processed in {residency_req} regions only."
                p.compliance.data_residency_statement = new_stmt
                p.compliance.gdpr_compliant = True
                notes.append(f"Fixed C001: Added explicit data residency statement for {residency_req}.")

        budget_req = requirements.get("constraints", {}).get("monthly_budget_usd")
        if budget_req:
            needs_c002 = False
            if not p.estimates:
                needs_c002 = True
            elif not p.estimates.estimate_band:
                needs_c002 = True
            if needs_c002:
                band = "medium"
                if budget_req <= 10000:
                    band = "low"
                elif budget_req > 30000:
                    band = "high"
                est_val = int(budget_req * 0.9)
                est_range = (int(budget_req * 0.8), int(budget_req * 1.0))
                if not p.estimates:
                    p.estimates = CostEstimate(
                        monthly_cost_usd=est_val,
                        estimate_range_usd_per_month=est_range,
                        estimate_band=band,
                        confidence="medium",
                    )
                else:
                    p.estimates.estimate_band = band
                    if not p.estimates.monthly_cost_usd:
                        p.estimates.monthly_cost_usd = est_val
                    if not p.estimates.estimate_range_usd_per_month:
                        p.estimates.estimate_range_usd_per_month = est_range
                    p.estimates.confidence = "medium"
                notes.append(f"Fixed C002: Computed cost estimate band '{band}' with medium confidence based on budget.")

        should_inject_redis = False
        cache_reason = ""
        if mode == "strategy" and requirements:
            traffic = requirements.get("traffic", {})
            ratio = traffic.get("read_write_ratio", "")
            if ratio:
                try:
                    read_part = int(ratio.split(":")[0])
                    if read_part >= 70:
                        should_inject_redis = True
                        cache_reason = f"Applied Strategy S2: Injected Redis cache due to high read ratio ({ratio})."
                except:
                    pass
        if apply_advice and not should_inject_redis:
            should_inject_redis = True
            cache_reason = "Applied Advice A002: Injected Redis cache component."

        if is_cache_missing(p):
            if should_inject_redis:
                if p.architecture.components.cache is None:
                    p.architecture.components.cache = {"type": "redis", "notes": cache_reason}
                elif isinstance(p.architecture.components.cache, dict):
                    p.architecture.components.cache["type"] = "redis"
                    p.architecture.components.cache["notes"] = cache_reason
                else:
                    p.architecture.components.cache.type = "redis"
                    p.architecture.components.cache.notes = cache_reason
                notes.append(cache_reason)
                p.risks.append(Risk(
                    title="Cache Complexity",
                    mitigation="Introduction of Redis increases operational complexity. Ensure eviction policies and consistency checks.",
                ))
            else:
                p.risks.append(Risk(
                    title="Missing Cache Layer",
                    mitigation="Cache unenabled: may impact read performance. Evaluate Redis for high-read paths.",
                ))
                notes.append("Fixed A002: Added Risk entry for missing cache.")

        if is_queue_missing(p):
            pass

    if not notes:
        print("No patchable violations found or candidate already compliant.")
        return False

    if p:
        p.patch_notes.extend(notes)
    if wf:
        wf.patch_notes.extend(notes)

    new_id = f"{cand.id}_patched"
    cand.id = new_id
    new_path = save_candidate(session_path, cand)
    print(f"Patched candidate saved to: {new_path.name}")
    print(f"Notes: {notes}")
    rebuild_index(session_path)
    print("Changes applied:")
    for n in notes:
        print(f"- {n}")
    return True
