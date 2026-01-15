from datetime import datetime
from typing import List, Optional, Dict, Any, Union, Literal, Tuple
from pydantic import BaseModel, Field
from enum import Enum

class Severity(str, Enum):
    HARD = "HARD"
    RISK = "RISK"
    ADVICE = "ADVICE"


# --- Workflow IR Models ---

class Agent(BaseModel):
    name: str
    role: str
    model: str
    tools: List[str] = Field(default_factory=list)
    budget: Dict[str, Union[int, float]] = Field(default_factory=dict) # e.g. {"max_turns": 6}

class Step(BaseModel):
    id: str
    agent: str
    action: str # plan|retrieve|synthesize|verify|code|review|decide
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)
    guards: Dict[str, Union[int, float, str]] = Field(default_factory=dict)

class Fallback(BaseModel):
    when: str
    do: str

class WorkflowBudget(BaseModel):
    max_total_turns: int
    max_total_tool_calls: int
    max_total_tokens: int

class Controls(BaseModel):
    budget: WorkflowBudget
    stop_conditions: List[str] = Field(default_factory=list)
    fallbacks: List[Fallback] = Field(default_factory=list)

class ArchitectureTest(BaseModel):
    name: str
    type: str
    threshold: float

class WorkflowAcceptance(BaseModel):
    tests: List[ArchitectureTest]

class WorkflowIR(BaseModel):
    title: str
    goal: str
    agents: List[Agent]
    steps: List[Step]
    controls: Controls
    acceptance: WorkflowAcceptance
    # Compatible fields for audit fallback
    summary: Optional[str] = None
    normalization_notes: List[str] = Field(default_factory=list)
    patch_notes: List[str] = Field(default_factory=list)

# --- Legacy Architecture v2 Models (kept for reference/mixins if needed) ---
# (Keeping Architecture/Proposal classes if we need smooth transition, 
#  but WorkflowIR replaces Proposal mainly)

class Compliance(BaseModel):
    gdpr_compliant: bool
    data_residency_statement: Optional[str] = None

class CostEstimate(BaseModel):
    monthly_cost_usd: Optional[float] = None
    estimate_range_usd_per_month: Optional[Union[Tuple[float, float], List[float]]] = None
    estimate_band: Optional[str] = None
    confidence: Optional[str] = None

class Proposal(BaseModel):
    # Reduced proposal just in case legacy code dep
    title: str
    summary: str
    architecture: Any = None
    slo: Any = None
    acceptance: Any = None
    experiments: Any = None
    risks: Any = None
    normalization_notes: List[str] = Field(default_factory=list)
    patch_notes: List[str] = Field(default_factory=list)
    strategy: Optional[str] = None
    compliance: Optional[Compliance] = None
    estimates: Optional[CostEstimate] = None

# --- Core Models ---

class Rule(BaseModel):
    id: str
    description: str
    severity: Severity
    condition: str
    fix_suggestion: Optional[str] = None

class Ruleset(BaseModel):
    rules: List[Rule]

class Candidate(BaseModel):
    id: str
    created_at: datetime = Field(default_factory=datetime.now)
    strategy: Optional[str] = None
    # v3 structure
    workflow_ir: Optional[WorkflowIR] = None
    # v2 structure (deprecated)
    proposal: Optional[Proposal] = None
    # v1 legacy support
    content: Optional[Dict[str, Any]] = None

class Violation(BaseModel):
    rule_id: str
    severity: Severity
    message: str
    fix_suggestion: Optional[str] = None

class AuditResult(BaseModel):
    candidate_id: str
    passed: bool
    violations: List[Violation]
    score: float

class SessionMetadata(BaseModel):
    name: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

# --- Metrics Models ---

class Metric(BaseModel):
    metric_key: str
    value: Union[int, float, bool, str]
    unit: Optional[str] = None
    source: Literal["proposal", "static_estimate", "evidence"]
    confidence: Literal["low", "medium", "high"]
    evidence_refs: List[str] = Field(default_factory=list)

class MetricsOutput(BaseModel):
    candidate_id: str
    generated_at: datetime = Field(default_factory=datetime.now)
    metrics: List[Metric]
    notes: List[str] = Field(default_factory=list)
    schema_version: str = "metrics_v1"
