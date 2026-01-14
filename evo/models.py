from datetime import datetime
from typing import List, Optional, Dict, Any, Union, Literal, Tuple
from pydantic import BaseModel, Field
from enum import Enum

class Severity(str, Enum):
    HARD = "HARD"
    RISK = "RISK"
    ADVICE = "ADVICE"

# --- Architecture v2 Models ---

class ComponentDetails(BaseModel):
    type: str
    instances: Optional[int] = None
    notes: Optional[str] = None

class Components(BaseModel):
    api: ComponentDetails
    db: ComponentDetails
    cache: Optional[ComponentDetails] = None
    queue: Optional[ComponentDetails] = None

class ReliabilityConfig(BaseModel):
    timeouts_ms: Dict[str, int] # e.g. {"client": 1000, "server": 5000}
    retries: Dict[str, Union[int, str]] # e.g. {"max_attempts": 3, "backoff": "exponential"}
    idempotency: Literal["required", "recommended", "not_supported"]

class Architecture(BaseModel):
    style: Literal["monolith", "modular-monolith", "microservices", "event-driven"]
    components: Components
    reliability: ReliabilityConfig

class SLO(BaseModel):
    p95_latency_ms: int
    error_rate: float
    availability: float

class Constraint(BaseModel):
    metric: str
    op: str
    threshold: Union[int, float]

class Acceptance(BaseModel):
    hard_constraints: List[Constraint]

class ExperimentConfig(BaseModel):
    tool: str
    duration_s: int
    target_rps: Optional[int] = None

class ChaosTest(BaseModel):
    fault: str
    target: str
    duration_s: int

class Experiments(BaseModel):
    load_test: ExperimentConfig
    chaos_test: List[ChaosTest]

class Risk(BaseModel):
    title: str
    mitigation: str

class Compliance(BaseModel):
    gdpr_compliant: bool
    data_residency_statement: Optional[str] = None
    notes: Optional[str] = None

class CostEstimate(BaseModel):
    monthly_cost_usd: Optional[int] = None
    estimate_range_usd_per_month: Optional[Tuple[int, int]] = None
    estimate_band: Literal["low", "medium", "high"] = "medium"
    confidence: Literal["low", "medium", "high"] = "low"
    notes: Optional[str] = None

class Proposal(BaseModel):
    title: str
    summary: str
    critical_paths: List[str] = Field(default_factory=list)
    architecture: Architecture
    slo: SLO
    acceptance: Acceptance
    experiments: Experiments
    risks: List[Risk]
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
    # v2 structure
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
