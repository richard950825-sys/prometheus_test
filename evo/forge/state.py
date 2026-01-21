from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ForgeState(BaseModel):
    user_requirements: Dict[str, Any] = Field(default_factory=dict)
    draft_workflows: List[Dict[str, Any]] = Field(default_factory=list)
    current_workflow: Optional[Dict[str, Any]] = None
    audit_report: Optional[Dict[str, Any]] = None
    simulation_summary: Optional[Dict[str, Any]] = None
    failure_cases: List[Dict[str, Any]] = Field(default_factory=list)
    required_fixes: List[Dict[str, Any]] = Field(default_factory=list)
    wdr_updates: List[Dict[str, Any]] = Field(default_factory=list)
    iteration_count: int = 0
    stop_reason: Optional[str] = None
