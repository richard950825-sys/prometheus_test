from typing import Any, Dict

from pydantic import BaseModel, Field


class WindTunnelSpec(BaseModel):
    sut_workflow: Dict[str, Any]
    scenario_pack: Dict[str, Any]
    fault_profile: Dict[str, Any]
    scorers: Dict[str, Any] = Field(default_factory=dict)
    run_config: Dict[str, Any] = Field(default_factory=dict)
