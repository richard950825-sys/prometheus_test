from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class WindTunnelReport(BaseModel):
    summary: Dict[str, Any]
    runs: List[Dict[str, Any]]
    stats: Dict[str, Any]
    failure_clusters: List[Dict[str, Any]] = Field(default_factory=list)
    regression: Optional[Dict[str, Any]] = None
    artifacts: Dict[str, Any] = Field(default_factory=dict)
