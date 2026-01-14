import os
import json
import re
from typing import Optional, Dict, Any
from google import genai
from dotenv import load_dotenv
from .models import Proposal, Candidate
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging (removed as generate_candidate uses print)
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

def generate_candidate(requirements: Dict[str, Any], attempt: int = 1, strategy_hint: Optional[str] = None) -> Optional[Candidate]:
    """
    Generates a single architecture candidate using Gemini with validation and retry.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY not found.")
        return None

    client = genai.Client(api_key=api_key)
    model_id = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    prompt = f"""
    You are an expert system architect.
    Based on the following requirements, generate ONE distinct architecture candidate proposal in valid JSON format.
    
    STRATEGY: {strategy_hint if strategy_hint else "Balanced"}
    (You MUST optimize the design for this strategy.)
    
    Requirements:
    {json.dumps(requirements, indent=2)}
    
    Constraints & Logic:
    1. Strategy '{strategy_hint}':
       - if 'cost-optimized': Minimize instances, use 'none' for cache/queue if feasible, focus on open source.
       - if 'reliability-optimized': MUST have cache and queue, multiple instances, strict timeouts (idempotency required).
       - if 'throughput-optimized': Async patterns, high target_rps, eventual consistency.
    2. Traffic: experiments.load_test.target_rps MUST depend on requirements.traffic.peak_rps (if present).
    3. Constraints: If 'data_residency' is specified, mention it in component notes.
    
    Output Schema:
    The output must be a single strict JSON object (NOT a list) matching this structure:
    {{
      "id": "string (unique)",
      "created_at": "ISO datetime",
      "proposal": {{
        "title": "string",
        "summary": "string",
        "strategy": "{strategy_hint if strategy_hint else 'balanced'}",
        "architecture": {{
          "style": "monolith|modular-monolith|microservices|event-driven",
          "components": {{
            "api": {{ "type": "http|grpc", "instances": int, "notes": "string" }},
            "db": {{ "type": "postgres|mysql|dynamodb|mongodb", "notes": "string" }},
            "cache": {{ "type": "redis|none|memcached", "notes": "string" }},
            "queue": {{ "type": "kafka|rabbitmq|sqs|none", "notes": "string" }}
          }},
          "reliability": {{
            "timeouts_ms": {{ "client": int, "server": int }},
            "retries": {{ "max_attempts": int, "backoff": "none|fixed|exponential" }},
            "idempotency": "required|recommended|not_supported"
          }}
        }},
        "slo": {{
          "p95_latency_ms": int,
          "error_rate": float,
          "availability": float
        }},
        "acceptance": {{
          "hard_constraints": [
            {{ "metric": "latency.p95_ms", "op": "<=", "threshold": int }},
            {{ "metric": "errors.rate", "op": "<=", "threshold": float }}
          ]
        }},
        "experiments": {{
          "load_test": {{ "tool": "k6", "duration_s": int, "target_rps": int }},
          "chaos_test": [ {{ "fault": "string", "target": "string", "duration_s": int }} ]
        }},
        "risks": [ {{ "title": "string", "mitigation": "string" }} ]
      }}
    }}
    
    SELF-CORRECTION CHECKLIST (MUST VERIFY):
    1. proposal.slo MUST align with requirements.slo.
    2. proposal.strategy MUST be '{strategy_hint if strategy_hint else 'balanced'}' (lowercase).
    3. JSON Format MUST be valid.
    
    Do not include markdown formatting.
    """

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config={
                "response_mime_type": "application/json"
            }
        )
        
        raw_text = response.text
        if not raw_text:
            raise ValueError("Empty response from Gemini")

        # Clean potential markdown (though mime_type helps)
        clean_text = re.sub(r"```json\s*|\s*```", "", raw_text).strip()
        data = json.loads(clean_text)
        
        # Validation
        # Check if it's a list by mistake
        if isinstance(data, list):
            data = data[0]
            
        # Pydantic Validate via Candidate (validation of Proposal nested field happens here)
        # Note: data should have 'id', 'created_at', 'proposal' keys as per schema.
        # But Candidate model has 'proposal' field. 
        # If Gemini returns the full object structure including 'id', we can try to parse it.
        # However, to be safe, we can reconstruct Candidate.
        
        # Let's try to validate the Inner Proposal first if it exists
        if "proposal" not in data:
            raise ValueError("Missing 'proposal' field in JSON.")
            
        proposal_obj = Proposal(**data["proposal"]) # This will raise ValidationError if missing fields
        
        # Construct Candidate
        candidate = Candidate(
            id=data.get("id") or f"gen_{int(datetime.now().timestamp())}",
            created_at=datetime.now(),
            proposal=proposal_obj,
            content=data.get("content") # pass through if generated
        )
        return candidate

    except Exception as e:
        print(f"Gemini generation failed (Attempt {attempt}): {e}")
        if attempt < 2:
            print("Retrying with feedback...")
            return generate_candidate(requirements, attempt + 1, strategy_hint)
        return None
