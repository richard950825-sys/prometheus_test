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
    You are an expert AI system architect.
    Based on the following requirements, generate ONE distinct Workflow IR candidate in valid JSON format.
    
    STRATEGY: {strategy_hint if strategy_hint else "Balanced"}
    (You MUST optimize the workflow design for this strategy.)
    
    Requirements:
    {json.dumps(requirements, indent=2)}
    
    Strategies:
    - if 'cost-optimized': Minimize tool usage, use cheaper models (gemini-flash), stricter budgets, fewer steps.
    - if 'reliability-optimized': Add 'verify' steps, 'critic' agents, robust fallbacks, higher max_retries.
    - if 'throughput-optimized': Maximize token budget, fewer verification steps, maximize parallel actions if potential.
    
    Output Schema:
    The output must be a single strict JSON object matching this structure:
    {{
      "id": "string (unique)",
      "created_at": "ISO datetime",
      "strategy": "{strategy_hint if strategy_hint else 'balanced'}",
      "workflow_ir": {{
        "title": "string",
        "goal": "string",
        "agents": [
          {{ "name": "planner|critic|coder", "role": "string", "model": "gemini-2.5-flash", "tools": ["retrieval","code"], "budget": {{ "max_turns": int }} }}
        ],
        "steps": [
          {{
            "id": "S1", "agent": "string", "action": "plan|retrieve|synthesize|verify|code",
            "inputs": ["string"], "outputs": ["string"],
            "guards": {{ "max_retries": 1, "timeout_s": 60 }}
          }}
        ],
        "controls": {{
          "budget": {{ "max_total_turns": 20, "max_total_tool_calls": 50, "max_total_tokens": 100000 }},
          "stop_conditions": ["answer_verified", "budget_exceeded"],
          "fallbacks": [ {{ "when": "retrieval_empty", "do": "broaden_query" }} ]
        }},
        "acceptance": {{
          "tests": [ {{ "name": "faithfulness", "type": "rag", "threshold": 0.8 }} ]
        }}
      }}
    }}
    
    SELF-CORRECTION CHECKLIST:
    1. workflow_ir.agents MUST have at least 1 agent.
    2. workflow_ir.steps MUST flow logically.
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

        # Clean potential markdown
        clean_text = re.sub(r"```json\s*|\s*```", "", raw_text).strip()
        data = json.loads(clean_text)
        
        # Validation
        if isinstance(data, list):
            data = data[0]
            
        from .models import WorkflowIR
        
        if "workflow_ir" not in data:
            # Fallback check if LLM returned just the inner object
            if "agents" in data and "steps" in data:
                wf_obj = WorkflowIR(**data)
                # Wrap it
                candidate = Candidate(
                    id=f"gen_{int(datetime.now().timestamp())}",
                    created_at=datetime.now(),
                    strategy=strategy_hint,
                    workflow_ir=wf_obj
                )
                return candidate
            raise ValueError("Missing 'workflow_ir' field in JSON.")
            
        wf_obj = WorkflowIR(**data["workflow_ir"])
        
        # Construct Candidate
        candidate = Candidate(
            id=data.get("id") or f"gen_{int(datetime.now().timestamp())}",
            created_at=datetime.now(),
            strategy=data.get("strategy") or strategy_hint,
            workflow_ir=wf_obj,
            proposal=None # Pivot away from proposal
        )
        return candidate

    except Exception as e:
        print(f"Gemini generation failed (Attempt {attempt}): {e}")
        if attempt < 2:
            print("Retrying with feedback...")
            return generate_candidate(requirements, attempt + 1, strategy_hint)
        return None
