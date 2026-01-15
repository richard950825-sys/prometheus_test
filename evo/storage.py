"""
Storage utilities for evo.
Handles all file I/O, candidate loading/saving, and index management.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from .models import Candidate


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
