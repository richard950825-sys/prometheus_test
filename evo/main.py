import typer
from .core import EvoCore
from typing import Optional

app = typer.Typer(help="Evo: A session-based generation and audit tool.")
core = EvoCore()

@app.command()
def init_session(name: str = typer.Option(..., help="Name of the session")):
    """Initialize a new session."""
    try:
        from pathlib import Path
        session_path = Path("sessions") / name
        if session_path.exists():
            typer.echo(f"Session directory '{session_path}' already exists.", err=True)
            typer.echo(f"Tip: You can run 'evo iterate --session {session_path}' directly, or use '--reset' to clear it.")
            return

        core.init_session(name)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def generate(
    session: str = typer.Option(..., help="Path to session directory"),
    n: int = typer.Option(3, help="Number of candidates to generate"),
    reset: bool = typer.Option(False, help="Clear existing candidates before generation")
):
    """Generate architecture candidates."""
    core.generate(session, n, reset)

@app.command()
def audit(
    session: str = typer.Option(..., help="Path to session directory"),
    include_patched: bool = typer.Option(False, help="Include patched candidates")
):
    """Audit candidates against ruleset."""
    core.audit(session, include_patched)

@app.command()
def report(
    session: str = typer.Option(..., help="Path to session directory"),
    include_patched: bool = typer.Option(False, help="Include patched candidates")
):
    """Generate session report."""
    core.report(session, include_patched)

@app.command()
def patch(
    session: str = typer.Option(..., help="Path to session directory"),
    candidate_id: str = typer.Option(..., help="Candidate ID to patch (or 'ALL')"),
    apply_advice: bool = typer.Option(False, help="Apply ADVICE level fixes"),
    mode: str = typer.Option("quick", help="Patch mode: 'quick' or 'strategy'")
):
    """Patch a candidate."""
    if candidate_id == "ALL":
        # Loop all unpatched
        import json
        from pathlib import Path
        s_dir = Path(session) / "candidates"
        for cf in s_dir.glob("cand_*.json"):
            if "_patched" in cf.name: continue
            cid = cf.stem.replace("cand_", "")
            try:
                core.patch(session, cid, apply_advice, mode)
            except Exception as e:
                print(f"Failed to patch {cid}: {e}")
    else:
        core.patch(session, candidate_id, apply_advice, mode)

@app.command()
def diff(
    session: str = typer.Option(..., help="Path to session directory"),
    base: str = typer.Option(..., help="Base candidate ID"),
    target: str = typer.Option(..., help="Target candidate ID")
):
    """Diff two candidates."""
    core.diff(session, base, target)

@app.command()
def recommend(
    session: str = typer.Option(..., help="Path to session directory"),
    include_patched: bool = typer.Option(False, help="Include patched candidates")
):
    """Recommend the best candidate."""
    core.recommend(session, include_patched)

@app.command()
def iterate(
    session: str = typer.Option(..., help="Path to session directory"),
    rounds: int = typer.Option(3, help="Number of evolution rounds"),
    population: int = typer.Option(3, help="Population size per round"),
    topk: int = typer.Option(1, help="Number of champions to carry over"),
    patch_mode: str = typer.Option("strategy", help="Patch mode: minimal, aggressive, strategy"),
    include_advice: bool = typer.Option(False, help="Include ADVICE level patches"),
    reset: bool = typer.Option(False, help="Reset evolution history"),
    allow_multi_patch: bool = typer.Option(False, help="Allow patching already patched candidates"),
    suite: Optional[str] = typer.Option(None, help="Name of evaluation suite to run")
):
    """Run iterative evolution engine."""
    try:
        core.iterate(session, rounds, population, topk, patch_mode, include_advice, reset, allow_multi_patch, suite=suite)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def metrics(
    session: str = typer.Option(..., help="Path to session directory"),
    include_patched: bool = typer.Option(False, help="Include patched candidates")
):
    """
    Generate outputs/metrics.json for candidates.
    """
    try:
        core.generate_metrics(session, include_patched=include_patched)
    except Exception as e:
         typer.echo(f"Error: {e}", err=True)
         raise typer.Exit(code=1)

@app.command()
def run(
    session: str = typer.Option(..., help="Path to session directory"),
    suite: str = typer.Option("rag_mini", help="Name of evaluation suite"),
    include_patched: bool = typer.Option(False, help="Include patched candidates")
):
    """Run simulated evaluation suite on candidates."""
    core = EvoCore()
    core.run_suite(session, suite, include_patched)

@app.command()
def index(
    session: str = typer.Option(..., help="Path to session directory"),
    rebuild: bool = typer.Option(False, help="Force rebuild of candidate index")
):
    """Manage candidate index."""
    if rebuild:
        core = EvoCore()
        core._update_candidate_index(session)
    else:
        typer.echo("Use --rebuild to regenerate the index.")

@app.command()
def migrate_ids(
    session: str = typer.Option(..., help="Path to session directory"),
    inplace: bool = typer.Option(False, help="Perform actual migration (otherwise dry run)")
):
    """Migrate legacy IDs to strict UUID format."""
    core = EvoCore()
    core.migrate_ids(session, inplace)

if __name__ == "__main__":
    app()
