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

# ... (generate, audit, etc. unchanged)

@app.command()
def iterate(
    session: str = typer.Option(..., help="Path to session directory"),
    rounds: int = typer.Option(3, help="Number of evolution rounds"),
    population: int = typer.Option(3, help="Population size per round"),
    topk: int = typer.Option(1, help="Top K candidates to refine (currently only 1 supported)"),
    patch_mode: str = typer.Option("strategy", help="Patch mode: 'quick' or 'strategy'"),
    include_advice: bool = typer.Option(False, help="Include Advice-level fixes"),
    reset: bool = typer.Option(False, "--reset/--no-reset", help="Reset session before starting (Use --reset to enable, --no-reset to disable)"),
    allow_multi_patch: bool = typer.Option(False, help="Allow patching a candidate multiple times (recursive patching)")
):
    """Run iterative evolution engine."""
    try:
        core.iterate(session, rounds, population, topk, patch_mode, include_advice, reset, allow_multi_patch)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
