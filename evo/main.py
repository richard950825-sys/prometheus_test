import typer
from .core import EvoCore
from typing import Optional

app = typer.Typer(help="Evo: A session-based generation and audit tool.")
core = EvoCore()

@app.command()
def init_session(name: str = typer.Option(..., help="Name of the session")):
    """Initialize a new session."""
    try:
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
    try:
        core.generate(session, n, reset)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def audit(
    session: str = typer.Option(..., help="Path to session directory"),
    include_patched: bool = typer.Option(False, help="Include patched candidates in audit")
):
    """Audit candidates in a session against ruleset."""
    try:
        core.audit(session, include_patched=include_patched)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def diff(
    session: str = typer.Option(..., help="Path to session directory"),
    base: str = typer.Option(..., help="ID of base candidate"),
    target: str = typer.Option(..., help="ID of target candidate")
):
    """Compare two candidates and generate a diff report."""
    try:
        core.diff(session, base, target)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def report(
    session: str = typer.Option(..., help="Path to session directory"),
    include_patched: bool = typer.Option(False, help="Include patched candidates in report")
):
    """Generate a report from audit results."""
    try:
        core.report(session, include_patched=include_patched)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def patch(
    session: str = typer.Option(..., help="Path to session directory"),
    candidate: str = typer.Option(..., help="ID of the candidate to patch"),
    apply_advice: bool = typer.Option(False, help="Apply optional advice fixes (e.g. inject Redis)")
):
    """Patch a candidate based on audit rules (R001, R002, R005, A002)."""
    try:
        core.patch(session, candidate, apply_advice=apply_advice)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
