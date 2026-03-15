"""
medguard CLI — start the server and manage configuration.

Usage:
    medguard                  # Start server on default port 8080
    medguard --port 9090      # Custom port
    medguard --host 127.0.0.1 # Bind to localhost only
    medguard --reload         # Auto-reload on code changes (dev mode)
    medguard check "text"     # Check text for PHI/safety issues
    medguard config           # Show current configuration
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="medguard",
        description="medguard — Healthcare LLM guardrails middleware",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Default: serve
    serve_parser = subparsers.add_parser("serve", help="Start the API server (default)")
    _add_serve_args(serve_parser)

    # check subcommand
    check_parser = subparsers.add_parser("check", help="Check text for safety issues")
    check_parser.add_argument("text", help="Text to check")

    # config subcommand
    subparsers.add_parser("config", help="Show current configuration")

    # Also allow serve args at top level (no subcommand = serve)
    _add_serve_args(parser)

    args = parser.parse_args()

    if args.command == "check":
        _run_check(args.text)
    elif args.command == "config":
        _show_config()
    else:
        _run_server(args)


def _add_serve_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--host", default=None, help="Bind host (default: from config)")
    p.add_argument("--port", type=int, default=None, help="Bind port (default: from config)")
    p.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    p.add_argument("--log-level", default=None, help="Log level (debug/info/warning/error)")


def _run_server(args) -> None:
    import uvicorn
    from medguard.config import get_config
    from medguard.core import MedGuard

    config = get_config()
    host = getattr(args, "host", None) or config.api.host
    port = getattr(args, "port", None) or config.api.port
    log_level = getattr(args, "log_level", None) or config.api.log_level.lower()
    reload = getattr(args, "reload", False)

    from rich.console import Console
    console = Console()
    console.print(f"\n[bold green]medguard[/bold green] v0.1.0")
    console.print(f"  API server starting on [cyan]http://{host}:{port}[/cyan]")
    console.print(f"  Docs: [cyan]http://{host}:{port}/docs[/cyan]")
    console.print(f"  Health: [cyan]http://{host}:{port}/v1/health[/cyan]\n")

    mg = MedGuard()
    app = mg.create_app()

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        reload=reload,
    )


def _run_check(text: str) -> None:
    import asyncio
    from rich.console import Console
    from rich.table import Table
    from medguard.core import MedGuard

    console = Console()
    mg = MedGuard()
    ctx = asyncio.run(mg.acheck(text))

    console.print("\n[bold]medguard safety check[/bold]")

    table = Table(show_header=True)
    table.add_column("Guardrail")
    table.add_column("Result")
    table.add_column("Details")

    if ctx.phi_result:
        status = "[red]TRIGGERED[/red]" if ctx.phi_result.phi_detected else "[green]CLEAN[/green]"
        details = f"{len(ctx.phi_result.matches)} entity(ies)" if ctx.phi_result.phi_detected else ""
        table.add_row("PHI Detection", status, details)

    if ctx.scope_result:
        status = "[yellow]OUT OF SCOPE[/yellow]" if not ctx.scope_result.in_scope else "[green]IN SCOPE[/green]"
        table.add_row("Scope", status, ctx.scope_result.category.value)

    if ctx.drug_result:
        status = "[red]BLOCKED[/red]" if ctx.drug_result.blocked else (
            "[yellow]WARNING[/yellow]" if ctx.drug_result.interactions else "[green]CLEAN[/green]"
        )
        details = f"{len(ctx.drug_result.interactions)} interaction(s)" if ctx.drug_result.interactions else ""
        table.add_row("Drug Safety", status, details)

    console.print(table)

    if ctx.blocked:
        console.print(f"\n[bold red]BLOCKED:[/bold red] {ctx.block_reason}")
    elif ctx.warnings:
        console.print(f"\n[bold yellow]Warnings:[/bold yellow]")
        for w in ctx.warnings:
            console.print(f"  • {w}")
    else:
        console.print("\n[bold green]All checks passed.[/bold green]")


def _show_config() -> None:
    from rich.console import Console
    from rich.syntax import Syntax
    from medguard.config import get_config

    config = get_config()
    console = Console()
    console.print("\n[bold]Current medguard configuration:[/bold]")
    console.print(Syntax(config.model_dump_json(indent=2), "json"))


if __name__ == "__main__":
    main()
