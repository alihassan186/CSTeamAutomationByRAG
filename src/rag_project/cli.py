from __future__ import annotations

import json
from typing import Optional

import typer
from dotenv import load_dotenv

from rag_project.ingest import ingest as ingest_index
from rag_project.rag import build_rag_chain
from rag_project.settings import get_settings


app = typer.Typer(add_completion=False, help="LangChain RAG CLI")


@app.command(name="ingest")
def ingest_cli(
    reset: bool = typer.Option(False, "--reset", help="Delete existing index and rebuild"),
):
    """Ingest documents from data/raw into a persistent Chroma index."""
    settings = get_settings()
    result = ingest_index(settings, reset=reset)
    typer.echo(json.dumps(result, indent=2))


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    show_sources: bool = typer.Option(False, "--show-sources", help="Return sources for retrieved chunks"),
):
    """Ask a single question."""
    settings = get_settings()
    chain = build_rag_chain(settings, show_sources=show_sources)
    result = chain.invoke(question)
    if isinstance(result, dict):
        typer.echo(json.dumps(result, indent=2))
    else:
        typer.echo(result)


@app.command()
def chat(
    show_sources: bool = typer.Option(False, "--show-sources", help="Return sources for each answer"),
):
    """Interactive chat loop (type 'exit' to quit)."""
    settings = get_settings()
    chain = build_rag_chain(settings, show_sources=show_sources)

    typer.echo("RAG chat started. Type 'exit' to quit.")
    while True:
        try:
            q = typer.prompt("You")
        except (EOFError, KeyboardInterrupt):
            typer.echo("\nbye")
            return

        if q.strip().lower() in {"exit", "quit"}:
            typer.echo("bye")
            return

        result = chain.invoke(q)
        if isinstance(result, dict):
            typer.echo(json.dumps(result, indent=2))
        else:
            typer.echo(result)


def main(argv: Optional[list[str]] = None) -> None:
    load_dotenv(override=False)
    app(standalone_mode=True, prog_name="rag")


if __name__ == "__main__":
    main()
