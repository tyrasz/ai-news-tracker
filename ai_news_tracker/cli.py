"""Command-line interface for the AI News Tracker."""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Confirm

from .models import init_db, FeedSource
from .embeddings import EmbeddingEngine
from .preferences import PreferenceLearner
from .recommender import NewsRecommender
from .sources import DEFAULT_FEEDS, FEED_CATEGORIES, ALL_FEEDS

console = Console()


def get_recommender(db_path: str = "news_tracker.db") -> tuple[NewsRecommender, any]:
    """Initialize and return the recommender with database session."""
    engine, Session = init_db(db_path)
    session = Session()
    embedding_engine = EmbeddingEngine()
    preference_learner = PreferenceLearner(embedding_engine)
    recommender = NewsRecommender(session, embedding_engine, preference_learner)
    return recommender, session


@click.group()
@click.option("--db", default="news_tracker.db", help="Path to database file")
@click.pass_context
def main(ctx, db):
    """AI News Tracker - Personalized news that learns your preferences."""
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db


@main.command()
@click.option("--all", "add_all", is_flag=True, help="Add all available news sources")
@click.option("--category", "-c", multiple=True, help="Add specific category (tech, world, us, politics, business, science, middle_east, europe, asia)")
@click.pass_context
def init(ctx, add_all, category):
    """Initialize the database with news sources.

    By default, adds tech sources. Use --all for all sources, or -c for specific categories.

    Examples:
        news init                    # Tech sources only
        news init --all              # All sources
        news init -c world -c us     # World and US news
    """
    recommender, session = get_recommender(ctx.obj["db_path"])

    # Determine which feeds to add
    if add_all:
        feeds_to_add = ALL_FEEDS
        console.print("[yellow]Adding all news sources...[/yellow]\n")
    elif category:
        feeds_to_add = []
        for cat in category:
            if cat in FEED_CATEGORIES:
                feeds_to_add.extend(FEED_CATEGORIES[cat])
                console.print(f"[yellow]Adding {cat} sources...[/yellow]")
            else:
                console.print(f"[red]Unknown category: {cat}[/red]")
                console.print(f"Available: {', '.join(FEED_CATEGORIES.keys())}")
                return
        console.print()
    else:
        feeds_to_add = DEFAULT_FEEDS

    # Add feeds
    added = 0
    for name, url in feeds_to_add:
        existing = session.query(FeedSource).filter_by(url=url).first()
        if not existing:
            recommender.add_feed_source(name, url)
            added += 1
            console.print(f"  Added: [cyan]{name}[/cyan]")

    console.print(f"\n[green]Initialized with {added} new sources.[/green]")
    console.print("Run [cyan]news fetch[/cyan] to download articles.")


@main.command()
def categories():
    """List available feed categories."""
    table = Table(title="Available Feed Categories")
    table.add_column("Category", style="cyan")
    table.add_column("Sources", justify="right")
    table.add_column("Examples")

    for cat, feeds in FEED_CATEGORIES.items():
        examples = ", ".join(f[0] for f in feeds[:3])
        if len(feeds) > 3:
            examples += "..."
        table.add_row(cat, str(len(feeds)), examples)

    console.print(table)
    console.print("\nUsage: [cyan]news init -c world -c politics[/cyan]")
    console.print("       [cyan]news init --all[/cyan] (add everything)")


@main.command()
@click.option("--content", is_flag=True, help="Also fetch full article content (slower)")
@click.pass_context
def fetch(ctx, content):
    """Fetch new articles from all feed sources."""
    console.print("[yellow]Fetching articles...[/yellow]")

    recommender, session = get_recommender(ctx.obj["db_path"])

    # Check if we have sources
    sources = session.query(FeedSource).filter_by(is_active=True).count()
    if sources == 0:
        console.print("[red]No feed sources configured. Run 'news init' first.[/red]")
        return

    with console.status("Downloading and processing articles..."):
        count = recommender.refresh_all_feeds(fetch_content=content)

    console.print(f"[green]Fetched {count} new articles.[/green]")


@main.command()
@click.option("--limit", "-n", default=10, help="Number of articles to show")
@click.option("--all", "show_all", is_flag=True, help="Include already read articles")
@click.option("--freshness", "-f", default=0.3, help="Freshness weight 0-1 (0=relevance only, 1=recency only)")
@click.option("--half-life", "-h", default=24.0, help="Hours until freshness drops to 50%")
@click.pass_context
def read(ctx, limit, show_all, freshness, half_life):
    """Show personalized article recommendations."""
    recommender, session = get_recommender(ctx.obj["db_path"])

    recommendations = recommender.get_recommendations(
        limit=limit,
        include_read=show_all,
        freshness_weight=freshness,
        freshness_half_life_hours=half_life,
    )

    if not recommendations:
        console.print("[yellow]No articles found. Run 'news fetch' to download articles.[/yellow]")
        return

    table = Table(title="Recommended Articles", show_lines=True)
    table.add_column("ID", style="dim", width=4)
    table.add_column("Score", width=6)
    table.add_column("Fresh", width=5)
    table.add_column("Source", style="cyan", width=15)
    table.add_column("Title", style="bold")

    for article, score, fresh in recommendations:
        score_color = "green" if score > 0.6 else "yellow" if score > 0.4 else "dim"
        fresh_color = "green" if fresh > 0.7 else "yellow" if fresh > 0.3 else "red"
        table.add_row(
            str(article.id),
            f"[{score_color}]{score:.2f}[/{score_color}]",
            f"[{fresh_color}]{fresh:.0%}[/{fresh_color}]",
            (article.source or "Unknown")[:15],
            article.title[:70],
        )

    console.print(table)
    console.print(f"\n[dim]Freshness weight: {freshness:.0%} | Half-life: {half_life:.0f}h[/dim]")
    console.print("Use [cyan]news like <ID>[/cyan] or [cyan]news dislike <ID>[/cyan] to train your preferences.")
    console.print("Use [cyan]news open <ID>[/cyan] to view an article.")


@main.command()
@click.argument("query", nargs=-1, required=True)
@click.option("--limit", "-n", default=10, help="Number of articles to show")
@click.option("--all", "show_all", is_flag=True, help="Include already read articles")
@click.option("--freshness", "-f", default=0.2, help="Freshness weight 0-1")
@click.option("--half-life", "-h", default=24.0, help="Hours until freshness drops to 50%")
@click.pass_context
def topic(ctx, query, limit, show_all, freshness, half_life):
    """Search for articles about a specific topic.

    Examples:
        news topic artificial intelligence
        news topic "climate change"
        news topic python programming -n 20
    """
    query_str = " ".join(query)
    recommender, session = get_recommender(ctx.obj["db_path"])

    console.print(f"[yellow]Searching for articles about:[/yellow] [bold]{query_str}[/bold]\n")

    results = recommender.search_by_topic(
        query=query_str,
        limit=limit,
        include_read=show_all,
        freshness_weight=freshness,
        freshness_half_life_hours=half_life,
    )

    if not results:
        console.print("[yellow]No matching articles found.[/yellow]")
        return

    table = Table(title=f"Articles about: {query_str}", show_lines=True)
    table.add_column("ID", style="dim", width=4)
    table.add_column("Match", width=5)
    table.add_column("Fresh", width=5)
    table.add_column("Source", style="cyan", width=15)
    table.add_column("Title", style="bold")

    for article, score, fresh in results:
        score_color = "green" if score > 0.6 else "yellow" if score > 0.4 else "dim"
        fresh_color = "green" if fresh > 0.7 else "yellow" if fresh > 0.3 else "red"
        table.add_row(
            str(article.id),
            f"[{score_color}]{score:.0%}[/{score_color}]",
            f"[{fresh_color}]{fresh:.0%}[/{fresh_color}]",
            (article.source or "Unknown")[:15],
            article.title[:70],
        )

    console.print(table)
    console.print(f"\n[dim]Freshness weight: {freshness:.0%} | Half-life: {half_life:.0f}h[/dim]")
    console.print("Use [cyan]news open <ID>[/cyan] to view an article.")


@main.command()
@click.argument("article_id", type=int)
@click.pass_context
def like(ctx, article_id):
    """Mark an article as liked to improve recommendations."""
    recommender, session = get_recommender(ctx.obj["db_path"])
    recommender.record_feedback(article_id, liked=True)
    console.print(f"[green]Liked article {article_id}. Your preferences have been updated.[/green]")


@main.command()
@click.argument("article_id", type=int)
@click.pass_context
def dislike(ctx, article_id):
    """Mark an article as disliked to improve recommendations."""
    recommender, session = get_recommender(ctx.obj["db_path"])
    recommender.record_feedback(article_id, liked=False)
    console.print(f"[red]Disliked article {article_id}. Your preferences have been updated.[/red]")


@main.command()
@click.argument("article_id", type=int)
@click.pass_context
def open(ctx, article_id):
    """View article details and open in browser."""
    import webbrowser
    from .models import Article

    recommender, session = get_recommender(ctx.obj["db_path"])
    article = session.query(Article).get(article_id)

    if not article:
        console.print(f"[red]Article {article_id} not found.[/red]")
        return

    # Display article details
    panel_content = f"""[bold]{article.title}[/bold]

[dim]Source:[/dim] {article.source or 'Unknown'}
[dim]Author:[/dim] {article.author or 'Unknown'}
[dim]Published:[/dim] {article.published_at or 'Unknown'}
[dim]URL:[/dim] {article.url}

{article.summary or article.content or '[No content available]'}"""

    console.print(Panel(panel_content[:2000], title="Article Details"))

    # Mark as read
    recommender.mark_read(article_id)

    # Offer to open in browser
    if Confirm.ask("Open in browser?"):
        webbrowser.open(article.url)


@main.command()
@click.argument("name")
@click.argument("url")
@click.pass_context
def add_source(ctx, name, url):
    """Add a new RSS/Atom feed source."""
    recommender, session = get_recommender(ctx.obj["db_path"])
    recommender.add_feed_source(name, url)
    console.print(f"[green]Added source: {name}[/green]")


@main.command()
@click.pass_context
def sources(ctx):
    """List all configured feed sources."""
    _, session = get_recommender(ctx.obj["db_path"])

    feed_sources = session.query(FeedSource).all()

    if not feed_sources:
        console.print("[yellow]No sources configured. Run 'news init' to add defaults.[/yellow]")
        return

    table = Table(title="Feed Sources")
    table.add_column("ID", style="dim")
    table.add_column("Name", style="cyan")
    table.add_column("URL")
    table.add_column("Active", justify="center")
    table.add_column("Last Fetched")

    for source in feed_sources:
        table.add_row(
            str(source.id),
            source.name,
            source.url[:50] + "..." if len(source.url) > 50 else source.url,
            "[green]Yes[/green]" if source.is_active else "[red]No[/red]",
            str(source.last_fetched or "Never"),
        )

    console.print(table)


@main.command()
@click.pass_context
def stats(ctx):
    """Show statistics about your news and preferences."""
    recommender, session = get_recommender(ctx.obj["db_path"])
    stats = recommender.get_stats()

    panel_content = f"""[bold]Article Stats[/bold]
  Total articles: {stats['total_articles']}
  Unread: {stats['unread_articles']}
  Liked: {stats['liked_articles']}
  Disliked: {stats['disliked_articles']}

[bold]Sources[/bold]
  Active feeds: {stats['active_sources']}

[bold]Preferences[/bold]
  Profile trained: {'[green]Yes[/green]' if stats['has_preferences'] else '[yellow]Not yet - like/dislike articles to train[/yellow]'}"""

    console.print(Panel(panel_content, title="News Tracker Stats"))


@main.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", "-p", default=8000, help="Port to run on")
@click.pass_context
def serve(ctx, host, port):
    """Start the web interface."""
    from .web import run_server
    db_path = ctx.obj["db_path"]
    console.print(f"[green]Starting web server at http://{host}:{port}[/green]")
    console.print(f"[dim]Using database: {db_path}[/dim]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")
    run_server(host=host, port=port, db_path=db_path)


if __name__ == "__main__":
    main()
