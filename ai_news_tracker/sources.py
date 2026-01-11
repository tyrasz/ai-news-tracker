"""News source ingestion from RSS feeds and web scraping."""

from __future__ import annotations

from datetime import datetime
from typing import Iterator
import feedparser
import httpx
import trafilatura
from dataclasses import dataclass


@dataclass
class RawArticle:
    """Raw article data before processing."""
    url: str
    title: str
    content: str | None = None
    summary: str | None = None
    source: str | None = None
    author: str | None = None
    published_at: datetime | None = None


class FeedFetcher:
    """Fetches articles from RSS/Atom feeds."""

    def __init__(self, timeout: float = 30.0):
        self.client = httpx.Client(timeout=timeout, follow_redirects=True)

    def fetch_feed(self, feed_url: str, source_name: str | None = None) -> Iterator[RawArticle]:
        """Parse an RSS/Atom feed and yield raw articles."""
        feed = feedparser.parse(feed_url)

        source = source_name or feed.feed.get("title", feed_url)

        for entry in feed.entries:
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published = datetime(*entry.published_parsed[:6])
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                published = datetime(*entry.updated_parsed[:6])

            yield RawArticle(
                url=entry.get("link", ""),
                title=entry.get("title", "Untitled"),
                summary=entry.get("summary"),
                source=source,
                author=entry.get("author"),
                published_at=published,
            )

    def fetch_full_content(self, url: str) -> str | None:
        """Fetch and extract full article content from a URL."""
        try:
            response = self.client.get(url)
            response.raise_for_status()
            content = trafilatura.extract(
                response.text,
                include_comments=False,
                include_tables=True,
                favor_precision=True,
            )
            return content
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            return None

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# Feed categories for easy setup
FEED_CATEGORIES = {
    "tech": [
        ("Hacker News", "https://hnrss.org/frontpage"),
        ("TechCrunch", "https://techcrunch.com/feed/"),
        ("Ars Technica", "https://feeds.arstechnica.com/arstechnica/index"),
        ("The Verge", "https://www.theverge.com/rss/index.xml"),
        ("Wired", "https://www.wired.com/feed/rss"),
        ("MIT Tech Review", "https://www.technologyreview.com/feed/"),
    ],
    "world": [
        ("Reuters World", "https://www.reutersagency.com/feed/?best-regions=world&post_type=best"),
        ("AP News", "https://rsshub.app/apnews/topics/world-news"),
        ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
        ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
        ("NPR World", "https://feeds.npr.org/1004/rss.xml"),
        ("The Guardian World", "https://www.theguardian.com/world/rss"),
    ],
    "us": [
        ("NPR News", "https://feeds.npr.org/1001/rss.xml"),
        ("Reuters US", "https://www.reutersagency.com/feed/?best-regions=usa&post_type=best"),
        ("AP US News", "https://rsshub.app/apnews/topics/us-news"),
        ("PBS NewsHour", "https://www.pbs.org/newshour/feeds/rss/headlines"),
        ("BBC US", "https://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml"),
    ],
    "politics": [
        ("Politico", "https://www.politico.com/rss/politicopicks.xml"),
        ("The Hill", "https://thehill.com/feed/"),
        ("Reuters Politics", "https://www.reutersagency.com/feed/?best-topics=political-general&post_type=best"),
        ("FiveThirtyEight", "https://fivethirtyeight.com/features/feed/"),
    ],
    "business": [
        ("Reuters Business", "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best"),
        ("CNBC", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
        ("Bloomberg", "https://feeds.bloomberg.com/markets/news.rss"),
        ("Financial Times", "https://www.ft.com/rss/home"),
        ("Economist", "https://www.economist.com/international/rss.xml"),
    ],
    "science": [
        ("Nature", "https://www.nature.com/nature.rss"),
        ("Science Daily", "https://www.sciencedaily.com/rss/all.xml"),
        ("Phys.org", "https://phys.org/rss-feed/"),
        ("New Scientist", "https://www.newscientist.com/feed/home/"),
        ("Scientific American", "https://rss.sciam.com/ScientificAmerican-Global"),
    ],
    "middle_east": [
        ("Al Jazeera Middle East", "https://www.aljazeera.com/xml/rss/all.xml"),
        ("BBC Middle East", "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml"),
        ("Reuters Middle East", "https://www.reutersagency.com/feed/?best-regions=middle-east&post_type=best"),
        ("The Guardian Middle East", "https://www.theguardian.com/world/middleeast/rss"),
    ],
    "europe": [
        ("BBC Europe", "https://feeds.bbci.co.uk/news/world/europe/rss.xml"),
        ("Reuters Europe", "https://www.reutersagency.com/feed/?best-regions=europe&post_type=best"),
        ("The Guardian Europe", "https://www.theguardian.com/world/europe-news/rss"),
        ("Euronews", "https://www.euronews.com/rss?level=theme&name=news"),
    ],
    "asia": [
        ("BBC Asia", "https://feeds.bbci.co.uk/news/world/asia/rss.xml"),
        ("Reuters Asia", "https://www.reutersagency.com/feed/?best-regions=asia&post_type=best"),
        ("The Guardian Asia", "https://www.theguardian.com/world/asia/rss"),
        ("South China Morning Post", "https://www.scmp.com/rss/91/feed"),
    ],
}

# Default feeds (tech only for backward compatibility)
DEFAULT_FEEDS = FEED_CATEGORIES["tech"]

# All feeds combined
ALL_FEEDS = []
for category_feeds in FEED_CATEGORIES.values():
    ALL_FEEDS.extend(category_feeds)
