"""OPML import/export for feed source portability."""

import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Tuple
from xml.dom import minidom

from .models import FeedSource


def export_opml(sources: List[FeedSource], title: str = "AI News Tracker Feeds") -> str:
    """
    Export feed sources to OPML format.

    Args:
        sources: List of FeedSource objects to export
        title: Title for the OPML document

    Returns:
        OPML XML string
    """
    opml = ET.Element("opml", version="2.0")

    head = ET.SubElement(opml, "head")
    ET.SubElement(head, "title").text = title
    ET.SubElement(head, "dateCreated").text = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")

    body = ET.SubElement(opml, "body")

    for source in sources:
        outline = ET.SubElement(
            body,
            "outline",
            type="rss",
            text=source.name,
            title=source.name,
            xmlUrl=source.url,
        )
        if source.feed_type:
            outline.set("feedType", source.feed_type)

    # Pretty print
    xml_str = ET.tostring(opml, encoding="unicode")
    dom = minidom.parseString(xml_str)
    return dom.toprettyxml(indent="  ")


def parse_opml(opml_content: str) -> List[Tuple[str, str, str]]:
    """
    Parse OPML content and extract feed sources.

    Args:
        opml_content: OPML XML string

    Returns:
        List of tuples (name, url, feed_type)
    """
    feeds = []

    try:
        root = ET.fromstring(opml_content)
    except ET.ParseError as e:
        raise ValueError(f"Invalid OPML format: {e}")

    # Find all outline elements with xmlUrl attribute
    for outline in root.iter("outline"):
        xml_url = outline.get("xmlUrl")
        if xml_url:
            name = outline.get("title") or outline.get("text") or xml_url
            feed_type = outline.get("feedType") or outline.get("type") or "rss"
            feeds.append((name, xml_url, feed_type))

    return feeds


def export_feeds_json(sources: List[FeedSource]) -> List[dict]:
    """
    Export feed sources to JSON-serializable format.

    Args:
        sources: List of FeedSource objects

    Returns:
        List of feed dictionaries
    """
    return [
        {
            "name": source.name,
            "url": source.url,
            "feed_type": source.feed_type,
            "is_active": source.is_active,
            "fetch_interval_minutes": source.fetch_interval_minutes,
        }
        for source in sources
    ]
