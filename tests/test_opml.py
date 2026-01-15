"""Tests for OPML import/export functionality."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ai_news_tracker.opml import export_opml, parse_opml, export_feeds_json


class TestExportOpml:
    """Tests for OPML export functionality."""

    def test_export_empty_sources(self):
        """Test exporting empty list of sources."""
        result = export_opml([])

        assert '<?xml version="1.0" ?>' in result
        assert '<opml version="2.0">' in result
        assert "<title>AI News Tracker Feeds</title>" in result
        assert "<body/>" in result or "<body>" in result

    def test_export_single_source(self):
        """Test exporting a single source."""
        source = MagicMock()
        source.name = "Test Feed"
        source.url = "https://example.com/feed"
        source.feed_type = "rss"

        result = export_opml([source])

        assert 'text="Test Feed"' in result
        assert 'title="Test Feed"' in result
        assert 'xmlUrl="https://example.com/feed"' in result
        assert 'feedType="rss"' in result
        assert 'type="rss"' in result

    def test_export_multiple_sources(self):
        """Test exporting multiple sources."""
        source1 = MagicMock()
        source1.name = "Feed One"
        source1.url = "https://example.com/feed1"
        source1.feed_type = "rss"

        source2 = MagicMock()
        source2.name = "Feed Two"
        source2.url = "https://example.com/feed2"
        source2.feed_type = "atom"

        result = export_opml([source1, source2])

        assert 'text="Feed One"' in result
        assert 'text="Feed Two"' in result
        assert 'xmlUrl="https://example.com/feed1"' in result
        assert 'xmlUrl="https://example.com/feed2"' in result

    def test_export_custom_title(self):
        """Test exporting with custom title."""
        result = export_opml([], title="My Custom Feeds")

        assert "<title>My Custom Feeds</title>" in result

    def test_export_source_without_feed_type(self):
        """Test exporting source without feed type."""
        source = MagicMock()
        source.name = "Test Feed"
        source.url = "https://example.com/feed"
        source.feed_type = None

        result = export_opml([source])

        assert 'text="Test Feed"' in result
        assert 'xmlUrl="https://example.com/feed"' in result
        # feedType should not be present when None
        assert 'feedType=' not in result

    def test_export_includes_date(self):
        """Test that export includes dateCreated."""
        result = export_opml([])

        assert "<dateCreated>" in result


class TestParseOpml:
    """Tests for OPML parsing functionality."""

    def test_parse_empty_opml(self):
        """Test parsing OPML with no feeds."""
        opml = """<?xml version="1.0" encoding="UTF-8"?>
        <opml version="2.0">
            <head><title>Empty</title></head>
            <body></body>
        </opml>"""

        result = parse_opml(opml)

        assert result == []

    def test_parse_single_feed(self):
        """Test parsing OPML with single feed."""
        opml = """<?xml version="1.0" encoding="UTF-8"?>
        <opml version="2.0">
            <head><title>Feeds</title></head>
            <body>
                <outline type="rss" text="Test Feed" title="Test Feed" xmlUrl="https://example.com/feed"/>
            </body>
        </opml>"""

        result = parse_opml(opml)

        assert len(result) == 1
        name, url, feed_type = result[0]
        assert name == "Test Feed"
        assert url == "https://example.com/feed"
        assert feed_type == "rss"

    def test_parse_multiple_feeds(self):
        """Test parsing OPML with multiple feeds."""
        opml = """<?xml version="1.0" encoding="UTF-8"?>
        <opml version="2.0">
            <head><title>Feeds</title></head>
            <body>
                <outline type="rss" text="Feed One" xmlUrl="https://example.com/feed1"/>
                <outline type="atom" text="Feed Two" xmlUrl="https://example.com/feed2"/>
            </body>
        </opml>"""

        result = parse_opml(opml)

        assert len(result) == 2
        assert result[0][0] == "Feed One"
        assert result[1][0] == "Feed Two"

    def test_parse_uses_title_over_text(self):
        """Test that title attribute takes precedence over text."""
        opml = """<?xml version="1.0" encoding="UTF-8"?>
        <opml version="2.0">
            <body>
                <outline text="Short Name" title="Full Title" xmlUrl="https://example.com/feed"/>
            </body>
        </opml>"""

        result = parse_opml(opml)

        assert result[0][0] == "Full Title"

    def test_parse_falls_back_to_text(self):
        """Test falling back to text when title is missing."""
        opml = """<?xml version="1.0" encoding="UTF-8"?>
        <opml version="2.0">
            <body>
                <outline text="Text Name" xmlUrl="https://example.com/feed"/>
            </body>
        </opml>"""

        result = parse_opml(opml)

        assert result[0][0] == "Text Name"

    def test_parse_falls_back_to_url(self):
        """Test falling back to URL when name is missing."""
        opml = """<?xml version="1.0" encoding="UTF-8"?>
        <opml version="2.0">
            <body>
                <outline xmlUrl="https://example.com/feed"/>
            </body>
        </opml>"""

        result = parse_opml(opml)

        assert result[0][0] == "https://example.com/feed"

    def test_parse_skips_outlines_without_url(self):
        """Test that outlines without xmlUrl are skipped."""
        opml = """<?xml version="1.0" encoding="UTF-8"?>
        <opml version="2.0">
            <body>
                <outline text="Category"/>
                <outline text="Feed" xmlUrl="https://example.com/feed"/>
            </body>
        </opml>"""

        result = parse_opml(opml)

        assert len(result) == 1
        assert result[0][1] == "https://example.com/feed"

    def test_parse_nested_outlines(self):
        """Test parsing nested outlines (categories)."""
        opml = """<?xml version="1.0" encoding="UTF-8"?>
        <opml version="2.0">
            <body>
                <outline text="Tech">
                    <outline text="Feed 1" xmlUrl="https://example.com/feed1"/>
                    <outline text="Feed 2" xmlUrl="https://example.com/feed2"/>
                </outline>
                <outline text="News">
                    <outline text="Feed 3" xmlUrl="https://example.com/feed3"/>
                </outline>
            </body>
        </opml>"""

        result = parse_opml(opml)

        assert len(result) == 3
        urls = [r[1] for r in result]
        assert "https://example.com/feed1" in urls
        assert "https://example.com/feed2" in urls
        assert "https://example.com/feed3" in urls

    def test_parse_uses_feedtype_attribute(self):
        """Test that feedType attribute is extracted."""
        opml = """<?xml version="1.0" encoding="UTF-8"?>
        <opml version="2.0">
            <body>
                <outline text="Feed" xmlUrl="https://example.com/feed" feedType="atom"/>
            </body>
        </opml>"""

        result = parse_opml(opml)

        assert result[0][2] == "atom"

    def test_parse_defaults_feed_type_to_type_or_rss(self):
        """Test feed type defaults."""
        opml = """<?xml version="1.0" encoding="UTF-8"?>
        <opml version="2.0">
            <body>
                <outline text="Feed 1" xmlUrl="https://example.com/feed1" type="rss"/>
                <outline text="Feed 2" xmlUrl="https://example.com/feed2"/>
            </body>
        </opml>"""

        result = parse_opml(opml)

        assert result[0][2] == "rss"
        assert result[1][2] == "rss"

    def test_parse_invalid_xml(self):
        """Test that invalid XML raises ValueError."""
        opml = "not valid xml"

        with pytest.raises(ValueError, match="Invalid OPML format"):
            parse_opml(opml)

    def test_parse_malformed_xml(self):
        """Test that malformed XML raises ValueError."""
        opml = "<opml><unclosed>"

        with pytest.raises(ValueError, match="Invalid OPML format"):
            parse_opml(opml)


class TestExportFeedsJson:
    """Tests for JSON export functionality."""

    def test_export_empty_sources(self):
        """Test exporting empty list."""
        result = export_feeds_json([])

        assert result == []

    def test_export_single_source(self):
        """Test exporting a single source."""
        source = MagicMock()
        source.name = "Test Feed"
        source.url = "https://example.com/feed"
        source.feed_type = "rss"
        source.is_active = True
        source.fetch_interval_minutes = 30

        result = export_feeds_json([source])

        assert len(result) == 1
        assert result[0]["name"] == "Test Feed"
        assert result[0]["url"] == "https://example.com/feed"
        assert result[0]["feed_type"] == "rss"
        assert result[0]["is_active"] is True
        assert result[0]["fetch_interval_minutes"] == 30

    def test_export_multiple_sources(self):
        """Test exporting multiple sources."""
        source1 = MagicMock()
        source1.name = "Feed One"
        source1.url = "https://example.com/feed1"
        source1.feed_type = "rss"
        source1.is_active = True
        source1.fetch_interval_minutes = 30

        source2 = MagicMock()
        source2.name = "Feed Two"
        source2.url = "https://example.com/feed2"
        source2.feed_type = "atom"
        source2.is_active = False
        source2.fetch_interval_minutes = 60

        result = export_feeds_json([source1, source2])

        assert len(result) == 2
        assert result[0]["name"] == "Feed One"
        assert result[1]["name"] == "Feed Two"
        assert result[1]["is_active"] is False

    def test_export_preserves_all_fields(self):
        """Test that all expected fields are preserved."""
        source = MagicMock()
        source.name = "Test"
        source.url = "https://example.com/feed"
        source.feed_type = None
        source.is_active = True
        source.fetch_interval_minutes = 15

        result = export_feeds_json([source])

        expected_keys = {"name", "url", "feed_type", "is_active", "fetch_interval_minutes"}
        assert set(result[0].keys()) == expected_keys
