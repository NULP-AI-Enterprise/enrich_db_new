"""
Tests for services/embeddings.py — build_embedding_text() function.

Exercises the text assembly logic: field inclusion, ordering, formatting, and
the 8000-character truncation ceiling.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from services.embeddings import build_embedding_text, _MAX_CHARS_PER_ITEM


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _item(**kwargs) -> dict:
    """Build an item dict with only the provided fields (plus mandatory title)."""
    base = {"title": "Default Title"}
    base.update(kwargs)
    return base


# ─── Title ────────────────────────────────────────────────────────────────────

class TestTitle:

    def test_title_is_always_first_in_output(self):
        item = _item(title="My Outlet", description="Some description.", category="News")
        result = build_embedding_text(item)
        assert result.startswith("My Outlet")

    def test_title_only_item_returns_just_the_title(self):
        item = {"title": "Solo Title"}
        result = build_embedding_text(item)
        assert result == "Solo Title"

    def test_empty_title_does_not_raise(self):
        item = {"title": ""}
        result = build_embedding_text(item)
        # Empty title — result may be empty string or just separators; must not raise
        assert isinstance(result, str)


# ─── Description ──────────────────────────────────────────────────────────────

class TestDescription:

    def test_description_is_included_when_present(self):
        description = "A leading technology publication."
        item = _item(description=description)
        result = build_embedding_text(item)
        assert description in result

    def test_description_is_omitted_when_none(self):
        item = _item(description=None)
        result = build_embedding_text(item)
        # No stray separators from missing description
        assert "None" not in result

    def test_description_is_omitted_when_missing(self):
        item = {"title": "T"}
        result = build_embedding_text(item)
        assert ". " not in result  # no separator from absent description


# ─── Category ─────────────────────────────────────────────────────────────────

class TestCategory:

    def test_category_is_formatted_as_category_colon_value(self):
        item = _item(category="Technology")
        result = build_embedding_text(item)
        assert "Category: Technology" in result

    def test_category_is_omitted_when_none(self):
        item = _item(category=None)
        result = build_embedding_text(item)
        assert "Category:" not in result

    def test_category_is_omitted_when_missing(self):
        item = {"title": "T"}
        result = build_embedding_text(item)
        assert "Category:" not in result


# ─── Tags ─────────────────────────────────────────────────────────────────────

class TestTags:

    def test_tags_are_included_as_comma_separated_list(self):
        item = _item(tags=["tech", "startup", "digital"])
        result = build_embedding_text(item)
        assert "Tags: tech, startup, digital" in result

    def test_tags_are_omitted_when_empty_list(self):
        item = _item(tags=[])
        result = build_embedding_text(item)
        assert "Tags:" not in result

    def test_tags_are_omitted_when_none(self):
        item = _item(tags=None)
        result = build_embedding_text(item)
        assert "Tags:" not in result

    def test_tags_are_omitted_when_missing(self):
        item = {"title": "T"}
        result = build_embedding_text(item)
        assert "Tags:" not in result


# ─── Audience interests ───────────────────────────────────────────────────────

class TestAudienceInterests:

    def test_audience_interests_are_included(self):
        item = _item(audience={"interests": ["AI", "startups", "technology"]})
        result = build_embedding_text(item)
        assert "Audience interests: AI, startups, technology" in result

    def test_audience_interests_omitted_when_empty(self):
        item = _item(audience={"interests": []})
        result = build_embedding_text(item)
        assert "Audience interests:" not in result

    def test_audience_interests_omitted_when_audience_is_none(self):
        item = _item(audience=None)
        result = build_embedding_text(item)
        assert "Audience interests:" not in result

    def test_audience_interests_omitted_when_audience_key_missing(self):
        item = {"title": "T"}
        result = build_embedding_text(item)
        assert "Audience interests:" not in result

    def test_audience_interests_omitted_when_interests_is_not_list(self):
        item = _item(audience={"interests": "tech enthusiasts"})
        result = build_embedding_text(item)
        assert "Audience interests:" not in result


# ─── Geographic coverage ──────────────────────────────────────────────────────

class TestGeographicCoverage:

    def test_geographic_coverage_is_included(self):
        item = _item(metrics={"geographic_coverage": ["Kyiv", "Lviv", "Kharkiv"]})
        result = build_embedding_text(item)
        assert "Geographic coverage: Kyiv, Lviv, Kharkiv" in result

    def test_geographic_coverage_omitted_when_empty_list(self):
        item = _item(metrics={"geographic_coverage": []})
        result = build_embedding_text(item)
        assert "Geographic coverage:" not in result

    def test_geographic_coverage_omitted_when_metrics_is_none(self):
        item = _item(metrics=None)
        result = build_embedding_text(item)
        assert "Geographic coverage:" not in result

    def test_geographic_coverage_omitted_when_not_a_list(self):
        item = _item(metrics={"geographic_coverage": "Ukraine"})
        result = build_embedding_text(item)
        assert "Geographic coverage:" not in result


# ─── Content topics ───────────────────────────────────────────────────────────

class TestContentTopics:

    def test_content_topics_are_included(self):
        item = _item(metrics={"content_topics": ["fintech", "blockchain", "crypto"]})
        result = build_embedding_text(item)
        assert "Content topics: fintech, blockchain, crypto" in result

    def test_content_topics_omitted_when_empty(self):
        item = _item(metrics={"content_topics": []})
        result = build_embedding_text(item)
        assert "Content topics:" not in result

    def test_content_topics_omitted_when_not_a_list(self):
        item = _item(metrics={"content_topics": "fintech"})
        result = build_embedding_text(item)
        assert "Content topics:" not in result


# ─── Truncation ───────────────────────────────────────────────────────────────

class TestTruncation:

    def test_output_never_exceeds_8000_chars(self):
        # Description alone is much longer than 8000 chars.
        long_description = "x" * 10_000
        item = _item(
            title="Big Outlet",
            description=long_description,
            category="News",
        )
        result = build_embedding_text(item)
        assert len(result) <= _MAX_CHARS_PER_ITEM

    def test_output_is_exactly_truncated_at_boundary(self):
        # Build a description that, together with title, will exceed 8000 chars.
        # The function joins with ". " so factor that in.
        long_description = "d" * 9000
        item = _item(title="T", description=long_description)
        result = build_embedding_text(item)
        assert len(result) == _MAX_CHARS_PER_ITEM

    def test_short_item_is_not_padded(self):
        item = _item(title="Short")
        result = build_embedding_text(item)
        assert len(result) < _MAX_CHARS_PER_ITEM
        assert len(result) == len("Short")


# ─── Combined / ordering ──────────────────────────────────────────────────────

class TestCombinedAndOrdering:

    def test_title_comes_before_description_before_category(self):
        item = _item(
            title="My Title",
            description="My description.",
            category="Business",
        )
        result = build_embedding_text(item)
        title_pos    = result.index("My Title")
        desc_pos     = result.index("My description.")
        category_pos = result.index("Category: Business")
        assert title_pos < desc_pos < category_pos

    def test_full_item_includes_all_expected_sections(self):
        item = {
            "title":       "Tech Weekly",
            "description": "A weekly digest for software engineers.",
            "category":    "Technology",
            "format_type": "Article",
            "language":    "en",
            "tags":        ["software", "engineering"],
            "audience": {
                "interests": ["open source", "cloud computing"],
            },
            "metrics": {
                "geographic_coverage": ["San Francisco", "New York"],
                "content_topics":      ["cloud", "devops"],
            },
        }
        result = build_embedding_text(item)

        assert "Tech Weekly" in result
        assert "A weekly digest for software engineers." in result
        assert "Category: Technology" in result
        assert "Tags: software, engineering" in result
        assert "Audience interests: open source, cloud computing" in result
        assert "Geographic coverage: San Francisco, New York" in result
        assert "Content topics: cloud, devops" in result

    def test_omitted_fields_leave_no_empty_separator_artefacts(self):
        """No leading/trailing '. ' or doubled '. . ' when optional fields are absent."""
        item = {"title": "Clean Outlet"}
        result = build_embedding_text(item)
        assert not result.startswith(". ")
        assert not result.endswith(". ")
        assert ". . " not in result
