"""
Tests for services/llm.py — _validate() and _mock() functions.

Import path trick: because the enrichment directory is not an installed package,
we insert its parent onto sys.path so that `from services.llm import ...` works
from any working directory.
"""

import sys
from pathlib import Path

# Add the enrichment root so `services.llm` resolves correctly.
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from services.llm import _validate, _mock, _VALID_CATEGORIES


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _minimal_valid_data(category: str = "News") -> dict:
    """Returns the smallest dict that passes _validate without defaulting fields."""
    return {
        "description": "A test outlet.",
        "category": category,
        "tags": ["tag1", "tag2"],
        "audience": {
            "age_range": "25-44",
            "interests": ["news", "politics"],
            "demographics": {
                "primary_language": "en",
                "geo": "unknown",
                "gender_split": "50M/50F",
            },
        },
        "metrics": {
            "geographic_coverage": ["Kyiv"],
            "publishing_frequency": "daily",
            "ad_formats_available": ["Banner"],
            "social_media_presence": "moderate",
        },
    }


# ─── _validate(): category field ──────────────────────────────────────────────

class TestValidateCategory:

    def test_valid_category_is_kept_unchanged(self):
        for category in _VALID_CATEGORIES:
            data = _minimal_valid_data(category)
            result = _validate(data, "Test Outlet")
            assert result["category"] == category, (
                f"Expected valid category '{category}' to be unchanged"
            )

    def test_invalid_category_is_replaced_with_news(self):
        data = _minimal_valid_data("InvalidCategory")
        result = _validate(data, "Test Outlet")
        assert result["category"] == "News"

    def test_none_category_is_replaced_with_news(self):
        data = _minimal_valid_data()
        data["category"] = None
        result = _validate(data, "Test Outlet")
        assert result["category"] == "News"

    def test_empty_string_category_is_replaced_with_news(self):
        data = _minimal_valid_data()
        data["category"] = ""
        result = _validate(data, "Test Outlet")
        assert result["category"] == "News"

    def test_all_valid_categories_are_accepted(self):
        """Sanity-check that _VALID_CATEGORIES itself is non-empty and meaningful."""
        assert len(_VALID_CATEGORIES) >= 10
        assert "News" in _VALID_CATEGORIES
        assert "Technology" in _VALID_CATEGORIES


# ─── _validate(): tags field ───────────────────────────────────────────────────

class TestValidateTags:

    def test_tags_longer_than_50_are_truncated_to_50(self):
        data = _minimal_valid_data()
        data["tags"] = [f"tag_{i}" for i in range(80)]
        result = _validate(data, "Test Outlet")
        # The real code truncates to [:50], not 10 — verified from source.
        assert len(result["tags"]) == 50

    def test_tags_shorter_than_50_are_kept_as_is(self):
        data = _minimal_valid_data()
        data["tags"] = ["alpha", "beta", "gamma"]
        result = _validate(data, "Test Outlet")
        assert result["tags"] == ["alpha", "beta", "gamma"]

    def test_missing_tags_field_results_in_empty_list(self):
        data = _minimal_valid_data()
        data.pop("tags", None)
        result = _validate(data, "Test Outlet")
        assert result["tags"] == []

    def test_none_tags_field_results_in_empty_list(self):
        data = _minimal_valid_data()
        data["tags"] = None
        result = _validate(data, "Test Outlet")
        assert result["tags"] == []

    def test_tags_are_coerced_to_strings(self):
        data = _minimal_valid_data()
        data["tags"] = [1, 2, 3]
        result = _validate(data, "Test Outlet")
        assert result["tags"] == ["1", "2", "3"]


# ─── _validate(): metrics.geographic_coverage field ───────────────────────────

class TestValidateGeographicCoverage:

    def test_list_geographic_coverage_is_kept(self):
        data = _minimal_valid_data()
        data["metrics"]["geographic_coverage"] = ["Kyiv", "Lviv"]
        result = _validate(data, "Test Outlet")
        assert result["metrics"]["geographic_coverage"] == ["Kyiv", "Lviv"]

    def test_string_geographic_coverage_is_replaced_with_empty_list(self):
        """A string is not a list — _validate must normalise it to []."""
        data = _minimal_valid_data()
        data["metrics"]["geographic_coverage"] = "Kyiv, Lviv"
        result = _validate(data, "Test Outlet")
        assert isinstance(result["metrics"]["geographic_coverage"], list)
        assert result["metrics"]["geographic_coverage"] == []

    def test_missing_geographic_coverage_defaults_to_empty_list(self):
        data = _minimal_valid_data()
        data["metrics"].pop("geographic_coverage", None)
        result = _validate(data, "Test Outlet")
        assert result["metrics"]["geographic_coverage"] == []

    def test_none_geographic_coverage_defaults_to_empty_list(self):
        data = _minimal_valid_data()
        data["metrics"]["geographic_coverage"] = None
        result = _validate(data, "Test Outlet")
        assert result["metrics"]["geographic_coverage"] == []


# ─── _validate(): audience.interests field ────────────────────────────────────

class TestValidateAudienceInterests:

    def test_existing_interests_list_is_preserved(self):
        data = _minimal_valid_data()
        data["audience"]["interests"] = ["tech", "startups"]
        result = _validate(data, "Test Outlet")
        assert result["audience"]["interests"] == ["tech", "startups"]

    def test_missing_audience_field_is_defaulted(self):
        """If 'audience' key is absent entirely, _validate inserts a default."""
        data = _minimal_valid_data()
        data.pop("audience")
        result = _validate(data, "Test Outlet")
        assert "audience" in result
        assert isinstance(result["audience"].get("interests"), list)


# ─── _validate(): stale pricing/reach fields stripped ─────────────────────────

class TestValidateStripLegacyFields:

    def test_reach_tier_is_removed(self):
        data = _minimal_valid_data()
        data["metrics"]["reach_tier"] = "national"
        result = _validate(data, "Test Outlet")
        assert "reach_tier" not in result["metrics"]

    def test_pricing_tier_is_removed(self):
        data = _minimal_valid_data()
        data["metrics"]["pricing_tier"] = "premium"
        result = _validate(data, "Test Outlet")
        assert "pricing_tier" not in result["metrics"]


# ─── _mock() ─────────────────────────────────────────────────────────────────

class TestMock:

    def test_mock_returns_all_required_top_level_keys(self):
        result = _mock("Tech Daily")
        required = {"description", "category", "tags", "audience", "metrics"}
        assert required.issubset(result.keys()), (
            f"Missing keys: {required - set(result.keys())}"
        )

    def test_mock_returns_valid_category(self):
        result = _mock("Sports Weekly")
        assert result["category"] in _VALID_CATEGORIES

    def test_mock_is_deterministic_for_same_title(self):
        """Same title must always produce the same category (seeded RNG)."""
        r1 = _mock("My Media Outlet")
        r2 = _mock("My Media Outlet")
        assert r1["category"] == r2["category"]

    def test_mock_may_produce_different_categories_for_different_titles(self):
        """With enough variation, different titles should not all share one category."""
        titles = [f"Outlet Number {i}" for i in range(30)]
        categories = {_mock(t)["category"] for t in titles}
        # If seeding works properly we should see at least 2 different categories
        assert len(categories) >= 2

    def test_mock_description_contains_title(self):
        title = "Digital Press Ukraine"
        result = _mock(title)
        assert title in result["description"]

    def test_mock_audience_has_required_structure(self):
        result = _mock("Test Outlet")
        audience = result["audience"]
        assert "age_range" in audience
        assert "interests" in audience
        assert isinstance(audience["interests"], list)

    def test_mock_metrics_has_required_structure(self):
        result = _mock("Test Outlet")
        metrics = result["metrics"]
        assert "geographic_coverage" in metrics
        assert "publishing_frequency" in metrics
        assert isinstance(metrics["geographic_coverage"], list)

    def test_mock_passes_validate_without_changes_to_category(self):
        """_mock output, when run through _validate, must keep the same valid category."""
        title = "Validated Mock Outlet"
        raw = _mock(title)
        original_category = raw["category"]
        validated = _validate(raw, title)
        assert validated["category"] == original_category
