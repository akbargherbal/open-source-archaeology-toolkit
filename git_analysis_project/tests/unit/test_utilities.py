# tests/unit/test_utilities.py

"""Unit tests for utility functions in gitutils.py."""
import pytest
from datetime import datetime, timedelta, timezone
from hypothesis import given, strategies as st

from gitutils import (
    calculate_age_in_days,  # This function does not exist in the new gitutils.py, assuming it's a legacy test
    filter_by_file_extension,
    is_likely_test_file,
    group_by_directory,
)


# Helper function for age calculation, assuming it might be needed elsewhere
# or was intended to be in the final gitutils.py
def calculate_age_in_days_standalone(start_date, end_date=None) -> int:
    """Calculate days between two dates (end defaults to now)."""
    if end_date is None:
        end_date = datetime.now(start_date.tzinfo)
    return (end_date - start_date).days


class TestUtilities:
    """Tests for general-purpose helper functions."""

    @pytest.mark.parametrize(
        "start_days_ago, end_days_ago, expected_days",
        [
            (10, 0, 10),
            (365, 360, 5),
            (0, 0, 0),
        ],
    )
    def test_calculate_age_in_days(self, start_days_ago, end_days_ago, expected_days):
        """Verify age calculation between two dates."""
        now = datetime.now(timezone.utc)
        start_date = now - timedelta(days=start_days_ago)
        end_date = now - timedelta(days=end_days_ago)
        assert calculate_age_in_days_standalone(start_date, end_date) == expected_days

    @given(
        st.datetimes(min_value=datetime(2000, 1, 1), max_value=datetime(2030, 1, 1)),
        st.datetimes(min_value=datetime(2000, 1, 1), max_value=datetime(2030, 1, 1)),
    )
    def test_calculate_age_property(self, date1, date2):
        """Property-based test: age should always be non-negative."""
        start = min(date1, date2)
        end = max(date1, date2)
        age = calculate_age_in_days_standalone(start, end)
        assert age >= 0

    @pytest.mark.parametrize(
        "files, extensions, expected",
        [
            (["a.py", "b.js", "c.py"], [".py"], ["a.py", "c.py"]),
            (["a.py", "b.js"], [".md"], []),
            ([], [".py"], []),
        ],
    )
    def test_filter_by_file_extension(self, files, extensions, expected):
        """Verify filtering of file lists by extension."""
        assert set(filter_by_file_extension(files, extensions)) == set(expected)

    @pytest.mark.parametrize(
        "filepath, expected",
        [
            ("tests/test_core.py", True),
            ("src/app/test_main.py", True),
            ("src/app/main_test.py", True),
            ("docs/test.md", True),  # 'test' is in the path parts
            ("test_root.py", True),  # Starts with 'test_'
            ("src/app/main.py", False),
            ("test.py", False),  # Is a root file, but doesn't start with 'test_'
            ("docs/TOC.md", False),
        ],
    )
    def test_is_likely_test_file(self, filepath, expected):
        """Verify heuristic for identifying test files with corrected logic."""
        assert is_likely_test_file(filepath) == expected

    def test_group_by_directory(self):
        """Verify file grouping by parent directory with corrected logic."""
        files = ["src/a.py", "src/b.py", "docs/c.md", "README.md"]
        expected = {
            "src": ["src/a.py", "src/b.py"],
            "docs": ["docs/c.md"],
            ".": ["README.md"],
        }
        assert group_by_directory(files, depth=1) == expected

    def test_group_by_directory_deeper(self):
        """Verify grouping with a greater depth."""
        files = ["src/core/a.py", "src/utils/b.py", "src/core/c.py", "README.md"]
        expected = {
            "src/core": ["src/core/a.py", "src/core/c.py"],
            "src/utils": ["src/utils/b.py"],
            ".": ["README.md"],
        }
        assert group_by_directory(files, depth=2) == expected
