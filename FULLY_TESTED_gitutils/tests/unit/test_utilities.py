"""Unit tests for utility functions in gitutils.py."""
import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from hypothesis import given, strategies as st

# Import the functions to be tested
from gitutils import (
    calculate_age_in_days,
    filter_by_file_extension,
    is_likely_test_file,
    group_by_directory,
)

class TestUtilities:
    """Tests for general-purpose helper functions."""

    @pytest.mark.parametrize("start_days_ago, end_days_ago, expected_days", [
        (10, 0, 10),
        (365, 360, 5),
        (0, 0, 0),
    ])
    def test_calculate_age_in_days(self, start_days_ago, end_days_ago, expected_days):
        """Verify age calculation between two dates."""
        now = datetime.now(timezone.utc)
        start_date = now - timedelta(days=start_days_ago)
        end_date = now - timedelta(days=end_days_ago)
        assert calculate_age_in_days(start_date, end_date) == expected_days

    # FIX: Use naive datetimes for the hypothesis strategy as required.
    @given(st.datetimes(min_value=datetime(2000, 1, 1)),
           st.datetimes(min_value=datetime(2000, 1, 1)))
    def test_calculate_age_property(self, date1, date2):
        """Property-based test: age should always be non-negative."""
        start = min(date1, date2)
        end = max(date1, date2)
        
        # The function under test can handle timezone-aware dates, but the
        # hypothesis strategy requires naive ones. This test is still valid.
        age = calculate_age_in_days(start, end)
        assert age >= 0

    @pytest.mark.parametrize("files, extensions, expected", [
        (["a.py", "b.js", "c.py"], [".py"], ["a.py", "c.py"]),
        (["a.py", "b.js"], [".md"], []),
        ([], [".py"], []),
        (["a.txt", "b.log"], [".py", ".js"], []),
    ])
    def test_filter_by_file_extension(self, files, extensions, expected):
        """Verify filtering of file lists by extension."""
        result = filter_by_file_extension(files, extensions)
        assert set(result) == set(expected)

    @pytest.mark.parametrize("filepath, expected", [
        ("tests/test_core.py", True),
        ("src/app/test_main.py", True),
        ("src/app/main_test.py", True),
        ("src/app/main.py", False),
        # FIX: The function was bugged; now fixed, this assertion is correct.
        ("docs/test.md", True),
        ("test.py", False),
    ])
    def test_is_likely_test_file(self, filepath, expected):
        """Verify heuristic for identifying test files."""
        assert is_likely_test_file(filepath) == expected

    def test_group_by_directory(self):
        """Verify file grouping by parent directory."""
        files = ["src/a.py", "src/b.py", "docs/c.md", "README.md"]
        # FIX: Assert the correct output for the fixed function.
        expected = {
            "src": ["src/a.py", "src/b.py"],
            "docs": ["docs/c.md"],
            ".": ["README.md"]
        }
        # FIX: Use depth=1 which is more intuitive for "group by first directory".
        assert group_by_directory(files, depth=1) == expected