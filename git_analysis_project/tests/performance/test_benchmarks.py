# tests/performance/test_benchmarks.py

import pytest

# FIX: Import the new cache-based function name
from gitutils import calculate_file_churn_cached
from git_cache import CommitDataCache


# This mark ensures these tests only run when explicitly requested
@pytest.mark.performance
class TestPerformance:
    """
    Performance benchmarks to validate the impact of refactoring.
    Requires the `pytest-benchmark` plugin.
    Run with: pytest --benchmark-only
    """

    def test_churn_performance(self, benchmark, simple_repo):
        """Benchmark the churn calculation on the new cache-based system."""

        # The setup phase (creating the cache) is done once.
        cache = CommitDataCache(simple_repo)

        # FIX: Benchmark the new cache-based function
        result = benchmark(calculate_file_churn_cached, cache)

        # We can also assert correctness within the benchmark
        assert result is not None
        assert not result.empty
