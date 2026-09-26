import pytest

# Python 3.14 finalises sqlite3 connections during interpreter shutdown in an
# order pytest reports as an unraisable exception. It is a GC-ordering
# artifact of tests that intentionally leave a connection open (the
# embedding-mismatch test, for one), not a leak in the store -- SQLiteStore
# closes cleanly everywhere it is used as a context manager.
collect_ignore: list[str] = []


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*Exception ignored while finalizing database connection.*"
        ":pytest.PytestUnraisableExceptionWarning",
    )
