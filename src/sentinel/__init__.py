"""Developer Sentinel - Jira to Claude Agent Orchestrator."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("developer-sentinel")
except PackageNotFoundError:
    # Package is not installed (e.g., running from source without pip install)
    __version__ = "0.0.0.dev0"

# Re-export DI container for convenience
from sentinel.container import SentinelContainer, create_container, create_test_container

__all__ = [
    "__version__",
    "SentinelContainer",
    "create_container",
    "create_test_container",
]
