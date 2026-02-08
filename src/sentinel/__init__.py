"""Developer Sentinel - Jira to Claude Agent Orchestrator."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("developer-sentinel")
except PackageNotFoundError:
    # Package is not installed (e.g., running from source without pip install)
    __version__ = "0.0.0.dev0"

# Re-export core public API
from sentinel.app import main
from sentinel.main import Sentinel

# NOTE: Update this list when adding new exports to this module.
__all__ = [
    "__version__",
    "Sentinel",
    "main",
]
