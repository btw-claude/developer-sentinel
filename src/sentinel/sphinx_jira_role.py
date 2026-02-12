"""Custom Sphinx role for linking Jira issue references in documentation.

Provides a ``:jira:`` role that converts Jira issue keys (e.g., ``DS-123``)
into clickable hyperlinks pointing to the configured Jira instance.

This extension was created as a follow-up from DS-1009, which identified the
need for a documentation-tooling mechanism to auto-link Jira references
instead of embedding raw URLs or plain-text issue keys in docstrings.
The evaluation (DS-1016) concluded that a lightweight custom Sphinx role —
rather than intersphinx (which maps *Sphinx* inventories, not Jira issue
trackers) — is the appropriate approach for this project.

Configuration:
    Add the extension to your Sphinx ``conf.py``::

        extensions = ["sentinel.sphinx_jira_role"]

    Set the base URL for your Jira instance::

        jira_base_url = "https://your-domain.atlassian.net"

    Then use the ``:jira:`` role in reStructuredText or docstrings::

        See :jira:`DS-1016` for details.

    This renders as a hyperlink to
    ``https://your-domain.atlassian.net/browse/DS-1016``.

Design decisions:
    * **Custom role over intersphinx**: Intersphinx maps between Sphinx
      documentation inventories (``objects.inv``).  Jira is not a Sphinx
      project and does not publish an inventory, so intersphinx cannot
      resolve Jira issue keys.  A custom role provides a purpose-built,
      zero-maintenance solution.
    * **Configurable base URL**: The Jira instance URL is set via
      ``jira_base_url`` in ``conf.py``, avoiding hard-coded URLs and
      enabling reuse across projects or Jira instances.
    * **Issue key validation**: The role validates that the issue key
      matches the standard Jira format (``PROJECT-123``) to catch typos
      early during documentation builds.

See Also:
    * DS-1009 — Code review that identified the need for this extension.
    * DS-999, DS-986 — Related issues referenced in the original inline
      note that was removed in DS-1009.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

from docutils import nodes
from docutils.parsers.rst import roles
from docutils.parsers.rst.states import Inliner

logger = logging.getLogger(__name__)

# Jira issue key pattern: 1-10 uppercase alpha letters followed by a hyphen and 1+ digits.
# Real Jira project keys are all-alpha uppercase (e.g., DS, PROJ, MYPROJECT).
# Tightened from [A-Z][A-Z0-9]{0,9} to [A-Z]{1,10} per DS-1019 code review feedback
# to reject alphanumeric project keys (e.g., A1B2C3-1) that are not valid Jira keys.
_JIRA_ISSUE_KEY_PATTERN = re.compile(r"^[A-Z]{1,10}-\d+$")

# Default Jira base URL (overridden via conf.py ``jira_base_url``).
_DEFAULT_JIRA_BASE_URL = "https://jira.example.com"


def _validate_issue_key(issue_key: str) -> bool:
    """Validate that a string matches the Jira issue key format.

    Args:
        issue_key: The string to validate (e.g., ``"DS-1016"``).

    Returns:
        ``True`` if the string is a valid Jira issue key, ``False`` otherwise.
    """
    return bool(_JIRA_ISSUE_KEY_PATTERN.match(issue_key))


def _jira_role_impl(
    name: str,
    rawtext: str,
    text: str,
    lineno: int,
    inliner: Inliner,
    options: Mapping[str, Any] | None = None,
    content: Sequence[str] | None = None,
) -> tuple[list[nodes.Node], list[nodes.system_message]]:
    """Create a hyperlink node for a Jira issue reference.

    This is the implementation function called by the adapter registered
    with ``register_local_role``.  The adapter forwards calls with the
    required non-optional signature; this function accepts ``None``
    defaults for ergonomic direct use in tests.

    Args:
        name: The role name used in the source (always ``"jira"``).
        rawtext: The entire role markup including the role name and
            backticks (e.g., ``:jira:`DS-1016```).
        text: The interpreted text content (the issue key, e.g.,
            ``"DS-1016"``).
        lineno: The line number where the role appears in the source.
        inliner: The ``Inliner`` object that called this function.
            Provides access to the document's reporter for warnings.
        options: Optional dict of directive options (unused).
        content: Optional list of content strings (unused).

    Returns:
        A two-element tuple:
        - A list containing one ``reference`` node (the hyperlink), or
          an empty list if validation fails.
        - A list of system messages (warnings for invalid keys), or an
          empty list on success.
    """
    issue_key = text.strip()

    # Validate the issue key format.
    if not _validate_issue_key(issue_key):
        msg = inliner.reporter.warning(
            f"Invalid Jira issue key format: {issue_key!r}. "
            f"Expected format: PROJECT-123 (e.g., DS-1016).",
            line=lineno,
        )
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]

    # Retrieve the Jira base URL from the Sphinx configuration.
    env = inliner.document.settings.env
    base_url = getattr(env.config, "jira_base_url", _DEFAULT_JIRA_BASE_URL)

    # Build the full URL: {base_url}/browse/{issue_key}
    url = f"{base_url.rstrip('/')}/browse/{issue_key}"

    # Create a reference node (hyperlink).
    node = nodes.reference(rawtext, issue_key, refuri=url)
    return [node], []


def jira_role(
    name: str,
    rawtext: str,
    text: str,
    lineno: int,
    inliner: Inliner,
    options: Mapping[str, Any],
    content: Sequence[str],
) -> tuple[Sequence[nodes.Node], Sequence[nodes.system_message]]:
    """Adapter for :func:`_jira_role_impl` matching the docutils role signature.

    ``register_local_role`` requires a callable with non-optional *options*
    and *content* parameters.  This thin adapter satisfies that contract and
    delegates to :func:`_jira_role_impl`.

    Args:
        name: The role name used in the source (always ``"jira"``).
        rawtext: The entire role markup including the role name and
            backticks (e.g., ``:jira:`DS-1016```).
        text: The interpreted text content (the issue key, e.g.,
            ``"DS-1016"``).
        lineno: The line number where the role appears in the source.
        inliner: The ``Inliner`` object that called this function.
        options: Dict of directive options (unused, forwarded).
        content: List of content strings (unused, forwarded).

    Returns:
        A two-element tuple of (node-list, message-list); see
        :func:`_jira_role_impl` for details.
    """
    return _jira_role_impl(name, rawtext, text, lineno, inliner, options, content)


def _check_jira_base_url(app: Any, config: Any) -> None:
    """Emit a Sphinx warning if jira_base_url is still the default placeholder.

    This ``config-inited`` callback helps catch forgotten configuration by
    warning during ``sphinx-build`` when ``jira_base_url`` has not been
    overridden from its ``example.com`` default.

    Added in DS-1019 per code review feedback on DS-1016 / PR #1026.

    Args:
        app: The Sphinx application object.
        config: The Sphinx configuration object.
    """
    base_url: str = getattr(config, "jira_base_url", _DEFAULT_JIRA_BASE_URL)
    if "example.com" in base_url:
        logger.warning(
            "jira_base_url is still set to the default placeholder (%s). "
            "Set jira_base_url in conf.py to your Jira instance URL.",
            base_url,
        )


def setup(app: Any) -> dict[str, Any]:
    """Register the ``:jira:`` role and its configuration with Sphinx.

    Called by Sphinx when the extension is loaded via ``conf.py``'s
    ``extensions`` list.

    Args:
        app: The Sphinx application object.

    Returns:
        Extension metadata dict with ``version`` and
        ``parallel_read_safe`` / ``parallel_write_safe`` flags.
    """
    # Register the configuration value for the Jira base URL.
    app.add_config_value("jira_base_url", _DEFAULT_JIRA_BASE_URL, "env")

    # Warn at build time if jira_base_url is still the default placeholder (DS-1019).
    app.connect("config-inited", _check_jira_base_url)

    # Register the :jira: role.
    # The types-docutils stubs declare an overly narrow return type
    # (Sequence[reference]) for the role callable, but docutils itself
    # accepts any Sequence[Node].  Suppress the false positive.
    roles.register_local_role("jira", jira_role)  # type: ignore[arg-type]

    return {
        "version": "0.1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
