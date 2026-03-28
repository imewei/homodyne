"""Completion parity tests.

Verifies that shell completion scripts are in sync with actual CLI arguments
defined in argparse parsers. Prevents silent drift when CLI flags are added
or removed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


def _extract_argparse_flags(parser) -> set[str]:
    """Extract all --long-flag names from an argparse parser."""
    flags = set()
    for action in parser._actions:
        for opt in action.option_strings:
            if opt.startswith("--"):
                flags.add(opt)
    return flags


def _extract_completion_flags_for_command(
    completion_path: Path, function_name: str
) -> set[str]:
    """Extract --flags from a specific bash completion function.

    Parses the completion script to find the function body and extracts
    all ``--flag-name`` tokens within it.

    Parameters
    ----------
    completion_path : Path
        Path to completion.sh
    function_name : str
        Bash function name (e.g., ``_homodyne``, ``_homodyne_config``)

    Returns
    -------
    set[str]
        Set of --flag names found in the function body.
    """
    content = completion_path.read_text(encoding="utf-8")

    # Find the function body: starts with "function_name() {" and ends at
    # the next "^}" at column 0.
    pattern = re.compile(
        rf"^{re.escape(function_name)}\(\)\s*\{{(.*?)^\}}",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(content)
    if match is None:
        pytest.fail(
            f"Completion function '{function_name}' not found in {completion_path}"
        )

    body = match.group(1)

    flags: set[str] = set()
    for flag_match in re.finditer(r"--[a-zA-Z][a-zA-Z0-9_-]*", body):
        flags.add(flag_match.group())
    return flags


def _extract_all_completion_flags(completion_path: Path) -> set[str]:
    """Extract every --flag token from the entire completion script."""
    content = completion_path.read_text(encoding="utf-8")
    flags: set[str] = set()
    for match in re.finditer(r"--[a-zA-Z][a-zA-Z0-9_-]*", content):
        flags.add(match.group())
    return flags


@pytest.fixture
def completion_script_path() -> Path:
    """Path to the shell completion script."""
    path = (
        Path(__file__).parent.parent.parent
        / "homodyne"
        / "runtime"
        / "shell"
        / "completion.sh"
    )
    if not path.exists():
        pytest.skip(f"completion.sh not found at {path}")
    return path


class TestHomodyneCompletionParity:
    """Verify homodyne CLI flags are present in shell completion."""

    # Flags auto-added by argparse that need not appear in completion
    SKIP_FLAGS = {"--help", "--version"}

    def test_main_cli_flags_in_completion(
        self, completion_script_path: Path
    ) -> None:
        """All homodyne main CLI flags should appear in _homodyne()."""
        from homodyne.cli.args_parser import create_parser

        parser = create_parser()
        argparse_flags = _extract_argparse_flags(parser) - self.SKIP_FLAGS
        completion_flags = _extract_completion_flags_for_command(
            completion_script_path, "_homodyne"
        )

        missing = argparse_flags - completion_flags
        if missing:
            pytest.fail(
                f"CLI flags missing from completion.sh _homodyne(): "
                f"{sorted(missing)}. "
                "Update homodyne/runtime/shell/completion.sh to include "
                "these flags."
            )

    def test_config_cli_flags_in_completion(
        self, completion_script_path: Path
    ) -> None:
        """All homodyne-config CLI flags should appear in _homodyne_config()."""
        from homodyne.cli.config_generator import create_parser

        parser = create_parser()
        argparse_flags = _extract_argparse_flags(parser) - self.SKIP_FLAGS
        completion_flags = _extract_completion_flags_for_command(
            completion_script_path, "_homodyne_config"
        )

        missing = argparse_flags - completion_flags
        if missing:
            pytest.fail(
                f"homodyne-config flags missing from completion.sh "
                f"_homodyne_config(): {sorted(missing)}. "
                "Update homodyne/runtime/shell/completion.sh to include "
                "these flags."
            )

    def test_completion_no_stale_main_flags(
        self, completion_script_path: Path
    ) -> None:
        """Flags in _homodyne() should correspond to actual argparse flags."""
        from homodyne.cli.args_parser import create_parser

        argparse_flags = _extract_argparse_flags(create_parser())
        completion_flags = _extract_completion_flags_for_command(
            completion_script_path, "_homodyne"
        )

        # Flags that legitimately appear in completion but not in argparse
        # (e.g., referenced in case-statement value lists, not as completable
        # options themselves).
        known_extras: set[str] = set()

        stale = completion_flags - argparse_flags - known_extras
        if stale:
            pytest.fail(
                f"Possibly stale flags in completion.sh _homodyne(): "
                f"{sorted(stale)}. "
                "These flags are in the completion script but not in "
                "the argparse parser. Remove them or add them to argparse."
            )

    def test_completion_no_stale_config_flags(
        self, completion_script_path: Path
    ) -> None:
        """Flags in _homodyne_config() should correspond to actual argparse flags."""
        from homodyne.cli.config_generator import create_parser

        argparse_flags = _extract_argparse_flags(create_parser())
        completion_flags = _extract_completion_flags_for_command(
            completion_script_path, "_homodyne_config"
        )

        known_extras: set[str] = set()

        stale = completion_flags - argparse_flags - known_extras
        if stale:
            pytest.fail(
                f"Possibly stale flags in completion.sh _homodyne_config(): "
                f"{sorted(stale)}. "
                "These flags are in the completion script but not in "
                "the argparse parser. Remove them or add them to argparse."
            )

    def test_method_choices_in_sync(
        self, completion_script_path: Path
    ) -> None:
        """The --method choices in completion should match argparse choices."""
        from homodyne.cli.args_parser import create_parser

        parser = create_parser()

        # Find the --method action and get its choices
        method_choices: set[str] | None = None
        for action in parser._actions:
            if "--method" in action.option_strings:
                method_choices = set(action.choices)
                break

        assert method_choices is not None, "--method action not found in parser"

        # Extract method choices from completion script
        content = completion_script_path.read_text(encoding="utf-8")

        # Look for: local methods="nlsq cmc both"
        match = re.search(r'local methods="([^"]*)"', content)
        assert match is not None, (
            'Could not find local methods="..." in completion.sh'
        )

        completion_methods = set(match.group(1).split())

        missing = method_choices - completion_methods
        extra = completion_methods - method_choices

        errors = []
        if missing:
            errors.append(
                f"Method choices missing from completion.sh: {sorted(missing)}"
            )
        if extra:
            errors.append(
                f"Stale method choices in completion.sh: {sorted(extra)}"
            )
        if errors:
            pytest.fail(". ".join(errors))


class TestCompletionAliases:
    """Verify shell aliases are defined in completion script."""

    EXPECTED_ALIASES = [
        "hm",
        "hconfig",
        "hm-nlsq",
        "hm-cmc",
        "hc-stat",
        "hc-flow",
        "hexp",
        "hsim",
        "hxla",
        "hsetup",
        "hclean",
    ]

    def test_aliases_present(self, completion_script_path: Path) -> None:
        """All expected aliases should be defined in completion.sh."""
        content = completion_script_path.read_text(encoding="utf-8")

        missing_aliases = []
        for alias in self.EXPECTED_ALIASES:
            # Look for alias definition: alias name=
            pattern = rf"^alias {re.escape(alias)}="
            if not re.search(pattern, content, re.MULTILINE):
                missing_aliases.append(alias)

        if missing_aliases:
            pytest.fail(
                f"Expected aliases missing from completion.sh: "
                f"{missing_aliases}"
            )

    def test_alias_completions_registered(
        self, completion_script_path: Path
    ) -> None:
        """All aliases should have a 'complete -F' registration."""
        content = completion_script_path.read_text(encoding="utf-8")

        missing_completions = []
        for alias in self.EXPECTED_ALIASES:
            pattern = rf"^complete -F \S+ {re.escape(alias)}$"
            if not re.search(pattern, content, re.MULTILINE):
                missing_completions.append(alias)

        if missing_completions:
            pytest.fail(
                f"Aliases missing 'complete -F' registration: "
                f"{missing_completions}"
            )
