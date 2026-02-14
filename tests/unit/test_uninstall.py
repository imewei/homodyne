"""Tests for uninstall_scripts._remove_homodyne_blocks state machine.

Validates:
- Single block removal (bash fi, fish end)
- Multiple blocks in one file
- Nested if/fi handling within blocks
- No-block passthrough (content unchanged)
- Partial block (missing end marker)
"""

from __future__ import annotations

from homodyne.uninstall_scripts import _remove_homodyne_blocks


class TestRemoveHomodyneBlocks:
    """Tests for the state-machine block remover."""

    def test_single_bash_block_removed(self):
        """Single completion block with fi terminator is removed."""
        content = (
            "# existing preamble\n"
            "export PATH\n"
            "# Homodyne shell completion (auto-added by homodyne-post-install)\n"
            "if [ -f some_script ]; then\n"
            "    source some_script\n"
            "fi\n"
            "# trailing content\n"
        )
        result = _remove_homodyne_blocks(content, "fi")
        assert "Homodyne" not in result
        assert "existing preamble" in result
        assert "trailing content" in result

    def test_single_fish_block_removed(self):
        """Single fish block with end terminator is removed."""
        content = (
            "set -x PATH\n"
            "# Homodyne XLA configuration (auto-added by homodyne-post-install)\n"
            "if test -f some_script\n"
            "    source some_script\n"
            "end\n"
            "set -x FOO\n"
        )
        result = _remove_homodyne_blocks(content, "end")
        assert "Homodyne" not in result
        assert "set -x PATH" in result
        assert "set -x FOO" in result

    def test_multiple_blocks_removed(self):
        """Both completion and XLA blocks removed from one file."""
        content = (
            "preamble\n"
            "# Homodyne shell completion (auto-added by homodyne-post-install)\n"
            "if [ -f a ]; then\n"
            "    source a\n"
            "fi\n"
            "middle\n"
            "# Homodyne XLA configuration (auto-added by homodyne-post-install)\n"
            "if [ -f b ]; then\n"
            "    source b\n"
            "fi\n"
            "end\n"
        )
        result = _remove_homodyne_blocks(content, "fi")
        assert "Homodyne" not in result
        assert "preamble" in result
        assert "middle" in result
        assert "end" in result  # literal "end" line preserved (not a fi marker)

    def test_nested_if_fi_handled(self):
        """Nested if/fi within a block doesn't prematurely end skip."""
        content = (
            "before\n"
            "# Homodyne shell completion (auto-added by homodyne-post-install)\n"
            "if [ -f script ]; then\n"
            '    if [ -z "$VAR" ]; then\n'
            "        echo nested\n"
            "    fi\n"
            "    source script\n"
            "fi\n"
            "after\n"
        )
        result = _remove_homodyne_blocks(content, "fi")
        assert "Homodyne" not in result
        assert "nested" not in result
        assert "before" in result
        assert "after" in result

    def test_no_block_passthrough(self):
        """Content without homodyne blocks passes through unchanged."""
        content = (
            "# Some other comment\n"
            "export PATH=/usr/bin:$PATH\n"
            "if [ -f ~/.bashrc ]; then\n"
            "    source ~/.bashrc\n"
            "fi\n"
        )
        result = _remove_homodyne_blocks(content, "fi")
        assert result == content

    def test_partial_block_missing_end_marker(self):
        """Block with missing end marker: header + body lines are stripped to EOF."""
        content = (
            "before\n"
            "# Homodyne shell completion (auto-added by homodyne-post-install)\n"
            "if [ -f script ]; then\n"
            "    source script\n"
            # No "fi" terminator — state machine stays in skip mode
        )
        result = _remove_homodyne_blocks(content, "fi")
        # All lines after the header should be consumed by skip mode
        assert "Homodyne" not in result
        assert "source script" not in result
        assert "before" in result
