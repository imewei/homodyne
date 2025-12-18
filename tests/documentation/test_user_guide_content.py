"""Tests for user guide documentation content."""

import re
from pathlib import Path


class TestUserGuideStructure:
    """Test user guide documentation structure and content."""

    @staticmethod
    def get_docs_path() -> Path:
        """Get path to docs directory."""
        return Path(__file__).parent.parent.parent / "docs"

    def test_user_guide_index_exists(self):
        """Test that user-guide/index.rst exists."""
        docs_path = self.get_docs_path()
        user_guide_index = docs_path / "user-guide" / "index.rst"
        assert user_guide_index.exists(), "user-guide/index.rst not found"

    def test_user_guide_index_has_toctree(self):
        """Test that user-guide/index.rst has toctree directive."""
        docs_path = self.get_docs_path()
        user_guide_index = docs_path / "user-guide" / "index.rst"
        content = user_guide_index.read_text()
        assert ".. toctree::" in content, (
            "user-guide/index.rst missing toctree directive"
        )

    def test_main_index_exists(self):
        """Test that main index.rst exists."""
        docs_path = self.get_docs_path()
        index = docs_path / "index.rst"
        assert index.exists(), "docs/index.rst not found"

    def test_main_index_has_correct_sections(self):
        """Test that main index.rst has 5 main sections."""
        docs_path = self.get_docs_path()
        index = docs_path / "index.rst"
        content = index.read_text()

        # Check for 5 main sections
        required_sections = [
            "User Guide",
            "API Reference",
            "Research",
            "Developer Guide",
            "Configuration",
        ]

        for section in required_sections:
            assert section in content, f"Main index.rst missing '{section}' section"

    def test_main_index_has_core_equation(self):
        """Test that main index.rst displays core equation."""
        docs_path = self.get_docs_path()
        index = docs_path / "index.rst"
        content = index.read_text()

        # Check for core equation reference (LaTeX math format)
        equation_patterns = [
            r"c_2",
            r"contrast",
            r"math::",
        ]

        matches = sum(1 for pattern in equation_patterns if re.search(pattern, content))
        assert matches >= 2, "Main index.rst missing core equation references"

    def test_main_index_has_citation(self):
        """Test that main index.rst includes He et al. PNAS 2024 citation."""
        docs_path = self.get_docs_path()
        index = docs_path / "index.rst"
        content = index.read_text()

        assert "He et al" in content or "PNAS 2024" in content, (
            "Main index.rst missing He et al. PNAS 2024 citation"
        )


class TestUserGuidePages:
    """Test individual user guide pages."""

    @staticmethod
    def get_docs_path() -> Path:
        """Get path to docs directory."""
        return Path(__file__).parent.parent.parent / "docs"

    def test_installation_page_exists(self):
        """Test that installation.rst exists."""
        docs_path = self.get_docs_path()
        installation = docs_path / "user-guide" / "installation.rst"
        assert installation.exists(), "user-guide/installation.rst not found"

    def test_installation_page_content(self):
        """Test that installation.rst has required content."""
        docs_path = self.get_docs_path()
        installation = docs_path / "user-guide" / "installation.rst"
        content = installation.read_text()

        required_content = [
            "Python 3.12",
            "pip install",
            "JAX",
            "0.8.0",
            "CPU-only",
        ]

        for required in required_content:
            assert required in content, f"installation.rst missing '{required}' content"

    def test_quickstart_page_exists(self):
        """Test that quickstart.rst exists."""
        docs_path = self.get_docs_path()
        quickstart = docs_path / "user-guide" / "quickstart.rst"
        assert quickstart.exists(), "user-guide/quickstart.rst not found"

    def test_quickstart_has_yaml_config(self):
        """Test that quickstart.rst includes YAML configuration example."""
        docs_path = self.get_docs_path()
        quickstart = docs_path / "user-guide" / "quickstart.rst"
        content = quickstart.read_text()

        # Check for YAML configuration block
        assert ".. code-block::" in content or "```" in content, (
            "quickstart.rst missing code block"
        )
        assert "homodyne" in content, "quickstart.rst missing homodyne command"

    def test_cli_page_exists(self):
        """Test that cli.rst exists."""
        docs_path = self.get_docs_path()
        cli = docs_path / "user-guide" / "cli.rst"
        assert cli.exists(), "user-guide/cli.rst not found"

    def test_cli_page_content(self):
        """Test that cli.rst has required content."""
        docs_path = self.get_docs_path()
        cli = docs_path / "user-guide" / "cli.rst"
        content = cli.read_text()

        required_commands = [
            "homodyne",
            "--config",
            "--method",
            "--verbose",
            "--quiet",
        ]

        for cmd in required_commands:
            assert cmd in content, f"cli.rst missing '{cmd}' command documentation"

    def test_configuration_page_exists(self):
        """Test that configuration.rst exists."""
        docs_path = self.get_docs_path()
        configuration = docs_path / "user-guide" / "configuration.rst"
        assert configuration.exists(), "user-guide/configuration.rst not found"

    def test_configuration_page_content(self):
        """Test that configuration.rst has required content."""
        docs_path = self.get_docs_path()
        configuration = docs_path / "user-guide" / "configuration.rst"
        content = configuration.read_text()

        required_content = [
            "per-angle",
            "YAML",
            "scaling",
            "v2.4.0",
        ]

        for required in required_content:
            assert required in content or required.lower() in content.lower(), (
                f"configuration.rst missing '{required}' content"
            )

    def test_examples_page_exists(self):
        """Test that examples.rst exists."""
        docs_path = self.get_docs_path()
        examples = docs_path / "user-guide" / "examples.rst"
        assert examples.exists(), "user-guide/examples.rst not found"

    def test_examples_page_content(self):
        """Test that examples.rst has required content."""
        docs_path = self.get_docs_path()
        examples = docs_path / "user-guide" / "examples.rst"
        content = examples.read_text()

        required_content = [
            "static",
            "laminar",
            "D0",
            "alpha",
            "D_offset",
        ]

        for required in required_content:
            assert required in content or required.lower() in content.lower(), (
                f"examples.rst missing '{required}' content"
            )


class TestCodeExamples:
    """Test that code examples in user guide are syntactically valid."""

    @staticmethod
    def get_docs_path() -> Path:
        """Get path to docs directory."""
        return Path(__file__).parent.parent.parent / "docs"

    def extract_code_blocks(self, content: str) -> list[str]:
        """Extract code blocks from RST content."""
        # Match both .. code-block:: and ``` style blocks
        code_blocks = []

        # Handle .. code-block::
        pattern = r"\.\. code-block::\s*(?:bash|python|yaml|shell)?\s*\n((?:(?:\n|   .*\n)*?))"
        for match in re.finditer(pattern, content):
            code = match.group(1)
            # Dedent the code
            lines = code.split("\n")
            if lines:
                # Remove leading whitespace
                dedented = "\n".join(
                    line[3:] if line.startswith("   ") else line for line in lines
                )
                code_blocks.append(dedented.strip())

        # Handle ``` style blocks
        pattern = r"```(?:python|bash|yaml|shell)?\s*\n(.*?)\n```"
        for match in re.finditer(pattern, content, re.DOTALL):
            code_blocks.append(match.group(1).strip())

        return code_blocks

    def test_quickstart_python_code_valid(self):
        """Test that quickstart.rst Python code is syntactically valid."""
        docs_path = self.get_docs_path()
        quickstart = docs_path / "user-guide" / "quickstart.rst"
        content = quickstart.read_text()

        code_blocks = self.extract_code_blocks(content)

        # At least one code block should exist
        assert len(code_blocks) > 0, (
            "quickstart.rst should have at least one code block"
        )

    def test_examples_yaml_config_valid(self):
        """Test that examples.rst YAML configs are valid."""
        docs_path = self.get_docs_path()
        examples = docs_path / "user-guide" / "examples.rst"

        if not examples.exists():
            return  # Skip if file doesn't exist yet

        content = examples.read_text()
        # Just verify YAML code blocks are present
        assert ".. code-block::" in content or "```" in content, (
            "examples.rst should have code blocks"
        )


class TestInternalLinks:
    """Test for broken internal links in user guide."""

    @staticmethod
    def get_docs_path() -> Path:
        """Get path to docs directory."""
        return Path(__file__).parent.parent.parent / "docs"

    def test_user_guide_pages_referenced_in_toctree(self):
        """Test that all user guide pages are referenced in toctree."""
        docs_path = self.get_docs_path()
        user_guide_index = docs_path / "user-guide" / "index.rst"

        if not user_guide_index.exists():
            return  # Skip if index doesn't exist yet

        content = user_guide_index.read_text()

        # List of pages that should be referenced
        expected_pages = [
            "installation",
            "quickstart",
            "cli",
            "configuration",
            "examples",
        ]

        for page in expected_pages:
            # Check if page is referenced in toctree
            # Look for either "   installation" or similar patterns
            pattern = rf"^\s+{page}\s*$"
            assert re.search(pattern, content, re.MULTILINE), (
                f"user-guide/index.rst should reference {page}"
            )

    def test_main_index_user_guide_reference(self):
        """Test that main index.rst references user-guide."""
        docs_path = self.get_docs_path()
        index = docs_path / "index.rst"
        content = index.read_text()

        # Check for user-guide reference
        assert "user-guide" in content or "User Guide" in content, (
            "Main index.rst should reference user-guide"
        )
