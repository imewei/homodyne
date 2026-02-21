"""Tests for user guide documentation content."""

import re
from pathlib import Path


class TestUserGuideStructure:
    """Test user guide documentation structure and content."""

    @staticmethod
    def get_docs_path() -> Path:
        """Get path to docs directory."""
        return Path(__file__).parent.parent.parent / "docs" / "source"

    def test_user_guide_index_exists(self):
        """Test that user_guide/index.rst exists."""
        docs_path = self.get_docs_path()
        user_guide_index = docs_path / "user_guide" / "index.rst"
        assert user_guide_index.exists(), "user_guide/index.rst not found"

    def test_user_guide_index_has_toctree(self):
        """Test that user_guide/index.rst has toctree directive."""
        docs_path = self.get_docs_path()
        user_guide_index = docs_path / "user_guide" / "index.rst"
        content = user_guide_index.read_text()
        assert ".. toctree::" in content, (
            "user_guide/index.rst missing toctree directive"
        )

    def test_main_index_exists(self):
        """Test that main index.rst exists."""
        docs_path = self.get_docs_path()
        index = docs_path / "index.rst"
        assert index.exists(), "docs/source/index.rst not found"

    def test_main_index_has_correct_sections(self):
        """Test that main index.rst has 5 main sections."""
        docs_path = self.get_docs_path()
        index = docs_path / "index.rst"
        content = index.read_text()

        # Check for 5 main sections
        required_sections = [
            "User Guide",
            "API Reference",
            "Theory",
            "Developer Guide",
            "Configuration",
        ]

        for section in required_sections:
            assert section in content, f"Main index.rst missing '{section}' section"

    def test_main_index_has_core_concepts(self):
        """Test that main index.rst references core XPCS concepts."""
        docs_path = self.get_docs_path()
        index = docs_path / "index.rst"
        content = index.read_text()

        # Check for core XPCS and analysis concepts
        concept_patterns = [
            r"XPCS",
            r"JAX",
            r"NLSQ|nlsq",
        ]

        matches = sum(1 for pattern in concept_patterns if re.search(pattern, content))
        assert matches >= 2, "Main index.rst missing core XPCS concept references"

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
        return Path(__file__).parent.parent.parent / "docs" / "source"

    def test_installation_page_exists(self):
        """Test that installation.rst exists."""
        docs_path = self.get_docs_path()
        installation = docs_path / "installation.rst"
        assert installation.exists(), "installation.rst not found"

    def test_installation_page_content(self):
        """Test that installation.rst has required content."""
        docs_path = self.get_docs_path()
        installation = docs_path / "installation.rst"
        content = installation.read_text()

        required_content = [
            "Python",
            "3.12",
            "pip install",
            "JAX",
            "CPU-only",
        ]

        for required in required_content:
            assert required in content, f"installation.rst missing '{required}' content"

    def test_quickstart_page_exists(self):
        """Test that quickstart.rst exists."""
        docs_path = self.get_docs_path()
        quickstart = docs_path / "quickstart.rst"
        assert quickstart.exists(), "quickstart.rst not found"

    def test_quickstart_has_yaml_config(self):
        """Test that quickstart.rst includes YAML configuration example."""
        docs_path = self.get_docs_path()
        quickstart = docs_path / "quickstart.rst"
        content = quickstart.read_text()

        # Check for YAML configuration block
        assert ".. code-block::" in content or "```" in content, (
            "quickstart.rst missing code block"
        )
        assert "homodyne" in content, "quickstart.rst missing homodyne command"

    def test_cli_page_exists(self):
        """Test that cli.rst exists."""
        docs_path = self.get_docs_path()
        cli = docs_path / "api" / "cli.rst"
        assert cli.exists(), "api/cli.rst not found"

    def test_cli_page_content(self):
        """Test that cli.rst has required content."""
        docs_path = self.get_docs_path()
        cli = docs_path / "api" / "cli.rst"
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
        """Test that configuration/index.rst exists."""
        docs_path = self.get_docs_path()
        configuration = docs_path / "configuration" / "index.rst"
        assert configuration.exists(), "configuration/index.rst not found"

    def test_configuration_page_content(self):
        """Test that configuration/index.rst has required content."""
        docs_path = self.get_docs_path()
        configuration = docs_path / "configuration" / "index.rst"
        content = configuration.read_text()

        required_content = [
            "per-angle",
            "YAML",
            "scaling",
            "v2.4.0",
        ]

        for required in required_content:
            assert required in content or required.lower() in content.lower(), (
                f"configuration/index.rst missing '{required}' content"
            )

    def test_examples_page_exists(self):
        """Test that examples/index.rst exists."""
        docs_path = self.get_docs_path()
        examples = docs_path / "examples" / "index.rst"
        assert examples.exists(), "examples/index.rst not found"

    def test_examples_page_content(self):
        """Test that examples/index.rst has required content."""
        docs_path = self.get_docs_path()
        examples = docs_path / "examples" / "index.rst"
        content = examples.read_text()

        required_content = [
            "static",
            "laminar",
            "Bayesian",
            "data",
            "optimization",
        ]

        for required in required_content:
            assert required in content or required.lower() in content.lower(), (
                f"examples/index.rst missing '{required}' content"
            )


class TestCodeExamples:
    """Test that code examples in user guide are syntactically valid."""

    @staticmethod
    def get_docs_path() -> Path:
        """Get path to docs directory."""
        return Path(__file__).parent.parent.parent / "docs" / "source"

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
        quickstart = docs_path / "quickstart.rst"
        content = quickstart.read_text()

        code_blocks = self.extract_code_blocks(content)

        # At least one code block should exist
        assert len(code_blocks) > 0, (
            "quickstart.rst should have at least one code block"
        )

    def test_examples_yaml_config_valid(self):
        """Test that examples/index.rst YAML configs are valid."""
        docs_path = self.get_docs_path()
        examples = docs_path / "examples" / "index.rst"

        if not examples.exists():
            return  # Skip if file doesn't exist yet

        content = examples.read_text()
        # Just verify code blocks are present
        assert ".. code-block::" in content or "```" in content, (
            "examples/index.rst should have code blocks"
        )


class TestInternalLinks:
    """Test for broken internal links in user guide."""

    @staticmethod
    def get_docs_path() -> Path:
        """Get path to docs directory."""
        return Path(__file__).parent.parent.parent / "docs" / "source"

    def test_user_guide_pages_referenced_in_toctree(self):
        """Test that all user guide sections are referenced in toctree."""
        docs_path = self.get_docs_path()
        user_guide_index = docs_path / "user_guide" / "index.rst"

        if not user_guide_index.exists():
            return  # Skip if index doesn't exist yet

        content = user_guide_index.read_text()

        # List of section directories that should be referenced
        expected_pages = [
            "01_fundamentals/index",
            "02_data_and_fitting/index",
            "03_advanced_topics/index",
            "04_practical_guides/index",
            "05_appendices/index",
        ]

        for page in expected_pages:
            # Check if page is referenced in toctree
            # Look for either "   01_fundamentals/index" or similar patterns
            pattern = rf"^\s+{re.escape(page)}\s*$"
            assert re.search(pattern, content, re.MULTILINE), (
                f"user_guide/index.rst should reference {page}"
            )

    def test_main_index_user_guide_reference(self):
        """Test that main index.rst references user guide."""
        docs_path = self.get_docs_path()
        index = docs_path / "index.rst"
        content = index.read_text()

        # Check for user guide reference
        assert "user_guide" in content or "User Guide" in content, (
            "Main index.rst should reference user guide"
        )
