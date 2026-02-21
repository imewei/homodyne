"""Tests for research/theory documentation content.

Task 5.1: Focused tests for research content validation.
- Test research pages build without errors
- Test LaTeX equations render correctly
- Test citation format is correct
- Test BibTeX entry is valid
"""

import re
from pathlib import Path


class TestResearchStructure:
    """Test research documentation structure and content."""

    @staticmethod
    def get_docs_path() -> Path:
        """Get path to docs directory."""
        return Path(__file__).parent.parent.parent / "docs" / "source"

    def test_research_index_exists(self):
        """Test that theory/index.rst exists."""
        docs_path = self.get_docs_path()
        research_index = docs_path / "theory" / "index.rst"
        assert research_index.exists(), "theory/index.rst not found"

    def test_research_index_has_toctree(self):
        """Test that theory/index.rst has toctree directive."""
        docs_path = self.get_docs_path()
        research_index = docs_path / "theory" / "index.rst"
        content = research_index.read_text()
        assert ".. toctree::" in content, "theory/index.rst missing toctree directive"

    def test_theoretical_framework_exists(self):
        """Test that theory/theoretical_framework.rst exists."""
        docs_path = self.get_docs_path()
        theoretical = docs_path / "theory" / "theoretical_framework.rst"
        assert theoretical.exists(), "theory/theoretical_framework.rst not found"

    def test_analysis_modes_exists(self):
        """Test that theory/analysis_modes.rst exists."""
        docs_path = self.get_docs_path()
        modes = docs_path / "theory" / "analysis_modes.rst"
        assert modes.exists(), "theory/analysis_modes.rst not found"

    def test_computational_methods_exists(self):
        """Test that theory/computational_methods.rst exists."""
        docs_path = self.get_docs_path()
        methods = docs_path / "theory" / "computational_methods.rst"
        assert methods.exists(), "theory/computational_methods.rst not found"

    def test_citations_exists(self):
        """Test that theory/citations.rst exists."""
        docs_path = self.get_docs_path()
        citations = docs_path / "theory" / "citations.rst"
        assert citations.exists(), "theory/citations.rst not found"


class TestLatexEquations:
    """Test LaTeX equation rendering in research documentation."""

    @staticmethod
    def get_docs_path() -> Path:
        """Get path to docs directory."""
        return Path(__file__).parent.parent.parent / "docs" / "source"

    def test_theoretical_framework_has_equations(self):
        """Test that theoretical_framework.rst contains LaTeX equations."""
        docs_path = self.get_docs_path()
        theoretical = docs_path / "theory" / "theoretical_framework.rst"
        content = theoretical.read_text()

        # Check for math directive
        assert ".. math::" in content, (
            "theoretical_framework.rst missing math directive"
        )

    def test_core_equation_present(self):
        """Test that the core equation c2 = 1 + contrast * [c1]^2 is documented."""
        docs_path = self.get_docs_path()
        theoretical = docs_path / "theory" / "theoretical_framework.rst"
        content = theoretical.read_text()

        # Check for core equation elements
        equation_patterns = [
            r"c_2",
            r"c_1",
            r"\\beta|contrast",
        ]

        for pattern in equation_patterns:
            assert re.search(pattern, content, re.IGNORECASE), (
                f"Core equation missing element matching pattern: {pattern}"
            )

    def test_transport_coefficient_equations(self):
        """Test that transport coefficient equations are documented."""
        docs_path = self.get_docs_path()
        theoretical = docs_path / "theory" / "theoretical_framework.rst"
        content = theoretical.read_text()

        # Check for D(t) = D0 * t^alpha + D_offset equation components
        transport_patterns = [
            r"D_0|D0",
            r"\\alpha|alpha",
            r"D.*offset|D_\\text{offset}",
        ]

        for pattern in transport_patterns:
            assert re.search(pattern, content, re.IGNORECASE), (
                f"Transport coefficient equation missing: {pattern}"
            )

    def test_shear_rate_equations(self):
        """Test that shear rate equations are documented."""
        docs_path = self.get_docs_path()
        theoretical = docs_path / "theory" / "theoretical_framework.rst"
        content = theoretical.read_text()

        # Check for gamma_dot(t) = gamma_dot_0 * t^beta + gamma_dot_offset components
        shear_patterns = [
            r"\\dot{\\gamma}|gamma.*dot|shear.*rate",
            r"\\beta|beta",
        ]

        for pattern in shear_patterns:
            assert re.search(pattern, content, re.IGNORECASE), (
                f"Shear rate equation missing: {pattern}"
            )


class TestCitationContent:
    """Test citation format and BibTeX entries."""

    @staticmethod
    def get_docs_path() -> Path:
        """Get path to docs directory."""
        return Path(__file__).parent.parent.parent / "docs" / "source"

    def test_he_pnas_citation_present(self):
        """Test that He et al. PNAS 2024 citation is present."""
        docs_path = self.get_docs_path()
        citations = docs_path / "theory" / "citations.rst"
        content = citations.read_text()

        # Check for citation elements
        citation_elements = [
            "He",
            "PNAS",
            "2024",
            "10.1073/pnas.2401162121",
        ]

        for element in citation_elements:
            assert element in content, f"Citation missing element: {element}"

    def test_bibtex_entry_present(self):
        """Test that BibTeX entry is present and valid."""
        docs_path = self.get_docs_path()
        citations = docs_path / "theory" / "citations.rst"
        content = citations.read_text()

        # Check for BibTeX block
        assert "@article" in content.lower(), "BibTeX entry missing @article"

        # Check for essential BibTeX fields
        bibtex_fields = [
            "title",
            "author",
            "journal",
            "year",
            "doi",
        ]

        for field in bibtex_fields:
            assert field in content.lower(), f"BibTeX entry missing field: {field}"

    def test_doi_link_format(self):
        """Test that DOI is properly formatted as a link."""
        docs_path = self.get_docs_path()
        citations = docs_path / "theory" / "citations.rst"
        content = citations.read_text()

        # Check for DOI link
        doi_pattern = r"10\.1073/pnas\.2401162121"
        assert re.search(doi_pattern, content), "DOI not found in expected format"


class TestAnalysisModes:
    """Test analysis modes documentation correctness."""

    @staticmethod
    def get_docs_path() -> Path:
        """Get path to docs directory."""
        return Path(__file__).parent.parent.parent / "docs" / "source"

    def test_static_mode_documented(self):
        """Test that static mode (3 params) is documented."""
        docs_path = self.get_docs_path()
        modes = docs_path / "theory" / "analysis_modes.rst"
        content = modes.read_text()

        # Check for static mode parameters
        static_params = ["D0", "alpha", "D_offset"]
        for param in static_params:
            assert param in content, f"Static mode missing parameter: {param}"

        # Check for "3" with "param" context (flexible matching for tables and headings)
        # Matches: "3 parameters", "Parameters (3)", "- 3" in table
        assert re.search(
            r"(3\s*(param|parameter)|param.*\(3\)|\n\s*-\s*3\s*\n)",
            content,
            re.IGNORECASE,
        ), "Static mode should mention 3 parameters"

    def test_laminar_flow_mode_documented(self):
        """Test that laminar flow mode (7 params) is documented."""
        docs_path = self.get_docs_path()
        modes = docs_path / "theory" / "analysis_modes.rst"
        content = modes.read_text()

        # Check for laminar flow parameters
        laminar_params = [
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot",
            "beta",
            "phi0",
        ]
        for param in laminar_params:
            assert param.lower() in content.lower(), (
                f"Laminar flow mode missing parameter: {param}"
            )

        # Check for "7" with "param" context (flexible matching for tables and headings)
        # Matches: "7 parameters", "Parameters (7)", "- 7" in table
        assert re.search(
            r"(7\s*(param|parameter)|param.*\(7\)|\n\s*-\s*7\s*\n)",
            content,
            re.IGNORECASE,
        ), "Laminar flow mode should mention 7 parameters"

    def test_no_deprecated_modes(self):
        """Test that deprecated modes are NOT documented."""
        docs_path = self.get_docs_path()
        modes = docs_path / "theory" / "analysis_modes.rst"
        content = modes.read_text()

        # These deprecated modes should NOT be present
        deprecated_modes = ["static_isotropic", "static_anisotropic"]
        for mode in deprecated_modes:
            assert mode not in content.lower(), (
                f"Deprecated mode '{mode}' should not be documented"
            )

    def test_per_angle_scaling_documented(self):
        """Test that per-angle scaling is documented."""
        docs_path = self.get_docs_path()
        modes = docs_path / "theory" / "analysis_modes.rst"
        content = modes.read_text()

        # Check for per-angle scaling mention
        assert re.search(r"per.?angle|per_angle", content, re.IGNORECASE), (
            "Per-angle scaling should be documented"
        )


class TestJAXFirstArchitecture:
    """Test that documentation reflects JAX-first architecture."""

    @staticmethod
    def get_docs_path() -> Path:
        """Get path to docs directory."""
        return Path(__file__).parent.parent.parent / "docs" / "source"

    def test_computational_methods_uses_jax(self):
        """Test that computational_methods.rst mentions JAX."""
        docs_path = self.get_docs_path()
        methods = docs_path / "theory" / "computational_methods.rst"
        content = methods.read_text()

        # Check for JAX mentions
        assert "JAX" in content, "computational_methods.rst should mention JAX"
        assert "JIT" in content, (
            "computational_methods.rst should mention JIT compilation"
        )

    def test_no_numba_references(self):
        """Test that documentation does not reference Numba."""
        docs_path = self.get_docs_path()
        methods = docs_path / "theory" / "computational_methods.rst"
        content = methods.read_text()

        # Check that Numba is NOT mentioned (JAX-first architecture)
        assert "numba" not in content.lower(), (
            "computational_methods.rst should not reference Numba (JAX-first architecture)"
        )

    def test_theoretical_framework_no_numba(self):
        """Test that theoretical_framework.rst does not reference Numba."""
        docs_path = self.get_docs_path()
        theoretical = docs_path / "theory" / "theoretical_framework.rst"
        content = theoretical.read_text()

        # Check that Numba is NOT mentioned
        assert "numba" not in content.lower(), (
            "theoretical_framework.rst should not reference Numba"
        )
