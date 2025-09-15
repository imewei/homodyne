"""
Results & Output System for Homodyne v2
=======================================

Comprehensive results processing and export system supporting:
- Multi-format export (YAML, JSON, NPZ, HDF5)
- Analysis summaries and reports
- Data validation and quality checks
- Publication-ready formatting

Main Components:
- exporters.py: Multi-format data exporters
- formatters.py: Data formatting and validation
- summarizers.py: Analysis summary generation
- reporters.py: Comprehensive report generation
"""

from homodyne.results.exporters import MultiFormatExporter, ResultExporter
from homodyne.results.formatters import DataValidator, ResultFormatter
from homodyne.results.reporters import ReportGenerator
from homodyne.results.summarizers import AnalysisSummarizer

__all__ = [
    "ResultExporter",
    "MultiFormatExporter",
    "ResultFormatter",
    "DataValidator",
    "AnalysisSummarizer",
    "ReportGenerator",
]
