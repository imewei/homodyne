"""
Analysis Workflows for Homodyne v2
==================================

Complete analysis orchestration system coordinating:
- Configuration loading and validation
- Data loading with dataset optimization
- Method execution (VI/MCMC/Hybrid)
- Results processing and export
- Visualization and reporting

Main Components:
- pipeline.py: Main analysis orchestrator
- executor.py: Method execution controller  
- results_manager.py: Results handling and export
- plotting_controller.py: Visualization coordination
"""

from homodyne.workflows.pipeline import AnalysisPipeline
from homodyne.workflows.executor import MethodExecutor
from homodyne.workflows.results_manager import ResultsManager
from homodyne.workflows.plotting_controller import PlottingController

__all__ = [
    "AnalysisPipeline",
    "MethodExecutor", 
    "ResultsManager",
    "PlottingController"
]