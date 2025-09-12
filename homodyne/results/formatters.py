"""
Data Formatting and Validation for Homodyne v2
===============================================

Comprehensive data formatting system with validation, standardization,
and quality assurance for analysis results.

Key Features:
- Result data validation and sanitization
- Format standardization across export formats
- Unit conversion and scaling utilities
- Data quality assessment and reporting
- Publication-ready formatting options
"""

from typing import Dict, Any, List, Optional, Union, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
import warnings
from datetime import datetime

from homodyne.utils.logging import get_logger
from homodyne.optimization.variational import VIResult
from homodyne.optimization.mcmc import MCMCResult
from homodyne.optimization.hybrid import HybridResult

logger = get_logger(__name__)

ResultType = Union[VIResult, MCMCResult, HybridResult]


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """
    Data validation issue container.
    
    Attributes:
        severity: Issue severity level
        category: Issue category (e.g., 'parameters', 'convergence')
        message: Human-readable issue description
        field: Specific field with issue (optional)
        suggested_fix: Suggested correction (optional)
    """
    severity: ValidationSeverity
    category: str
    message: str
    field: Optional[str] = None
    suggested_fix: Optional[str] = None


@dataclass
class ValidationReport:
    """
    Comprehensive validation report.
    
    Attributes:
        is_valid: Overall validation status
        issues: List of validation issues
        summary: Summary statistics
        recommendations: List of recommendations
    """
    is_valid: bool
    issues: List[ValidationIssue]
    summary: Dict[str, int]
    recommendations: List[str]


class DataValidator:
    """
    Data validation engine for analysis results.
    
    Provides comprehensive validation with physics-based checks,
    statistical validation, and data quality assessment.
    """
    
    def __init__(self):
        """Initialize data validator."""
        self.physics_constraints = self._get_physics_constraints()
        self.statistical_thresholds = self._get_statistical_thresholds()
    
    def validate_result(self, result: ResultType) -> ValidationReport:
        """
        Validate analysis result comprehensively.
        
        Args:
            result: Analysis result to validate
            
        Returns:
            Comprehensive validation report
        """
        logger.debug(f"Validating {result.__class__.__name__} result")
        
        issues = []
        
        # Core validation checks
        issues.extend(self._validate_basic_structure(result))
        issues.extend(self._validate_parameters(result))
        issues.extend(self._validate_quality_metrics(result))
        issues.extend(self._validate_convergence(result))
        
        # Method-specific validation
        if isinstance(result, VIResult):
            issues.extend(self._validate_vi_specific(result))
        elif isinstance(result, MCMCResult):
            issues.extend(self._validate_mcmc_specific(result))
        elif isinstance(result, HybridResult):
            issues.extend(self._validate_hybrid_specific(result))
        
        # Statistical validation
        issues.extend(self._validate_statistical_properties(result))
        
        # Create summary
        summary = self._create_issue_summary(issues)
        
        # Determine overall validation status
        critical_errors = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        errors = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        is_valid = len(critical_errors) == 0 and len(errors) == 0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues)
        
        report = ValidationReport(
            is_valid=is_valid,
            issues=issues,
            summary=summary,
            recommendations=recommendations
        )
        
        logger.debug(f"Validation completed: {summary}")
        return report
    
    def _validate_basic_structure(self, result: ResultType) -> List[ValidationIssue]:
        """
        Validate basic result structure.
        
        Args:
            result: Result to validate
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # Check for required attributes
        required_attrs = ['mean_params']
        for attr in required_attrs:
            if not hasattr(result, attr):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="structure",
                    message=f"Missing required attribute: {attr}",
                    field=attr,
                    suggested_fix=f"Ensure {attr} is set during analysis"
                ))
        
        # Check parameter dimensions
        if hasattr(result, 'mean_params'):
            params = result.mean_params
            if params is not None:
                if hasattr(params, '__len__'):
                    param_count = len(params)
                    if param_count not in [3, 7]:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            category="parameters",
                            message=f"Unexpected parameter count: {param_count} (expected 3 or 7)",
                            field="mean_params",
                            suggested_fix="Check analysis mode configuration"
                        ))
        
        return issues
    
    def _validate_parameters(self, result: ResultType) -> List[ValidationIssue]:
        """
        Validate parameter values against physics constraints.
        
        Args:
            result: Result to validate
            
        Returns:
            List of validation issues
        """
        issues = []
        
        if not hasattr(result, 'mean_params') or result.mean_params is None:
            return issues
        
        params = result.mean_params
        if hasattr(params, '__len__'):
            param_names = self._get_parameter_names(len(params))
            
            for i, (param_name, param_value) in enumerate(zip(param_names, params)):
                # Check for invalid values
                if not np.isfinite(param_value):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="parameters",
                        message=f"Parameter {param_name} has invalid value: {param_value}",
                        field=f"mean_params[{i}]",
                        suggested_fix="Check for numerical issues in optimization"
                    ))
                    continue
                
                # Check physics constraints
                constraint = self.physics_constraints.get(param_name, {})
                
                min_val = constraint.get('min')
                if min_val is not None and param_value < min_val:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="parameters",
                        message=f"Parameter {param_name} = {param_value:.4f} below physical minimum {min_val}",
                        field=f"mean_params[{i}]",
                        suggested_fix="Check initial conditions or parameter bounds"
                    ))
                
                max_val = constraint.get('max')
                if max_val is not None and param_value > max_val:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        category="parameters",
                        message=f"Parameter {param_name} = {param_value:.4f} above typical maximum {max_val}",
                        field=f"mean_params[{i}]"
                    ))
        
        return issues
    
    def _validate_quality_metrics(self, result: ResultType) -> List[ValidationIssue]:
        """
        Validate quality metrics.
        
        Args:
            result: Result to validate
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # Check chi-squared if available
        if hasattr(result, 'chi_squared'):
            chi_sq = result.chi_squared
            if not np.isfinite(chi_sq):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="quality",
                    message=f"Invalid chi-squared value: {chi_sq}",
                    field="chi_squared",
                    suggested_fix="Check data quality and fitting procedure"
                ))
            elif chi_sq > 10.0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="quality",
                    message=f"High chi-squared value: {chi_sq:.3f} (>10)",
                    field="chi_squared",
                    suggested_fix="Check model appropriateness and data quality"
                ))
            elif chi_sq < 0.1:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="quality",
                    message=f"Very low chi-squared: {chi_sq:.3f} (<0.1) - possible overfitting",
                    field="chi_squared"
                ))
        
        # Check ELBO if available (VI methods)
        if hasattr(result, 'final_elbo'):
            elbo = result.final_elbo
            if not np.isfinite(elbo):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="quality",
                    message=f"Invalid ELBO value: {elbo}",
                    field="final_elbo",
                    suggested_fix="Check VI optimization settings"
                ))
        
        return issues
    
    def _validate_convergence(self, result: ResultType) -> List[ValidationIssue]:
        """
        Validate convergence indicators.
        
        Args:
            result: Result to validate
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # VI convergence check
        if hasattr(result, 'converged'):
            if not result.converged:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="convergence",
                    message="VI optimization did not converge",
                    field="converged",
                    suggested_fix="Increase iterations or adjust learning rate"
                ))
        
        return issues
    
    def _validate_vi_specific(self, result: VIResult) -> List[ValidationIssue]:
        """
        Validate VI-specific attributes.
        
        Args:
            result: VI result to validate
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # Check ELBO history if available
        if hasattr(result, 'elbo_history') and result.elbo_history:
            elbo_history = np.array(result.elbo_history)
            
            # Check for monotonic improvement (with some tolerance)
            if len(elbo_history) > 10:  # Only check if enough history
                recent_trend = elbo_history[-10:] - elbo_history[-11:-1]
                if np.mean(recent_trend) < -1e-6:  # Decreasing trend
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        category="convergence",
                        message="ELBO shows recent decreasing trend",
                        field="elbo_history",
                        suggested_fix="Consider reducing learning rate"
                    ))
        
        return issues
    
    def _validate_mcmc_specific(self, result: MCMCResult) -> List[ValidationIssue]:
        """
        Validate MCMC-specific attributes.
        
        Args:
            result: MCMC result to validate
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # Check R-hat diagnostics
        if hasattr(result, 'r_hat') and result.r_hat is not None:
            r_hat = np.array(result.r_hat)
            
            max_r_hat = np.max(r_hat)
            if max_r_hat > 1.1:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="convergence",
                    message=f"High R-hat detected: {max_r_hat:.3f} (>1.1)",
                    field="r_hat",
                    suggested_fix="Increase warmup samples or run longer chains"
                ))
            elif max_r_hat > 1.05:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="convergence",
                    message=f"Moderate R-hat: {max_r_hat:.3f} (>1.05)",
                    field="r_hat"
                ))
        
        # Check effective sample size
        if hasattr(result, 'ess') and result.ess is not None:
            ess = np.array(result.ess)
            
            min_ess = np.min(ess)
            n_samples = getattr(result, 'n_samples', 1000)
            
            if min_ess < n_samples * 0.1:  # Less than 10% effective
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="efficiency",
                    message=f"Low effective sample size: {min_ess:.0f} (<10% of total)",
                    field="ess",
                    suggested_fix="Increase chain length or improve proposal distribution"
                ))
        
        return issues
    
    def _validate_hybrid_specific(self, result: HybridResult) -> List[ValidationIssue]:
        """
        Validate hybrid-specific attributes.
        
        Args:
            result: Hybrid result to validate
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # Validate sub-results if available
        if hasattr(result, 'vi_result') and result.vi_result:
            issues.extend(self._validate_vi_specific(result.vi_result))
        
        if hasattr(result, 'mcmc_result') and result.mcmc_result:
            issues.extend(self._validate_mcmc_specific(result.mcmc_result))
        
        # Check recommendation consistency
        if hasattr(result, 'recommended_method'):
            recommendation = result.recommended_method
            if recommendation not in ['vi', 'mcmc', 'hybrid']:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="recommendation",
                    message=f"Unexpected method recommendation: {recommendation}",
                    field="recommended_method"
                ))
        
        return issues
    
    def _validate_statistical_properties(self, result: ResultType) -> List[ValidationIssue]:
        """
        Validate statistical properties of results.
        
        Args:
            result: Result to validate
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # Check parameter uncertainties if available
        if hasattr(result, 'std_params') and result.std_params is not None:
            std_params = np.array(result.std_params)
            mean_params = np.array(result.mean_params)
            
            # Check for unreasonably small uncertainties
            rel_uncertainties = std_params / np.abs(mean_params)
            small_uncertainty_mask = rel_uncertainties < 1e-6
            
            if np.any(small_uncertainty_mask):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="uncertainty",
                    message="Some parameters have very small uncertainties (<1e-6 relative)",
                    field="std_params",
                    suggested_fix="Check if uncertainties are physically reasonable"
                ))
            
            # Check for unreasonably large uncertainties
            large_uncertainty_mask = rel_uncertainties > 1.0
            if np.any(large_uncertainty_mask):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="uncertainty",
                    message="Some parameters have large uncertainties (>100% relative)",
                    field="std_params",
                    suggested_fix="Consider longer sampling or check data quality"
                ))
        
        return issues
    
    def _create_issue_summary(self, issues: List[ValidationIssue]) -> Dict[str, int]:
        """
        Create summary statistics of validation issues.
        
        Args:
            issues: List of validation issues
            
        Returns:
            Summary statistics dictionary
        """
        summary = {
            'total': len(issues),
            'critical': 0,
            'error': 0,
            'warning': 0,
            'info': 0
        }
        
        for issue in issues:
            summary[issue.severity.value] += 1
        
        return summary
    
    def _generate_recommendations(self, issues: List[ValidationIssue]) -> List[str]:
        """
        Generate recommendations based on validation issues.
        
        Args:
            issues: List of validation issues
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Group issues by category
        category_counts = {}
        for issue in issues:
            category_counts[issue.category] = category_counts.get(issue.category, 0) + 1
        
        # Generate category-specific recommendations
        if category_counts.get('convergence', 0) > 2:
            recommendations.append("Multiple convergence issues detected - consider adjusting optimization settings")
        
        if category_counts.get('parameters', 0) > 2:
            recommendations.append("Several parameter issues found - check physics constraints and initial conditions")
        
        if category_counts.get('quality', 0) > 0:
            recommendations.append("Quality metrics suggest checking model appropriateness and data quality")
        
        # Add specific fix suggestions
        for issue in issues:
            if issue.suggested_fix and issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
                recommendations.append(f"{issue.category}: {issue.suggested_fix}")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _get_parameter_names(self, count: int) -> List[str]:
        """
        Get parameter names for given count.
        
        Args:
            count: Number of parameters
            
        Returns:
            List of parameter names
        """
        if count == 3:
            return ['D0', 'alpha', 'D_offset']
        elif count == 7:
            return ['D0', 'alpha', 'D_offset', 'gamma_dot_0', 'beta', 'gamma_dot_offset', 'phi_0']
        else:
            return [f'param_{i}' for i in range(count)]
    
    def _get_physics_constraints(self) -> Dict[str, Dict[str, float]]:
        """
        Get physics-based parameter constraints.
        
        Returns:
            Dictionary of parameter constraints
        """
        return {
            'D0': {'min': 0.0, 'max': 1e-6},
            'alpha': {'min': 0.0, 'max': 2.0},
            'D_offset': {'min': 0.0, 'max': 1e-6},
            'gamma_dot_0': {'min': 0.0, 'max': 1000.0},
            'beta': {'min': -2.0, 'max': 2.0},
            'gamma_dot_offset': {'min': 0.0, 'max': 100.0},
            'phi_0': {'min': 0.0, 'max': 2 * np.pi}
        }
    
    def _get_statistical_thresholds(self) -> Dict[str, float]:
        """
        Get statistical validation thresholds.
        
        Returns:
            Dictionary of statistical thresholds
        """
        return {
            'chi_squared_high': 10.0,
            'chi_squared_low': 0.1,
            'r_hat_warning': 1.1,
            'r_hat_info': 1.05,
            'ess_fraction_warning': 0.1,
            'relative_uncertainty_small': 1e-6,
            'relative_uncertainty_large': 1.0
        }


class ResultFormatter:
    """
    Result data formatter with standardization and presentation options.
    
    Provides consistent formatting across different output formats with
    publication-ready options and unit conversion utilities.
    """
    
    def __init__(self, precision: int = 4, scientific_notation: bool = False):
        """
        Initialize result formatter.
        
        Args:
            precision: Number of decimal places for formatting
            scientific_notation: Use scientific notation for small/large numbers
        """
        self.precision = precision
        self.scientific_notation = scientific_notation
        self.unit_conversions = self._get_unit_conversions()
    
    def format_result(self, result: ResultType, style: str = 'standard') -> Dict[str, Any]:
        """
        Format analysis result for presentation.
        
        Args:
            result: Analysis result to format
            style: Formatting style ('standard', 'publication', 'compact')
            
        Returns:
            Formatted result dictionary
        """
        if style == 'publication':
            return self._format_publication_style(result)
        elif style == 'compact':
            return self._format_compact_style(result)
        else:
            return self._format_standard_style(result)
    
    def _format_standard_style(self, result: ResultType) -> Dict[str, Any]:
        """
        Format result in standard style.
        
        Args:
            result: Result to format
            
        Returns:
            Standard formatted result
        """
        formatted = {
            'method': result.__class__.__name__.replace('Result', '').upper(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Format parameters
        if hasattr(result, 'mean_params'):
            formatted['parameters'] = self._format_parameters(result)
        
        # Format quality metrics
        if hasattr(result, 'chi_squared'):
            formatted['chi_squared'] = self._format_number(result.chi_squared)
        
        if hasattr(result, 'final_elbo'):
            formatted['final_elbo'] = self._format_number(result.final_elbo)
        
        return formatted
    
    def _format_publication_style(self, result: ResultType) -> Dict[str, Any]:
        """
        Format result for publication.
        
        Args:
            result: Result to format
            
        Returns:
            Publication-ready formatted result
        """
        formatted = {
            'Method': result.__class__.__name__.replace('Result', ''),
            'Analysis Date': datetime.now().strftime('%B %d, %Y')
        }
        
        # Format parameters with uncertainties
        if hasattr(result, 'mean_params'):
            params = self._format_parameters_with_units(result)
            formatted['Parameters'] = params
        
        # Format quality metrics
        quality = {}
        if hasattr(result, 'chi_squared'):
            quality['χ²'] = self._format_number(result.chi_squared, scientific=True)
        
        if hasattr(result, 'final_elbo'):
            quality['ELBO'] = self._format_number(result.final_elbo, scientific=True)
        
        if quality:
            formatted['Quality Metrics'] = quality
        
        return formatted
    
    def _format_compact_style(self, result: ResultType) -> Dict[str, Any]:
        """
        Format result in compact style.
        
        Args:
            result: Result to format
            
        Returns:
            Compact formatted result
        """
        method = result.__class__.__name__.replace('Result', '').lower()
        
        # Create compact summary
        summary = f"{method}"
        
        if hasattr(result, 'chi_squared'):
            summary += f", χ²={self._format_number(result.chi_squared, precision=2)}"
        
        if hasattr(result, 'converged') and hasattr(result, 'converged'):
            status = "✓" if result.converged else "⚠"
            summary += f" {status}"
        
        return {'summary': summary, 'method': method}
    
    def _format_parameters(self, result: ResultType) -> Dict[str, Any]:
        """
        Format parameter values with uncertainties.
        
        Args:
            result: Result with parameters
            
        Returns:
            Formatted parameters dictionary
        """
        if not hasattr(result, 'mean_params') or result.mean_params is None:
            return {}
        
        mean_params = np.array(result.mean_params)
        param_names = self._get_parameter_names(len(mean_params))
        
        formatted_params = {}
        
        for i, (name, value) in enumerate(zip(param_names, mean_params)):
            param_info = {
                'value': self._format_number(value),
                'units': self._get_parameter_units(name)
            }
            
            # Add uncertainty if available
            if hasattr(result, 'std_params') and result.std_params is not None:
                std_params = np.array(result.std_params)
                if i < len(std_params):
                    param_info['uncertainty'] = self._format_number(std_params[i])
                    param_info['formatted'] = f"{param_info['value']} ± {param_info['uncertainty']}"
            
            formatted_params[name] = param_info
        
        return formatted_params
    
    def _format_parameters_with_units(self, result: ResultType) -> Dict[str, str]:
        """
        Format parameters with proper units for publication.
        
        Args:
            result: Result with parameters
            
        Returns:
            Dictionary of formatted parameter strings
        """
        params = self._format_parameters(result)
        formatted = {}
        
        for name, info in params.items():
            value_str = info.get('formatted', str(info['value']))
            units = info.get('units', '')
            
            if units:
                formatted[name] = f"{value_str} {units}"
            else:
                formatted[name] = value_str
        
        return formatted
    
    def _format_number(self, 
                      value: float,
                      precision: Optional[int] = None,
                      scientific: Optional[bool] = None) -> str:
        """
        Format numerical value according to settings.
        
        Args:
            value: Numerical value to format
            precision: Override precision (optional)
            scientific: Override scientific notation (optional)
            
        Returns:
            Formatted number string
        """
        if not np.isfinite(value):
            return str(value)
        
        prec = precision if precision is not None else self.precision
        use_sci = scientific if scientific is not None else self.scientific_notation
        
        # Determine if scientific notation should be used
        if use_sci or abs(value) < 10**(-prec) or abs(value) >= 10**(prec+2):
            return f"{value:.{prec}e}"
        else:
            return f"{value:.{prec}f}"
    
    def _get_parameter_names(self, count: int) -> List[str]:
        """
        Get parameter names for given count.
        
        Args:
            count: Number of parameters
            
        Returns:
            List of parameter names
        """
        if count == 3:
            return ['D₀', 'α', 'D_offset']
        elif count == 7:
            return ['D₀', 'α', 'D_offset', 'γ̇₀', 'β', 'γ̇_offset', 'φ₀']
        else:
            return [f'param_{i}' for i in range(count)]
    
    def _get_parameter_units(self, param_name: str) -> str:
        """
        Get units for parameter.
        
        Args:
            param_name: Parameter name
            
        Returns:
            Units string
        """
        unit_map = {
            'D₀': 'μm²/s',
            'D_offset': 'μm²/s',
            'α': '',  # dimensionless
            'γ̇₀': 's⁻¹',
            'γ̇_offset': 's⁻¹',
            'β': '',  # dimensionless
            'φ₀': 'rad'
        }
        
        return unit_map.get(param_name, '')
    
    def _get_unit_conversions(self) -> Dict[str, Dict[str, float]]:
        """
        Get unit conversion factors.
        
        Returns:
            Dictionary of conversion factors
        """
        return {
            'diffusion': {
                'm²/s_to_μm²/s': 1e12,
                'cm²/s_to_μm²/s': 1e8
            },
            'time': {
                's_to_ms': 1000,
                's_to_μs': 1e6
            }
        }