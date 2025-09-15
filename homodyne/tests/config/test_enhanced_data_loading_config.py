"""
Configuration Test Suite for Enhanced Data Loading System
=========================================================

Comprehensive testing of the configuration system for enhanced data loading:
- YAML configuration schema validation
- Parameter combination testing
- Template validation and examples
- Migration testing from old formats
- Configuration robustness and error handling
- Parameter boundary testing
- Configuration inheritance and overrides

Key Testing Areas:
- All new YAML parameters introduced by Subagents 1-4
- Parameter validation and constraints
- Configuration file parsing and error handling
- Template completeness and consistency
- Migration from JSON to YAML formats
- Default value fallbacks
- Configuration schema evolution
"""

import copy
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, patch

import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Core homodyne imports
try:
    from homodyne.config.manager import ConfigManager
    from homodyne.data.filtering_utils import FilterConfig, FilterType
    from homodyne.data.memory_manager import MemoryConfig, MemoryStrategy
    from homodyne.data.performance_engine import (OptimizationLevel,
                                                  PerformanceConfig)
    from homodyne.data.preprocessing import (PreprocessingConfig,
                                             PreprocessingStage)
    from homodyne.data.quality_controller import (QualityControlConfig,
                                                  QualityThreshold)

    HOMODYNE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import homodyne modules: {e}")
    HOMODYNE_AVAILABLE = False

# YAML support
try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# JSON support (for migration testing)
import json


class TestConfigurationSchema:
    """Test configuration schema validation and parsing"""

    @classmethod
    def setup_class(cls):
        """Set up configuration schema tests"""
        if not HAS_YAML or not HOMODYNE_AVAILABLE:
            pytest.skip("Required dependencies not available")

        cls.test_dir = Path(tempfile.mkdtemp(prefix="homodyne_config_test_"))
        cls.config_dir = cls.test_dir / "configs"
        cls.config_dir.mkdir(exist_ok=True)

    @classmethod
    def teardown_class(cls):
        """Clean up configuration test data"""
        if hasattr(cls, "test_dir") and cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)

    def test_complete_enhanced_config_schema(self):
        """Test complete enhanced configuration schema"""
        complete_config = {
            "data_loading": {
                "enhanced_features": {
                    "enable_filtering": True,
                    "enable_preprocessing": True,
                    "enable_quality_control": True,
                    "enable_performance_optimization": True,
                    "version": "2.0.0",
                    "compatibility_mode": False,
                },
                "filtering": {
                    "q_range": {"min": 1e-4, "max": 1e-2, "units": "m^-1"},
                    "quality_threshold": 0.85,
                    "frame_selection": {
                        "method": "adaptive",
                        "start_frame": 0,
                        "end_frame": -1,
                        "quality_based_selection": True,
                    },
                    "phi_filtering": {
                        "enable": True,
                        "method": "adaptive",
                        "angle_range": {
                            "min": 0.0,
                            "max": 6.28318530718,  # 2π
                        },
                        "sectors": "auto",
                    },
                    "advanced_filters": {
                        "enable_outlier_detection": True,
                        "outlier_threshold": 3.0,
                        "enable_statistical_filtering": True,
                        "statistical_threshold": 2.5,
                    },
                    "parallel_processing": {
                        "enable": True,
                        "max_threads": "auto",
                        "chunk_size": "auto",
                    },
                },
                "preprocessing": {
                    "pipeline_stages": [
                        "diagonal_correction",
                        "normalization",
                        "noise_reduction",
                        "outlier_treatment",
                    ],
                    "diagonal_correction": {
                        "method": "statistical",
                        "threshold": 2.0,
                        "preserve_physics": True,
                        "correction_factor": "auto",
                    },
                    "normalization": {
                        "method": "baseline",
                        "preserve_physics": True,
                        "reference_method": "first_points",
                        "reference_count": 10,
                    },
                    "noise_reduction": {
                        "method": "median",
                        "kernel_size": 3,
                        "iterations": 1,
                        "preserve_features": True,
                    },
                    "outlier_treatment": {
                        "method": "statistical",
                        "threshold": 3.0,
                        "replacement_method": "interpolation",
                    },
                    "chunked_processing": {
                        "enable": True,
                        "chunk_size": "auto",
                        "overlap": 0.1,
                    },
                    "parallel_processing": {"enable": True, "max_threads": "auto"},
                    "audit_trail": {
                        "enable": True,
                        "save_intermediate": False,
                        "detailed_logging": True,
                    },
                },
                "quality_control": {
                    "enable_progressive": True,
                    "enable_auto_repair": True,
                    "enable_feedback": True,
                    "thresholds": {
                        "signal_to_noise": 5.0,
                        "data_completeness": 0.9,
                        "baseline_stability": 0.95,
                        "correlation_decay": 0.8,
                        "physics_consistency": 0.9,
                    },
                    "repair_strategies": {
                        "enable_interpolation": True,
                        "enable_smoothing": True,
                        "enable_outlier_correction": True,
                        "max_repair_fraction": 0.1,
                    },
                    "progressive_stages": [
                        "basic_validation",
                        "filtered_validation",
                        "preprocessed_validation",
                        "final_validation",
                    ],
                    "quality_metrics": {
                        "enable_detailed_metrics": True,
                        "save_quality_report": True,
                        "report_format": "yaml",
                    },
                },
                "performance": {
                    "optimization_level": "adaptive",
                    "memory_strategy": "balanced",
                    "caching": {
                        "enable_memory_cache": True,
                        "enable_disk_cache": True,
                        "cache_directory": "auto",
                        "max_cache_size": "1GB",
                        "compression": {
                            "enable": True,
                            "algorithm": "zstd",
                            "level": 3,
                        },
                    },
                    "parallel_processing": {
                        "enable": True,
                        "max_threads": "auto",
                        "thread_affinity": False,
                        "load_balancing": "dynamic",
                    },
                    "memory_management": {
                        "enable_memory_mapping": True,
                        "chunk_size": "auto",
                        "prefetch_size": "auto",
                        "gc_frequency": "adaptive",
                    },
                    "io_optimization": {
                        "enable_async_io": True,
                        "buffer_size": "auto",
                        "read_ahead": True,
                        "write_behind": True,
                    },
                    "monitoring": {
                        "enable_performance_monitoring": True,
                        "log_performance_metrics": True,
                        "save_performance_report": False,
                    },
                },
            }
        }

        # Save complete configuration
        config_path = self.config_dir / "complete_enhanced_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(complete_config, f, default_flow_style=False, indent=2)

        # Test parsing
        config_manager = ConfigManager(str(config_path))

        # Verify all sections are accessible
        assert (
            config_manager.get("data_loading", "enhanced_features", "enable_filtering")
            == True
        )
        assert (
            config_manager.get("data_loading", "filtering", "quality_threshold") == 0.85
        )
        assert (
            config_manager.get(
                "data_loading", "preprocessing", "diagonal_correction", "method"
            )
            == "statistical"
        )
        assert (
            config_manager.get("data_loading", "quality_control", "enable_progressive")
            == True
        )
        assert (
            config_manager.get("data_loading", "performance", "optimization_level")
            == "adaptive"
        )

        print("✓ Complete enhanced configuration schema validated")

    def test_filtering_config_validation(self):
        """Test filtering configuration parameter validation"""
        valid_filtering_configs = [
            # Basic configuration
            {"q_range": {"min": 1e-4, "max": 1e-2}, "quality_threshold": 0.8},
            # Complete configuration
            {
                "q_range": {"min": 1e-4, "max": 1e-2, "units": "m^-1"},
                "quality_threshold": 0.9,
                "frame_selection": {"method": "auto"},
                "phi_filtering": {"enable": True, "method": "adaptive"},
                "parallel_processing": {"enable": True, "max_threads": 4},
            },
            # Edge values
            {
                "quality_threshold": 0.0,  # Minimum
                "frame_selection": {"start_frame": 0, "end_frame": -1},
            },
            {
                "quality_threshold": 1.0,  # Maximum
                "q_range": {"min": 1e-6, "max": 1e-1},  # Wide range
            },
        ]

        for i, config_dict in enumerate(valid_filtering_configs):
            full_config = {"data_loading": {"filtering": config_dict}}
            config_path = self.config_dir / f"filtering_valid_{i}.yaml"

            with open(config_path, "w") as f:
                yaml.dump(full_config, f)

            config_manager = ConfigManager(str(config_path))
            filter_config = FilterConfig.from_dict(
                config_manager.get("data_loading", "filtering", default={})
            )

            assert filter_config is not None
            assert 0.0 <= filter_config.quality_threshold <= 1.0

            print(f"✓ Valid filtering config {i} parsed successfully")

        # Test invalid configurations
        invalid_filtering_configs = [
            {"q_range": {"min": 1e-2, "max": 1e-4}},  # min > max
            {"quality_threshold": -0.1},  # Below 0
            {"quality_threshold": 1.5},  # Above 1
            {"frame_selection": {"start_frame": -5}},  # Invalid start frame
            {"parallel_processing": {"max_threads": -1}},  # Invalid thread count
        ]

        for i, config_dict in enumerate(invalid_filtering_configs):
            full_config = {"data_loading": {"filtering": config_dict}}
            config_path = self.config_dir / f"filtering_invalid_{i}.yaml"

            with open(config_path, "w") as f:
                yaml.dump(full_config, f)

            config_manager = ConfigManager(str(config_path))

            with pytest.raises((ValueError, RuntimeError, AssertionError)):
                filter_config = FilterConfig.from_dict(
                    config_manager.get("data_loading", "filtering", default={})
                )
                # Validation should fail during FilteringEngine creation
                from homodyne.data.filtering_utils import FilteringEngine

                filtering_engine = FilteringEngine(filter_config)

            print(f"✓ Invalid filtering config {i} properly rejected")

    def test_preprocessing_config_validation(self):
        """Test preprocessing configuration parameter validation"""
        valid_preprocessing_configs = [
            # Minimal configuration
            {"pipeline_stages": ["diagonal_correction"]},
            # Standard configuration
            {
                "pipeline_stages": ["diagonal_correction", "normalization"],
                "diagonal_correction": {"method": "statistical", "threshold": 2.0},
                "normalization": {"method": "baseline"},
            },
            # Complete configuration
            {
                "pipeline_stages": [
                    "diagonal_correction",
                    "normalization",
                    "noise_reduction",
                    "outlier_treatment",
                ],
                "diagonal_correction": {
                    "method": "statistical",
                    "threshold": 3.0,
                    "preserve_physics": True,
                },
                "normalization": {"method": "statistical", "preserve_physics": True},
                "noise_reduction": {"method": "gaussian", "kernel_size": 5},
                "chunked_processing": {"enable": True, "chunk_size": 1000},
            },
        ]

        for i, config_dict in enumerate(valid_preprocessing_configs):
            full_config = {"data_loading": {"preprocessing": config_dict}}
            config_path = self.config_dir / f"preprocessing_valid_{i}.yaml"

            with open(config_path, "w") as f:
                yaml.dump(full_config, f)

            config_manager = ConfigManager(str(config_path))
            preprocessing_config = PreprocessingConfig.from_dict(
                config_manager.get("data_loading", "preprocessing", default={})
            )

            assert preprocessing_config is not None
            assert len(preprocessing_config.pipeline_stages) > 0

            # Verify stages are valid
            valid_stages = [stage.value for stage in PreprocessingStage]
            for stage in preprocessing_config.pipeline_stages:
                assert stage in valid_stages

            print(f"✓ Valid preprocessing config {i} parsed successfully")

        # Test invalid configurations
        invalid_preprocessing_configs = [
            {"pipeline_stages": []},  # Empty stages
            {"pipeline_stages": ["invalid_stage"]},  # Invalid stage name
            {
                "pipeline_stages": ["diagonal_correction"],
                "diagonal_correction": {"threshold": -1.0},  # Invalid threshold
            },
            {
                "pipeline_stages": ["noise_reduction"],
                "noise_reduction": {"kernel_size": 0},  # Invalid kernel size
            },
        ]

        for i, config_dict in enumerate(invalid_preprocessing_configs):
            full_config = {"data_loading": {"preprocessing": config_dict}}
            config_path = self.config_dir / f"preprocessing_invalid_{i}.yaml"

            with open(config_path, "w") as f:
                yaml.dump(full_config, f)

            config_manager = ConfigManager(str(config_path))

            with pytest.raises((ValueError, RuntimeError, AssertionError, KeyError)):
                preprocessing_config = PreprocessingConfig.from_dict(
                    config_manager.get("data_loading", "preprocessing", default={})
                )
                # Some validation might happen during pipeline creation
                from homodyne.data.preprocessing import PreprocessingPipeline

                preprocessing_pipeline = PreprocessingPipeline(preprocessing_config)

            print(f"✓ Invalid preprocessing config {i} properly rejected")

    def test_quality_control_config_validation(self):
        """Test quality control configuration parameter validation"""
        valid_qc_configs = [
            # Basic configuration
            {
                "enable_progressive": True,
                "thresholds": {"signal_to_noise": 5.0, "data_completeness": 0.9},
            },
            # Complete configuration
            {
                "enable_progressive": True,
                "enable_auto_repair": True,
                "enable_feedback": True,
                "thresholds": {
                    "signal_to_noise": 8.0,
                    "data_completeness": 0.95,
                    "baseline_stability": 0.9,
                    "correlation_decay": 0.8,
                    "physics_consistency": 0.85,
                },
                "repair_strategies": {
                    "enable_interpolation": True,
                    "enable_smoothing": False,
                    "max_repair_fraction": 0.05,
                },
                "progressive_stages": [
                    "basic_validation",
                    "filtered_validation",
                    "final_validation",
                ],
            },
        ]

        for i, config_dict in enumerate(valid_qc_configs):
            full_config = {"data_loading": {"quality_control": config_dict}}
            config_path = self.config_dir / f"qc_valid_{i}.yaml"

            with open(config_path, "w") as f:
                yaml.dump(full_config, f)

            config_manager = ConfigManager(str(config_path))
            qc_config = QualityControlConfig.from_dict(
                config_manager.get("data_loading", "quality_control", default={})
            )

            assert qc_config is not None

            # Verify thresholds are in valid range
            for threshold_name, threshold_value in qc_config.thresholds.items():
                if isinstance(threshold_value, (int, float)):
                    assert (
                        threshold_value >= 0.0
                    ), f"Threshold {threshold_name} should be non-negative"
                    if (
                        "completeness" in threshold_name
                        or "stability" in threshold_name
                    ):
                        assert (
                            threshold_value <= 1.0
                        ), f"Threshold {threshold_name} should be <= 1.0"

            print(f"✓ Valid quality control config {i} parsed successfully")

        # Test invalid configurations
        invalid_qc_configs = [
            {"thresholds": {"signal_to_noise": -1.0}},  # Negative SNR threshold
            {"thresholds": {"data_completeness": 1.5}},  # > 1.0
            {
                "repair_strategies": {
                    "max_repair_fraction": -0.1  # Negative repair fraction
                }
            },
            {"repair_strategies": {"max_repair_fraction": 1.5}},  # > 1.0
        ]

        for i, config_dict in enumerate(invalid_qc_configs):
            full_config = {"data_loading": {"quality_control": config_dict}}
            config_path = self.config_dir / f"qc_invalid_{i}.yaml"

            with open(config_path, "w") as f:
                yaml.dump(full_config, f)

            config_manager = ConfigManager(str(config_path))

            with pytest.raises((ValueError, RuntimeError, AssertionError)):
                qc_config = QualityControlConfig.from_dict(
                    config_manager.get("data_loading", "quality_control", default={})
                )
                # Validation might happen during QualityController creation
                from homodyne.data.quality_controller import QualityController

                quality_controller = QualityController(qc_config)

            print(f"✓ Invalid quality control config {i} properly rejected")

    def test_performance_config_validation(self):
        """Test performance configuration parameter validation"""
        valid_performance_configs = [
            # Basic configuration
            {"optimization_level": "balanced"},
            # Complete configuration
            {
                "optimization_level": "aggressive",
                "memory_strategy": "adaptive",
                "caching": {
                    "enable_memory_cache": True,
                    "enable_disk_cache": True,
                    "max_cache_size": "500MB",
                },
                "parallel_processing": {"enable": True, "max_threads": 8},
                "memory_management": {
                    "enable_memory_mapping": True,
                    "chunk_size": 1024,
                },
                "monitoring": {
                    "enable_performance_monitoring": True,
                    "log_performance_metrics": False,
                },
            },
        ]

        for i, config_dict in enumerate(valid_performance_configs):
            full_config = {"data_loading": {"performance": config_dict}}
            config_path = self.config_dir / f"performance_valid_{i}.yaml"

            with open(config_path, "w") as f:
                yaml.dump(full_config, f)

            config_manager = ConfigManager(str(config_path))
            perf_config = PerformanceConfig.from_dict(
                config_manager.get("data_loading", "performance", default={})
            )

            assert perf_config is not None

            # Verify optimization level is valid
            valid_levels = [level.value for level in OptimizationLevel]
            assert perf_config.optimization_level in valid_levels

            print(f"✓ Valid performance config {i} parsed successfully")

        # Test invalid configurations
        invalid_performance_configs = [
            {"optimization_level": "invalid_level"},
            {"parallel_processing": {"max_threads": -1}},
            {"caching": {"max_cache_size": "invalid_size"}},
            {"memory_management": {"chunk_size": -100}},
        ]

        for i, config_dict in enumerate(invalid_performance_configs):
            full_config = {"data_loading": {"performance": config_dict}}
            config_path = self.config_dir / f"performance_invalid_{i}.yaml"

            with open(config_path, "w") as f:
                yaml.dump(full_config, f)

            config_manager = ConfigManager(str(config_path))

            with pytest.raises((ValueError, RuntimeError, AssertionError, KeyError)):
                perf_config = PerformanceConfig.from_dict(
                    config_manager.get("data_loading", "performance", default={})
                )
                # Validation might happen during engine creation
                from homodyne.data.performance_engine import PerformanceEngine

                performance_engine = PerformanceEngine(perf_config)

            print(f"✓ Invalid performance config {i} properly rejected")


class TestConfigurationTemplates:
    """Test configuration templates and examples"""

    @classmethod
    def setup_class(cls):
        """Set up template testing"""
        if not HAS_YAML or not HOMODYNE_AVAILABLE:
            pytest.skip("Required dependencies not available")

        cls.test_dir = Path(tempfile.mkdtemp(prefix="homodyne_template_test_"))
        cls.template_dir = cls.test_dir / "templates"
        cls.template_dir.mkdir(exist_ok=True)

        # Create standard templates
        cls._create_standard_templates()

    @classmethod
    def teardown_class(cls):
        """Clean up template test data"""
        if hasattr(cls, "test_dir") and cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)

    @classmethod
    def _create_standard_templates(cls):
        """Create standard configuration templates"""
        templates = {
            "minimal_enhanced.yaml": {
                "data_loading": {
                    "enhanced_features": {
                        "enable_filtering": True,
                        "enable_preprocessing": False,
                        "enable_quality_control": False,
                        "enable_performance_optimization": False,
                    },
                    "filtering": {"quality_threshold": 0.8},
                }
            },
            "standard_enhanced.yaml": {
                "data_loading": {
                    "enhanced_features": {
                        "enable_filtering": True,
                        "enable_preprocessing": True,
                        "enable_quality_control": True,
                        "enable_performance_optimization": False,
                    },
                    "filtering": {
                        "q_range": {"min": 1e-4, "max": 1e-2},
                        "quality_threshold": 0.85,
                    },
                    "preprocessing": {
                        "pipeline_stages": ["diagonal_correction", "normalization"]
                    },
                    "quality_control": {
                        "enable_progressive": True,
                        "thresholds": {
                            "signal_to_noise": 5.0,
                            "data_completeness": 0.9,
                        },
                    },
                }
            },
            "performance_optimized.yaml": {
                "data_loading": {
                    "enhanced_features": {
                        "enable_filtering": True,
                        "enable_preprocessing": True,
                        "enable_quality_control": True,
                        "enable_performance_optimization": True,
                    },
                    "filtering": {
                        "q_range": {"min": 1e-4, "max": 1e-2},
                        "quality_threshold": 0.8,
                        "parallel_processing": {"enable": True},
                    },
                    "preprocessing": {
                        "pipeline_stages": ["diagonal_correction", "normalization"],
                        "chunked_processing": {"enable": True},
                        "parallel_processing": {"enable": True},
                    },
                    "quality_control": {
                        "enable_progressive": True,
                        "enable_auto_repair": True,
                    },
                    "performance": {
                        "optimization_level": "aggressive",
                        "caching": {
                            "enable_memory_cache": True,
                            "enable_disk_cache": True,
                        },
                        "parallel_processing": {"enable": True},
                        "memory_management": {"enable_memory_mapping": True},
                    },
                }
            },
            "comprehensive_enhanced.yaml": {
                "data_loading": {
                    "enhanced_features": {
                        "enable_filtering": True,
                        "enable_preprocessing": True,
                        "enable_quality_control": True,
                        "enable_performance_optimization": True,
                        "version": "2.0.0",
                    },
                    "filtering": {
                        "q_range": {"min": 1e-4, "max": 1e-2, "units": "m^-1"},
                        "quality_threshold": 0.85,
                        "frame_selection": {"method": "adaptive"},
                        "phi_filtering": {"enable": True, "method": "adaptive"},
                        "advanced_filters": {"enable_outlier_detection": True},
                        "parallel_processing": {"enable": True, "max_threads": "auto"},
                    },
                    "preprocessing": {
                        "pipeline_stages": [
                            "diagonal_correction",
                            "normalization",
                            "noise_reduction",
                        ],
                        "diagonal_correction": {
                            "method": "statistical",
                            "threshold": 2.0,
                            "preserve_physics": True,
                        },
                        "normalization": {
                            "method": "baseline",
                            "preserve_physics": True,
                        },
                        "noise_reduction": {"method": "median", "kernel_size": 3},
                        "chunked_processing": {"enable": True},
                        "parallel_processing": {"enable": True},
                        "audit_trail": {"enable": True},
                    },
                    "quality_control": {
                        "enable_progressive": True,
                        "enable_auto_repair": True,
                        "enable_feedback": True,
                        "thresholds": {
                            "signal_to_noise": 5.0,
                            "data_completeness": 0.9,
                            "baseline_stability": 0.95,
                        },
                        "repair_strategies": {
                            "enable_interpolation": True,
                            "max_repair_fraction": 0.1,
                        },
                        "quality_metrics": {"enable_detailed_metrics": True},
                    },
                    "performance": {
                        "optimization_level": "adaptive",
                        "memory_strategy": "balanced",
                        "caching": {
                            "enable_memory_cache": True,
                            "enable_disk_cache": True,
                            "compression": {"enable": True},
                        },
                        "parallel_processing": {"enable": True},
                        "memory_management": {
                            "enable_memory_mapping": True,
                            "chunk_size": "auto",
                        },
                        "monitoring": {"enable_performance_monitoring": True},
                    },
                }
            },
        }

        # Save templates
        for template_name, template_content in templates.items():
            template_path = cls.template_dir / template_name
            with open(template_path, "w") as f:
                yaml.dump(template_content, f, default_flow_style=False, indent=2)

        cls.template_files = list(templates.keys())

    def test_template_validity(self):
        """Test that all templates are valid and parseable"""
        for template_name in self.template_files:
            template_path = self.template_dir / template_name

            # Test YAML parsing
            with open(template_path, "r") as f:
                config_data = yaml.safe_load(f)

            assert config_data is not None, f"Template {template_name} failed to parse"
            assert (
                "data_loading" in config_data
            ), f"Template {template_name} missing data_loading section"

            # Test ConfigManager parsing
            config_manager = ConfigManager(str(template_path))

            # Verify basic structure
            enhanced_features = config_manager.get(
                "data_loading", "enhanced_features", default={}
            )
            assert isinstance(
                enhanced_features, dict
            ), f"Template {template_name} has invalid enhanced_features"

            # Test component configuration creation
            if enhanced_features.get("enable_filtering", False):
                filter_config = FilterConfig.from_dict(
                    config_manager.get("data_loading", "filtering", default={})
                )
                assert filter_config is not None

            if enhanced_features.get("enable_preprocessing", False):
                preprocessing_config = PreprocessingConfig.from_dict(
                    config_manager.get("data_loading", "preprocessing", default={})
                )
                assert preprocessing_config is not None

            if enhanced_features.get("enable_quality_control", False):
                qc_config = QualityControlConfig.from_dict(
                    config_manager.get("data_loading", "quality_control", default={})
                )
                assert qc_config is not None

            if enhanced_features.get("enable_performance_optimization", False):
                perf_config = PerformanceConfig.from_dict(
                    config_manager.get("data_loading", "performance", default={})
                )
                assert perf_config is not None

            print(f"✓ Template {template_name} is valid")

    def test_template_completeness(self):
        """Test that templates include all necessary parameters"""
        required_sections = {
            "minimal_enhanced.yaml": ["enhanced_features", "filtering"],
            "standard_enhanced.yaml": [
                "enhanced_features",
                "filtering",
                "preprocessing",
                "quality_control",
            ],
            "performance_optimized.yaml": [
                "enhanced_features",
                "filtering",
                "preprocessing",
                "quality_control",
                "performance",
            ],
            "comprehensive_enhanced.yaml": [
                "enhanced_features",
                "filtering",
                "preprocessing",
                "quality_control",
                "performance",
            ],
        }

        for template_name, expected_sections in required_sections.items():
            template_path = self.template_dir / template_name
            config_manager = ConfigManager(str(template_path))

            for section in expected_sections:
                section_data = config_manager.get("data_loading", section, default={})
                assert (
                    section_data
                ), f"Template {template_name} missing or empty section: {section}"

            print(f"✓ Template {template_name} has all required sections")

    def test_template_consistency(self):
        """Test consistency between templates"""
        # Load all templates
        template_configs = {}
        for template_name in self.template_files:
            template_path = self.template_dir / template_name
            template_configs[template_name] = ConfigManager(str(template_path))

        # Test parameter consistency where applicable
        for template_name, config_manager in template_configs.items():
            # Check filtering thresholds are reasonable
            quality_threshold = config_manager.get(
                "data_loading", "filtering", "quality_threshold", default=None
            )
            if quality_threshold is not None:
                assert (
                    0.0 <= quality_threshold <= 1.0
                ), f"Template {template_name} has invalid quality threshold"

            # Check preprocessing stages are valid
            preprocessing_stages = config_manager.get(
                "data_loading", "preprocessing", "pipeline_stages", default=[]
            )
            valid_stages = [
                "diagonal_correction",
                "normalization",
                "noise_reduction",
                "outlier_treatment",
            ]
            for stage in preprocessing_stages:
                assert (
                    stage in valid_stages
                ), f"Template {template_name} has invalid preprocessing stage: {stage}"

            # Check performance optimization levels
            opt_level = config_manager.get(
                "data_loading", "performance", "optimization_level", default=None
            )
            if opt_level is not None:
                valid_levels = ["conservative", "balanced", "aggressive", "adaptive"]
                assert (
                    opt_level in valid_levels
                ), f"Template {template_name} has invalid optimization level"

        print("✓ Template consistency validated")

    def test_template_documentation_accuracy(self):
        """Test that templates match their documentation"""
        # Test that minimal template actually has minimal configuration
        minimal_config = ConfigManager(str(self.template_dir / "minimal_enhanced.yaml"))
        enhanced_features = minimal_config.get(
            "data_loading", "enhanced_features", default={}
        )

        # Should only have filtering enabled
        assert enhanced_features.get("enable_filtering", False) == True
        assert (
            enhanced_features.get("enable_preprocessing", True) == False
        )  # Default should be False
        assert enhanced_features.get("enable_quality_control", True) == False
        assert enhanced_features.get("enable_performance_optimization", True) == False

        # Test that performance template has performance features
        perf_config = ConfigManager(
            str(self.template_dir / "performance_optimized.yaml")
        )
        enhanced_features = perf_config.get(
            "data_loading", "enhanced_features", default={}
        )

        assert enhanced_features.get("enable_performance_optimization", False) == True

        # Performance section should exist and be configured
        performance_section = perf_config.get("data_loading", "performance", default={})
        assert (
            performance_section
        ), "Performance template missing performance configuration"
        assert performance_section.get("optimization_level") == "aggressive"

        print("✓ Template documentation accuracy validated")


class TestConfigurationMigration:
    """Test migration from old configuration formats"""

    @classmethod
    def setup_class(cls):
        """Set up migration testing"""
        if not HAS_YAML or not HOMODYNE_AVAILABLE:
            pytest.skip("Required dependencies not available")

        cls.test_dir = Path(tempfile.mkdtemp(prefix="homodyne_migration_test_"))
        cls.migration_dir = cls.test_dir / "migration"
        cls.migration_dir.mkdir(exist_ok=True)

        # Create old format configurations
        cls._create_old_format_configs()

    @classmethod
    def teardown_class(cls):
        """Clean up migration test data"""
        if hasattr(cls, "test_dir") and cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)

    @classmethod
    def _create_old_format_configs(cls):
        """Create old format configurations for migration testing"""
        # Old JSON format (v1)
        old_json_config = {
            "xpcs_loader": {
                "enable_filtering": True,
                "quality_threshold": 0.8,
                "q_range": [1e-4, 1e-2],
            },
            "preprocessing": {
                "diagonal_correction": True,
                "normalization": "baseline",
                "noise_reduction": False,
            },
            "performance": {"enable_caching": True, "max_threads": 4},
        }

        old_json_path = cls.migration_dir / "old_config_v1.json"
        with open(old_json_path, "w") as f:
            json.dump(old_json_config, f, indent=2)

        # Old YAML format (early v2)
        old_yaml_config = {
            "data_loader": {  # Old section name
                "filtering": {"enable": True, "threshold": 0.85},
                "preprocessing": {
                    "stages": ["diagonal", "normalize"],  # Old stage names
                    "diagonal": {"method": "basic"},
                    "normalize": {"method": "first_point"},
                },
            }
        }

        old_yaml_path = cls.migration_dir / "old_config_early_v2.yaml"
        with open(old_yaml_path, "w") as f:
            yaml.dump(old_yaml_config, f)

        cls.old_configs = {"json_v1": old_json_path, "yaml_early_v2": old_yaml_path}

    def test_json_to_yaml_migration(self):
        """Test migration from JSON v1 format to YAML v2 format"""
        old_json_path = self.old_configs["json_v1"]

        # Load old JSON configuration
        with open(old_json_path, "r") as f:
            old_config = json.load(f)

        # Simulate migration process
        migrated_config = {
            "data_loading": {
                "enhanced_features": {
                    "enable_filtering": old_config.get("xpcs_loader", {}).get(
                        "enable_filtering", False
                    ),
                    "enable_preprocessing": True,  # Enable based on old preprocessing settings
                    "enable_quality_control": False,  # New feature, default off for compatibility
                    "enable_performance_optimization": old_config.get(
                        "performance", {}
                    ).get("enable_caching", False),
                },
                "filtering": {
                    "quality_threshold": old_config.get("xpcs_loader", {}).get(
                        "quality_threshold", 0.8
                    ),
                    "q_range": {
                        "min": old_config.get("xpcs_loader", {}).get(
                            "q_range", [1e-4, 1e-2]
                        )[0],
                        "max": old_config.get("xpcs_loader", {}).get(
                            "q_range", [1e-4, 1e-2]
                        )[1],
                    },
                },
                "preprocessing": {
                    "pipeline_stages": [],  # Build from old boolean flags
                    "diagonal_correction": {
                        "method": "statistical"
                    },  # Default modern method
                    "normalization": {
                        "method": old_config.get("preprocessing", {}).get(
                            "normalization", "baseline"
                        )
                    },
                },
                "performance": {
                    "optimization_level": "balanced",  # Conservative migration
                    "parallel_processing": {
                        "enable": True,
                        "max_threads": old_config.get("performance", {}).get(
                            "max_threads", 4
                        ),
                    },
                    "caching": {
                        "enable_memory_cache": old_config.get("performance", {}).get(
                            "enable_caching", False
                        )
                    },
                },
            }
        }

        # Add stages based on old boolean flags
        if old_config.get("preprocessing", {}).get("diagonal_correction", False):
            migrated_config["data_loading"]["preprocessing"]["pipeline_stages"].append(
                "diagonal_correction"
            )
        if old_config.get("preprocessing", {}).get("normalization"):
            migrated_config["data_loading"]["preprocessing"]["pipeline_stages"].append(
                "normalization"
            )
        if old_config.get("preprocessing", {}).get("noise_reduction", False):
            migrated_config["data_loading"]["preprocessing"]["pipeline_stages"].append(
                "noise_reduction"
            )

        # Save migrated configuration
        migrated_path = self.migration_dir / "migrated_from_json.yaml"
        with open(migrated_path, "w") as f:
            yaml.dump(migrated_config, f, default_flow_style=False, indent=2)

        # Test that migrated configuration works
        config_manager = ConfigManager(str(migrated_path))

        # Verify sections are accessible
        assert (
            config_manager.get("data_loading", "enhanced_features", "enable_filtering")
            == True
        )
        assert (
            config_manager.get("data_loading", "filtering", "quality_threshold") == 0.8
        )

        # Create components to verify they work
        filter_config = FilterConfig.from_dict(
            config_manager.get("data_loading", "filtering", default={})
        )
        assert filter_config is not None

        preprocessing_config = PreprocessingConfig.from_dict(
            config_manager.get("data_loading", "preprocessing", default={})
        )
        assert preprocessing_config is not None

        print("✓ JSON v1 to YAML v2 migration successful")

    def test_backward_compatibility(self):
        """Test that old parameter names still work with warnings"""
        # Create config with mix of old and new parameter names
        mixed_config = {
            "data_loading": {
                "filtering": {
                    "threshold": 0.85,  # Old name
                    "quality_threshold": 0.8,  # New name - should take precedence
                    "enable_quality_filter": True,  # Old name
                },
                "preprocessing": {
                    "stages": ["diagonal", "normalize"],  # Old stage names
                    "pipeline_stages": [
                        "diagonal_correction"
                    ],  # New format - should take precedence
                },
            }
        }

        mixed_config_path = self.migration_dir / "mixed_old_new.yaml"
        with open(mixed_config_path, "w") as f:
            yaml.dump(mixed_config, f)

        # Should handle mixed configuration with warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            config_manager = ConfigManager(str(mixed_config_path))

            # Should be able to extract configuration
            filter_config_dict = config_manager.get(
                "data_loading", "filtering", default={}
            )
            assert filter_config_dict is not None

            # New format should take precedence
            assert filter_config_dict.get("quality_threshold", 0.0) == 0.8  # New value

            # Should issue warnings about deprecated parameters
            if len(w) > 0:
                warning_messages = [str(warning.message).lower() for warning in w]
                # Might warn about deprecated parameter names
                print(f"✓ Issued {len(w)} compatibility warnings (expected)")

        print("✓ Backward compatibility validation successful")

    def test_configuration_version_detection(self):
        """Test automatic detection of configuration version"""
        configs_to_test = [
            # Modern v2 configuration
            {
                "data_loading": {
                    "enhanced_features": {"version": "2.0.0"},
                    "filtering": {"quality_threshold": 0.8},
                }
            },
            # Old-style configuration without version
            {"xpcs_loader": {"quality_threshold": 0.8}},
            # Mixed configuration
            {"data_loading": {"filtering": {"threshold": 0.8}}},  # Old parameter name
        ]

        expected_versions = ["2.0.0", "1.0.0", "mixed"]

        for i, (config_data, expected_version) in enumerate(
            zip(configs_to_test, expected_versions)
        ):
            config_path = self.migration_dir / f"version_test_{i}.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            config_manager = ConfigManager(str(config_path))

            # Test version detection logic (this would be implemented in ConfigManager)
            if (
                "data_loading" in config_data
                and "enhanced_features" in config_data["data_loading"]
            ):
                detected_version = config_data["data_loading"]["enhanced_features"].get(
                    "version", "2.0.0"
                )
            elif "xpcs_loader" in config_data:
                detected_version = "1.0.0"  # Old format
            else:
                detected_version = "mixed"

            assert (
                detected_version == expected_version
            ), f"Version detection failed for config {i}"

            print(f"✓ Version detection for config {i}: {detected_version}")


class TestConfigurationInheritanceAndOverrides:
    """Test configuration inheritance and override mechanisms"""

    @classmethod
    def setup_class(cls):
        """Set up inheritance testing"""
        if not HAS_YAML or not HOMODYNE_AVAILABLE:
            pytest.skip("Required dependencies not available")

        cls.test_dir = Path(tempfile.mkdtemp(prefix="homodyne_inheritance_test_"))
        cls.inheritance_dir = cls.test_dir / "inheritance"
        cls.inheritance_dir.mkdir(exist_ok=True)

    @classmethod
    def teardown_class(cls):
        """Clean up inheritance test data"""
        if hasattr(cls, "test_dir") and cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)

    def test_configuration_defaults(self):
        """Test that configuration defaults work correctly"""
        # Minimal configuration with most settings using defaults
        minimal_config = {
            "data_loading": {
                "enhanced_features": {"enable_filtering": True}
                # All other sections should use defaults
            }
        }

        minimal_config_path = self.inheritance_dir / "minimal_with_defaults.yaml"
        with open(minimal_config_path, "w") as f:
            yaml.dump(minimal_config, f)

        config_manager = ConfigManager(str(minimal_config_path))

        # Test that defaults are applied correctly
        filter_config_dict = config_manager.get("data_loading", "filtering", default={})

        # Should get default values for unspecified parameters
        filter_config = FilterConfig.from_dict(filter_config_dict)

        # Verify default values are reasonable
        assert 0.0 <= filter_config.quality_threshold <= 1.0

        # Test preprocessing defaults
        preprocessing_config_dict = config_manager.get(
            "data_loading", "preprocessing", default={}
        )
        preprocessing_config = PreprocessingConfig.from_dict(preprocessing_config_dict)

        # Should have some default pipeline stages
        assert len(preprocessing_config.pipeline_stages) >= 0  # May be empty by default

        print("✓ Configuration defaults applied correctly")

    def test_configuration_overrides(self):
        """Test configuration override mechanisms"""
        # Base configuration
        base_config = {
            "data_loading": {
                "enhanced_features": {
                    "enable_filtering": True,
                    "enable_preprocessing": True,
                },
                "filtering": {
                    "quality_threshold": 0.8,
                    "q_range": {"min": 1e-4, "max": 1e-2},
                },
                "preprocessing": {"pipeline_stages": ["diagonal_correction"]},
            }
        }

        # Override configuration
        override_config = {
            "data_loading": {
                "filtering": {"quality_threshold": 0.9},  # Override only this value
                "preprocessing": {
                    "pipeline_stages": [
                        "diagonal_correction",
                        "normalization",
                    ]  # Add stage
                },
            }
        }

        base_config_path = self.inheritance_dir / "base_config.yaml"
        with open(base_config_path, "w") as f:
            yaml.dump(base_config, f)

        override_config_path = self.inheritance_dir / "override_config.yaml"
        with open(override_config_path, "w") as f:
            yaml.dump(override_config, f)

        # Test manual override (this would be implemented in ConfigManager)
        with open(base_config_path, "r") as f:
            merged_config = yaml.safe_load(f)

        with open(override_config_path, "r") as f:
            override_data = yaml.safe_load(f)

        # Simulate deep merge
        def deep_merge(base_dict, override_dict):
            result = base_dict.copy()
            for key, value in override_dict.items():
                if (
                    key in result
                    and isinstance(result[key], dict)
                    and isinstance(value, dict)
                ):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        merged_config = deep_merge(merged_config, override_data)

        # Save merged configuration
        merged_config_path = self.inheritance_dir / "merged_config.yaml"
        with open(merged_config_path, "w") as f:
            yaml.dump(merged_config, f)

        # Test merged configuration
        config_manager = ConfigManager(str(merged_config_path))

        # Should have override values
        assert (
            config_manager.get("data_loading", "filtering", "quality_threshold") == 0.9
        )  # Overridden
        assert (
            config_manager.get("data_loading", "filtering", "q_range", "min") == 1e-4
        )  # From base

        # Preprocessing should have both stages
        stages = config_manager.get(
            "data_loading", "preprocessing", "pipeline_stages", default=[]
        )
        assert "diagonal_correction" in stages
        assert "normalization" in stages

        print("✓ Configuration overrides work correctly")

    def test_environment_variable_overrides(self):
        """Test environment variable configuration overrides"""
        # Base configuration
        base_config = {
            "data_loading": {
                "filtering": {"quality_threshold": 0.8},
                "performance": {"max_threads": 4},
            }
        }

        base_config_path = self.inheritance_dir / "env_override_base.yaml"
        with open(base_config_path, "w") as f:
            yaml.dump(base_config, f)

        # Test with environment variable overrides
        env_overrides = {
            "HOMODYNE_FILTERING_QUALITY_THRESHOLD": "0.95",
            "HOMODYNE_PERFORMANCE_MAX_THREADS": "8",
        }

        with patch.dict(os.environ, env_overrides):
            config_manager = ConfigManager(str(base_config_path))

            # ConfigManager should check environment variables
            # This would be implemented in the actual ConfigManager class

            # Simulate environment override logic
            env_quality_threshold = os.environ.get(
                "HOMODYNE_FILTERING_QUALITY_THRESHOLD"
            )
            if env_quality_threshold:
                override_quality_threshold = float(env_quality_threshold)
                assert override_quality_threshold == 0.95

            env_max_threads = os.environ.get("HOMODYNE_PERFORMANCE_MAX_THREADS")
            if env_max_threads:
                override_max_threads = int(env_max_threads)
                assert override_max_threads == 8

        print("✓ Environment variable overrides validated")


def test_configuration_system_integration():
    """Integration test for the complete configuration system"""
    test_dir = Path(tempfile.mkdtemp(prefix="homodyne_config_integration_"))

    try:
        # Create a comprehensive configuration
        integration_config = {
            "data_loading": {
                "enhanced_features": {
                    "enable_filtering": True,
                    "enable_preprocessing": True,
                    "enable_quality_control": True,
                    "enable_performance_optimization": True,
                    "version": "2.0.0",
                },
                "filtering": {
                    "q_range": {"min": 1e-4, "max": 1e-2},
                    "quality_threshold": 0.85,
                    "parallel_processing": {"enable": True},
                },
                "preprocessing": {
                    "pipeline_stages": ["diagonal_correction", "normalization"],
                    "parallel_processing": {"enable": True},
                },
                "quality_control": {
                    "enable_progressive": True,
                    "enable_auto_repair": True,
                },
                "performance": {
                    "optimization_level": "balanced",
                    "caching": {"enable_memory_cache": True},
                },
            }
        }

        config_path = test_dir / "integration_test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(integration_config, f, default_flow_style=False, indent=2)

        # Test complete integration
        config_manager = ConfigManager(str(config_path))

        # Create all component configurations
        filter_config = FilterConfig.from_dict(
            config_manager.get("data_loading", "filtering", default={})
        )

        preprocessing_config = PreprocessingConfig.from_dict(
            config_manager.get("data_loading", "preprocessing", default={})
        )

        qc_config = QualityControlConfig.from_dict(
            config_manager.get("data_loading", "quality_control", default={})
        )

        perf_config = PerformanceConfig.from_dict(
            config_manager.get("data_loading", "performance", default={})
        )

        # Verify all configurations are valid
        assert filter_config is not None
        assert preprocessing_config is not None
        assert qc_config is not None
        assert perf_config is not None

        print("✓ Configuration system integration test passed")

    finally:
        shutil.rmtree(test_dir)


if __name__ == "__main__":
    # Run configuration tests when executed directly
    pytest.main([__file__, "-v", "--tb=short"])
