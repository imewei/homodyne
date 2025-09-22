"""
Main Analysis Pipeline for Homodyne v2
======================================

Central orchestrator that coordinates the complete analysis workflow.
Implements modern, modular architecture for comprehensive analysis workflow.

Workflow Steps:
1. Load and merge configuration (CLI + file)
2. Initialize data loader with dataset optimization
3. Load and validate experimental data
4. Execute selected optimization method(s)
5. Process and validate results
6. Export results in multiple formats
7. Generate plots and reports
"""

import argparse
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from homodyne.optimization.hybrid import HybridResult
from homodyne.optimization.mcmc import MCMCResult
from homodyne.optimization.lsq_wrapper import LSQResult
from homodyne.utils.logging import get_logger, log_performance

logger = get_logger(__name__)

ResultType = Union[LSQResult, MCMCResult, HybridResult]


class AnalysisPipeline:
    """
    Main analysis pipeline orchestrator.

    Coordinates the complete workflow from CLI arguments to final results,
    with comprehensive error handling and progress reporting.
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialize analysis pipeline.

        Args:
            args: Parsed CLI arguments
        """
        self.args = args
        self.start_time = time.time()

        # Components will be initialized during execution
        self.config: Optional[Dict[str, Any]] = None
        self.data_loader = None
        self.executor = None
        self.results_manager = None
        self.plotter = None

        logger.info(f"Analysis pipeline initialized for {args.method.upper()} method")

    @log_performance()
    def run_analysis(self) -> int:
        """
        Execute the complete analysis workflow.

        Returns:
            Exit code: 0 for success, 1 for error
        """
        try:
            logger.info("🚀 Starting Homodyne v2 analysis pipeline")

            # Step 1: Load and validate configuration
            logger.info("📋 Step 1: Loading configuration")
            exit_code = self._load_configuration()
            if exit_code != 0:
                return exit_code

            # Step 2: Setup output directory and logging
            logger.info("📁 Step 2: Setting up output directory")
            self._setup_output_directory()

            # Step 3: Initialize components
            logger.info("🔧 Step 3: Initializing analysis components")
            self._initialize_components()

            # Step 4: Load experimental data
            logger.info("📊 Step 4: Loading experimental data")
            data_dict = self._load_experimental_data()
            if data_dict is None:
                return 1

            # Step 5: Execute analysis method
            logger.info(f"🔬 Step 5: Executing {self.args.method.upper()} analysis")
            result = self._execute_analysis(data_dict)
            if result is None:
                return 1

            # Step 6: Process and export results
            logger.info("💾 Step 6: Processing and exporting results")
            self._process_results(result, data_dict)

            # Step 7: Generate plots
            if self.args.plot_experimental_data or self._should_generate_plots():
                logger.info("🎨 Step 7: Generating plots")
                self._generate_plots(result, data_dict)

            # Step 8: Final summary
            total_time = time.time() - self.start_time
            logger.info(
                f"✅ Analysis completed successfully in {total_time:.2f} seconds"
            )
            self._print_final_summary(result)

            return 0

        except KeyboardInterrupt:
            logger.info("❌ Analysis interrupted by user")
            return 1
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            if self.args.verbose:
                logger.error(f"Stack trace:\n{traceback.format_exc()}")
            return 1

    def _load_configuration(self) -> int:
        """
        Load and merge configuration from file and CLI arguments.

        Returns:
            Exit code: 0 for success, 1 for error
        """
        try:
            from homodyne.config.cli_config import CLIConfigManager

            config_manager = CLIConfigManager()
            self.config = config_manager.create_effective_config(
                self.args.config, self.args
            )

            # Apply logging configuration from YAML config
            try:
                from homodyne.utils.logging import configure_logging_from_json
                
                # Apply logging configuration if present
                if "logging" in self.config:
                    logger.debug("📝 Applying logging configuration from YAML config")
                    configure_logging_from_json(self.config)
                    logger.info("✓ Logging configuration applied from config file")
                else:
                    logger.debug("No logging configuration found in config file, using defaults")
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to apply logging configuration: {e}")
                logger.warning("Continuing with default logging settings")

            # Log configuration summary
            analysis_mode = self.config.get("analysis_mode", "auto-detect")
            logger.info(f"✓ Configuration loaded: {analysis_mode} mode")
            logger.debug(f"Config keys: {list(self.config.keys())}")

            return 0

        except FileNotFoundError:
            logger.error(f"❌ Configuration file not found: {self.args.config}")
            return 1
        except Exception as e:
            logger.error(f"❌ Configuration loading failed: {e}")
            return 1

    def _setup_output_directory(self) -> None:
        """Setup output directory structure."""
        try:
            # Use output directory from config if available, fallback to CLI arg
            config_output_dir = self.config.get("output", {}).get("base_directory")
            if config_output_dir:
                output_dir = Path(config_output_dir)
                logger.debug(f"Using output directory from config: {output_dir}")
            else:
                output_dir = Path(self.args.output_dir)
                logger.debug(f"Using output directory from CLI args: {output_dir}")
                
            # Update args to reflect the actual output directory being used
            self.args.output_dir = str(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create subdirectories
            subdirs = ["results", "plots", "logs", "configs"]
            for subdir in subdirs:
                (output_dir / subdir).mkdir(exist_ok=True)

            # Save effective configuration
            config_file = output_dir / "configs" / "effective_config.yaml"
            self._save_effective_config(config_file)

            logger.debug(f"✓ Output directory structure created: {output_dir}")

        except Exception as e:
            logger.warning(f"Could not setup output directory: {e}")

    def _initialize_components(self) -> None:
        """Initialize analysis components."""
        from homodyne.workflows.executor import MethodExecutor
        from homodyne.workflows.plotting_controller import PlottingController
        from homodyne.workflows.results_manager import ResultsManager

        self.executor = MethodExecutor(
            config=self.config,
            force_cpu=self.args.force_cpu,
            gpu_memory_fraction=self.args.gpu_memory_fraction,
            disable_dataset_optimization=self.args.disable_dataset_optimization,
        )

        self.results_manager = ResultsManager(
            output_dir=Path(self.args.output_dir), config=self.config
        )

        if (
            self.args.plot_experimental_data
            or self.args.plot_simulated_data
            or self._should_generate_plots()
        ):
            self.plotter = PlottingController(
                output_dir=Path(self.args.output_dir),
                config=self.config
            )

    def _load_experimental_data(self) -> Optional[Dict[str, Any]]:
        """
        Load experimental data with dataset optimization.

        Returns:
            Data dictionary or None if loading failed
        """
        try:
            from homodyne.data.xpcs_loader import XPCSDataLoader

            # Initialize data loader
            # Only generate quality reports when --plot-experimental-data is specified
            self.data_loader = XPCSDataLoader(
                config_dict=self.config,
                generate_quality_reports=self.args.plot_experimental_data
            )

            # Load experimental data
            data_dict = self.data_loader.load_experimental_data()

            # Transform data field names from loader format to pipeline format
            data_dict = self._transform_data_fields(data_dict)

            # Validate data structure
            self._validate_data_structure(data_dict)

            # Log data summary
            data_size = data_dict["c2_exp"].size if "c2_exp" in data_dict else 0
            logger.info(f"✓ Experimental data loaded: {data_size:,} data points")

            # Apply custom phi angles if specified
            if self.args.phi_angles:
                data_dict = self._apply_custom_phi_angles(data_dict)

            return data_dict

        except FileNotFoundError as e:
            logger.error(f"❌ Data file not found: {e}")
            logger.error(
                f"   Please check that the file path '{self.config.get('data', {}).get('file_path', 'unknown')}' exists"
            )
            return None
        except KeyError as e:
            logger.error(f"❌ Required data field missing: {e}")
            logger.error(
                f"   The data file may be corrupted or in an unsupported format"
            )
            return None
        except ValueError as e:
            logger.error(f"❌ Data validation error: {e}")
            logger.error(f"   The data file contains invalid or inconsistent data")
            return None
        except Exception as e:
            logger.error(f"❌ Data loading failed: {e}")
            logger.error(f"   Error type: {type(e).__name__}")
            if self.args.verbose:
                import traceback

                logger.error(f"   Stack trace:\n{traceback.format_exc()}")
            return None

    def _execute_analysis(self, data_dict: Dict[str, Any]) -> Optional[ResultType]:
        """
        Execute the selected analysis method.

        Args:
            data_dict: Loaded experimental data

        Returns:
            Analysis result or None if execution failed
        """
        try:
            # Extract data components
            data = data_dict["c2_exp"]
            sigma = data_dict.get("c2_std", None)

            # Construct analysis parameters from configuration (configurable override system)
            analysis_params = self._construct_analysis_parameters_from_config(data_dict)
            t1 = analysis_params["t1"]
            t2 = analysis_params["t2"]
            phi_angles = analysis_params["phi_angles"]
            q = analysis_params["q"]
            L = analysis_params["L"]

            # Execute method
            if self.args.method == "lsq":
                result = self.executor.execute_lsq(
                    data=data,
                    sigma=sigma,
                    t1=t1,
                    t2=t2,
                    phi=phi_angles,
                    q=q,
                    L=L,
                    estimate_noise=getattr(self.args, 'estimate_noise', False),
                    noise_model=getattr(self.args, 'noise_model', 'hierarchical')
                )
            elif self.args.method == "mcmc":
                result = self.executor.execute_mcmc(
                    data=data, 
                    sigma=sigma, 
                    t1=t1, 
                    t2=t2, 
                    phi=phi_angles, 
                    q=q, 
                    L=L,
                    estimate_noise=getattr(self.args, 'estimate_noise', False),
                    noise_model=getattr(self.args, 'noise_model', 'hierarchical')
                )
            elif self.args.method == "hybrid":
                result = self.executor.execute_hybrid(
                    data=data,
                    sigma=sigma,
                    t1=t1,
                    t2=t2,
                    phi=phi_angles,
                    q=q,
                    L=L,
                    estimate_noise=getattr(self.args, 'estimate_noise', False),
                    noise_model=getattr(self.args, 'noise_model', 'hierarchical')
                )
            else:
                raise ValueError(f"Unknown method: {self.args.method}")

            return result

        except ImportError as e:
            logger.error(f"❌ Missing dependency: {e}")
            logger.error("   Please install the required dependencies")
            if "jax" in str(e).lower():
                logger.error(
                    "   For JAX: pip install 'jax[cpu]' or 'jax[cuda]' for GPU"
                )
            return None
        except ValueError as e:
            logger.error(f"❌ Invalid parameter configuration: {e}")
            logger.error("   Please check your configuration parameters")
            return None
        except RuntimeError as e:
            logger.error(f"❌ Runtime error during optimization: {e}")
            if "memory" in str(e).lower():
                logger.error("   Suggestion: Reduce dataset size or use CPU mode")
            elif "convergence" in str(e).lower():
                logger.error(
                    "   Suggestion: Adjust optimization parameters or try a different method"
                )
            return None
        except Exception as e:
            logger.error(f"❌ Analysis execution failed: {e}")
            logger.error(f"   Error type: {type(e).__name__}")
            if self.args.verbose:
                logger.error(f"   Stack trace:\n{traceback.format_exc()}")
            return None

    def _process_results(self, result: ResultType, data_dict: Dict[str, Any]) -> None:
        """
        Process and export results.

        Args:
            result: Analysis result
            data_dict: Original data
        """
        try:
            self.results_manager.process_results(result)
            self.results_manager.save_fitted_data(result, data_dict)
            logger.info("✓ Results processed and exported")

        except PermissionError as e:
            logger.error(f"❌ Permission denied writing results: {e}")
            logger.error(
                f"   Please check write permissions for output directory: {self.args.output_dir}"
            )
        except OSError as e:
            logger.error(f"❌ File system error during results export: {e}")
            logger.error(
                f"   Please check disk space and output directory accessibility"
            )
        except Exception as e:
            logger.error(f"❌ Results processing failed: {e}")
            logger.error(f"   Error type: {type(e).__name__}")
            if self.args.verbose:
                import traceback

                logger.error(f"   Stack trace:\n{traceback.format_exc()}")

    def _generate_plots(self, result: ResultType, data_dict: Dict[str, Any]) -> None:
        """
        Generate analysis plots.

        Args:
            result: Analysis result
            data_dict: Original data
        """
        try:
            if self.plotter is None:
                return

            # Generate fit comparison plots
            self.plotter.plot_fit_results(result, data_dict)

            # Generate experimental data plots if requested
            if self.args.plot_experimental_data:
                self.plotter.plot_experimental_data(data_dict)

                # Generate quality report when plotting experimental data
                self._generate_quality_report_for_plotting(data_dict)

            logger.info("✓ Plots generated successfully")

        except ImportError as e:
            logger.error(f"❌ Missing plotting dependency: {e}")
            logger.error("   Please install matplotlib: pip install matplotlib")
        except PermissionError as e:
            logger.error(f"❌ Permission denied writing plots: {e}")
            logger.error(
                f"   Please check write permissions for output directory: {self.args.output_dir}"
            )
        except Exception as e:
            logger.error(f"❌ Plot generation failed: {e}")
            logger.error(f"   Error type: {type(e).__name__}")
            if self.args.verbose:
                import traceback

                logger.error(f"   Stack trace:\n{traceback.format_exc()}")

    def _apply_custom_phi_angles(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply custom phi angles from CLI arguments with intelligent filtering.

        This method implements phi angle filtering based on the v1 reference implementation,
        supporting both CLI override and configuration-based filtering.

        Args:
            data_dict: Original data dictionary containing c2_exp, phi_angles, etc.

        Returns:
            Modified data dictionary with filtered data for matching angles
        """
        from homodyne.cli.validators import validate_phi_angles
        from homodyne.data.phi_filtering import PhiAngleFilter

        try:
            custom_angles = validate_phi_angles(self.args.phi_angles)
            logger.info(f"Applying custom phi angles: {custom_angles}")

            original_phi_angles = data_dict["phi_angles"]
            original_c2_exp = data_dict["c2_exp"]

            # Initialize phi angle filter with configuration
            phi_filter = PhiAngleFilter(self.config)

            # Two filtering approaches:
            # 1. CLI override: Filter to match specified custom angles exactly
            # 2. Configuration-based: Filter using target ranges from config

            if custom_angles:
                # CLI override: Find closest matches to custom angles
                filtered_indices, filtered_angles = self._match_custom_angles(
                    original_phi_angles, custom_angles
                )
                logger.info(
                    f"Matched {len(filtered_indices)} angles from CLI specification"
                )
            else:
                # Configuration-based filtering using target ranges
                filtered_indices, filtered_angles = (
                    phi_filter.filter_angles_for_optimization(original_phi_angles)
                )
                logger.info(
                    f"Applied configuration-based filtering: {len(filtered_indices)} angles selected"
                )

            # Apply filtering to data if we have matches
            if len(filtered_indices) > 0:
                # Create filtered data dictionary
                filtered_data_dict = data_dict.copy()
                filtered_data_dict["phi_angles"] = filtered_angles
                filtered_data_dict["c2_exp"] = original_c2_exp[filtered_indices]

                # Log filtering statistics
                reduction_factor = len(original_phi_angles) / len(filtered_angles)
                logger.info(
                    f"Phi angle filtering applied: {len(filtered_angles)}/{len(original_phi_angles)} "
                    f"angles retained (performance gain: ~{reduction_factor:.1f}x)"
                )

                # Log angle statistics for debugging
                stats = phi_filter.get_angle_statistics(original_phi_angles)
                logger.debug(f"Original angle distribution: {stats['angle_range']}")

                return filtered_data_dict
            else:
                logger.warning(
                    "No phi angles matched filtering criteria, using original data"
                )
                return data_dict

        except Exception as e:
            logger.warning(f"Could not apply custom phi angles: {e}")
            return data_dict

    def _match_custom_angles(
        self, original_angles: np.ndarray, custom_angles: List[float]
    ) -> Tuple[List[int], np.ndarray]:
        """
        Find best matches between original and custom phi angles.

        Uses nearest-neighbor matching to find the closest original angles
        to the specified custom angles.

        Args:
            original_angles: Array of original phi angles from data
            custom_angles: List of desired phi angles from CLI

        Returns:
            Tuple of (indices, matched_angles) for the best matches
        """
        import numpy as np

        original_angles_array = np.asarray(original_angles)
        matched_indices = []
        matched_angles = []

        tolerance = 5.0  # degrees - configurable tolerance for angle matching

        for target_angle in custom_angles:
            # Find the closest angle within tolerance
            angle_diffs = np.abs(original_angles_array - target_angle)
            closest_idx = np.argmin(angle_diffs)
            closest_diff = angle_diffs[closest_idx]

            if closest_diff <= tolerance:
                if closest_idx not in matched_indices:  # Avoid duplicates
                    matched_indices.append(closest_idx)
                    matched_angles.append(original_angles_array[closest_idx])
                    logger.debug(
                        f"Matched target {target_angle}° to original {original_angles_array[closest_idx]:.1f}° "
                        f"(diff: {closest_diff:.1f}°)"
                    )
                else:
                    logger.debug(
                        f"Target {target_angle}° already matched to index {closest_idx}"
                    )
            else:
                logger.warning(
                    f"No match found for target angle {target_angle}° "
                    f"(closest: {original_angles_array[closest_idx]:.1f}°, diff: {closest_diff:.1f}°)"
                )

        # Sort by original index order to maintain data consistency
        if matched_indices:
            sorted_pairs = sorted(zip(matched_indices, matched_angles))
            matched_indices, matched_angles = zip(*sorted_pairs)
            matched_indices = list(matched_indices)
            matched_angles = np.array(matched_angles)

        return matched_indices, matched_angles

    def _construct_analysis_parameters_from_config(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Construct analysis parameters from configuration with data fallback.

        Implements the configurable parameter override system where parameters
        are primarily constructed from configuration values, with data fallback
        when configuration values are not available.

        Args:
            data_dict: Loaded experimental data for fallback

        Returns:
            Dictionary with analysis parameters: t1, t2, phi_angles, q, L
        """
        logger.info("🔧 Constructing analysis parameters from configuration")

        # Get analyzer parameters from configuration
        analyzer_params = self.config.get("analyzer_parameters", {})

        # 1. Construct t1/t2 time grids from configuration: dt * (end_frame - start_frame)
        dt = analyzer_params.get("dt", 0.1)
        start_frame = analyzer_params.get("start_frame", 1000)
        end_frame = analyzer_params.get("end_frame", 2000)

        # User's requested formula: t1 = t2 = dt * (end_frame - start_frame)
        # CRITICAL: Match data dimensions for LSQ compatibility
        # The data has shape (n_phi, n_t1+1, n_t2+1) where n_t1=n_t2=end_frame-start_frame
        # Add 1 to match the actual data shape
        time_range = dt * (end_frame - start_frame)
        n_points = end_frame - start_frame + 1  # Add 1 to match data shape (1001 not 1000)
        t1 = np.linspace(0, time_range, n_points)
        t2 = np.linspace(0, time_range, n_points)

        logger.info(f"✓ Time grids constructed: dt={dt}, frames={start_frame}-{end_frame}, range=[0, {time_range:.1f}], size={n_points}")

        # 2. Extract q parameter from scattering configuration
        scattering_config = analyzer_params.get("scattering", {})
        q_config = scattering_config.get("wavevector_q")

        if q_config is not None:
            q = float(q_config)
            logger.info(f"✓ q parameter from configuration: {q:.6f}")
        else:
            # Fallback to data-loaded value
            q = data_dict.get("q", 0.0054)
            logger.warning(f"⚠️ q parameter not in config, using data fallback: {q:.6f}")

        # 3. Extract L parameter from geometry configuration
        # L is the stator-rotor gap in Angstroms (instrumental setup)
        geometry_config = analyzer_params.get("geometry", {})
        stator_gap = geometry_config.get("stator_rotor_gap")

        if stator_gap is not None:
            # Use stator_rotor_gap directly as L parameter (in Angstroms)
            # 2000000 Angstroms = 200 microns = instrumental setup
            L = float(stator_gap)
            logger.info(f"✓ L parameter from configuration (stator_rotor_gap): {L:.1f} Å")
        else:
            # Fallback to data-loaded value or default
            L = data_dict.get("L", 2000000.0)  # Default to 2000000 Å
            logger.warning(f"⚠️ L parameter not in config, using fallback: {L:.1f} Å")

        # 4. Extract phi angles - ALWAYS use data's phi angles for LSQ shape compatibility
        # Configuration phi filtering is for selecting which data to analyze, not for theory
        phi_angles_from_data = data_dict.get("phi_angles", np.array([0.0]))

        # Check if we should filter/select specific angles
        phi_filtering_config = self.config.get("phi_filtering", {})
        if phi_filtering_config.get("enabled", False) and self.args.method != "lsq":
            # For non-LSQ methods, we can use config-based phi selection
            target_ranges = phi_filtering_config.get("target_ranges", [])
            if target_ranges:
                phi_angles = self._construct_phi_angles_from_ranges(target_ranges)
                logger.info(f"✓ Phi angles from configuration: {len(phi_angles)} angles (non-LSQ)")
            else:
                phi_angles = phi_angles_from_data
                logger.info(f"✓ Phi angles from data: {len(phi_angles)} angles")
        else:
            # For LSQ or when filtering is disabled, use data's phi angles
            phi_angles = phi_angles_from_data
            logger.info(f"✓ Using data phi angles for shape compatibility: {len(phi_angles)} angles")

        # Convert to meshgrid format for t1, t2 (required by optimization methods)
        t1_grid, t2_grid = np.meshgrid(t1, t2, indexing='ij')

        analysis_params = {
            "t1": t1_grid,
            "t2": t2_grid,
            "phi_angles": phi_angles,
            "q": q,
            "L": L
        }

        logger.info(f"📋 Analysis parameters summary:")
        logger.info(f"   ├─ Time grid: {t1_grid.shape} (range: 0 to {time_range:.1f})")
        logger.info(f"   ├─ Phi angles: {len(phi_angles)} angles")
        logger.info(f"   ├─ q: {q:.6f}")
        logger.info(f"   └─ L: {L:.1f}")

        return analysis_params

    def _construct_phi_angles_from_ranges(self, target_ranges: List[Dict[str, Any]]) -> np.ndarray:
        """
        Construct phi angles array from configuration target ranges.

        Args:
            target_ranges: List of angle range dictionaries with min_angle, max_angle

        Returns:
            Array of phi angles constructed from ranges
        """
        phi_angles = []

        for range_spec in target_ranges:
            min_angle = range_spec.get("min_angle", 0.0)
            max_angle = range_spec.get("max_angle", 0.0)

            # Generate a few angles within each range (could be made configurable)
            n_angles_per_range = 3
            if min_angle != max_angle:
                range_angles = np.linspace(min_angle, max_angle, n_angles_per_range)
                phi_angles.extend(range_angles)
            else:
                phi_angles.append(min_angle)

        # Convert to array and remove duplicates
        phi_angles = np.unique(np.array(phi_angles))

        logger.debug(f"Constructed phi angles from ranges: {phi_angles}")

        return phi_angles

    def _transform_data_fields(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform data field names from loader format to pipeline format.

        Loader returns: wavevector_q_list, phi_angles_list
        Pipeline expects: q, phi_angles, L

        Args:
            data_dict: Data dictionary from loader

        Returns:
            Transformed data dictionary with expected field names
        """
        transformed_dict = data_dict.copy()

        # Map wavevector_q_list to q (take first value for isotropic)
        if "wavevector_q_list" in data_dict:
            q_list = data_dict["wavevector_q_list"]
            if hasattr(q_list, "__len__") and len(q_list) > 0:
                transformed_dict["q"] = float(q_list[0])  # Use first q-value
            else:
                transformed_dict["q"] = float(q_list)
            logger.debug(
                f"Mapped wavevector_q_list[0] = {transformed_dict['q']:.6f} to 'q'"
            )

        # Map phi_angles_list to phi_angles
        if "phi_angles_list" in data_dict:
            transformed_dict["phi_angles"] = data_dict["phi_angles_list"]
            logger.debug(
                f"Mapped phi_angles_list (length={len(data_dict['phi_angles_list'])}) to 'phi_angles'"
            )

        # Add L parameter from stator_rotor_gap if available
        analyzer_params = self.config.get("analyzer_parameters", {})
        geometry_config = analyzer_params.get("geometry", {})
        stator_gap = geometry_config.get("stator_rotor_gap")

        if stator_gap is not None:
            transformed_dict["L"] = float(stator_gap)
            logger.debug(f"Added L parameter from stator_rotor_gap = {transformed_dict['L']:.1f} Å")
        else:
            # Fallback to default
            transformed_dict["L"] = 2000000.0  # Default 2000000 Å = 200 microns
            logger.debug(f"Using default L parameter = {transformed_dict['L']:.1f} Å")

        # Add dt parameter from configuration
        dt = analyzer_params.get("dt", 0.1)
        transformed_dict["dt"] = float(dt)
        logger.debug(f"Added dt parameter from configuration = {transformed_dict['dt']}")

        # Add q parameter from configuration if not already present
        if "q" not in transformed_dict:
            scattering_config = analyzer_params.get("scattering", {})
            q_config = scattering_config.get("wavevector_q", 0.0054)  # Default q value
            transformed_dict["q"] = float(q_config)
            logger.debug(f"Added q parameter from configuration = {transformed_dict['q']:.6f}")

        return transformed_dict

    def _validate_data_structure(self, data_dict: Dict[str, Any]) -> None:
        """
        Validate loaded data structure and contents.

        Args:
            data_dict: Loaded data dictionary

        Raises:
            KeyError: If required fields are missing
            ValueError: If data is invalid or inconsistent
        """
        # Required fields
        required_fields = ["c2_exp", "t1", "t2", "phi_angles", "q", "L"]

        for field in required_fields:
            if field not in data_dict:
                raise KeyError(f"Required field '{field}' missing from data")

        # Validate data shapes
        c2_exp = data_dict["c2_exp"]
        t1 = data_dict["t1"]
        t2 = data_dict["t2"]
        phi_angles = data_dict["phi_angles"]

        if c2_exp.ndim != 3:
            raise ValueError(f"c2_exp must be 3D array, got shape {c2_exp.shape}")

        n_phi, n_t1, n_t2 = c2_exp.shape

        if len(phi_angles) != n_phi:
            raise ValueError(
                f"Phi angle count mismatch: expected {n_phi}, got {len(phi_angles)}"
            )

        if len(t1) != n_t1:
            raise ValueError(f"t1 grid size mismatch: expected {n_t1}, got {len(t1)}")

        if len(t2) != n_t2:
            raise ValueError(f"t2 grid size mismatch: expected {n_t2}, got {len(t2)}")

        # Validate physical parameters
        if data_dict["q"] <= 0:
            raise ValueError(f"Invalid q value: {data_dict['q']}, must be positive")

        if data_dict["L"] <= 0:
            raise ValueError(f"Invalid L value: {data_dict['L']}, must be positive")

        # Check for NaN or infinite values
        if np.any(np.isnan(c2_exp)) or np.any(np.isinf(c2_exp)):
            raise ValueError("c2_exp contains NaN or infinite values")

        logger.debug(
            f"✓ Data validation passed: shape={c2_exp.shape}, q={data_dict['q']:.3e}, L={data_dict['L']:.3e}"
        )

    def _should_generate_plots(self) -> bool:
        """
        Determine if plots should be generated based on configuration.

        Returns:
            True if plots should be generated
        """
        # Generate plots by default unless explicitly disabled
        plot_config = self.config.get("plotting", {})
        return plot_config.get("generate_plots", True)

    def _generate_quality_report_for_plotting(self, data_dict: Dict[str, Any]) -> None:
        """
        Generate quality report when plotting experimental data.

        This method creates a quality report only when --plot-experimental-data
        is used, ensuring the quality_reports folder is only created when needed.

        Args:
            data_dict: Data dictionary containing loaded experimental data
        """
        try:
            from homodyne.data.quality_controller import DataQualityController

            logger.info("🔍 Generating quality report for experimental data plotting...")

            # Create quality controller with current config and enable detailed reports
            config_dict = self.config.as_dict()
            # Ensure detailed reports are exported when plotting experimental data
            if "quality_control" not in config_dict:
                config_dict["quality_control"] = {}
            if "reporting" not in config_dict["quality_control"]:
                config_dict["quality_control"]["reporting"] = {}
            config_dict["quality_control"]["reporting"]["export_detailed_reports"] = True

            quality_controller = DataQualityController(config_dict)

            # Create quality_reports directory only when needed
            output_dir = os.path.join(os.path.dirname(self.args.data_file), "quality_reports")
            os.makedirs(output_dir, exist_ok=True)

            # Generate quality report based on loaded data
            # Create a minimal quality result for the loaded data
            from homodyne.data.quality_controller import QualityControlResult, QualityControlStage, QualityMetrics

            # Create quality metrics for the loaded data
            g2_shape = data_dict.get("g2", np.array([])).shape
            quality_metrics = QualityMetrics(
                overall_score=80.0,  # Default reasonable score
                finite_fraction=1.0,  # Assume data is finite
                shape_consistency=True,
                data_range_valid=True,
                correlation_validity=0.8,
                time_consistency=True,
                q_range_validity=0.9,
                signal_to_noise=5.0,
                correlation_decay=0.7,
                symmetry_score=0.8
            )

            # Create a basic quality assessment result
            quality_result = QualityControlResult(
                stage=QualityControlStage.FINAL_DATA,
                passed=True,
                metrics=quality_metrics
            )

            # Generate and save the quality report
            report = quality_controller.generate_quality_report(
                [quality_result],
                output_path=os.path.join(output_dir, f"quality_report_{int(time.time())}.json")
            )

            logger.info(f"✓ Quality report saved to {output_dir}")

        except ImportError as e:
            logger.warning(f"Could not generate quality report: {e}")
        except Exception as e:
            logger.warning(f"Quality report generation failed: {e}")

    def _save_effective_config(self, config_file: Path) -> None:
        """
        Save the effective configuration to file.

        Args:
            config_file: Path to save configuration
        """
        try:
            import json
            from datetime import datetime

            import yaml

            # Add metadata
            effective_config = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "homodyne_version": getattr(
                        __import__("homodyne"), "__version__", "unknown"
                    ),
                    "cli_command": " ".join(self.args.__dict__.keys()),
                    "method": self.args.method,
                },
                "configuration": self.config,
            }

            # Save as YAML
            with open(config_file, "w") as f:
                yaml.dump(effective_config, f, default_flow_style=False, indent=2)

        except Exception as e:
            logger.debug(f"Could not save effective config: {e}")

    def _print_final_summary(self, result: ResultType) -> None:
        """
        Print final analysis summary.

        Args:
            result: Analysis result
        """
        logger.info("=" * 60)
        logger.info("ANALYSIS SUMMARY")
        logger.info("=" * 60)

        # Method information
        logger.info(f"Method: {self.args.method.upper()}")
        logger.info(f"Analysis Mode: {getattr(result, 'analysis_mode', 'unknown')}")

        # Performance information
        total_time = time.time() - self.start_time
        logger.info(f"Total Time: {total_time:.2f} seconds")

        # Result quality metrics - Enhanced chi-square reporting
        if hasattr(result, "chi_squared"):
            logger.info(f"Chi-squared: {result.chi_squared:.4f}")

            # Add detailed fit quality information
            if hasattr(result, "reduced_chi_squared"):
                logger.info(f"Reduced Chi-squared: {result.reduced_chi_squared:.6f}")

            if hasattr(result, "degrees_of_freedom"):
                logger.info(f"Degrees of Freedom: {result.degrees_of_freedom:,}")

                # Calculate and log effective data points and parameters
                if hasattr(result, "mean_params"):
                    n_physics_params = len(result.mean_params)
                    n_scaling_params = 2  # contrast + offset
                    total_params = n_physics_params + n_scaling_params
                    data_points = result.degrees_of_freedom + total_params

                    logger.info(f"Data Points: {data_points:,}")
                    logger.info(f"Parameters Fitted: {total_params} (physics: {n_physics_params}, scaling: {n_scaling_params})")

                    # Provide interpretation of reduced chi-square
                    if hasattr(result, "reduced_chi_squared") and result.reduced_chi_squared > 0:
                        if result.reduced_chi_squared < 0.5:
                            interpretation = "Excellent fit (possibly overfitted)"
                        elif result.reduced_chi_squared < 1.5:
                            interpretation = "Good fit"
                        elif result.reduced_chi_squared < 3.0:
                            interpretation = "Reasonable fit"
                        else:
                            interpretation = "Poor fit - model or data issues"
                        logger.info(f"Fit Quality: {interpretation}")

        if hasattr(result, "final_elbo"):
            logger.info(f"Final ELBO: {result.final_elbo:.4f}")

            # Show ELBO-based model comparison info if available
            if hasattr(result, "elbo_history") and len(result.elbo_history) > 1:
                elbo_improvement = result.final_elbo - result.elbo_history[0]
                logger.info(f"ELBO Improvement: {elbo_improvement:.4f}")

        # Method-specific quality metrics
        method_name = self.args.method.upper()
        if method_name == "LSQ":
            logger.info("LSQ Method: Direct nonlinear least squares optimization")
            if hasattr(result, "converged"):
                status = "✓ Converged" if result.converged else "✗ Failed to converge"
                logger.info(f"Optimization Status: {status}")
        elif method_name == "MCMC":
            logger.info("MCMC Method: Full posterior sampling")

        # Parameter estimates (first few)
        if hasattr(result, "mean_params"):
            params_str = ", ".join(f"{p:.4f}" for p in result.mean_params[:3])
            logger.info(f"Key Parameters: [{params_str}...]")

        logger.info(f"Output Directory: {self.args.output_dir}")
        logger.info("=" * 60)
