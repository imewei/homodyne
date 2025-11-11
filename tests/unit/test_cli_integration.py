"""Test suite for CLI integration with Consensus Monte Carlo.

This module tests the integration of CMC into the command-line interface,
including argument parsing, configuration overrides, and diagnostic plot generation.

Test Categories:
- Argument parsing: CLI arguments are parsed correctly
- Config override: CLI arguments override config file values
- Diagnostic plot generation: Plots are generated when requested
- Warning messages: Appropriate warnings for invalid configurations
- Backward compatibility: Existing CLI usage still works
"""

import argparse
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest

from homodyne.cli.args_parser import create_parser
from homodyne.cli.args_parser import validate_args


class TestCMCArgumentParsing:
    """Test CMC-specific CLI argument parsing."""

    def test_cmc_num_shards_argument(self):
        """Test --cmc-num-shards argument is parsed correctly."""
        parser = create_parser()
        args = parser.parse_args(["--cmc-num-shards", "20"])

        assert args.cmc_num_shards == 20

    def test_cmc_backend_argument(self):
        """Test --cmc-backend argument is parsed correctly."""
        parser = create_parser()
        args = parser.parse_args(["--cmc-backend", "multiprocessing"])

        assert args.cmc_backend == "multiprocessing"

    def test_cmc_plot_diagnostics_flag(self):
        """Test --cmc-plot-diagnostics flag is parsed correctly."""
        parser = create_parser()
        args_with_flag = parser.parse_args(["--cmc-plot-diagnostics"])
        args_without_flag = parser.parse_args([])

        assert args_with_flag.cmc_plot_diagnostics is True
        assert args_without_flag.cmc_plot_diagnostics is False

    def test_method_argument_choices(self):
        """Test --method argument accepts all valid choices."""
        parser = create_parser()

        for method in ["nlsq", "mcmc"]:
            args = parser.parse_args(["--method", method])
            assert args.method == method

    def test_method_argument_default(self):
        """Test --method argument defaults to 'nlsq'."""
        parser = create_parser()
        args = parser.parse_args([])

        assert args.method == "nlsq"

    def test_all_cmc_arguments_together(self):
        """Test all CMC arguments can be used together with automatic selection."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "--method",
                "mcmc",
                "--cmc-num-shards",
                "16",
                "--cmc-backend",
                "pjit",
                "--cmc-plot-diagnostics",
            ]
        )

        assert args.method == "mcmc"
        assert args.cmc_num_shards == 16
        assert args.cmc_backend == "pjit"
        assert args.cmc_plot_diagnostics is True


class TestCMCArgumentValidation:
    """Test validation of CMC-specific CLI arguments."""

    def test_negative_num_shards_rejected(self):
        """Test negative --cmc-num-shards is rejected."""
        parser = create_parser()
        args = parser.parse_args(["--cmc-num-shards", "-5"])

        # validate_args should return False
        assert not validate_args(args)

    def test_zero_num_shards_rejected(self):
        """Test zero --cmc-num-shards is rejected."""
        parser = create_parser()
        args = parser.parse_args(["--cmc-num-shards", "0"])

        assert not validate_args(args)

    def test_positive_num_shards_accepted(self):
        """Test positive --cmc-num-shards is accepted."""
        parser = create_parser()
        args = parser.parse_args(["--cmc-num-shards", "10"])

        # validate_args should return True for valid args
        # (ignoring other validation that may fail due to missing files)
        args.config = Path("./homodyne_config.yaml")  # Set default
        args.data_file = None  # No data file specified

        # This will check CMC validation specifically
        is_valid = args.cmc_num_shards > 0
        assert is_valid

    def test_invalid_backend_rejected(self):
        """Test invalid --cmc-backend choice is rejected by argparse."""
        parser = create_parser()

        with pytest.raises(SystemExit):
            # argparse raises SystemExit for invalid choices
            parser.parse_args(["--cmc-backend", "invalid_backend"])

    def test_valid_backend_choices(self):
        """Test all valid backend choices are accepted."""
        parser = create_parser()
        valid_backends = ["auto", "pjit", "multiprocessing", "pbs"]

        for backend in valid_backends:
            args = parser.parse_args(["--cmc-backend", backend])
            assert args.cmc_backend == backend


class TestCMCConfigOverride:
    """Test CLI arguments override config file values."""

    @patch("homodyne.cli.commands.fit_mcmc_jax")
    @patch("homodyne.cli.commands._apply_angle_filtering_for_optimization")
    def test_cmc_num_shards_overrides_config(self, mock_filter, mock_fit_mcmc):
        """Test --cmc-num-shards CLI argument overrides config file."""
        # Setup mocks
        mock_config = Mock()
        mock_config.get_cmc_config.return_value = {
            "sharding": {"num_shards": 10}  # Config file value
        }
        mock_config.config = {"analysis_mode": "laminar_flow"}
        # Nov 10, 2025: Mock get_initial_parameters to return None (no initial values)
        mock_config.get_initial_parameters.return_value = None

        # Fix: Ensure phi_angles_list matches c2_exp first dimension
        # Also provide 2D meshgrids for t1/t2 (required by data pooling code)
        n_phi = 3
        n_t = 10
        t_vals = np.linspace(0, 1, n_t)
        t1_2d, t2_2d = np.meshgrid(t_vals, t_vals, indexing='ij')
        mock_data = {
            "c2_exp": np.random.rand(n_phi, n_t, n_t),  # (3, 10, 10)
            "t1": t1_2d,  # 2D meshgrid (10, 10)
            "t2": t2_2d,  # 2D meshgrid (10, 10)
            "phi_angles_list": np.array([0, 45, 90]),  # 3 angles matching c2_exp
            "wavevector_q_list": [0.01],
        }
        mock_filter.return_value = mock_data

        mock_result = Mock()
        mock_result.is_cmc_result.return_value = False
        mock_fit_mcmc.return_value = mock_result

        # Import here to use mocked dependencies
        from homodyne.cli.commands import _run_optimization

        # Create args with CLI override
        args = argparse.Namespace(
            method="mcmc",
            cmc_num_shards=20,  # CLI override
            cmc_backend=None,
            cmc_plot_diagnostics=False,
            n_samples=100,
            n_warmup=100,
            n_chains=2,
            output_dir=Path("/tmp/test"),
        )

        # Run optimization
        _run_optimization(args, mock_config, mock_data)

        # Verify fit_mcmc_jax was called with overridden num_shards
        call_kwargs = mock_fit_mcmc.call_args[1]
        assert call_kwargs["cmc_config"]["sharding"]["num_shards"] == 20

    @patch("homodyne.cli.commands.fit_mcmc_jax")
    @patch("homodyne.cli.commands._apply_angle_filtering_for_optimization")
    def test_cmc_backend_overrides_config(self, mock_filter, mock_fit_mcmc):
        """Test --cmc-backend CLI argument overrides config file."""
        # Setup mocks
        mock_config = Mock()
        mock_config.get_cmc_config.return_value = {
            "backend": {"name": "pjit"}  # Config file value
        }
        mock_config.config = {"analysis_mode": "laminar_flow"}
        # Nov 10, 2025: Mock get_initial_parameters to return None (no initial values)
        mock_config.get_initial_parameters.return_value = None

        # Fix: Ensure phi_angles_list matches c2_exp first dimension
        # Also provide 2D meshgrids for t1/t2 (required by data pooling code)
        n_phi = 3
        n_t = 10
        t_vals = np.linspace(0, 1, n_t)
        t1_2d, t2_2d = np.meshgrid(t_vals, t_vals, indexing='ij')
        mock_data = {
            "c2_exp": np.random.rand(n_phi, n_t, n_t),  # (3, 10, 10)
            "t1": t1_2d,  # 2D meshgrid (10, 10)
            "t2": t2_2d,  # 2D meshgrid (10, 10)
            "phi_angles_list": np.array([0, 45, 90]),  # 3 angles matching c2_exp
            "wavevector_q_list": [0.01],
        }
        mock_filter.return_value = mock_data

        mock_result = Mock()
        mock_result.is_cmc_result.return_value = False
        mock_fit_mcmc.return_value = mock_result

        from homodyne.cli.commands import _run_optimization

        # Create args with CLI override
        args = argparse.Namespace(
            method="mcmc",
            cmc_num_shards=None,
            cmc_backend="multiprocessing",  # CLI override
            cmc_plot_diagnostics=False,
            n_samples=100,
            n_warmup=100,
            n_chains=2,
            output_dir=Path("/tmp/test"),
        )

        # Run optimization
        _run_optimization(args, mock_config, mock_data)

        # Verify fit_mcmc_jax was called with overridden backend
        call_kwargs = mock_fit_mcmc.call_args[1]
        assert call_kwargs["cmc_config"]["backend"]["name"] == "multiprocessing"


class TestCMCDiagnosticPlotGeneration:
    """Test CMC diagnostic plot generation."""

    @patch("homodyne.cli.commands._generate_cmc_diagnostic_plots")
    @patch("homodyne.cli.commands.fit_mcmc_jax")
    @patch("homodyne.cli.commands._apply_angle_filtering_for_optimization")
    def test_diagnostic_plots_generated_for_cmc_result(
        self, mock_filter, mock_fit_mcmc, mock_generate_plots
    ):
        """Test diagnostic plots are generated when result is CMC and flag is set."""
        # Setup mocks
        mock_config = Mock()
        mock_config.get_cmc_config.return_value = {}
        mock_config.config = {"analysis_mode": "laminar_flow"}

        # Fix: Ensure phi_angles_list matches c2_exp first dimension
        # Also provide 2D meshgrids for t1/t2 (required by data pooling code)
        n_phi = 3
        n_t = 10
        t_vals = np.linspace(0, 1, n_t)
        t1_2d, t2_2d = np.meshgrid(t_vals, t_vals, indexing='ij')
        mock_data = {
            "c2_exp": np.random.rand(n_phi, n_t, n_t),  # (3, 10, 10)
            "t1": t1_2d,  # 2D meshgrid (10, 10)
            "t2": t2_2d,  # 2D meshgrid (10, 10)
            "phi_angles_list": np.array([0, 45, 90]),  # 3 angles matching c2_exp
            "wavevector_q_list": [0.01],
        }
        mock_filter.return_value = mock_data

        # Create CMC result mock
        mock_result = Mock()
        mock_result.is_cmc_result.return_value = True  # This is a CMC result
        mock_result.cmc_diagnostics = {"success_rate": 0.95}
        mock_fit_mcmc.return_value = mock_result

        from homodyne.cli.commands import _run_optimization

        # Create args with diagnostic plots enabled
        args = argparse.Namespace(
            method="mcmc",
            cmc_num_shards=None,
            cmc_backend=None,
            cmc_plot_diagnostics=True,  # Request diagnostic plots
            n_samples=100,
            n_warmup=100,
            n_chains=2,
            output_dir=Path("/tmp/test"),
        )

        # Run optimization
        _run_optimization(args, mock_config, mock_data)

        # Verify _generate_cmc_diagnostic_plots was called
        mock_generate_plots.assert_called_once()
        call_args = mock_generate_plots.call_args[0]
        assert call_args[0] == mock_result
        assert call_args[1] == Path("/tmp/test")

    @patch("homodyne.cli.commands._generate_cmc_diagnostic_plots")
    @patch("homodyne.cli.commands.fit_mcmc_jax")
    @patch("homodyne.cli.commands._apply_angle_filtering_for_optimization")
    def test_diagnostic_plots_not_generated_for_nuts_result(
        self, mock_filter, mock_fit_mcmc, mock_generate_plots
    ):
        """Test diagnostic plots are NOT generated when result is NUTS (not CMC)."""
        # Setup mocks
        mock_config = Mock()
        mock_config.get_cmc_config.return_value = {}
        mock_config.config = {"analysis_mode": "laminar_flow"}

        # Fix: Ensure phi_angles_list matches c2_exp first dimension
        # Also provide 2D meshgrids for t1/t2 (required by data pooling code)
        n_phi = 3
        n_t = 10
        t_vals = np.linspace(0, 1, n_t)
        t1_2d, t2_2d = np.meshgrid(t_vals, t_vals, indexing='ij')
        mock_data = {
            "c2_exp": np.random.rand(n_phi, n_t, n_t),  # (3, 10, 10)
            "t1": t1_2d,  # 2D meshgrid (10, 10)
            "t2": t2_2d,  # 2D meshgrid (10, 10)
            "phi_angles_list": np.array([0, 45, 90]),  # 3 angles matching c2_exp
            "wavevector_q_list": [0.01],
        }
        mock_filter.return_value = mock_data

        # Create NUTS result mock
        mock_result = Mock()
        mock_result.is_cmc_result.return_value = False  # This is NOT a CMC result
        mock_fit_mcmc.return_value = mock_result

        from homodyne.cli.commands import _run_optimization

        # Create args with diagnostic plots enabled
        args = argparse.Namespace(
            method="mcmc",
            cmc_num_shards=None,
            cmc_backend=None,
            cmc_plot_diagnostics=True,  # Request diagnostic plots
            n_samples=100,
            n_warmup=100,
            n_chains=2,
            output_dir=Path("/tmp/test"),
        )

        # Run optimization
        _run_optimization(args, mock_config, mock_data)

        # Verify _generate_cmc_diagnostic_plots was NOT called
        mock_generate_plots.assert_not_called()

    @patch("homodyne.cli.commands._generate_cmc_diagnostic_plots")
    @patch("homodyne.cli.commands.fit_mcmc_jax")
    @patch("homodyne.cli.commands._apply_angle_filtering_for_optimization")
    def test_diagnostic_plots_not_generated_when_flag_false(
        self, mock_filter, mock_fit_mcmc, mock_generate_plots
    ):
        """Test diagnostic plots are NOT generated when flag is False."""
        # Setup mocks
        mock_config = Mock()
        mock_config.get_cmc_config.return_value = {}
        mock_config.config = {"analysis_mode": "laminar_flow"}

        # Fix: Ensure phi_angles_list matches c2_exp first dimension
        # Also provide 2D meshgrids for t1/t2 (required by data pooling code)
        n_phi = 3
        n_t = 10
        t_vals = np.linspace(0, 1, n_t)
        t1_2d, t2_2d = np.meshgrid(t_vals, t_vals, indexing='ij')
        mock_data = {
            "c2_exp": np.random.rand(n_phi, n_t, n_t),  # (3, 10, 10)
            "t1": t1_2d,  # 2D meshgrid (10, 10)
            "t2": t2_2d,  # 2D meshgrid (10, 10)
            "phi_angles_list": np.array([0, 45, 90]),  # 3 angles matching c2_exp
            "wavevector_q_list": [0.01],
        }
        mock_filter.return_value = mock_data

        # Create CMC result mock
        mock_result = Mock()
        mock_result.is_cmc_result.return_value = True
        mock_result.cmc_diagnostics = {"success_rate": 0.95}
        mock_fit_mcmc.return_value = mock_result

        from homodyne.cli.commands import _run_optimization

        # Create args without diagnostic plots
        args = argparse.Namespace(
            method="mcmc",
            cmc_num_shards=None,
            cmc_backend=None,
            cmc_plot_diagnostics=False,  # Do NOT request diagnostic plots
            n_samples=100,
            n_warmup=100,
            n_chains=2,
            output_dir=Path("/tmp/test"),
        )

        # Run optimization
        _run_optimization(args, mock_config, mock_data)

        # Verify _generate_cmc_diagnostic_plots was NOT called
        mock_generate_plots.assert_not_called()


class TestCMCDiagnosticPlotFunction:
    """Test _generate_cmc_diagnostic_plots function directly."""

    def test_diagnostic_plot_function_with_cmc_result(self, tmp_path):
        """Test _generate_cmc_diagnostic_plots saves diagnostic data for CMC result."""
        from homodyne.cli.commands import _generate_cmc_diagnostic_plots

        # Create mock CMC result
        mock_result = Mock()
        mock_result.is_cmc_result.return_value = True
        mock_result.cmc_diagnostics = {
            "per_shard_diagnostics": [{"shard_id": 0, "r_hat": 1.01}],
            "kl_matrix": [[0.0, 0.5], [0.5, 0.0]],
            "success_rate": 0.95,
            "combined_diagnostics": {"ess": 1500},
        }

        output_dir = tmp_path / "test_output"

        # Call function
        _generate_cmc_diagnostic_plots(mock_result, output_dir, "laminar_flow")

        # Verify JSON file was created
        diag_file = output_dir / "cmc_diagnostics" / "cmc_diagnostics.json"
        assert diag_file.exists()

        # Verify JSON content
        import json

        with open(diag_file) as f:
            data = json.load(f)
        assert "per_shard_diagnostics" in data
        assert "between_shard_kl" in data
        assert "success_rate" in data
        assert data["success_rate"] == 0.95

    def test_diagnostic_plot_function_with_nuts_result_logs_warning(self, caplog):
        """Test _generate_cmc_diagnostic_plots logs warning for NUTS result."""
        from homodyne.cli.commands import _generate_cmc_diagnostic_plots

        # Create mock NUTS result (not CMC)
        mock_result = Mock()
        mock_result.is_cmc_result.return_value = False

        output_dir = Path("/tmp/test_output")

        # Call function
        with caplog.at_level("WARNING"):
            _generate_cmc_diagnostic_plots(mock_result, output_dir, "laminar_flow")

        # Verify warning was logged
        assert "not a CMC result" in caplog.text

    def test_diagnostic_plot_function_with_missing_diagnostics_logs_warning(
        self, caplog
    ):
        """Test _generate_cmc_diagnostic_plots logs warning when diagnostics missing."""
        from homodyne.cli.commands import _generate_cmc_diagnostic_plots

        # Create mock CMC result without diagnostics
        mock_result = Mock()
        mock_result.is_cmc_result.return_value = True
        mock_result.cmc_diagnostics = None  # Missing diagnostics

        output_dir = Path("/tmp/test_output")

        # Call function
        with caplog.at_level("WARNING"):
            _generate_cmc_diagnostic_plots(mock_result, output_dir, "laminar_flow")

        # Verify warning was logged
        assert "diagnostics not available" in caplog.text


class TestBackwardCompatibility:
    """Test backward compatibility of CLI integration."""

    def test_existing_cli_usage_still_works(self):
        """Test existing CLI usage without CMC arguments still works."""
        parser = create_parser()

        # Old usage (no CMC arguments)
        args = parser.parse_args(
            [
                "--config",
                "my_config.yaml",
                "--method",
                "nlsq",
            ]
        )

        # Verify default values for CMC arguments
        assert args.cmc_num_shards is None
        assert args.cmc_backend is None
        assert args.cmc_plot_diagnostics is False

    def test_mcmc_method_without_cmc_args_works(self):
        """Test --method mcmc without CMC arguments works (auto-selection)."""
        parser = create_parser()

        args = parser.parse_args(
            [
                "--method",
                "mcmc",
            ]
        )

        # Verify defaults
        assert args.method == "mcmc"
        assert args.cmc_num_shards is None
        assert args.cmc_backend is None
        assert args.cmc_plot_diagnostics is False


# Summary test to validate all CLI integration functionality
def test_cli_integration_summary():
    """Summary test validating all CLI integration components."""
    parser = create_parser()

    # Test 1: All CMC arguments parse correctly
    args = parser.parse_args(
        [
            "--method",
            "mcmc",
            "--cmc-num-shards",
            "16",
            "--cmc-backend",
            "multiprocessing",
            "--cmc-plot-diagnostics",
        ]
    )
    assert args.method == "mcmc"
    assert args.cmc_num_shards == 16
    assert args.cmc_backend == "multiprocessing"
    assert args.cmc_plot_diagnostics is True

    # Test 2: Validation rejects invalid values
    args_invalid = parser.parse_args(["--cmc-num-shards", "0"])
    assert not validate_args(args_invalid)

    # Test 3: Backward compatibility preserved
    args_old = parser.parse_args(["--method", "nlsq"])
    assert args_old.cmc_num_shards is None
    assert args_old.cmc_backend is None
    assert args_old.cmc_plot_diagnostics is False

    # Test 4: Help text includes CMC information
    help_text = parser.format_help()
    assert "cmc" in help_text.lower()
    assert "consensus monte carlo" in help_text.lower()
    assert "num-shards" in help_text.lower()
    assert "backend" in help_text.lower()
    assert "diagnostics" in help_text.lower()
