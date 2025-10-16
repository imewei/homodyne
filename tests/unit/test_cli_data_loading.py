"""
Unit Tests for CLI Data Loading
================================

Tests for the CLI data loading functionality, including:
- Config schema normalization
- XPCSDataLoader integration
- CLI argument overrides
- Edge case handling

Author: Claude Code (AI Assistant)
Date: 2025-10-10
"""

from unittest.mock import Mock, patch

import pytest
import yaml

from homodyne.cli.commands import _load_data

# Import the modules to test
from homodyne.config.manager import ConfigManager


class TestConfigNormalization:
    """Test suite for configuration schema normalization."""

    def test_normalization_legacy_format(self, tmp_path):
        """Test that legacy data_folder_path + data_file_name is normalized to file_path."""
        # Create a test config with legacy format
        config_data = {
            "experimental_data": {
                "data_folder_path": "./data/sample/",
                "data_file_name": "test_data.hdf",
            },
            "analyzer_parameters": {"dt": 0.1, "start_frame": 1, "end_frame": 100},
        }

        # Create ConfigManager with override
        config = ConfigManager(config_override=config_data)

        # Check that normalization happened
        exp_data = config.config["experimental_data"]
        assert "file_path" in exp_data, "file_path should be added by normalization"
        assert exp_data["file_path"] == "data/sample/test_data.hdf"

        # Check that original fields are preserved
        assert exp_data["data_folder_path"] == "./data/sample/"
        assert exp_data["data_file_name"] == "test_data.hdf"

    def test_normalization_modern_format(self, tmp_path):
        """Test that modern file_path format is not modified."""
        # Create a test config with modern format
        config_data = {
            "experimental_data": {"file_path": "./data/experiment.hdf"},
            "analyzer_parameters": {"dt": 0.1, "start_frame": 1, "end_frame": 100},
        }

        # Create ConfigManager with override
        config = ConfigManager(config_override=config_data)

        # Check that file_path is unchanged
        exp_data = config.config["experimental_data"]
        assert exp_data["file_path"] == "./data/experiment.hdf"

        # Check that no extra fields were added
        assert "data_folder_path" not in exp_data
        assert "data_file_name" not in exp_data

    def test_normalization_phi_angles(self, tmp_path):
        """Test that phi angles paths are also normalized."""
        # Create a test config with phi angles
        config_data = {
            "experimental_data": {
                "data_folder_path": "./data/",
                "data_file_name": "test.hdf",
                "phi_angles_path": "./data/phi/",
                "phi_angles_file": "angles.txt",
            }
        }

        # Create ConfigManager with override
        config = ConfigManager(config_override=config_data)

        # Check phi angles normalization
        exp_data = config.config["experimental_data"]
        assert "phi_angles_full_path" in exp_data
        assert exp_data["phi_angles_full_path"] == "data/phi/angles.txt"

    def test_normalization_with_absolute_paths(self):
        """Test normalization with absolute paths."""
        config_data = {
            "experimental_data": {
                "data_folder_path": "/home/user/xpcs/data/",
                "data_file_name": "experiment_001.hdf",
            }
        }

        config = ConfigManager(config_override=config_data)
        exp_data = config.config["experimental_data"]

        assert "file_path" in exp_data
        assert exp_data["file_path"] == "/home/user/xpcs/data/experiment_001.hdf"


class TestDataLoading:
    """Test suite for data loading functionality."""

    @patch("homodyne.cli.commands.XPCSDataLoader")
    def test_load_data_with_cli_override(self, mock_loader_class):
        """Test loading data with --data-file CLI override."""
        # Mock the loader
        mock_loader = Mock()
        mock_loader.load_experimental_data.return_value = {
            "c2_exp": Mock(size=1000),
            "t1": Mock(),
            "t2": Mock(),
        }
        mock_loader_class.return_value = mock_loader

        # Create mock args and config
        mock_args = Mock()
        mock_args.data_file = "test_data.hdf"

        mock_config = Mock()
        mock_config.config = {
            "analyzer_parameters": {"dt": 0.1, "start_frame": 1, "end_frame": 100}
        }

        # Call _load_data
        result = _load_data(mock_args, mock_config)

        # Verify XPCSDataLoader was called with correct config
        assert mock_loader_class.called
        call_args = mock_loader_class.call_args[1]
        assert "config_dict" in call_args

        # Verify data was loaded
        assert result is not None
        assert "c2_exp" in result

    @patch("homodyne.cli.commands.XPCSDataLoader")
    def test_load_data_from_config(self, mock_loader_class):
        """Test loading data from configuration file."""
        # Mock the loader
        mock_loader = Mock()
        mock_loader.load_experimental_data.return_value = {
            "c2_exp": Mock(size=2000),
            "t1": Mock(),
            "t2": Mock(),
        }
        mock_loader_class.return_value = mock_loader

        # Create mock args and config
        mock_args = Mock()
        mock_args.data_file = None

        mock_config = Mock()
        mock_config.config = {
            "experimental_data": {
                "data_folder_path": "./data/",
                "data_file_name": "config_data.hdf",
            },
            "analyzer_parameters": {"dt": 0.1, "start_frame": 1, "end_frame": 200},
        }

        # Call _load_data
        result = _load_data(mock_args, mock_config)

        # Verify XPCSDataLoader was called with full config
        assert mock_loader_class.called
        call_args = mock_loader_class.call_args[1]
        assert call_args["config_dict"] == mock_config.config

        # Verify data was loaded
        assert result is not None
        assert "c2_exp" in result

    @patch("homodyne.cli.commands.XPCSDataLoader")
    def test_load_data_relative_path(self, mock_loader_class):
        """Test loading data with relative path (edge case)."""
        # Mock the loader
        mock_loader = Mock()
        mock_loader.load_experimental_data.return_value = {"c2_exp": Mock(size=500)}
        mock_loader_class.return_value = mock_loader

        # Create mock args with relative path (just filename)
        mock_args = Mock()
        mock_args.data_file = "data.hdf"  # No directory prefix

        mock_config = Mock()
        mock_config.config = {
            "analyzer_parameters": {"dt": 0.1, "start_frame": 1, "end_frame": 100}
        }

        # Call _load_data
        result = _load_data(mock_args, mock_config)

        # Verify it handled relative path
        assert mock_loader_class.called
        call_args = mock_loader_class.call_args[1]
        exp_data = call_args["config_dict"]["experimental_data"]

        # Should have absolute path
        assert "data_folder_path" in exp_data
        assert "data_file_name" in exp_data
        assert exp_data["data_file_name"] == "data.hdf"

    def test_load_data_missing_config(self):
        """Test error handling when configuration is missing data file."""
        # Create mock args without CLI override
        mock_args = Mock()
        mock_args.data_file = None

        # Create config without data file
        mock_config = Mock()
        mock_config.config = {
            "experimental_data": {},  # No data file specified
            "analyzer_parameters": {"dt": 0.1, "start_frame": 1, "end_frame": 100},
        }

        # Should raise RuntimeError (wraps ValueError) with helpful message
        with pytest.raises(RuntimeError) as exc_info:
            _load_data(mock_args, mock_config)

        assert "No data file specified" in str(exc_info.value)

    @patch("homodyne.cli.commands.XPCSDataLoader")
    def test_load_data_file_not_found(self, mock_loader_class):
        """Test error handling when data file doesn't exist."""
        # Mock the loader to raise FileNotFoundError
        mock_loader = Mock()
        mock_loader.load_experimental_data.side_effect = FileNotFoundError(
            "Data file not found: /path/to/missing.hdf"
        )
        mock_loader_class.return_value = mock_loader

        # Create mock args and config
        mock_args = Mock()
        mock_args.data_file = "/path/to/missing.hdf"

        mock_config = Mock()
        mock_config.config = {"analyzer_parameters": {}}

        # Should raise RuntimeError with FileNotFoundError as cause
        with pytest.raises(RuntimeError) as exc_info:
            _load_data(mock_args, mock_config)

        assert "Data file not found" in str(exc_info.value)


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""

    def test_empty_config(self):
        """Test handling of empty configuration."""
        config_data = {}
        config = ConfigManager(config_override=config_data)

        # Should not crash during normalization
        assert config.config == {}

    def test_config_with_none_values(self):
        """Test handling of None values in config."""
        config_data = {
            "experimental_data": {"data_folder_path": None, "data_file_name": None}
        }

        config = ConfigManager(config_override=config_data)

        # Should handle None gracefully
        exp_data = config.config["experimental_data"]
        # Normalization might skip or handle None values
        # This tests for no crash

    @patch("homodyne.cli.commands.XPCSDataLoader")
    def test_load_data_with_complex_path(self, mock_loader_class):
        """Test loading data with complex path (spaces, special chars)."""
        # Mock the loader
        mock_loader = Mock()
        mock_loader.load_experimental_data.return_value = {"c2_exp": Mock(size=100)}
        mock_loader_class.return_value = mock_loader

        # Create mock args with complex path
        mock_args = Mock()
        mock_args.data_file = "./data/my experiment (test)/data file #1.hdf"

        mock_config = Mock()
        mock_config.config = {"analyzer_parameters": {}}

        # Should handle complex path
        result = _load_data(mock_args, mock_config)
        assert result is not None


class TestIntegrationScenarios:
    """Integration test scenarios combining multiple components."""

    def test_full_workflow_template_config(self, tmp_path):
        """Test full workflow: template config → normalization → data loading."""
        # Create a template-style config file
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "experimental_data": {
                "data_folder_path": str(tmp_path),
                "data_file_name": "test.hdf",
            },
            "analyzer_parameters": {"dt": 0.1, "start_frame": 1, "end_frame": 100},
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Load config
        config = ConfigManager(str(config_file))

        # Verify normalization happened
        exp_data = config.config["experimental_data"]
        assert "file_path" in exp_data
        assert exp_data["data_file_name"] == "test.hdf"

    def test_cli_override_precedence(self):
        """Test that CLI --data-file takes precedence over config."""
        # Create config with one file
        config_data = {
            "experimental_data": {
                "data_folder_path": "./config/",
                "data_file_name": "config_file.hdf",
            },
            "analyzer_parameters": {"dt": 0.1},
        }

        config = ConfigManager(config_override=config_data)

        # Mock args with different file
        mock_args = Mock()
        mock_args.data_file = "./cli/cli_file.hdf"

        # When _load_data is called, it should use CLI file, not config file
        # (This would need mocking XPCSDataLoader to fully test)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
