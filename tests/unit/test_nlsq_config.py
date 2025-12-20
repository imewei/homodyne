"""Unit tests for homodyne.optimization.nlsq.config module.

Tests the NLSQConfig dataclass for parsing and validating NLSQ configuration.
"""

from homodyne.optimization.nlsq.config import NLSQConfig


class TestNLSQConfigDefaults:
    """Test NLSQConfig default values."""

    def test_default_loss_function(self):
        """Test default loss function is soft_l1."""
        config = NLSQConfig()
        assert config.loss == "soft_l1"

    def test_default_convergence_settings(self):
        """Test default convergence tolerances."""
        config = NLSQConfig()
        assert config.max_iterations == 1000
        assert config.ftol == 1e-8
        assert config.xtol == 1e-8
        assert config.gtol == 1e-8

    def test_default_scaling_settings(self):
        """Test default scaling settings."""
        config = NLSQConfig()
        assert config.x_scale == "jac"
        assert config.x_scale_map is None
        assert config.trust_region_scale == 1.0

    def test_default_diagnostics_enabled(self):
        """Test diagnostics enabled by default."""
        config = NLSQConfig()
        assert config.enable_diagnostics is True

    def test_default_streaming_settings(self):
        """Test default streaming optimizer settings."""
        config = NLSQConfig()
        assert config.enable_streaming is True
        assert config.streaming_chunk_size == 50000

    def test_default_stratified_settings(self):
        """Test default stratified optimization settings."""
        config = NLSQConfig()
        assert config.enable_stratified is True
        assert config.target_chunk_size == 100000

    def test_default_recovery_settings(self):
        """Test default recovery settings."""
        config = NLSQConfig()
        assert config.enable_recovery is True
        assert config.max_recovery_attempts == 3

    def test_default_hybrid_streaming_settings(self):
        """Test default hybrid streaming settings."""
        config = NLSQConfig()
        assert config.enable_hybrid_streaming is True
        assert config.hybrid_normalize is True
        assert config.hybrid_normalization_strategy == "bounds"
        assert config.hybrid_warmup_iterations == 100
        assert config.hybrid_max_warmup_iterations == 500
        assert config.hybrid_warmup_learning_rate == 0.001
        assert config.hybrid_gauss_newton_max_iterations == 50
        assert config.hybrid_gauss_newton_tol == 1e-8
        assert config.hybrid_chunk_size == 50000
        assert config.hybrid_trust_region_initial == 1.0
        assert config.hybrid_regularization_factor == 1e-10
        assert config.hybrid_enable_checkpoints is True
        assert config.hybrid_checkpoint_frequency == 100
        assert config.hybrid_validate_numerics is True


class TestNLSQConfigFromDict:
    """Test NLSQConfig.from_dict class method."""

    def test_from_empty_dict(self):
        """Test creating config from empty dictionary uses defaults."""
        config = NLSQConfig.from_dict({})
        assert config.loss == "soft_l1"
        assert config.max_iterations == 1000
        assert config.enable_diagnostics is True

    def test_from_dict_with_loss_function(self):
        """Test parsing loss function from dict."""
        config = NLSQConfig.from_dict({"loss": "huber"})
        assert config.loss == "huber"

    def test_from_dict_with_convergence_settings(self):
        """Test parsing convergence settings from dict."""
        config = NLSQConfig.from_dict(
            {
                "max_iterations": 500,
                "tolerance": 1e-6,  # maps to ftol
                "xtol": 1e-7,
                "gtol": 1e-9,
            }
        )
        assert config.max_iterations == 500
        assert config.ftol == 1e-6
        assert config.xtol == 1e-7
        assert config.gtol == 1e-9

    def test_from_dict_with_nested_diagnostics(self):
        """Test parsing nested diagnostics section."""
        config = NLSQConfig.from_dict({"diagnostics": {"enable": False}})
        assert config.enable_diagnostics is False

    def test_from_dict_with_nested_streaming(self):
        """Test parsing nested streaming section."""
        config = NLSQConfig.from_dict(
            {
                "streaming": {
                    "enable": False,
                    "chunk_size": 25000,
                }
            }
        )
        assert config.enable_streaming is False
        assert config.streaming_chunk_size == 25000

    def test_from_dict_with_nested_stratified(self):
        """Test parsing nested stratified section."""
        config = NLSQConfig.from_dict(
            {
                "stratified": {
                    "enable": False,
                    "target_chunk_size": 200000,
                }
            }
        )
        assert config.enable_stratified is False
        assert config.target_chunk_size == 200000

    def test_from_dict_with_nested_recovery(self):
        """Test parsing nested recovery section."""
        config = NLSQConfig.from_dict(
            {
                "recovery": {
                    "enable": False,
                    "max_attempts": 5,
                }
            }
        )
        assert config.enable_recovery is False
        assert config.max_recovery_attempts == 5

    def test_from_dict_with_nested_hybrid_streaming(self):
        """Test parsing nested hybrid_streaming section."""
        config = NLSQConfig.from_dict(
            {
                "hybrid_streaming": {
                    "enable": False,
                    "normalize": False,
                    "normalization_strategy": "p0",
                    "warmup_iterations": 50,
                    "max_warmup_iterations": 200,
                    "warmup_learning_rate": 0.01,
                    "gauss_newton_max_iterations": 25,
                    "gauss_newton_tol": 1e-6,
                    "chunk_size": 100000,
                    "trust_region_initial": 2.0,
                    "regularization_factor": 1e-8,
                    "enable_checkpoints": False,
                    "checkpoint_frequency": 50,
                    "validate_numerics": False,
                }
            }
        )
        assert config.enable_hybrid_streaming is False
        assert config.hybrid_normalize is False
        assert config.hybrid_normalization_strategy == "p0"
        assert config.hybrid_warmup_iterations == 50
        assert config.hybrid_max_warmup_iterations == 200
        assert config.hybrid_warmup_learning_rate == 0.01
        assert config.hybrid_gauss_newton_max_iterations == 25
        assert config.hybrid_gauss_newton_tol == 1e-6
        assert config.hybrid_chunk_size == 100000
        assert config.hybrid_trust_region_initial == 2.0
        assert config.hybrid_regularization_factor == 1e-8
        assert config.hybrid_enable_checkpoints is False
        assert config.hybrid_checkpoint_frequency == 50
        assert config.hybrid_validate_numerics is False

    def test_from_dict_with_x_scale_map(self):
        """Test parsing x_scale_map from dict."""
        config = NLSQConfig.from_dict(
            {
                "x_scale": "jac",
                "x_scale_map": {"D0": 1e4, "alpha": 1.0},
            }
        )
        assert config.x_scale == "jac"
        assert config.x_scale_map == {"D0": 1e4, "alpha": 1.0}

    def test_from_dict_float_conversion(self):
        """Test that numeric values are properly converted to float."""
        config = NLSQConfig.from_dict(
            {
                "trust_region_scale": "2.5",  # string should be converted
                "tolerance": "1e-10",
            }
        )
        assert config.trust_region_scale == 2.5
        assert config.ftol == 1e-10


class TestNLSQConfigValidation:
    """Test NLSQConfig.validate method."""

    def test_valid_config_returns_empty_list(self):
        """Test that valid config returns no errors."""
        config = NLSQConfig()
        errors = config.validate()
        assert errors == []

    def test_invalid_loss_function(self):
        """Test validation catches invalid loss function."""
        config = NLSQConfig(loss="invalid_loss")
        errors = config.validate()
        assert len(errors) == 1
        assert "loss must be one of" in errors[0]

    def test_all_valid_loss_functions(self):
        """Test all valid loss functions pass validation."""
        valid_losses = ["linear", "soft_l1", "huber", "cauchy", "arctan"]
        for loss in valid_losses:
            config = NLSQConfig(loss=loss)
            errors = config.validate()
            assert errors == [], f"Loss '{loss}' should be valid"

    def test_invalid_trust_region_scale(self):
        """Test validation catches non-positive trust_region_scale."""
        config = NLSQConfig(trust_region_scale=0)
        errors = config.validate()
        assert any("trust_region_scale must be positive" in e for e in errors)

        config = NLSQConfig(trust_region_scale=-1.0)
        errors = config.validate()
        assert any("trust_region_scale must be positive" in e for e in errors)

    def test_invalid_tolerance_values(self):
        """Test validation catches non-positive tolerances."""
        config = NLSQConfig(ftol=0)
        errors = config.validate()
        assert any("ftol must be positive" in e for e in errors)

        config = NLSQConfig(xtol=-1e-8)
        errors = config.validate()
        assert any("xtol must be positive" in e for e in errors)

        config = NLSQConfig(gtol=0)
        errors = config.validate()
        assert any("gtol must be positive" in e for e in errors)

    def test_invalid_max_iterations(self):
        """Test validation catches non-positive max_iterations."""
        config = NLSQConfig(max_iterations=0)
        errors = config.validate()
        assert any("max_iterations must be positive" in e for e in errors)

    def test_invalid_chunk_sizes(self):
        """Test validation catches non-positive chunk sizes."""
        config = NLSQConfig(streaming_chunk_size=0)
        errors = config.validate()
        assert any("streaming_chunk_size must be positive" in e for e in errors)

        config = NLSQConfig(target_chunk_size=-100)
        errors = config.validate()
        assert any("target_chunk_size must be positive" in e for e in errors)

    def test_invalid_max_recovery_attempts(self):
        """Test validation catches negative max_recovery_attempts."""
        config = NLSQConfig(max_recovery_attempts=-1)
        errors = config.validate()
        assert any("max_recovery_attempts must be non-negative" in e for e in errors)

    def test_zero_recovery_attempts_valid(self):
        """Test zero recovery attempts is valid (disables recovery)."""
        config = NLSQConfig(max_recovery_attempts=0)
        errors = config.validate()
        assert not any("max_recovery_attempts" in e for e in errors)

    def test_invalid_hybrid_normalization_strategy(self):
        """Test validation catches invalid normalization strategy."""
        config = NLSQConfig(hybrid_normalization_strategy="invalid")
        errors = config.validate()
        assert any("hybrid_normalization_strategy must be one of" in e for e in errors)

    def test_all_valid_normalization_strategies(self):
        """Test all valid normalization strategies pass validation."""
        strategies = ["auto", "bounds", "p0", "none"]
        for strategy in strategies:
            config = NLSQConfig(hybrid_normalization_strategy=strategy)
            errors = config.validate()
            assert not any("hybrid_normalization_strategy" in e for e in errors)

    def test_invalid_hybrid_warmup_settings(self):
        """Test validation catches invalid hybrid warmup settings."""
        config = NLSQConfig(hybrid_warmup_iterations=0)
        errors = config.validate()
        assert any("hybrid_warmup_iterations must be positive" in e for e in errors)

        config = NLSQConfig(hybrid_max_warmup_iterations=-1)
        errors = config.validate()
        assert any("hybrid_max_warmup_iterations must be positive" in e for e in errors)

        config = NLSQConfig(hybrid_warmup_learning_rate=0)
        errors = config.validate()
        assert any("hybrid_warmup_learning_rate must be positive" in e for e in errors)

    def test_invalid_hybrid_gauss_newton_settings(self):
        """Test validation catches invalid Gauss-Newton settings."""
        config = NLSQConfig(hybrid_gauss_newton_max_iterations=0)
        errors = config.validate()
        assert any(
            "hybrid_gauss_newton_max_iterations must be positive" in e for e in errors
        )

        config = NLSQConfig(hybrid_gauss_newton_tol=-1e-8)
        errors = config.validate()
        assert any("hybrid_gauss_newton_tol must be positive" in e for e in errors)

    def test_invalid_hybrid_chunk_size(self):
        """Test validation catches invalid hybrid chunk size."""
        config = NLSQConfig(hybrid_chunk_size=0)
        errors = config.validate()
        assert any("hybrid_chunk_size must be positive" in e for e in errors)

    def test_multiple_validation_errors(self):
        """Test that multiple errors are all captured."""
        config = NLSQConfig(
            loss="invalid",
            ftol=0,
            max_iterations=-1,
            streaming_chunk_size=0,
        )
        errors = config.validate()
        assert len(errors) >= 4


class TestNLSQConfigIsValid:
    """Test NLSQConfig.is_valid method."""

    def test_is_valid_with_defaults(self):
        """Test default config is valid."""
        config = NLSQConfig()
        assert config.is_valid() is True

    def test_is_valid_with_custom_valid_config(self):
        """Test custom valid config is valid."""
        config = NLSQConfig(
            loss="huber",
            max_iterations=500,
            ftol=1e-6,
        )
        assert config.is_valid() is True

    def test_is_valid_with_invalid_config(self):
        """Test invalid config returns False."""
        config = NLSQConfig(loss="invalid")
        assert config.is_valid() is False


class TestNLSQConfigToDict:
    """Test NLSQConfig.to_dict method."""

    def test_to_dict_contains_all_fields(self):
        """Test to_dict includes all configuration fields."""
        config = NLSQConfig()
        d = config.to_dict()

        # Top-level fields
        assert "loss" in d
        assert "trust_region_scale" in d
        assert "max_iterations" in d
        assert "ftol" in d
        assert "xtol" in d
        assert "gtol" in d
        assert "x_scale" in d
        assert "x_scale_map" in d

        # Nested sections
        assert "diagnostics" in d
        assert "streaming" in d
        assert "stratified" in d
        assert "recovery" in d
        assert "hybrid_streaming" in d

    def test_to_dict_nested_diagnostics_structure(self):
        """Test diagnostics section structure."""
        config = NLSQConfig(enable_diagnostics=False)
        d = config.to_dict()
        assert d["diagnostics"]["enable"] is False

    def test_to_dict_nested_streaming_structure(self):
        """Test streaming section structure."""
        config = NLSQConfig(enable_streaming=False, streaming_chunk_size=25000)
        d = config.to_dict()
        assert d["streaming"]["enable"] is False
        assert d["streaming"]["chunk_size"] == 25000

    def test_to_dict_nested_stratified_structure(self):
        """Test stratified section structure."""
        config = NLSQConfig(enable_stratified=False, target_chunk_size=200000)
        d = config.to_dict()
        assert d["stratified"]["enable"] is False
        assert d["stratified"]["target_chunk_size"] == 200000

    def test_to_dict_nested_recovery_structure(self):
        """Test recovery section structure."""
        config = NLSQConfig(enable_recovery=False, max_recovery_attempts=5)
        d = config.to_dict()
        assert d["recovery"]["enable"] is False
        assert d["recovery"]["max_attempts"] == 5

    def test_to_dict_nested_hybrid_streaming_structure(self):
        """Test hybrid_streaming section structure."""
        config = NLSQConfig(
            enable_hybrid_streaming=False,
            hybrid_normalize=False,
            hybrid_normalization_strategy="p0",
            hybrid_warmup_iterations=50,
        )
        d = config.to_dict()
        hs = d["hybrid_streaming"]
        assert hs["enable"] is False
        assert hs["normalize"] is False
        assert hs["normalization_strategy"] == "p0"
        assert hs["warmup_iterations"] == 50

    def test_to_dict_roundtrip(self):
        """Test that to_dict -> from_dict preserves values.

        Note: ftol in to_dict maps to 'tolerance' in from_dict,
        so we need to manually adjust for roundtrip.
        """
        original = NLSQConfig(
            loss="huber",
            max_iterations=500,
            ftol=1e-6,
            enable_streaming=False,
            streaming_chunk_size=25000,
        )
        d = original.to_dict()

        # Adjust for YAML config naming convention
        # (to_dict uses 'ftol', from_dict expects 'tolerance')
        d["tolerance"] = d.pop("ftol")

        restored = NLSQConfig.from_dict(d)

        assert restored.loss == original.loss
        assert restored.max_iterations == original.max_iterations
        assert restored.ftol == original.ftol
        assert restored.enable_streaming == original.enable_streaming
        assert restored.streaming_chunk_size == original.streaming_chunk_size


class TestNLSQConfigValidationErrors:
    """Test validation error field is updated correctly."""

    def test_validation_errors_field_updated(self):
        """Test that _validation_errors is populated after validate()."""
        config = NLSQConfig(loss="invalid")
        _ = config.validate()
        assert len(config._validation_errors) > 0

    def test_validation_errors_cleared_on_revalidation(self):
        """Test that _validation_errors is updated on each validate() call."""
        config = NLSQConfig(loss="invalid")
        errors1 = config.validate()
        assert len(errors1) > 0

        # Change to valid
        config.loss = "soft_l1"
        errors2 = config.validate()
        assert errors2 == []
        assert config._validation_errors == []
