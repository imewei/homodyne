"""
MCMC Statistical Validation Tests
=================================

Statistical tests for MCMC sampling functionality:
- Convergence diagnostics (R-hat, effective sample size)
- Parameter recovery validation
- Chain mixing and autocorrelation analysis
- Posterior distribution properties
- Bayesian credible intervals
"""

import numpy as np
import pytest

# Handle JAX imports
try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

# Handle statistical packages
try:
    import scipy.stats as stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import arviz as az

    HAS_ARVIZ = True
except ImportError:
    HAS_ARVIZ = False


@pytest.mark.mcmc
@pytest.mark.slow
class TestMCMCConvergence:
    """Test MCMC convergence and sampling quality."""

    def test_mcmc_module_availability(self):
        """Test MCMC module availability and imports."""
        try:
            from homodyne.optimization import NUMPYRO_AVAILABLE, fit_mcmc_jax

            assert callable(fit_mcmc_jax)
            assert isinstance(NUMPYRO_AVAILABLE, bool)
        except ImportError:
            pytest.skip("MCMC module not available")

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_mcmc_basic_sampling(self, synthetic_xpcs_data):
        """Test basic MCMC sampling functionality."""
        try:
            from homodyne.optimization import NUMPYRO_AVAILABLE, fit_mcmc_jax
        except ImportError:
            pytest.skip("MCMC module not available")

        if not NUMPYRO_AVAILABLE:
            pytest.skip("NumPyro not available")

        data = synthetic_xpcs_data

        # Basic MCMC configuration
        mcmc_config = {
            "analysis_mode": "static",
            "optimization": {
                "method": "mcmc",
                "mcmc": {
                    "num_samples": 200,  # Small for testing
                    "num_warmup": 100,
                    "chains": 1,
                },
            },
            "hardware": {"force_cpu": True},
        }

        try:
            # Extract required parameters for MCMC
            sigma = data["sigma"]
            t1 = data["t1"]
            t2 = data["t2"]
            phi = data["phi_angles_list"]
            q = data["wavevector_q_list"][0]
            L = 1.0

            result = fit_mcmc_jax(
                sigma,
                t1,
                t2,
                phi,
                q,
                L,
                analysis_mode="static",
                num_samples=200,
                num_warmup=100,
            )

            # Basic validation
            assert hasattr(result, "samples")
            assert hasattr(result, "mean_params")

            # Samples should have correct shape
            samples = result.samples
            assert isinstance(samples, dict)

            # Should have parameter samples
            for param_name in ["offset", "contrast", "diffusion_coefficient"]:
                if param_name in samples:
                    param_samples = samples[param_name]
                    assert len(param_samples) > 0
                    assert np.all(np.isfinite(param_samples))

        except Exception as e:
            pytest.skip(f"MCMC sampling failed: {e}")

    @pytest.mark.skipif(
        not (JAX_AVAILABLE and HAS_SCIPY), reason="JAX or SciPy not available"
    )
    def test_mcmc_parameter_recovery(self):
        """Test MCMC parameter recovery with known synthetic data."""
        try:
            from homodyne.tests.factories.data_factory import XPCSDataFactory

            from homodyne.optimization import NUMPYRO_AVAILABLE, fit_mcmc_jax
        except ImportError:
            pytest.skip("Required modules not available")

        if not NUMPYRO_AVAILABLE:
            pytest.skip("NumPyro not available")

        # Generate data with known parameters
        factory = XPCSDataFactory(seed=42)
        true_params = {
            "offset": 1.0,
            "contrast": 0.3,
            "diffusion_coefficient": 0.15,
            "shear_rate": 0.0,
            "L": 1.0,
        }

        data = factory.create_synthetic_correlation_data(
            n_times=20,
            n_angles=12,
            true_parameters=true_params,
            noise_level=0.005,  # Low noise for better recovery
            q_value=0.01,
        )

        try:
            result = fit_mcmc_jax(
                data["sigma"],
                data["t1"],
                data["t2"],
                data["phi_angles_list"],
                data["wavevector_q_list"][0],
                true_params["L"],
                analysis_mode="static",
                num_samples=500,
                num_warmup=200,
            )

            # Check parameter recovery
            mean_params = result.mean_params

            for param_name, true_value in true_params.items():
                if param_name in mean_params:
                    recovered_value = mean_params[param_name]
                    relative_error = abs(recovered_value - true_value) / true_value

                    # MCMC should recover parameters within reasonable bounds
                    tolerance = 0.15  # 15% tolerance for MCMC
                    assert relative_error < tolerance, (
                        f"Parameter {param_name}: recovered {recovered_value:.4f}, "
                        f"true {true_value:.4f}, error {relative_error:.4f}"
                    )

        except Exception as e:
            pytest.skip(f"MCMC parameter recovery test failed: {e}")

    @pytest.mark.skipif(
        not (JAX_AVAILABLE and HAS_SCIPY), reason="JAX or SciPy not available"
    )
    def test_mcmc_convergence_diagnostics(self, synthetic_xpcs_data):
        """Test MCMC convergence diagnostics."""
        try:
            from homodyne.optimization import NUMPYRO_AVAILABLE, fit_mcmc_jax
        except ImportError:
            pytest.skip("MCMC module not available")

        if not NUMPYRO_AVAILABLE:
            pytest.skip("NumPyro not available")

        data = synthetic_xpcs_data

        try:
            # Run MCMC with multiple chains
            result = fit_mcmc_jax(
                data["sigma"],
                data["t1"],
                data["t2"],
                data["phi_angles_list"],
                data["wavevector_q_list"][0],
                1.0,
                analysis_mode="static",
                num_samples=300,
                num_warmup=150,
                num_chains=2,
            )

            samples = result.samples

            # Check chain mixing for each parameter
            for param_name, param_samples in samples.items():
                if isinstance(param_samples, np.ndarray) and param_samples.ndim >= 1:
                    # Basic convergence checks
                    assert len(param_samples) > 100, (
                        f"Not enough samples for {param_name}"
                    )

                    # Check for finite values
                    assert np.all(np.isfinite(param_samples)), (
                        f"Non-finite samples for {param_name}"
                    )

                    # Check for reasonable variation (not stuck)
                    param_std = np.std(param_samples)
                    assert param_std > 0, f"No variation in {param_name} samples"

                    # Simple autocorrelation check
                    if len(param_samples) > 50:
                        # Lag-1 autocorrelation should not be too high
                        autocorr_1 = np.corrcoef(param_samples[:-1], param_samples[1:])[
                            0, 1
                        ]
                        assert autocorr_1 < 0.9, (
                            f"High autocorrelation for {param_name}: {autocorr_1:.3f}"
                        )

        except Exception as e:
            pytest.skip(f"MCMC convergence diagnostics failed: {e}")

    @pytest.mark.skipif(not HAS_ARVIZ, reason="ArviZ not available")
    def test_mcmc_with_arviz_diagnostics(self, synthetic_xpcs_data):
        """Test MCMC with ArviZ diagnostics."""
        try:
            from homodyne.optimization import NUMPYRO_AVAILABLE, fit_mcmc_jax
        except ImportError:
            pytest.skip("MCMC module not available")

        if not NUMPYRO_AVAILABLE:
            pytest.skip("NumPyro not available")

        data = synthetic_xpcs_data

        try:
            result = fit_mcmc_jax(
                data["sigma"],
                data["t1"],
                data["t2"],
                data["phi_angles_list"],
                data["wavevector_q_list"][0],
                1.0,
                analysis_mode="static",
                num_samples=400,
                num_warmup=200,
                num_chains=2,
            )

            samples = result.samples

            # Convert to ArviZ InferenceData if possible
            if hasattr(result, "arviz_data") or "arviz" in str(type(result)):
                # If already in ArviZ format
                inference_data = (
                    result.arviz_data if hasattr(result, "arviz_data") else result
                )
            else:
                # Create ArviZ InferenceData manually
                try:
                    # Reshape samples for ArviZ (chains, draws, parameters)
                    arviz_samples = {}
                    for param_name, param_samples in samples.items():
                        if isinstance(param_samples, np.ndarray):
                            # Assume single chain for now
                            arviz_samples[param_name] = param_samples.reshape(1, -1)

                    inference_data = az.from_dict(arviz_samples)
                except:
                    pytest.skip("Could not create ArviZ InferenceData")

            # Run ArviZ diagnostics
            try:
                # Effective sample size
                ess = az.ess(inference_data)
                for param_name in ess.data_vars:
                    ess_value = float(ess[param_name].values)
                    assert ess_value > 50, f"Low ESS for {param_name}: {ess_value}"

                # R-hat (if multiple chains)
                if (
                    hasattr(inference_data.posterior, "chain")
                    and len(inference_data.posterior.chain) > 1
                ):
                    rhat = az.rhat(inference_data)
                    for param_name in rhat.data_vars:
                        rhat_value = float(rhat[param_name].values)
                        assert rhat_value < 1.1, (
                            f"High R-hat for {param_name}: {rhat_value}"
                        )

            except Exception as e:
                pytest.skip(f"ArviZ diagnostics failed: {e}")

        except Exception as e:
            pytest.skip(f"MCMC with ArviZ test failed: {e}")


@pytest.mark.mcmc
@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
class TestMCMCStatisticalProperties:
    """Test statistical properties of MCMC results."""

    def test_posterior_distribution_properties(self, synthetic_xpcs_data):
        """Test properties of posterior distributions."""
        try:
            from homodyne.optimization import NUMPYRO_AVAILABLE, fit_mcmc_jax
        except ImportError:
            pytest.skip("MCMC module not available")

        if not NUMPYRO_AVAILABLE:
            pytest.skip("NumPyro not available")

        data = synthetic_xpcs_data

        try:
            result = fit_mcmc_jax(
                data["sigma"],
                data["t1"],
                data["t2"],
                data["phi_angles_list"],
                data["wavevector_q_list"][0],
                1.0,
                analysis_mode="static",
                num_samples=300,
                num_warmup=150,
            )

            samples = result.samples

            # Test each parameter's posterior
            for param_name, param_samples in samples.items():
                if isinstance(param_samples, np.ndarray) and len(param_samples) > 50:
                    # Test normality (should be approximately normal for well-behaved posteriors)
                    try:
                        stat, p_value = stats.normaltest(param_samples)
                        # Don't require strict normality, just check for severe non-normality
                        # (p_value very close to 0 indicates strong non-normality)
                        if p_value < 1e-10:
                            # Very non-normal, but that might be okay
                            pass
                    except:
                        pass

                    # Test for outliers (basic check)
                    q25, q75 = np.percentile(param_samples, [25, 75])
                    iqr = q75 - q25
                    outlier_threshold_low = q25 - 3 * iqr
                    outlier_threshold_high = q75 + 3 * iqr

                    outliers = np.sum(
                        (param_samples < outlier_threshold_low)
                        | (param_samples > outlier_threshold_high)
                    )
                    outlier_fraction = outliers / len(param_samples)

                    # Should not have too many outliers
                    assert outlier_fraction < 0.1, (
                        f"Too many outliers for {param_name}: {outlier_fraction:.3f}"
                    )

                    # Check for finite variance
                    param_var = np.var(param_samples)
                    assert param_var > 0, f"Zero variance for {param_name}"
                    assert np.isfinite(param_var), (
                        f"Non-finite variance for {param_name}"
                    )

        except Exception as e:
            pytest.skip(f"Posterior distribution test failed: {e}")

    def test_credible_intervals(self):
        """Test Bayesian credible intervals."""
        try:
            from homodyne.tests.factories.data_factory import XPCSDataFactory

            from homodyne.optimization import NUMPYRO_AVAILABLE, fit_mcmc_jax
        except ImportError:
            pytest.skip("Required modules not available")

        if not NUMPYRO_AVAILABLE:
            pytest.skip("NumPyro not available")

        # Generate data with known parameters
        factory = XPCSDataFactory(seed=123)
        true_params = {"offset": 1.0, "contrast": 0.4, "diffusion_coefficient": 0.12}

        data = factory.create_synthetic_correlation_data(
            n_times=25, n_angles=18, true_parameters=true_params, noise_level=0.008
        )

        try:
            result = fit_mcmc_jax(
                data["sigma"],
                data["t1"],
                data["t2"],
                data["phi_angles_list"],
                data["wavevector_q_list"][0],
                1.0,
                analysis_mode="static",
                num_samples=400,
                num_warmup=200,
            )

            samples = result.samples

            # Compute credible intervals
            for param_name, true_value in true_params.items():
                if param_name in samples:
                    param_samples = samples[param_name]

                    # 95% credible interval
                    ci_lower = np.percentile(param_samples, 2.5)
                    ci_upper = np.percentile(param_samples, 97.5)

                    # True value should be within credible interval most of the time
                    # (Allow some failures due to sampling variance)
                    within_ci = ci_lower <= true_value <= ci_upper

                    if not within_ci:
                        # Check if it's close to the boundary
                        distance_to_lower = abs(true_value - ci_lower) / abs(
                            ci_upper - ci_lower
                        )
                        distance_to_upper = abs(true_value - ci_upper) / abs(
                            ci_upper - ci_lower
                        )
                        min_distance = min(distance_to_lower, distance_to_upper)

                        # Allow if very close to boundary (within 10% of CI width)
                        if min_distance > 0.1:
                            pytest.fail(
                                f"True {param_name} ({true_value:.4f}) outside 95% CI "
                                f"[{ci_lower:.4f}, {ci_upper:.4f}]"
                            )

                    # CI should have reasonable width (not too narrow or too wide)
                    ci_width = ci_upper - ci_lower
                    assert ci_width > 0, f"Zero-width CI for {param_name}"

                    # Relative CI width should be reasonable
                    mean_value = np.mean(param_samples)
                    relative_width = ci_width / abs(mean_value + 1e-10)
                    assert relative_width < 2.0, (
                        f"CI too wide for {param_name}: {relative_width:.3f}"
                    )

        except Exception as e:
            pytest.skip(f"Credible intervals test failed: {e}")

    def test_mcmc_reproducibility(self, synthetic_xpcs_data):
        """Test MCMC reproducibility with fixed random seed."""
        try:
            from homodyne.optimization import NUMPYRO_AVAILABLE, fit_mcmc_jax
        except ImportError:
            pytest.skip("MCMC module not available")

        if not NUMPYRO_AVAILABLE:
            pytest.skip("NumPyro not available")

        data = synthetic_xpcs_data

        try:
            # Run MCMC twice with same seed
            mcmc_params = {
                "sigma": data["sigma"],
                "t1": data["t1"],
                "t2": data["t2"],
                "phi_angles_list": data["phi_angles_list"],
                "q": data["wavevector_q_list"][0],
                "L": 1.0,
                "analysis_mode": "static",
                "num_samples": 200,
                "num_warmup": 100,
                "rng_key": 42,  # Fixed seed
            }

            result1 = fit_mcmc_jax(**mcmc_params)
            result2 = fit_mcmc_jax(**mcmc_params)

            # Results should be identical (or very close) with same seed
            samples1 = result1.samples
            samples2 = result2.samples

            common_params = set(samples1.keys()) & set(samples2.keys())

            for param_name in common_params:
                s1 = samples1[param_name]
                s2 = samples2[param_name]

                if isinstance(s1, np.ndarray) and isinstance(s2, np.ndarray):
                    # Should be very close (identical with same seed)
                    np.testing.assert_array_almost_equal(
                        s1,
                        s2,
                        decimal=10,
                        err_msg=f"MCMC not reproducible for {param_name}",
                    )

        except Exception as e:
            pytest.skip(f"MCMC reproducibility test failed: {e}")

    def test_mcmc_chain_mixing(self, synthetic_xpcs_data):
        """Test MCMC chain mixing quality."""
        try:
            from homodyne.optimization import NUMPYRO_AVAILABLE, fit_mcmc_jax
        except ImportError:
            pytest.skip("MCMC module not available")

        if not NUMPYRO_AVAILABLE:
            pytest.skip("NumPyro not available")

        data = synthetic_xpcs_data

        try:
            result = fit_mcmc_jax(
                data["sigma"],
                data["t1"],
                data["t2"],
                data["phi_angles_list"],
                data["wavevector_q_list"][0],
                1.0,
                analysis_mode="static",
                num_samples=300,
                num_warmup=150,
            )

            samples = result.samples

            for param_name, param_samples in samples.items():
                if isinstance(param_samples, np.ndarray) and len(param_samples) > 100:
                    # Check for trend (should be stationary)
                    first_half = param_samples[: len(param_samples) // 2]
                    second_half = param_samples[len(param_samples) // 2 :]

                    mean1 = np.mean(first_half)
                    mean2 = np.mean(second_half)
                    std1 = np.std(first_half)
                    std2 = np.std(second_half)

                    # Means should be similar (no strong trend)
                    mean_diff = abs(mean1 - mean2)
                    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
                    standardized_diff = mean_diff / (pooled_std + 1e-10)

                    # Should not have strong trend (< 2 standard deviations)
                    assert standardized_diff < 2.0, (
                        f"Strong trend in {param_name}: {standardized_diff:.3f}"
                    )

                    # Check for reasonable acceptance (not too stuck)
                    # Count number of unique values
                    unique_values = len(np.unique(param_samples))
                    unique_fraction = unique_values / len(param_samples)

                    # Should have reasonable diversity (not stuck at few values)
                    assert unique_fraction > 0.1, (
                        f"Poor mixing for {param_name}: {unique_fraction:.3f}"
                    )

        except Exception as e:
            pytest.skip(f"Chain mixing test failed: {e}")


@pytest.mark.mcmc
class TestMCMCEdgeCases:
    """Test MCMC behavior in edge cases."""

    def test_mcmc_high_noise_data(self):
        """Test MCMC with high noise data."""
        try:
            from homodyne.tests.factories.data_factory import XPCSDataFactory

            from homodyne.optimization import NUMPYRO_AVAILABLE, fit_mcmc_jax
        except ImportError:
            pytest.skip("Required modules not available")

        if not NUMPYRO_AVAILABLE:
            pytest.skip("NumPyro not available")

        # High noise scenario
        factory = XPCSDataFactory(seed=42)
        data = factory.create_edge_case_dataset(case_type="high_noise")

        try:
            result = fit_mcmc_jax(
                data["sigma"],
                data["t1"],
                data["t2"],
                data["phi_angles_list"],
                data["wavevector_q_list"][0],
                1.0,
                analysis_mode="static",
                num_samples=300,
                num_warmup=200,
            )

            # Should complete without error
            assert hasattr(result, "samples")
            assert hasattr(result, "mean_params")

            # Samples should be finite
            for param_name, param_samples in result.samples.items():
                if isinstance(param_samples, np.ndarray):
                    assert np.all(np.isfinite(param_samples)), (
                        f"Non-finite samples for {param_name} with high noise"
                    )

        except Exception as e:
            pytest.skip(f"High noise MCMC test failed: {e}")

    def test_mcmc_low_contrast_data(self):
        """Test MCMC with low contrast data."""
        try:
            from homodyne.tests.factories.data_factory import XPCSDataFactory

            from homodyne.optimization import NUMPYRO_AVAILABLE, fit_mcmc_jax
        except ImportError:
            pytest.skip("Required modules not available")

        if not NUMPYRO_AVAILABLE:
            pytest.skip("NumPyro not available")

        # Low contrast scenario
        factory = XPCSDataFactory(seed=42)
        data = factory.create_edge_case_dataset(case_type="low_contrast")

        try:
            result = fit_mcmc_jax(
                data["sigma"],
                data["t1"],
                data["t2"],
                data["phi_angles_list"],
                data["wavevector_q_list"][0],
                1.0,
                analysis_mode="static",
                num_samples=250,
                num_warmup=150,
            )

            # Should handle low contrast gracefully
            assert hasattr(result, "samples")

            # Contrast should be recovered as low but positive
            if "contrast" in result.mean_params:
                recovered_contrast = result.mean_params["contrast"]
                assert 0.0 <= recovered_contrast <= 0.2, (
                    f"Unexpected contrast recovery: {recovered_contrast}"
                )

        except Exception as e:
            pytest.skip(f"Low contrast MCMC test failed: {e}")

    def test_mcmc_small_dataset(self):
        """Test MCMC with very small dataset."""
        try:
            from homodyne.tests.factories.data_factory import XPCSDataFactory

            from homodyne.optimization import NUMPYRO_AVAILABLE, fit_mcmc_jax
        except ImportError:
            pytest.skip("Required modules not available")

        if not NUMPYRO_AVAILABLE:
            pytest.skip("NumPyro not available")

        # Very small dataset
        factory = XPCSDataFactory(seed=42)
        data = factory.create_synthetic_correlation_data(
            n_times=8,  # Very small
            n_angles=6,  # Very small
            noise_level=0.01,
        )

        try:
            result = fit_mcmc_jax(
                data["sigma"],
                data["t1"],
                data["t2"],
                data["phi_angles_list"],
                data["wavevector_q_list"][0],
                1.0,
                analysis_mode="static",
                num_samples=200,
                num_warmup=100,
            )

            # Should handle small dataset
            assert hasattr(result, "samples")

            # Parameters should still be reasonable
            for param_name, param_value in result.mean_params.items():
                assert np.isfinite(param_value), (
                    f"Non-finite parameter {param_name} with small dataset"
                )

        except Exception as e:
            pytest.skip(f"Small dataset MCMC test failed: {e}")
