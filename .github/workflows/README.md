# GitHub Workflows

Consolidated CI/CD workflows for the Homodyne project.

## Workflows

### ci.yml - Continuous Integration

**Triggers:** Push to main/develop, Pull Requests

**Jobs:**

- Lint & format (ruff + black)
- **Test CPU (multi-platform):** Ubuntu, Windows, macOS with Python 3.12 (CPU-only JAX)
- **Test GPU (disabled):** Linux with CUDA support (requires self-hosted runner)
- Build package
- Deploy documentation (main branch only)

**Duration:** ~10-15 minutes

**Platform Support:**
- All tests run on CPU-only JAX (cross-platform compatible)
- GPU tests available but disabled by default (requires CUDA-enabled runner)

______________________________________________________________________

### quality.yml - Code Quality

**Triggers:** Push to main, Weekly (Sundays at 3 AM UTC), Manual

**Jobs:**

- Security scan (bandit, safety, pip-audit)
- Type checking (mypy)
- Test coverage (pytest-cov)
- Complexity analysis (radon, xenon)

**Duration:** ~20 minutes

______________________________________________________________________

### release.yml - Release Automation

**Triggers:** Version tags (v\*.*.*), Manual

**Jobs:**

- Validate version format
- Run full test suite
- Build distributions
- Publish to PyPI (trusted publishing)
- Create GitHub release with changelog

**Duration:** ~25 minutes

______________________________________________________________________

## Usage

### Triggering CI

CI runs automatically on every push and pull request.

### Running Quality Checks

```bash
# Via GitHub UI: Actions → Code Quality → Run workflow
```

### Creating a Release

```bash
git tag v1.2.3
git push origin v1.2.3
```

______________________________________________________________________

## Configuration

**Python Version:** 3.12 (tested: 3.12.3)

**Key Dependencies:**

- ruff ≥0.13.3
- black ≥25.9.0
- pytest ≥8.4.2
- mypy ≥1.18.2

**Secrets:** PyPI trusted publishing (configured in repository settings)

______________________________________________________________________

## Artifacts

- **CI:** Package distributions, coverage reports
- **Quality:** Security reports, coverage HTML, complexity metrics
- **Release:** Distributions (uploaded to PyPI and GitHub)
