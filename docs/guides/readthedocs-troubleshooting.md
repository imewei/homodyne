# ReadTheDocs Troubleshooting Guide

Comprehensive troubleshooting for Homodyne documentation deployment on ReadTheDocs.

## Quick Diagnostics

### 1. Check RTD Build Status

**Access:** https://readthedocs.org/dashboard/homodyne/

**Status Indicators:**

- 🟢 Green: Build successful
- 🟡 Yellow: Building in progress
- 🔴 Red: Build failed
- ⚫ Gray: No builds yet

**Check Latest Build:**

1. Click "Builds" in RTD Dashboard
1. Look at timestamp of most recent build
1. Click build ID to view detailed logs
1. Scroll to bottom for final status and error messages

### 2. Run Local Build

**Test Sphinx build locally before pushing:**

```bash
# Navigate to docs directory
cd docs

# Clean previous builds
make clean

# Build HTML
make html

# Check output
ls -la _build/html/
```

**If local build fails:**

- Fix the issue before pushing to GitHub
- Ensure all dependencies are installed: `pip install -r requirements.txt`

### 3. Check GitHub Actions

**Access:** https://github.com/imewei/homodyne/actions/workflows/docs.yml

**This gives quick feedback before RTD:**

1. Click latest "Documentation" workflow run
1. See build status (✓ or ✗)
1. Expand "Build Documentation" step to see details
1. Check "Link checker" step for broken links

## Common Issues and Solutions

### Issue 1: Build Fails with "ModuleNotFoundError"

**Error Message:**

```
ModuleNotFoundError: No module named 'homodyne'
```

**Root Cause:** The homodyne package is not installed in the RTD environment.

**Solution:**

1. **Verify `.readthedocs.yaml`:**

   ```yaml
   build:
     jobs:
       post_install:
         - pip install -e .
   ```

1. **Verify `docs/requirements.txt`:**

   - Should NOT contain `homodyne` (installed via pyproject.toml)
   - Should contain: `sphinx`, `sphinx-rtd-theme`, etc.

1. **Test locally:**

   ```bash
   pip install -e .
   cd docs
   make html
   ```

1. **Rebuild on RTD:**

   - Go to RTD Dashboard → Builds
   - Click "Build Version" next to your branch
   - Wait for completion and check logs

### Issue 2: Build Fails with "No module named 'jax'"

**Error Message:**

```
ModuleNotFoundError: No module named 'jax'
```

**Root Cause:** JAX is not installed in RTD environment (may not be in
docs/requirements.txt or pyproject.toml).

**Solution:**

1. **Add JAX to `docs/requirements.txt`:**

   ```
   # Required for building (must match package dependencies)
   jax>=0.8.0,<0.9.0
   jaxlib>=0.8.0,<0.9.0
   ```

1. **Verify it's CPU-only:**

   - Do NOT use `jax[cuda12-local]` (RTD is CPU-only)
   - Plain `jax` defaults to CPU backend (correct for RTD)

1. **Rebuild:**

   - Push changes or manually rebuild on RTD

### Issue 3: "sphinx-rtd-theme" Not Found

**Error Message:**

```
Extension error:
Could not import extension sphinx_rtd_theme (exception: No module named 'sphinx_rtd_theme')
```

**Root Cause:** Theme not listed in `docs/requirements.txt`.

**Solution:**

1. **Verify `docs/requirements.txt` contains:**

   ```
   sphinx-rtd-theme>=2.0.0,<3.0.0
   ```

1. **Verify `docs/conf.py` contains:**

   ```python
   html_theme = 'sphinx_rtd_theme'
   ```

1. **Rebuild on RTD**

### Issue 4: Build Timeout (>15 minutes)

**Error Message:**

```
Build exceeded time limit of 900 seconds
```

**Root Cause:** Build takes too long, likely due to:

- Large datasets being processed
- Slow autodoc generation
- Many large images or diagrams

**Solutions:**

**Option A: Optimize Sphinx Build**

1. Remove unnecessary autodoc sections
1. Disable image generation if not critical
1. Move large examples to separate pages

**Option B: Increase Timeout**

1. Go to RTD Admin → Advanced Settings
1. Set "Build timeout" to 1200 (20 minutes)
1. Only increase if build is genuinely useful

**Option C: Disable Expensive Features** In `docs/conf.py`:

```python
# Only generate autodoc for public APIs
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': False,  # Skip undocumented members
    'show-inheritance': True,
}
```

### Issue 5: Documentation Not Updating After Push

**Symptom:** Changes pushed to GitHub but RTD still shows old version.

**Diagnosis Steps:**

1. **Check GitHub Webhook:**

   ```
   GitHub Settings → Webhooks
   ```

   - Should have entry from `readthedocs.com` or `readthedocs.org`
   - Status should be green (successful)
   - Click to see recent deliveries

1. **Check RTD Build:**

   - Go to RTD Dashboard → Builds
   - Check timestamp of latest build
   - Does it match your push time?

1. **Check Browser Cache:**

   - Force refresh: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (Mac)
   - Check in incognito window
   - Clear browser cookies for homodyne.readthedocs.io

**Solution:**

If webhook exists and browser cache cleared:

1. **Manual Rebuild on RTD:**

   - Dashboard → Builds
   - Click "Build Version" next to your branch
   - Wait 2-5 minutes

1. **If Still Doesn't Update:**

   - Check build logs for errors
   - Verify push was successful: `git log --oneline origin/main` (should show your
     commit)
   - Check RTD email for build notifications

1. **Recreate Webhook (Last Resort):**

   - RTD Admin → Integrations
   - Remove old integration
   - Click "Connect with GitHub"
   - Follow OAuth flow
   - RTD auto-creates new webhook

### Issue 6: Search Not Working

**Symptom:** Search box shows but returns no results.

**Common Causes:**

- Build failed (search index not created)
- RTD Free tier limitation (search disabled)
- Theme not properly configured

**Solutions:**

1. **Verify Build Succeeded:**

   - Check RTD Dashboard for red indicators
   - Review latest build log for errors

1. **Check RTD Tier:**

   - Free tier has limitations
   - Professional tier has better search
   - Consider upgrading if search is critical

1. **Verify Sphinx Configuration:** In `docs/conf.py`:

   ```python
   html_search_language = 'en'
   ```

1. **Rebuild Search Index:**

   - Wait 24+ hours after successful build
   - Search index updates after stable builds

### Issue 7: Theme Not Applied or Looks Wrong

**Symptom:** Documentation displays in default theme, not RTD theme.

**Causes:**

- Theme CSS not loaded
- Sphinx version incompatible
- Browser cache issue

**Solutions:**

1. **Verify Configuration:** Check `docs/conf.py`:

   ```python
   html_theme = 'sphinx_rtd_theme'

   # Optional: theme options
   html_theme_options = {
       'logo_only': False,
       'display_version': True,
       'prev_next_buttons_location': 'bottom',
       'style_external_links': False,
       'vcs_pageview_mode': '',
       'style_nav_header_background': '#2980B9',
   }
   ```

1. **Verify Theme Installation:**

   ```bash
   pip install sphinx-rtd-theme>=2.0.0
   ```

1. **Rebuild:**

   - Local: `cd docs && make clean && make html`
   - RTD: Manual rebuild via dashboard

1. **Clear Caches:**

   - Browser cache (Ctrl+Shift+Del)
   - RTD cache (Dashboard → Builds → Clear Cache)

### Issue 8: Broken Links in Link Checker

**Symptom:** GitHub Actions reports broken links in documentation.

**Check Results:**

- Action artifacts: Download `linkcheck-results`
- File: `output.txt` lists all broken links

**Common Issues:**

| Issue | Solution | |-------|----------| | External links offline | Mark as `ignore` in
Sphinx config | | Broken internal links | Fix file paths or anchor names | | Links to
GitHub that require auth | Mark as ignored (known issue) | | Links that sometimes
redirect | May be transient, recheck |

**Fix Broken Links:**

1. **Identify broken link:**

   ```
   output.txt:
   https://example.com/page [broken] - 404 Not Found
   docs/file.rst: line 42
   ```

1. **Fix in source file:**

   ```rst
   .. _correct-link:

   `Corrected Link <https://correct-url.com>`_
   ```

1. **Ignore external links (if needed):** In `docs/conf.py`:

   ```python
   linkcheck_ignore = [
       'https://zenodo.org/.*',  # Sometimes slow
       'https://doi.org/.*',      # External DOI links
   ]
   ```

### Issue 9: PDF/ePub Builds Missing

**Symptom:** PDF or ePub downloads not available on RTD.

**Solutions:**

1. **Enable in RTD Settings:**

   - Admin → Advanced Settings
   - Under "Formats":
     - Check "PDF"
     - Check "ePub"

1. **Verify Sphinx Configuration:** Check `docs/conf.py` has:

   ```python
   latex_elements = {
       'papersize': 'letterpaper',
       'pointsize': '10pt',
       'fontpkg': r'\usepackage[utf8]{inputenc}',
   }
   ```

1. **Test Locally:**

   ```bash
   cd docs
   make pdf      # Requires LaTeX
   make epub
   ```

1. **If Local PDF Build Fails:**

   - May require LaTeX installation: `apt-get install texlive-latex-base`
   - PDF might not build on RTD (OK - feature limitation)

### Issue 10: Version Selector Not Working

**Symptom:** Version dropdown shows but doesn't switch versions.

**Solutions:**

1. **Ensure Multiple Versions Exist:**

   - RTD Admin → Versions
   - At least 2 versions should be "Active"
   - Try enabling an older version/tag

1. **Verify Default Version:**

   - RTD Admin → Versions
   - Click version to edit
   - Set as "Default"

1. **Build All Versions:**

   - Some versions may not have builds
   - Click "Build" next to each version

## Performance Troubleshooting

### Slow Documentation Builds

**Benchmark:**

- Target: 2-5 minutes
- Acceptable: \<10 minutes
- Problem: >15 minutes (hits timeout)

**Optimization Steps:**

1. **Profile Sphinx Build:**

   ```bash
   cd docs
   sphinx-build -b html -v -j 4 . _build/html
   # Check output for slow steps
   ```

1. **Disable Unnecessary Extensions:** In `docs/conf.py`, comment out:

   - Unused Sphinx extensions
   - Heavy autodoc configurations
   - Slow external integrations

1. **Optimize Images:**

   - Compress images: `optipng`, `jpegoptim`
   - Use WebP format for modern browsers
   - Limit image dimensions

1. **Parallel Builds:** Ensure `.readthedocs.yaml` uses:

   ```yaml
   build:
     jobs:
       post_build:
         - sphinx-build -b html -j auto docs/ docs/_build/html
   ```

## Advanced Debugging

### Enable Verbose Logging on RTD

Unfortunately, RTD doesn't expose full build logs. To get more info:

1. **GitHub Actions Provides More Details:**

   - Run: `make html` and `make linkcheck`
   - See full Sphinx output
   - Better for debugging

1. **Request Build Logs:**

   - RTD Premium supports log downloads
   - Free tier shows limited logs

### Local Full Build

Replicate RTD build locally:

```bash
# Fresh environment (recommended)
python -m venv /tmp/rtd_test
source /tmp/rtd_test/bin/activate

# Install like RTD does
pip install -r docs/requirements.txt
pip install -e .

# Build exactly like RTD
cd docs
python -m sphinx -b html -W . _build/html
```

The `-W` flag converts warnings to errors (helps catch issues RTD might miss).

## Escalation Path

If you can't resolve the issue:

1. **Check GitHub Issues:** https://github.com/imewei/homodyne/issues

   - Search for similar issues
   - Create new issue with:
     - RTD build log excerpt
     - GitHub Actions workflow output
     - Local build output

1. **ReadTheDocs Support:**

   - https://docs.readthedocs.io/en/stable/support/

1. **Community:**

   - Sphinx documentation: https://www.sphinx-doc.org/
   - RTD community: https://github.com/readthedocs

## Prevention Tips

To avoid common issues:

1. **Local Testing:**

   - Always run `make html` before pushing
   - Check for warnings/errors

1. **GitHub Actions:**

   - Review Actions workflow output on PR
   - Fix issues before merging

1. **Regular Monitoring:**

   - Check RTD Dashboard weekly
   - Inspect build logs monthly
   - Monitor external link health

1. **Keep Dependencies Updated:**

   - Update Sphinx quarterly
   - Test after updates locally

1. **Documentation Review:**

   - Review docs before each release
   - Fix warnings in docs (don't suppress)

## Checklist for New Deployments

- [ ] `.readthedocs.yaml` is valid YAML
- [ ] `docs/requirements.txt` contains all dependencies
- [ ] `docs/conf.py` is syntactically correct
- [ ] Local `make html` succeeds with 0 errors
- [ ] GitHub Actions docs workflow passes
- [ ] RTD project imported and configured
- [ ] First RTD build succeeded
- [ ] Documentation appears at https://homodyne.readthedocs.io
- [ ] Search functionality works
- [ ] Version selector shows at least 2 versions
- [ ] PDF/ePub downloads available

## Related Documentation

- GitHub Actions Workflow: `.github/workflows/docs.yml`
- Sphinx Configuration: `docs/conf.py`
- Dependencies: `docs/requirements.txt`
- ReadTheDocs Config: `.readthedocs.yaml`
- Setup Guide: `docs/guides/readthedocs-setup.md`
