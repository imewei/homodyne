# ReadTheDocs Deployment Setup Guide

This guide provides step-by-step instructions for deploying Homodyne documentation to ReadTheDocs (RTD).

## Overview

ReadTheDocs is a continuous documentation deployment platform that automatically builds and hosts your documentation whenever you push changes to GitHub.

**Current Status:** Ready for ReadTheDocs deployment
- **Configuration File:** `.readthedocs.yaml` (already configured)
- **Sphinx Setup:** Complete with RTD theme
- **GitHub Integration:** Requires manual web interface setup

## Prerequisites

Before starting, ensure you have:

1. A GitHub account with access to the `homodyne` repository
2. A ReadTheDocs account (free at https://readthedocs.org)
3. Admin access to the GitHub repository

## Step-by-Step Setup Instructions

### Step 1: Import Project into ReadTheDocs

1. **Visit ReadTheDocs:**
   - Go to https://readthedocs.org
   - Sign in with your GitHub account (or create a free account)

2. **Import the Homodyne Repository:**
   - Click "Import a Project" or go to https://readthedocs.org/dashboard/import/
   - Select "Import from GitHub"
   - Search for `homodyne` or navigate to `imewei/homodyne`
   - Click "Create" to import the repository

3. **Verify Initial Configuration:**
   - Project Name: `homodyne`
   - Project URL: Will be assigned as `homodyne.readthedocs.io`
   - Repository: `imewei/homodyne`
   - Repository Type: Git

### Step 2: Configure Project Settings

After importing, configure these settings:

1. **Navigate to Project Dashboard:**
   - Log into ReadTheDocs
   - Go to https://readthedocs.org/dashboard/homodyne/

2. **Admin → Settings:**
   - **Project Details:**
     - Name: `homodyne`
     - Slug: `homodyne`
     - Description: "JAX-first homodyne scattering analysis for XPCS under nonequilibrium conditions"
     - Repository: `imewei/homodyne`
     - Repository Type: `Git`
     - Language: `English`
     - Programming Language: `Python`

   - **Default Version:**
     - Default Branch: `main`
     - Default Version: `latest`

   - **Build Settings:**
     - Check: "Build on commit to main branch"
     - Check: "Build pull request previews"
     - Build Timeout: `900` seconds (15 minutes)
     - Privacy Level: `Public`

### Step 3: Configure Advanced Settings

1. **Admin → Advanced Settings:**

   - **Build Configuration:**
     - Build jobs in use: Check all selected formats
     - Auto-cancel builds: Enable (cancel old builds on new commits)
     - Architecture: Keep default (x86-64)

   - **Documentation Formats:**
     - HTML: Enabled (default)
     - PDF: Enabled
     - ePub: Enabled

   - **Install Dependencies:**
     - Requirements file: `docs/requirements.txt`
     - Use Python Virtual Environments: Enabled (default)

   - **Python Settings:**
     - Python Version: 3.12
     - Install Project: Use setup.py/pyproject.toml

   - **Additional Build Settings:**
     - Build timeout: 900 seconds (15 minutes)
     - Number of builds to keep: 3
     - Show version warning: Enabled

### Step 4: Configure Webhook (Optional - Usually Automatic)

ReadTheDocs usually auto-configures the GitHub webhook. Verify it's set up:

1. **GitHub Webhook Verification:**
   - Go to your GitHub repository: https://github.com/imewei/homodyne
   - Settings → Webhooks
   - Look for a webhook from `readthedocs.com` or `readthedocs.org`
   - Verify it's active

2. **If Webhook Is Missing:**
   - Go to ReadTheDocs Admin → Integrations
   - Click "Integrate with ReadTheDocs"
   - Follow the GitHub OAuth flow
   - ReadTheDocs will auto-create the webhook

### Step 5: Configure Version Control

1. **Admin → Versions:**

   - **Active Versions:**
     - Enable `main` (latest development version)
     - Enable `latest` (always points to newest release tag)

   - **Tag Versions:**
     - Pattern: `v*` (matches tags like v2.0.0, v3.0.0, etc.)
     - Auto-create from tags: Enabled

2. **Default Version:**
   - Set `latest` as the default version users see

### Step 6: Configure Build Notifications (Optional)

1. **Admin → Notifications:**

   - **Email Notifications:**
     - Build failures: Your email
     - Build status changes: Enable

   - **Slack/Discord Integration (Optional):**
     - Click "Add Integration"
     - Enter webhook URL for Slack/Discord
     - Test the integration

## Verification Checklist

After configuration, verify the setup works:

### Manual Build Test

1. **Trigger Initial Build:**
   - Go to ReadTheDocs Dashboard: https://readthedocs.org/dashboard/homodyne/
   - Click "Build Version" next to `latest` or `main`
   - Wait for build to complete (usually 2-5 minutes)
   - Check build logs for errors

2. **Verify Documentation:**
   - Visit https://homodyne.readthedocs.io
   - Navigate main menu items:
     - User Guide ✓
     - API Reference ✓
     - Developer Guide ✓
     - Configuration Templates ✓
     - Advanced Topics ✓
     - Troubleshooting ✓
   - Test search functionality (top-right magnifying glass)
   - Test version selector (bottom-left)

### Automated Deployment Test

1. **Make a Test Commit:**
   ```bash
   # Make a minor change to documentation
   echo "Test line" >> docs/index.rst
   git add docs/index.rst
   git commit -m "test: verify readthedocs auto-deployment"
   git push origin main
   ```

2. **Verify Auto-Build:**
   - Check ReadTheDocs Dashboard within 30 seconds
   - Should see "Building" status for the new commit
   - Wait for build to complete (2-5 minutes)
   - Verify updated documentation appears at https://homodyne.readthedocs.io

3. **Revert Test Commit:**
   ```bash
   git reset --soft HEAD~1
   git restore docs/index.rst
   git commit -m "test: revert readthedocs verification commit"
   git push origin main
   ```

## Configuration File Reference

The project uses `.readthedocs.yaml` v2 configuration. See current configuration:

```yaml
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.12"
  jobs:
    post_install:
      - pip install -e .

python:
  install:
    - requirements: docs/requirements.txt
    - method: pip
      path: .

sphinx:
  configuration: docs/conf.py
  fail_on_warning: false

formats:
  - pdf
  - epub
```

**Key Settings:**
- **OS:** Ubuntu 22.04 (stable, well-tested)
- **Python:** 3.12 (matches project requirements)
- **Sphinx Config:** `docs/conf.py` (standard location)
- **Build Formats:** HTML (default), PDF, ePub
- **Warnings:** Non-fatal (build succeeds even with warnings)

## Troubleshooting

### Build Failures

**Common Issues and Solutions:**

| Issue | Cause | Solution |
|-------|-------|----------|
| "ModuleNotFoundError: No module named 'homodyne'" | Package not installed | Ensure `pip install -e .` in post_install |
| "Error when building documentation" | Missing dependencies | Check `docs/requirements.txt` completeness |
| "Cannot import from jax" | JAX not in docs requirements | Add `jax>=0.8.0` to `docs/requirements.txt` |
| "Build takes >15 minutes" | Timeout | Optimize Sphinx build or increase timeout to 20 minutes |
| "Theme not found: sphinx_rtd_theme" | Theme not installed | Verify `sphinx-rtd-theme` in `docs/requirements.txt` |

### Documentation Not Updating

**If changes don't appear after push:**

1. **Check Build Status:**
   - Go to ReadTheDocs Dashboard
   - Look for "Building..." status
   - Check build logs for errors

2. **Manual Rebuild:**
   - Click "Build Version" next to your branch/version
   - Wait for build to complete
   - Clear browser cache (Ctrl+Shift+Del) and reload

3. **Check Webhook:**
   - Verify GitHub webhook is active (GitHub Settings → Webhooks)
   - Check ReadTheDocs Integrations page

### Search Not Working

**Common Causes:**
- HTML build failed
- Elasticsearch index not built
- RTD Free tier limitation

**Solutions:**
- Rebuild documentation
- Check build logs for errors
- Search works best after 24 hours of stable builds

## GitHub Actions Integration

The project also includes a GitHub Actions workflow (`.github/workflows/docs.yml`) that:

1. **Builds documentation on every PR and push to main/develop**
2. **Runs Sphinx link checker to find broken links**
3. **Generates build reports and comments on PRs**
4. **Stores build artifacts for inspection**

This provides quick feedback on documentation changes before ReadTheDocs deployment.

## Monitoring and Maintenance

### Regular Checks

- **Weekly:** Review ReadTheDocs build status
- **Monthly:** Check link checker results
- **Per Release:** Build and test PDF/ePub formats

### Version Management

ReadTheDocs maintains multiple versions:
- `latest` - Always points to newest release tag
- `main` - Development version (bleeding edge)
- Release tags (v2.0.0, v2.1.0, etc.)

Users can switch versions using the version selector (bottom-left of any page).

## Next Steps

After successful ReadTheDocs setup:

1. **Update README.md** with documentation badge
2. **Monitor first few builds** for warnings/errors
3. **Optimize build performance** if needed
4. **Configure email notifications** for build failures
5. **Test version switching** functionality
6. **Document any RTD-specific customizations**

## Additional Resources

- **ReadTheDocs Official Guide:** https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html
- **Configuration File Reference:** https://docs.readthedocs.io/en/stable/config-file/v2.html
- **Sphinx Documentation:** https://www.sphinx-doc.org/
- **RTD Theme Docs:** https://sphinx-rtd-theme.readthedocs.io/

## Support

For issues:

1. **Check RTD Dashboard logs** - Most informative source
2. **Review GitHub Actions workflow results** - Early error detection
3. **Check Sphinx build output locally** - `cd docs && make html`
4. **Consult Homodyne documentation** in `docs/troubleshooting/`

## Contacts

- **Homodyne Issues:** https://github.com/imewei/homodyne/issues
- **ReadTheDocs Support:** https://docs.readthedocs.io/en/stable/support/
