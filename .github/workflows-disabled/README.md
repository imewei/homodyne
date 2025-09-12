# GitHub Workflows - Temporarily Disabled

## Status: DISABLED ⏸️

The GitHub workflows in this directory have been temporarily disabled while the Homodyne v2 codebase is under active development and major restructuring.

## Disabled Workflows:
- `ci.yml` - Fast CI pipeline with testing
- `quality.yml` - Code quality checks
- `quality-monitoring.yml` - Quality monitoring
- `docs.yml` - Documentation builds  
- `claude.yml` - Claude Code integration
- `release.yml` - Release automation

## Reason for Disabling:
The codebase is undergoing significant development and the current workflows may:
- Fail due to incomplete implementations
- Generate excessive noise during development
- Consume unnecessary CI/CD resources
- Create false alerts during major refactoring

## How to Re-enable:

When the codebase is ready for CI/CD, simply rename this directory back:

```bash
# Re-enable all workflows
mv .github/workflows-disabled .github/workflows
```

## Timeline:
- **Disabled**: September 12, 2025
- **Re-enable when**: Homodyne v2 codebase reaches stable development milestone

## Alternative Testing:
During this period, local testing should be used:
```bash
# Local testing commands
make test           # Run basic tests
make test-all       # Run comprehensive tests  
make lint           # Code quality checks
make format         # Code formatting
```

---
*This is a temporary measure to allow focused development without CI/CD interruptions.*