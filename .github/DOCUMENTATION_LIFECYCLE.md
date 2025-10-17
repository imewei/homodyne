# Documentation Lifecycle Policy

**Purpose**: Prevent accumulation of temporary diagnostic files while preserving
valuable technical knowledge.

**Established**: November 17, 2025 **Last Updated**: November 17, 2025

______________________________________________________________________

## Overview

This policy defines how temporary diagnostic documentation moves through its lifecycle
from creation during active debugging to eventual archival or deletion, ensuring the
repository maintains high signal-to-noise ratio while preserving valuable
troubleshooting context.

## File Categories

### 1. Permanent Documentation

**Location**: `docs/`, `README.md`, `CLAUDE.md`, `CHANGELOG.md`

**Characteristics**:

- Long-term value for all developers
- Regularly maintained and updated
- Part of official project documentation
- Versioned and reviewed

**Examples**:

- User guides and tutorials
- API documentation
- Architecture overviews
- Troubleshooting guides (reusable patterns)

**Review Cadence**: Quarterly updates with releases

______________________________________________________________________

### 2. Temporary Diagnostic Files

**Location**: Root directory (for active debugging only)

**Characteristics**:

- Created during active troubleshooting
- Exploratory and verbose
- Time-bound (specific to one issue)
- Not intended for long-term reference

**Naming Convention**: `[ISSUE]_[TYPE].md`

- Examples: `NLSQ_FIX_PLAN.md`, `DEBUG_REPORT.md`, `DIAGNOSIS_SUMMARY.md`
- Use UPPERCASE to indicate temporary status

**Retention**: **Maximum 30 days** after issue resolution

**Disposition Path**: Extract knowledge → Archive → Delete

**Examples**:

- Bug diagnosis reports
- Fix implementation plans
- Integration test guides
- Performance investigation logs

______________________________________________________________________

### 3. Archived Documentation

**Location**: `docs/archive/YYYY-MM-description/`

**Characteristics**:

- Historical value for similar future issues
- Complete context preservation
- Searchable via file system
- Includes README explaining status

**Naming Convention**: Use original filenames within dated directory

**Retention**: **6 months** from archival date

**Review Cadence**: Quarterly

**Examples**:

- Complex debugging sessions
- Integration troubleshooting episodes
- Architecture decision rationale (before final doc)

______________________________________________________________________

## Workflow

### Stage 1: During Active Debugging

**Actions**:

- ✅ Create diagnostic files in root directory
- ✅ Use descriptive uppercase names (e.g., `GPU_OOM_DIAGNOSIS.md`)
- ✅ Be verbose and exploratory - document everything
- ✅ Include timestamps, log snippets, code diffs
- ✅ Track attempted fixes and their results

**Guidelines**:

- Focus on capturing information, not polishing
- Include "dead ends" - they're valuable context
- Link to relevant code locations and commits
- Document environment details (OS, versions, hardware)

**Duration**: While actively debugging (no time limit during active work)

______________________________________________________________________

### Stage 2: After Issue Resolution (Within 7 days)

**Actions**:

1. **Extract valuable knowledge**:

   - Reusable diagnostic approaches → `docs/troubleshooting/`
   - Known pitfalls or gotchas → CLAUDE.md Known Issues section
   - API compatibility notes → CLAUDE.md or module docstrings

1. **Document in code**:

   - Add detailed commit messages with fix context
   - Add code comments explaining "why" for complex fixes
   - Update relevant module docstrings

1. **Decide disposition**:

   - Archive if unique diagnostic value
   - Delete if fully redundant with permanent docs

**Guidelines**:

- Don't let diagnostic files languish - process promptly
- Extract the "lesson learned," not just the fix
- Make knowledge discoverable (don't bury in git history)

______________________________________________________________________

### Stage 3: Cleanup (Within 30 days)

**Actions for Archive**:

1. Create directory: `docs/archive/YYYY-MM-issue-description/`
1. Move relevant diagnostic files to archive
1. Create archive README with:
   - Status (RESOLVED/ONGOING/DEFERRED)
   - Summary of issues and fixes
   - Links to current documentation
   - Quick links to commits and code
   - Retention review date

**Actions for Delete**:

1. Verify knowledge extracted to permanent docs
1. Verify commit messages contain necessary context
1. Delete files
1. Note deletion in commit message with rationale

**Commit Message Format**:

```
docs: [archive|remove] Oct 2025 diagnostic files

Knowledge preserved in:
- docs/troubleshooting/ (reusable guides)
- CLAUDE.md (known issues)
- Git commits (implementation details)

Files [archived|deleted]:
- FILE1.md
- FILE2.md

Rationale: [Brief explanation]
```

______________________________________________________________________

### Stage 4: Quarterly Archive Review

**Schedule**: Every 3 months (Jan, Apr, Jul, Oct)

**Actions**:

1. Review all directories in `docs/archive/` older than 6 months
1. For each archive:
   - Check if issue recurred → Keep if yes
   - Check if still referenced → Keep if yes
   - Check if knowledge fully migrated → Delete if yes
1. Delete obsolete archives
1. Update retention dates for kept archives

**Criteria for Keeping**:

- ✅ Unique diagnostic methodology not documented elsewhere
- ✅ Still referenced in issues or discussions
- ✅ Recurring problem pattern
- ✅ Complex integration scenario likely to recur

**Criteria for Deleting**:

- ✅ Fully covered in permanent documentation
- ✅ Fix stable for 6+ months with no recurrence
- ✅ No longer relevant (deprecated dependencies, etc.)
- ✅ Better documentation now exists

______________________________________________________________________

## Decision Criteria

### When to Archive (Not Delete)

**Archive if file contains**:

- ✅ Unique diagnostic methodology or approach
- ✅ Complex troubleshooting sequence (>5 attempts)
- ✅ Architectural decision rationale
- ✅ Integration guide for external libraries
- ✅ Performance investigation with benchmarks
- ✅ Multiple failed approaches before solution

**Example**: NLSQ integration troubleshooting with 5 different root cause hypotheses

______________________________________________________________________

### When to Delete

**Delete if**:

- ✅ Fully covered in CLAUDE.md or permanent docs
- ✅ Details preserved in git commit messages
- ✅ Testing guides superseded by automated tests
- ✅ Fix is stable and well-understood
- ✅ Simple fix with no diagnostic complexity

**Example**: Single-line bug fix with obvious cause and solution

______________________________________________________________________

### When to Extract to Permanent Docs

**Extract if**:

- ✅ Reusable troubleshooting pattern
- ✅ Known pitfall in API or framework
- ✅ Performance optimization lesson
- ✅ Common error with non-obvious fix
- ✅ Coordinate system or convention explanation

**Destination**:

- `docs/troubleshooting/` → Reusable diagnostic guides
- `CLAUDE.md` → Known Issues section
- Module docstrings → API gotchas
- `docs/guides/` → User-facing best practices

**Example**: "Diagnosing Silent Optimization Failures" guide extracted from NLSQ
debugging

______________________________________________________________________

## Examples

### Good Practice (Follows Policy)

```
Timeline:
2025-10-17: Create NLSQ_FIX_PLAN.md during debugging
2025-10-18: Fix implemented, committed with detailed message
2025-10-24: Knowledge extracted to docs/troubleshooting/
2025-10-24: Archive remaining diagnostic files
2025-11-17: Root directory clean ✓
2026-04-17: Quarterly review - archive kept (unique methodology)
```

**Result**: Knowledge preserved, repository clean, sustainable

______________________________________________________________________

### Bad Practice (Violates Policy)

```
Timeline:
2025-10-17: Create NLSQ_FIX_PLAN.md
2025-12-17: File still in root, 60 days later
2026-01-17: File still in root, 90 days old
2026-02-17: New developer confused: "Is this still an issue?"
```

**Result**: Repository clutter, confusion, lost context

______________________________________________________________________

## Enforcement

### Pre-commit Hook (Optional)

```bash
# .git/hooks/pre-commit
# Warn if diagnostic files older than 30 days in root

find . -maxdepth 1 -name "*_DIAGNOSIS.md" -o -name "*_FIX_*.md" -o -name "*_DEBUG_*.md" | \
while read file; do
  age=$(( ($(date +%s) - $(stat -f%m "$file")) / 86400 ))
  if [ $age -gt 30 ]; then
    echo "WARNING: $file is $age days old - consider archiving per DOCUMENTATION_LIFECYCLE.md"
  fi
done
```

### Quarterly Cleanup Issue

Create GitHub issue template: `.github/ISSUE_TEMPLATE/quarterly-docs-cleanup.md`

```markdown
---
name: Quarterly Documentation Cleanup
about: Review and clean up archived documentation
title: 'Q[X] 2025 Documentation Lifecycle Review'
labels: documentation, maintenance
assignees: ''
---

## Quarterly Archive Review

**Quarter**: Q[X] 2025
**Review Date**: YYYY-MM-DD

### Archives to Review

List all `docs/archive/` directories older than 6 months:

- [ ] `docs/archive/YYYY-MM-description/` (Age: X months)
  - Status: [Keep | Delete]
  - Reason: [...]

### Actions Taken

- Kept: X archives (list reasons)
- Deleted: Y archives (list reasons)
- Updated retention dates: Z archives

### Notes

[Any observations about documentation process]
```

### PR Review Checklist

Add to PR template:

```markdown
- [ ] No temporary diagnostic files added to root (check for *_DIAGNOSIS.md, *_FIX_*.md, etc.)
- [ ] Temporary docs properly archived or deleted if resolution PR
- [ ] Knowledge extracted to permanent docs if applicable
```

______________________________________________________________________

## Metrics and Success Criteria

### Repository Health Metrics

**Target State**:

- Root directory: ≤ 5 markdown files (README, CHANGELOG, CLAUDE, CONTRIBUTING,
  CODE_OF_CONDUCT)
- Archive directories: ≤ 10 at any time
- Average archive age: < 4 months
- Diagnostic file lifetime: < 21 days (70th percentile)

**Red Flags**:

- ❌ Diagnostic files in root > 60 days old
- ❌ More than 20 archive directories
- ❌ Archives older than 12 months
- ❌ No quarterly review in 6+ months

### Process Metrics

**Measure**:

- Time from issue resolution to cleanup (target: < 7 days)
- Percentage of diagnostics properly archived vs. deleted (target: 30% archived, 70%
  extracted+deleted)
- Recurrence of archived issues (measure usefulness)
- Developer feedback on finding troubleshooting info

______________________________________________________________________

## FAQ

### Q: What if I'm not sure whether to archive or delete?

**A**: Default to archive. It's easier to delete later during quarterly review than to
recover lost context.

### Q: Can I create diagnostic files in subdirectories?

**A**: No. Temporary diagnostics should be in root for visibility. This "forces" cleanup
because they're obvious.

### Q: What about investigation notebooks or test scripts?

**A**: Different lifecycle. These go in `tests/` or `experiments/` and follow code
lifecycle, not documentation lifecycle.

### Q: What if the issue recurs after archival?

**A**: Extract the diagnostic approach to permanent docs in `docs/troubleshooting/`.
Recurring issues need permanent solutions.

### Q: Who enforces this policy?

**A**: Everyone. During PR reviews, ask: "Should this diagnostic file be cleaned up?"
Senior devs review quarterly.

______________________________________________________________________

## Related Documentation

- **Example cleanup**: Git commits from November 17, 2025 (this policy's first
  application)
- **Archive example**: `docs/archive/2025-10-nlsq-integration/`
- **Troubleshooting guides**: `docs/troubleshooting/`
- **Known issues**: See CLAUDE.md

______________________________________________________________________

## Revision History

| Date | Version | Changes | |------|---------|---------| | 2025-11-17 | 1.0 | Initial
policy established after October 2025 NLSQ integration cleanup |

______________________________________________________________________

**Questions or Suggestions?** Open an issue with label `documentation` to discuss policy
improvements.
