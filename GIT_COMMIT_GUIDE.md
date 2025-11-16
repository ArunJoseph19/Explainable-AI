# Git Commit Guide

This file contains the recommended git commands to commit the repository cleanup.

## Before Committing

1. **Review Changes:**
```bash
git status
git diff
```

2. **Check for Sensitive Data:**
```bash
# Make sure no API keys are committed
grep -r "HOLISTIC_AI" . --exclude-dir=.git --exclude="*.md"
grep -r "sk-" . --exclude-dir=.git --exclude="*.md"
```

3. **Verify .gitignore is working:**
```bash
git check-ignore -v outputs/
git check-ignore -v venv/
```

## Staging Changes

```bash
# Stage the new structure
git add dashboards/
git add src/
git add notebooks/
git add scripts/
git add assets/
git add config/
git add outputs/.gitkeep

# Stage documentation
git add README.md
git add CONTRIBUTING.md
git add QUICKSTART.md
git add CLEANUP_SUMMARY.md
git add .gitignore
git add setup.sh

# Stage modified requirements if needed
git add requirements.txt
```

## Review Deletions

```bash
# This will show files that were deleted
git status

# If you see files that should be removed from git:
git rm core/ react_agent/ Flux/ -r
git rm *.png ngrok* core.zip lost+found/ -f 2>/dev/null || true
```

## Commit

```bash
# Create a comprehensive commit
git commit -m "Major refactor: Reorganize repository structure

- Organize code into logical directories (dashboards/, src/, notebooks/, outputs/)
- Move source files to src/ with subdirectories (agents/, core/, flux/)
- Consolidate all Jupyter notebooks in notebooks/
- Centralize outputs in outputs/ directory
- Remove duplicate folders and temporary files
- Add comprehensive documentation (README, CONTRIBUTING, QUICKSTART)
- Add .gitignore for Python projects
- Add setup.sh automation script
- Update all import paths to reflect new structure
- Fix dashboard paths and configuration

Breaking changes:
- Dashboard launch commands updated
- Output directory changed to outputs/experiments/
- Import paths updated for modular structure

See CLEANUP_SUMMARY.md for complete details."
```

## Push to GitHub

```bash
# Push to main branch
git push origin main

# Or if using a feature branch
git checkout -b refactor/repository-structure
git push origin refactor/repository-structure
```

## After Pushing

1. **Update GitHub Repository Settings:**
   - Add repository description
   - Add topics/tags: `ai`, `image-generation`, `explainable-ai`, `flux`, `diffusion-models`
   - Update repository website link if applicable

2. **Create Release (Optional):**
```bash
git tag -a v1.0.0 -m "Version 1.0.0 - Major repository reorganization"
git push origin v1.0.0
```

3. **Update GitHub README:**
   - The new README.md will automatically display
   - Check that badges are rendering correctly
   - Verify all internal links work

## Verification Checklist

After pushing, verify on GitHub:
- [ ] README displays correctly
- [ ] Directory structure is clean
- [ ] No sensitive files (API keys, large binaries) committed
- [ ] .gitignore is working (venv/, outputs/ not tracked)
- [ ] All documentation files are visible
- [ ] setup.sh is executable
- [ ] Project looks professional

## Rolling Back (If Needed)

```bash
# If something went wrong
git reset --soft HEAD~1  # Undo last commit, keep changes
git reset --hard HEAD~1  # Undo last commit, discard changes

# Restore specific file
git checkout HEAD -- path/to/file
```

## Notes

- **Branch Protection:** Consider setting up branch protection rules
- **CI/CD:** Add GitHub Actions for automated testing
- **Documentation:** Keep README updated as features evolve
- **Contributors:** Add CODEOWNERS file if needed

---

**Ready to commit! 🚀**
