---
name: git-workflow
description: Help with Git operations including commits, branching, merging, rebasing, and PR workflows. Follows the project's existing commit conventions.
license: MIT
compatibility: opencode
metadata:
  audience: developers
  workflow: version-control
---

## What I do
- Draft commit messages following the project's conventions
- Create and manage branches with consistent naming
- Prepare pull requests with clear descriptions
- Resolve merge conflicts with guidance
- Clean up git history (squash, rebase, reword)
- Analyze git diff for staged/unstaged changes

## When to use me
Use me when asked to "commit this", "create a PR", "rebase onto main", "fix merge conflict", or "clean up git history".

## How I work
1. Check `git log` for the project's commit message style
2. Verify which files are tracked, staged, and modified
3. Follow the Git Safety Protocol:
   - NEVER run destructive commands (push --force, hard reset) without explicit request
   - NEVER skip hooks without explicit request
   - NEVER ammend pushed commits without explicit request
4. Draft concise, conventional commit messages
5. Add only relevant files — never commit secrets (.env, credentials, tokens)

## Branch naming conventions
- Feature: `feature/<description>`
- Bug fix: `fix/<description>`
- Refactor: `refactor/<description>`
- Docs: `docs/<description>`
- If the project uses different conventions, follow those instead
