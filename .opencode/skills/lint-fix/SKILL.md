---
name: lint-fix
description: Detect and fix linting, formatting, and type-checking errors. Identifies the project's linter configuration and applies fixes that match the codebase style.
license: MIT
compatibility: opencode
metadata:
  audience: developers
  workflow: code-quality
---

## What I do
- Detect the project's linter/formatter configuration (eslint, ruff, pylint, prettier, etc.)
- Run lint/format/typecheck commands and parse errors
- Apply fixes for auto-fixable issues
- Manually fix issues that require semantic understanding
- Ensure changes match the codebase's existing style

## When to use me
Use me when asked to "fix lint errors", "fix type errors", "format this code", or "clean up warnings".

## How I work
1. Look for linter config files (pyproject.toml, .eslintrc, .prettierrc, etc.)
2. Run the appropriate lint/typecheck command
3. For auto-fixable errors: use the tool's fix flag (--fix, --write, etc.)
4. For semantic errors: read the code, understand the intent, apply the fix
5. Re-run the linter to verify all issues are resolved
6. Never change code behavior while fixing lint/type issues

## Common linters by ecosystem
| Language | Linter | Fix flag |
|----------|--------|----------|
| Python | ruff | --fix |
| Python | mypy / pyright | (typecheck, no auto-fix) |
| JavaScript/TS | eslint | --fix |
| JavaScript/TS | prettier | --write |
| Go | gofmt / go vet | -w |
| Rust | clippy | --fix |
