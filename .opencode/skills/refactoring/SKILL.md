---
name: refactoring
description: Refactor code to improve readability, reduce duplication, and simplify complexity without changing external behavior. Follows existing codebase conventions.
license: MIT
compatibility: opencode
metadata:
  audience: developers
  workflow: refactoring
---

## What I do
- Extract duplicated logic into shared functions/utilities
- Simplify complex conditionals (guard clauses, early returns, lookup tables)
- Improve variable/function naming for clarity
- Break down large functions into smaller, focused ones
- Remove dead code and unnecessary abstractions
- Convert magic numbers to named constants

## When to use me
Use me when asked to "refactor X", "clean up this code", "simplify", or "DRY up".

## How I work
1. Read the target code and understand its behavior
2. Identify the specific issues: duplication, complexity, naming, structure
3. Verify the exact same behavior is preserved
4. Apply changes incrementally, one refactoring step at a time
5. Never change external API signatures unless explicitly requested
6. Ensure existing test patterns are compatible with the refactored code

## Principles
- **Behavior preserving**: Refactoring should not change what the code does
- **Incremental**: Make small, verifiable changes
- **Convention-first**: Match existing patterns in the codebase
- **Readability over cleverness**: Favor clarity over brevity
