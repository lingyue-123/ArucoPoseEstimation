---
name: test-generation
description: Generate comprehensive unit tests for functions, classes, or modules. Detects the testing framework in use and follows existing test patterns.
license: MIT
compatibility: opencode
metadata:
  audience: developers
  workflow: testing
---

## What I do
- Detect the project's test framework and conventions
- Generate tests covering: happy path, edge cases, error conditions, boundary values
- Use existing test fixtures, mocks, and helpers from the codebase
- Follow the exact same patterns as existing tests (naming, structure, assertions)

## When to use me
Use me when asked to "write tests for X", "add test coverage", or "generate unit tests".

## How I work
1. Identify the test framework in use (pytest, jest, unittest, go test, etc.)
2. Study 2-3 existing test files for patterns, imports, and conventions
3. Analyze the target function/module signature, behavior, and dependencies
4. Generate tests with clear arrange-act-assert structure
5. Include docstrings/comments matching existing test style

## Test coverage checklist
- [ ] Normal / happy path inputs
- [ ] Edge cases (empty, null, zero, max, min)
- [ ] Error / exception paths
- [ ] Boundary values
- [ ] Type coercion edge cases (for dynamic languages)
