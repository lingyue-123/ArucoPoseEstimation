---
name: code-review
description: Review code for bugs, security vulnerabilities, performance issues, and code quality. Analyze diffs or specific files and provide actionable feedback.
license: MIT
compatibility: opencode
metadata:
  audience: developers
  workflow: code-review
---

## What I do
- Analyze code for logical bugs, edge cases, and error handling gaps
- Identify security vulnerabilities (injection, XSS, unsafe deserialization, secrets exposure)
- Spot performance bottlenecks (N+1 queries, unnecessary allocations, blocking I/O)
- Check adherence to existing codebase conventions (naming, patterns, imports)
- Flag missing tests, unclear variable names, and excessive complexity

## When to use me
Use me when asked to "review this code", "check for bugs", "audit for security", or "is this code safe?".

## How I work
1. Read the target files or git diff
2. Examine surrounding code for context and conventions
3. Report findings grouped by severity: critical, warning, info
4. For each issue: state the problem, explain the risk, and propose a fix
5. Only flag real issues — do not nitpick trivial style choices

## What I check
| Category | Examples |
|----------|----------|
| Logic | Off-by-one, inverted conditions, missing null checks, race conditions |
| Security | SQL injection, XSS, hardcoded secrets, unsafe eval, path traversal |
| Performance | N+1 queries, redundant loops, large allocations in hot paths |
| Robustness | Missing error handling, exception swallowing, timeout absence |
| Conventions | Inconsistent naming, wrong import style, deviation from existing patterns |
