---
name: documentation
description: Generate docstrings, API documentation, type annotations, and inline comments. Follows the project's existing documentation style and conventions.
license: MIT
compatibility: opencode
metadata:
  audience: developers
  workflow: documentation
---

## What I do
- Generate docstrings for functions, classes, and modules
- Add type annotations where missing
- Write or update README sections
- Document API endpoints (params, responses, auth)
- Add clarifying inline comments for complex logic
- Generate usage examples

## When to use me
Use me when asked to "document X", "add docstrings", "generate docs", or "explain the API".

## How I work
1. Detect the project's documentation style (Google, NumPy, Sphinx, JSDoc, etc.)
2. Study existing documented code for formatting conventions
3. Use the project's type annotation patterns (if any)
4. Only document what's missing or unclear — don't add noise
5. For docstrings: include types, descriptions, params, returns, raises, examples
6. For APIs: describe endpoints, request/response schemas, auth requirements, error codes

## Documentation priorities
1. Public API surfaces (exported functions, classes, endpoints)
2. Complex algorithms with non-obvious logic
3. Configuration and environment variables
4. Module-level overviews for large files
