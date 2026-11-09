---
name: python-module
description: Create a module in a team Python package following the project structure and code style. Use when a new Python module, class, or set of functions is requested.
---

# Skill: Add a Python module

Description: Create a new module in a team Python package following the project structure and code style defined in the `commons` knowledge base ([scm.x5.ru/product-hr-analytics/hrmax-aa/commons](https://scm.x5.ru/product-hr-analytics/hrmax-aa/commons)). Project-agnostic: replace `my_project` and `<package_name>` with the actual package name. Apply this skill when a new module, class or set of functions is requested.

## Steps

1. Place the new module in `src/<package_name>/` with a `*_utils.py` name for integration functionality (for example, `keycloak_utils.py`) or a descriptive name for domain functionality (for example, `hashing.py`).
2. Update the project `README.md` with a line describing the new module in the `src` section.
3. Write the module complying with the agent rules in [`rules/`](../../rules/):
    - full type annotations (`disallow_untyped_defs` is enabled in mypy);
    - pydantic v2 with `ConfigDict(arbitrary_types_allowed=True)` for runtime-validated models;
    - docstrings in imperative mood starting on the quote line, sections separated by blank lines;
    - double quotes, lowercase variable names, uppercase configuration constants;
    - full-path imports, except for allowed `typing` / `typing_extensions` / `collections.abc` contents and `Path`;
    - sensitive connection parameters read from env files, never hard-coded.
4. If the package already provides wrappers for an external service or storage, reuse them instead of raw client APIs.
5. Add tests under `tests/` using pytest with the project plugins; reusable fixtures go into `conftest.py`; tests must be independent for parallel execution.
6. Run the style compliance skill ([`../style-compliance/SKILL.md`](../style-compliance/SKILL.md)) and fix all findings.
7. If the module implements complex logic, add a document to `docs/` in English with title case headings and Harvard-style bibliographic references; scientific references in code are documented in the project `README.md`.

## Guardrails

- Do not add ORM (SQLAlchemy) based data access for large-scale computations; interact with storages via SQL with `.format()` for object names and placeholders for parameters.
- Do not add dependencies not alphabetically ordered in `pyproject.toml`.
- Do not expose private API of external libraries without a presence check and `warnings.warn` fallback.
