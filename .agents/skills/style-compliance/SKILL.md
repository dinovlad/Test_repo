---
name: style-compliance
description: Verify Python code, tests, notebooks, and documentation against the team style guide. Use when checking style or preparing a commit or merge request.
---

# Skill: Style compliance check

Description: Verify that modified or newly created Python code, tests and notebooks comply with the team styleguide from the `commons` knowledge base ([scm.x5.ru/product-hr-analytics/hrmax-aa/commons](https://scm.x5.ru/product-hr-analytics/hrmax-aa/commons)). Project-agnostic: applies to any team repository. Apply this skill before preparing a commit or merge request.

## Steps

1. Run Ruff checks over the changed files, including notebooks and tests, via `ruff check --fix` followed by `ruff format`; do not use flake8, black or isort.
2. Run type checking via `mypy` for Python files and `nbqa mypy` for notebooks.
3. Run spelling checks via `codespell` for `.py` and `.md` files and `nbqa codespell` for notebooks; if the project has a pre-commit config, reuse the ignore lists defined there.
4. Execute tests with `pytest` and confirm they pass independently (they run under pytest-xdist and pytest-randomly).
5. Review the diff for styleguide-specific conventions that tools do not catch:
    - double quotes for string values;
    - built-in types instead of `typing` aliases, `|` instead of `Union` / `Optional`;
    - `@override` decorator on overriding methods;
    - extended assert form `assert expression1, expression2`;
    - docstrings in imperative mood, opening on the same line as the quotes, with periods on descriptions of methods, functions, attributes, return values and exceptions;
    - required docstring sections: `Args:` for at least one non-`self`/`cls` argument, `Raises:` for explicitly raised exceptions (excluding `assert` statements) or unhandled exceptions from called functions, `Returns:` when the return value type is not `None`;
    - newline after the argument name in `Args:` / `Attributes:` and after the exception class name in `Raises:`; single reasons as plain text, multiple reasons as bullet points;
    - backticks around references to methods, functions and attributes in comments;
    - no f-strings in SQL query text; object names through `.format()`, parameters through placeholders (`$1`, `?`);
    - `orjson` / `joblib` instead of `json` / `pickle`, parquet with pyarrow engine and zstd compression for datasets;
    - no stored outputs in notebooks unless marked `# SAVE OUTPUT`;
    - in-place contract: chaining only when a modified object is returned, otherwise `None`.

## Guardrails

- Do not weaken lint, typing or test settings in `pyproject.toml` to make checks pass.
- Do not commit credentials, datasets, serialized objects, caches or project outputs.
