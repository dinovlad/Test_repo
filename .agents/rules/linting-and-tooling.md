# Linting, formatting and type checking

- Before committing, always run Ruff (lint and format, including notebooks and tests) and mypy; do not use flake8, black or isort separately.
- Use nbQA for notebooks (`nbqa mypy notebook.ipynb`, `nbqa codespell`).
- Settings are shared across team projects in `pyproject.toml`; do not introduce per-project deviations without a documented rationale.
- Configure VS Code with `formatOnSave` and Ruff as the default formatter, remove Pylint, and set `"mypy-type-checker.cwd": "${nearestConfig}"`.

# Environment and dependencies

- List dependencies in `[project] dependencies` in alphabetical order.
- Install dev dependencies with `pip install -e ".[dev]"`; for sibling team projects, use editable installs without the `dev` extra, for example `pip install -e ../my-sibling-project/`.
- Use the `Makefile` targets for environment and kernel setup; environment files are per-platform (for example, `environment-macos.yml`, `environment-linux.yml`).
