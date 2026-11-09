---
name: env-setup
description: Set up a local development environment for a team Python repository. Use when the environment must be created, refreshed, or validated.
---

# Skill: Environment setup

Description: Set up a local development environment for any team Python repository in accordance with the conventions from the `commons` knowledge base ([scm.x5.ru/product-hr-analytics/hrmax-aa/commons](https://scm.x5.ru/product-hr-analytics/hrmax-aa/commons)). Project-agnostic. Apply this skill when the environment must be created, refreshed or validated.

## Steps

1. Confirm the target Python version is 3.11 as pinned in the project `pyproject.toml`.
2. Detect the platform and select the matching environment file (for example, `environment-macos.yml` or `environment-linux.yml`; single `environment.yml` where provided).
3. Create the conda environment and Jupyter kernel with the project `Makefile` install target; the environment lives locally in `env/` (one project – one conda environment – one kernel), never committed to version control.
4. Install the package with dev dependencies: `pip install -e ".[dev]"`; for sibling team projects use editable installs without the `dev` extra, for example `pip install -e ../my-sibling-project/`.
5. Create the required env files with connection parameters in the parent directory of the consuming project, following the templates in the project `README.md` where provided; env files are excluded from version control.
6. Verify the checks run: `ruff check`, `ruff format --check`, `mypy` and `pytest`.

## Guardrails

- Do not commit env files, the `env/` directory or any credentials.
- Do not install the package without the conda environment; kernel and environment are created together.
- On JupyterHub, respect CPU limits applied via the conda activate script in the project `Makefile` where present.
