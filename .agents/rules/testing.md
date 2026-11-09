# Testing

- Use pytest with the plugins: pytest-cov, pytest-mock, pytest-randomly, pytest-sugar, pytest-timeout, pytest-xdist.
- Tests must be independent so they can run in parallel with pytest-xdist.
- Store reusable fixtures in `conftest.py`; nested test module structures are allowed and do not require `__init__.py` files because of `--import-mode=importlib`.
- Run tests with plain `pytest`; configuration lives in `pyproject.toml` under `[tool.pytest.ini_options]`.
