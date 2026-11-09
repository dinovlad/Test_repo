# Project layout

- Follow the canonical project structure:

```
my-project/
├── .gitignore
├── Makefile
├── README.md
├── environment.yml
├── pyproject.toml
├── env/
├── docs/
│   ├── design_doc.md
│   └── other_doc.md
├── src/
│   └── my_project/
│       ├── __init__.py
│       ├── py.typed
│       ├── models/
│       │   ├── __init__.py
│       │   └── model.py
│       └── config/
│           ├── __init__.py
│           └── settings.py
├── notebooks/
│   ├── exploration.ipynb
│   ├── training.ipynb
│   └── evaluation.ipynb
└── tests/
    └── my_project/
        └── test_module.py
```

- One project – one conda environment – one kernel; the environment lives in the local `env/` directory and is never committed.
- Target Python version is 3.11 (constrained by PySpark and MLflow integrations).
- Do not commit any of the following: credentials, API keys, datasets (`.parquet`, `.csv`), serialized objects (`.joblib`, `.pkl`), environment artifacts, caches, or project outputs (`.xlsx`, `.png`, `.log`).
- Keep `py.typed` in the package and declared in `pyproject.toml` under `[tool.setuptools.package-data]` for correct mypy interaction.
- Data-preparation projects additionally contain `sql/` for parameterized query scripts and `src/<package_name>/build_table` for executing them.
