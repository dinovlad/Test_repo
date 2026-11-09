# Code style

- Adhere to PEP 8, then Google Python Style Guide, then pytest Good Integration Practices (in that order of precedence); documented deviations are fixed in `pyproject.toml`.
- Use double quotes for string values.
- Variable names are lowercase; configuration constants (global seeds, role models) are uppercase.
- Import modules by full paths; only these are exceptioned by established practice: contents of `typing` / `typing_extensions` (`Any`, `Literal`, `Self` etc.), contents of `collections.abc`, and `Path` from `pathlib`.
- Prefer built-in types over `typing` aliases; use `|` instead of `Union` / `Optional`; in unions order types as built-in, then custom, then `None`.
- Explicitly use the `@override` decorator for overriding methods, including abstract method implementations.
- Write assertions in the extended form: `assert expression1, expression2`.
- Use the public API of external libraries by default; if a private function/method/attribute is required (for example, for efficiency), add a safety check for its presence, since its logic may change without notice.
- Use the `match` statement unreachable case explicitly via `assert_never(unreachable)`.
