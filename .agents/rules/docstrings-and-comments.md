# Docstrings and comments

- Multi-line docstrings consist of complete sentences, start on the same line as the opening quotes, and closing quotes go on a new line when the body spans two or more lines.
- Sections are required as follows:
    - `Args:` when the function or method has at least one argument (other than `self` / `cls` in methods);
    - `Raises:` when at least one exception is explicitly raised in the function body (`assert` statements do not count) or exceptions raised by other calls inside the function are not handled;
    - `Returns:` when the return value type is not `None`.
- In the `Args:` and `Attributes:` sections, the description starts on the line following the argument name:

```python
Args:
    x:
        An argument.
```

- In the `Raises:` section, the description starts on the line following the exception class name; a single reason is written as plain text, multiple reasons are listed as bullet points:

```python
Raises:
    ValueError:
        `x` is invalid.
```

```python
Raises:
    ValueError:
        - `x` is invalid.
        - `y` is negative.
```
- Docstrings are written in the imperative mood ("Calculate...", not "Calculates..."), per PEP 257.
- Block and multi-line comments start with a capital letter; in-line comments start with a lowercase letter.
- Descriptions of methods, functions, attributes, return values and exceptions in docstrings end with a period.
- Wrap references to methods, functions and attributes in backticks; write mathematical symbols and formulas in loose TeX style without special delimiters.
- Separate section names and section bodies in docstrings with a blank line.
