# Serialization and artifacts

- Formats: JSON, joblib, pickle (extension `.pkl`, not `.pickle`).
- Use `orjson` and `joblib` instead of standard `json` and `pickle`; for objects not natively picklable, wrap with `joblib.wrap_non_picklable_objects()`.
- For numpy and sklearn objects, use the joblib format.
- Store datasets extracted directly from storages as parquet with the pyarrow engine and zstd compression, `index=False`.
- Name file objects `file`, not `f` (for example, `with open("study.pkl", "wb") as file:`).
- Do not store cell outputs in notebooks (logs, tables, plots), except for notebooks marked `# SAVE OUTPUT` at the top.

# Validation

- Use `pydantic` for runtime type validation; configure models with the v2 pattern:

```python
model_config = pydantic_config.ConfigDict(arbitrary_types_allowed=True)
```
