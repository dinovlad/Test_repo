# SQL in Python

- Do not use f-strings for query text; pass object names (databases, schemas, tables) through `.format()` and query parameters through placeholders (`$1` for PostgreSQL/Greenplum, `?` for Trino) with parameterized queries.
- For long queries with CTEs or nested queries, use named placeholders for object names.
- Queries not supporting parameterization (for example, `CREATE TABLE`) receive their parameters through `.format()`.
- Do not use ORM approaches (SQLAlchemy) in the context of distributed computations on large data; prefer Trino over Hive for Hadoop.
