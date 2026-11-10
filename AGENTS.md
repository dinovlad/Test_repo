# Repository Instructions

These instructions apply to this repository and its subdirectories.

## Rules to Load for Every Chat

Before starting work, read all of the following rule files and apply their relevant
requirements throughout the task. Resolve these paths from the repository root:

- [Language and communication](.agents/rules/language-and-communication.md)
- [Security and documentation](.agents/rules/security.md)
- [Project layout](.agents/rules/project-layout.md)
- [Code style](.agents/rules/code-style.md)
- [Docstrings and comments](.agents/rules/docstrings-and-comments.md)
- [In-place transformations](.agents/rules/in-place-transformations.md)
- [Serialization and artifacts](.agents/rules/serialization-and-artifacts.md)
- [SQL in Python](.agents/rules/sql-in-python.md)
- [Testing](.agents/rules/testing.md)
- [Linting and tooling](.agents/rules/linting-and-tooling.md)

The files contain shared team conventions. Apply Python, notebook, SQL, and
environment requirements when that technology is involved in the requested work.
Do not scaffold a Python project or create an environment solely to load these
instructions. Follow the user's current request and higher-priority instructions
if they conflict with the shared conventions.

## Project-Specific Exceptions

The user has explicitly retained the following project conventions:

- Keep development dependencies in `[dependency-groups].dev` and the documented
  `uv` workflow. Do not replace them with a `dev` extra, add the shared pytest
  plugin suite, or introduce conda, Makefile, or editor setup solely for rule
  compliance.
- Unit tests may use assertions without messages, as stated in the code-style
  rule. Production assertions still require messages.
- Follow the supplied Google-style docstring layout: put section content
  immediately after its heading, with argument or exception names and their
  descriptions on separate lines. Keep blank lines between sections. Exception
  conditions may use complete prose rather than mandatory bullet lists.
- Keep estimator logic in the single `src/labor_cost_optimization/att.py` module
  and its design document in `docs/att.md`; do not scaffold the generic
  project-layout example solely for compliance.

## Skills

Repository skills live in `.agents/skills/`. When the user names a skill or the
task matches its purpose, read its `SKILL.md` and follow the applicable workflow.
Read referenced rules or skills as needed. If skill discovery is unavailable in
the client, use the paths below directly.

| Skill | When to Use | Instructions |
| --- | --- | --- |
| `env-setup` | Creating, refreshing, or validating a Python development environment | [.agents/skills/env-setup/SKILL.md](.agents/skills/env-setup/SKILL.md) |
| `python-module` | Adding a Python module, class, or functions | [.agents/skills/python-module/SKILL.md](.agents/skills/python-module/SKILL.md) |
| `style-compliance` | Checking style or preparing changes for a commit or merge request | [.agents/skills/style-compliance/SKILL.md](.agents/skills/style-compliance/SKILL.md) |
| `commit-readiness` | Preparing implemented changes for a commit or merge request | [.agents/skills/commit-readiness/SKILL.md](.agents/skills/commit-readiness/SKILL.md) |

Apply skills within the requested task's scope; their presence does not request
every workflow on every chat. In particular, instructions describing environment
creation, commits, pushes, or merge requests do not independently authorize those
actions. Run checks appropriate to the repository and changed files, and report
checks that are unavailable or not applicable rather than claiming they passed.
