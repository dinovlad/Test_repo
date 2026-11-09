---
name: commit-readiness
description: Prepare a commit or merge request according to team conventions. Use when preparing implemented changes for a commit or merge request.
---

# Skill: Commit and merge request readiness

Description: Prepare a commit or merge request in any team repository in accordance with the conventions from the `commons` knowledge base ([scm.x5.ru/product-hr-analytics/hrmax-aa/commons](https://scm.x5.ru/product-hr-analytics/hrmax-aa/commons)). Project-agnostic. Apply this skill after implementing changes.

## Steps

1. Run the style compliance skill ([`../style-compliance/SKILL.md`](../style-compliance/SKILL.md)): Ruff, mypy, codespell and pytest must pass before committing.
2. Verify the changes touch only files appropriate for version control; confirm that `.gitignore` excludes credentials, datasets (`.parquet`, `.csv`), serialized objects (`.joblib`, `.pkl`), environment artifacts, caches and outputs (`.xlsx`, `.png`, `.log`).
3. Stage the changes and write a commit message in American English as incomplete sentences with a clear scope, for example `Add retry policy to data client`.
4. Determine the Kaiten card for the change; if none exists, propose using the semantically related epic card or creating a new card. Name the branch `kaiten-<card id>-<essence of change>` with hyphens.
5. Create the merge request with the title `kaiten-<card id>: <essence of change>`, enable the option to delete the source branch after merge, and keep the main branch linear with fast-forward merges (squash commits only for long-running initiatives outside specific Kaiten tasks).
6. After the merge request is approved and merged, remind about removing the dev branch locally.

## Guardrails

- Do not push directly to the main branch; all changes go through merge requests.
- Do not include credentials or secrets in commit messages, MR descriptions or branch names.
- For notebook diffs, recommend `nbdime` instead of raw `git diff`.
