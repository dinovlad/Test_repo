# Test_repo
A repository for investigating Git and GitHub abilities.

## Agent Rules and Skills

[AGENTS.md](AGENTS.md) is the entry point for Codex instructions in this repository.
New Codex chats opened in this repository load it automatically; it directs the
agent to read the ten shared rule files in [.agents/rules/](.agents/rules/) and
apply the relevant requirements.

Four skills are available in [.agents/skills/](.agents/skills/): `env-setup`,
`python-module`, `style-compliance`, and `commit-readiness`. Each has a `SKILL.md`
with discovery metadata. Codex selects skills when the task matches their
description, or you can request one explicitly, for example `$python-module`.
The full skill workflow is loaded when needed.

The rules and skill bodies come from the supplied `agents.zip`; skill metadata
was added for discovery, and macOS archive metadata was omitted. Edit the files
under `.agents/` to maintain them, and update `AGENTS.md` when adding or removing
rules or skills.

These instructions apply to chats working in a checkout containing these files.
Commit and merge them into the branch used for new checkouts to make them
available there as well. They do not configure chats outside this repository.
