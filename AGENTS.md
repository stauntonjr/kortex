# Kortex Agent Instructions

These instructions apply to agents working in this repository across tools that
support `AGENTS.md`.

## Canonical project documents

- Use `docs/spec.md` as the architecture source of truth.
- Use `docs/backlog.md` as the execution backlog source of truth.
- Do not treat external chat logs as the project spec once the in-repo docs are
  present.

## GitHub planning workflow

Use the GitHub CLI (`gh`) for issue, milestone, and project management when the
user asks to track work or when new durable follow-up work is discovered.

Repository:

- `stauntonjr/kortex`

Project board:

- `Kortex Roadmap`
- URL: `https://github.com/users/stauntonjr/projects/2`
- Project number: `2`

Milestones:

- `Phase 0`
- `Phase 1`
- `Phase 2`
- `Phase 3`
- `Phase 4`
- `Phase 5`
- `Phase 6`

Project fields:

- `Status`: `Todo`, `In Progress`, `Done`
- `Phase`: `Phase 0` through `Phase 6`
- `Area`: `gateway`, `memory`, `typedb`, `ingestion`, `agent`, `docs`, `ops`
- `Priority`: `P0`, `P1`, `P2`

## Required issue hygiene

Before creating a new issue:

- Search existing issues to avoid duplicates.
- Check whether the work already maps to an existing backlog issue.
- Prefer updating or commenting on an existing issue instead of creating a near
  duplicate.

When creating a new issue:

- Create it with `gh issue create --repo stauntonjr/kortex ...`.
- Add an appropriate label such as `enhancement`, `bug`, or `documentation`.
- Assign the correct milestone by phase.
- Add the issue to the `Kortex Roadmap` project.
- Set `Status`, `Phase`, `Area`, and `Priority` on the project item.

When updating existing work:

- Move `Status` to `In Progress` only when the issue is actively being worked.
- Move `Status` to `Done` only when the work is actually complete or the issue
  is being closed.
- Close stale umbrella issues if they are superseded by a structured issue set,
  and leave a redirect comment.

## Preferred commands

Search existing issues:

```bash
gh issue list --repo stauntonjr/kortex --state all --limit 100
```

Create an issue:

```bash
gh issue create --repo stauntonjr/kortex --title "..." --body-file /path/to/body.md --label enhancement
```

Assign a milestone:

```bash
gh issue edit <issue-number> --repo stauntonjr/kortex --milestone "Phase 1"
```

Add an issue to the project and get the project item ID:

```bash
gh project item-add 2 --owner stauntonjr --url https://github.com/stauntonjr/kortex/issues/<issue-number> --format json --jq '.id'
```

Inspect project fields before editing item metadata:

```bash
gh project field-list 2 --owner stauntonjr --format json
```

If project commands fail due to auth scope, refresh GitHub CLI auth with:

```bash
gh auth refresh -s read:project -s project
```

## Planning alignment

- Keep TypeDB work explicit in planning; it is a real project priority.
- Treat chat history as a first-class memory domain in both the graph and vector
  layers, not as code-only metadata.
- Preserve the phased execution order from `docs/backlog.md` unless the user
  explicitly reprioritizes it.
