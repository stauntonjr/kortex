# Copilot Instructions For Kortex

Use `docs/spec.md` as the architecture source of truth and `docs/backlog.md` as
the execution backlog source of truth.

Also read `AGENTS.md` at the repository root. That file is the canonical shared
workflow for issue tracking, milestones, and project-board maintenance.

When the user asks to track work or when durable follow-up work is discovered:

- Use `gh` against `stauntonjr/kortex`.
- Search before creating duplicate issues.
- Add new issues to the `Kortex Roadmap` project.
- Set milestone, `Status`, `Phase`, `Area`, and `Priority` to keep the roadmap
  organized.
- Prefer updating existing backlog issues when the work already maps cleanly.

Keep TypeDB and chat-history memory work visible in planning. Chat history is a
first-class memory domain in Kortex, not an optional future extra.