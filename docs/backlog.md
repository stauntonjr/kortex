# Kortex Backlog

This backlog is ordered by execution value, not by how exciting the ideas are.
The first priority is to make Kortex stable and truthful. The second is to make
its memory useful. The third is to make it more autonomous.

## Phase 0: Stabilize The Current Baseline

### B0.1 Fix gateway completeness

- Restore or implement the missing helper functions used by the gateway runtime.
- Remove duplicate lifecycle definitions.
- Ensure process cleanup is deterministic on shutdown.

Acceptance criteria:

- The gateway imports successfully.
- The gateway test module can import every referenced symbol.
- A mocked request can pass through the proxy path.

### B0.2 Align packaging and runtime dependencies

- Make `compose.yaml`, `requirements.txt`, and `pyproject.toml` agree.
- Ensure the gateway container installs everything it imports.
- Align healthcheck paths with actual endpoints.

Acceptance criteria:

- The compose gateway container can start without missing-module errors.
- Healthchecks probe real endpoints.

### B0.3 Fix model naming and config drift

- Align embedding model names across gateway config, ingestion defaults, and
  runtime expectations.
- Remove stale env vars and dead config where practical.

Acceptance criteria:

- Default embedding requests route to the intended embedding backend.

## Phase 1: Establish A Real Memory Foundation

### B1.1 Expand the TypeDB schema beyond code-only storage

- Add entities and relations for chat sessions, turns, artifacts, directives,
  and provenance links.
- Keep the initial schema small and enforceable.

Acceptance criteria:

- The schema can represent code entities and chat history in one graph.
- Example inserts exist for both domains.

### B1.2 Define chat-history ingestion

- Create a pipeline that stores chat transcripts in both TypeDB and Qdrant.
- Preserve session metadata, turn order, role, and extracted references.

Acceptance criteria:

- A sample session can be ingested into the graph.
- The same session can be embedded for semantic search.

### B1.3 Make retrieval actually query storage

- Replace formatting-only helpers with real retrieval from TypeDB and Qdrant.
- Support separate recall modes for code, chat, and project artifacts.

Acceptance criteria:

- Retrieval returns persisted nodes rather than caller-supplied mock nodes.
- Context assembly remains bounded by depth and token budget.

## Phase 2: Replace Naive Code Ingestion

### B2.1 Add AST-backed code extraction

- Introduce Tree-sitter, `ast-grep`, or another structural parser.
- Extract repo, file, symbol, and line-span anchors.

Acceptance criteria:

- Python source ingestion no longer relies only on character chunking.
- Parsed entities include source anchors.

### B2.2 Model code relationships explicitly

- Add imports, calls, ownership, and file membership links.
- Keep the first pass deterministic rather than LLM-derived.

Acceptance criteria:

- One repository can be ingested with structural edges visible in TypeDB.

## Phase 3: Make The Gateway Useful As A Product Surface

### B3.1 Formalize routing policy

- Define explicit routing rules for coder, embedding, midweight, and heavyweight
  models.
- Support fallback behavior when callers omit `model`.

Acceptance criteria:

- Requests without a model route predictably.
- Embedding requests cannot accidentally land on the code model.

### B3.2 Add lifecycle controls and visibility

- Expose active model state, readiness, and recent routing decisions.
- Consider idle shutdown for heavyweight models.

Acceptance criteria:

- Operators can tell which model is online and why.

### B3.3 Add focused gateway tests

- Cover model swap locking, retries, error translation, and readiness behavior.

Acceptance criteria:

- Gateway behavior is verified without live model runtimes.

## Phase 4: Orchestrate Retrieval And Agent Behavior

### B4.1 Upgrade the agent from shell to real orchestrator

- Use LangGraph or equivalent only after retrieval primitives are stable.
- Introduce nodes for classification, retrieval, execution, and response.

Acceptance criteria:

- The agent can retrieve memory and route through the gateway end to end.

### B4.2 Add tool-facing memory access

- Evaluate `typedb-mcp` as the structured tool interface.
- Decide whether the repo should wrap it or depend on it externally.

Acceptance criteria:

- Agents can query structured memory through a stable interface.

### B4.3 Background compaction for chat history

- Compact sessions into durable summaries without losing graph links.
- Extract directives or stable preferences as reviewable candidates.

Acceptance criteria:

- Chat history remains searchable at raw-turn and compacted levels.

## Phase 5: Improve Operations And Developer Experience

### B5.1 Improve docs and setup truth

- Expand the README.
- Document model roles, ports, dependencies, and development flows.

Acceptance criteria:

- A new contributor can understand the stack without reading external chats.

### B5.2 Add CI for import, unit, and schema validation

- Run unit tests and basic validation on pull requests.
- Add syntax or schema checks for TypeQL where feasible.

Acceptance criteria:

- PRs catch missing imports, broken tests, and schema regressions automatically.

### B5.3 Add observability intentionally

- Add Prometheus and Grafana for infra and model telemetry.
- Add Langfuse or equivalent only when trace data has a defined consumer.

Acceptance criteria:

- Telemetry answers specific operator questions instead of existing as unused
  tooling.

## Phase 6: Package And Extend

### B6.1 Add AI Workbench packaging

- Revisit `.project/spec.yaml` after runtime behavior is stable.
- Validate the spec with Workbench before making it central to onboarding.

Acceptance criteria:

- Workbench import works from a clean clone.

### B6.2 Evaluate `nvidia-sync` integration

- Treat this as deployment automation, not core architecture.

Acceptance criteria:

- Kortex can be provisioned repeatably without changing runtime semantics.

### B6.3 Evaluate advanced graph additions only after usage proves the need

- `Graphiti`: optional if temporal memory semantics become painful to model.
- `typedb-mcp`: recommended if agents need direct structured tool access.
- `ast-grep`: recommended for structural search and repair.
- OpenClaw or OpenShell: optional, out-of-band management layer rather than a
  replacement for the gateway.

Acceptance criteria:

- Each addition solves a demonstrated problem in Kortex rather than broadening
  scope by default.

## Recommended Issue Queue

Open these first, in order:

1. Fix gateway completeness and missing helper implementations.
2. Align compose, requirements, and pyproject dependency truth.
3. Extend TypeDB schema to include chat sessions and directives.
4. Implement chat-history ingestion into TypeDB and Qdrant.
5. Replace naive code chunking with AST-backed entity extraction.
6. Implement real hybrid retrieval from TypeDB plus Qdrant.
7. Add focused gateway and retrieval tests.
8. Add LangGraph orchestration after retrieval is real.
9. Evaluate `typedb-mcp` and `ast-grep` as feature additions.
10. Add Prometheus, Grafana, and Langfuse only after baseline stability.
