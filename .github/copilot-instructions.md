# Copilot Instructions

Start with `CLAUDE.md`. It is the primary repository-specific instruction file for workflow, commands, artifact layout, and execution conventions.

Also treat these steering docs as authoritative when changing code or experiment flow:

- `documents/steering/product.md`
- `documents/steering/structure.md`
- `documents/steering/tech.md`

Key reminders for Copilot sessions:

- Use `make dev`, `make lint`, `make test`, and the `make phase{N}` targets from `CLAUDE.md` / `Makefile`.
- Run experiment phases via `ml_run.sh` (directly or through the Makefile wrappers), and use `--detach` for long runs.
- Preserve the phase-gate workflow: phase scripts write metrics/figures plus `artifacts/metadata/phase{N}.json`, and downstream work should treat those JSON files as the checkpoint contract.
- Follow the existing logging and artifact conventions instead of inventing new output locations or ad hoc scripts.

Keep this file intentionally brief. Update the authoritative guidance in `CLAUDE.md` and `documents/steering/` first, and only keep stable Copilot-specific pointers here.
