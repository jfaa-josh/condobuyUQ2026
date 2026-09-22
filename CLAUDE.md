# Claude Code Instructions

## Project
- Project name, purpose, and key decisions: see the memory files (and `README.md`).
- Python version is pinned in `.python-version` and `pyproject.toml` (`requires-python`); keep them in sync.
- Dependencies are managed with `uv` (`uv add <pkg>`, `uv add --dev <pkg>`, `uv sync`). Do not use `pip install`.
- Source code lives in `src/`. Run tools through the project venv (`uv run ...`).
- The implementation plan for `src/` lives in `.claude/implementation-plan.md` (gitignored, local-only). Read it at
  the start of any session touching implementation work, and keep it current: update it whenever the plan changes
  (new decisions, scope changes, completed steps, open questions) rather than letting it go stale.

## Memory & Context Preservation

**Save context proactively during every conversation.** Context compression is aggressive — do not wait until the end of a session to save important information.

### When to save to memory
- After any significant decision, architectural choice, or design discussion
- When the user explains *why* something is done a certain way
- When a bug is diagnosed and root-caused (save the cause, not just the fix)
- When the user corrects your approach or confirms it was right
- When starting a new feature or significant task (save the goal and approach)
- At natural milestones (e.g., after completing a subtask)

### What to save
- User preferences and working style corrections → `feedback_*.md`
- Project decisions, goals, and motivations → `project_*.md`
- External resources and where to find things → `reference_*.md`

Use this project's Claude memory directory (the auto-memory location provided in the session's system prompt). Write memories as if the next conversation starts completely cold; do not rely on chat history as the primary context source.
