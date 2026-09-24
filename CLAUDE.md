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

**Save context proactively during every conversation.** Context compression is aggressive — do not wait until the end of a session, or a natural pause, to save important information. A compaction (or any other context loss) can happen between any two turns, with no warning, so treat every turn that contains something worth keeping as if it were the last turn you'll get: write the memory file **in that same turn**, before moving on to the next request, not batched up for "later."

### When to save to memory
- After any significant decision, architectural choice, or design discussion
- When the user explains *why* something is done a certain way
- When a bug is diagnosed and root-caused (save the cause, not just the fix)
- When the user corrects your approach or confirms it was right
- When starting a new feature or significant task (save the goal and approach)
- At natural milestones (e.g., after completing a subtask)
- **Any one-off behavioral instruction or standing constraint the user gives you** ("don't do X yourself," "always ask before Y," quota/rate limits to respect, etc.) — these are easy to miss because they don't look like "project decisions," but they're exactly the kind of thing that silently stops being followed once it's out of context. Save these as `feedback_*.md` the moment they're given, not after the task they came up in is finished.

### What to save
- User preferences and working style corrections → `feedback_*.md`
- Project decisions, goals, and motivations → `project_*.md`
- External resources and where to find things → `reference_*.md`

Use this project's Claude memory directory (the auto-memory location provided in the session's system prompt). Write memories as if the next conversation starts completely cold; do not rely on chat history as the primary context source.
