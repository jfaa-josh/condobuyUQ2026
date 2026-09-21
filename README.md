# Project Template (Python + uv + VS Code)

A generic starting point: pinned Python, `uv`-managed virtual environment, pre-commit checks (ruff, mypy),
VS Code auto-selecting `.venv`, and a `CLAUDE.md` for Claude Code. No runtime packages are included.

## Starting a new project from this template

1. **Copy the template folder** to the new location, rename it to the project name, and open it in VS Code
   (*File > Open Folder*). Do all the remaining steps in the VS Code integrated terminal (*Terminal > New Terminal*).
   Start a fresh git history:
   ```powershell
   git init
   ```
2. **Edit `pyproject.toml`:** set `name`, `description`, and `authors`.
3. **Edit `LICENSE.md`:** in the "Licensed Work" line, change the repository URL to this project's repo
   (currently `https://github.com/jfaa-josh/stock-ops`), and check the other project-specific details.
4. **Python version** (default 3.13): to change it, edit *all* of these together —
   `.python-version` (e.g. `3.12.10`), `requires-python` (e.g. `~=3.12.0`), ruff `target-version` (e.g. `py312`),
   and mypy `python_version` (e.g. `3.12`). The pre-commit hook fails if `.python-version` and `requires-python`
   disagree.
5. **Add packages** (updates `pyproject.toml` and `uv.lock`, and installs into `.venv`):
   ```powershell
   uv add numpy pandas          # runtime dependencies
   uv add --dev pytest          # dev-only tools
   ```
6. **Create the venv** (installs Python if needed, plus the dev tools already listed; `uv add` does this too):
   ```powershell
   uv sync
   ```
7. **Load the venv in VS Code.** Close VS Code completely and reopen the project folder. Then open a new terminal
   (*Terminal > New Terminal*) and check that the prompt starts with `(project-name)`, which means the venv is active.
8. **Install the git hooks:**
   ```powershell
   uv run pre-commit install
   ```
9. **Make and push the first commit.** Create an empty repository on GitHub (no README, .gitignore, or license), then
   run the following, replacing the URL with your repo's. The pre-commit hooks run on this commit; if they change files
   or fail, re-run `git add .` and `git commit` until it passes.
   ```powershell
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/jfaa-josh/<project-name>.git
   git push -u origin HEAD
   ```
   `git remote add` is what connects your local repo to GitHub: `git init` only creates a local repo that knows nothing
   about GitHub, and `origin` is the conventional name for the remote. `push -u origin HEAD` uploads the current branch
   and remembers it, so later you can just run `git push`.
10. **Start coding.** Create your package under `src/`, e.g. `src/my_package/__init__.py` and
    `src/my_package/main.py`. Run with `uv run python -m my_package.main`, or press F5 in VS Code
    (*Python: Current File*).
11. **Replace this README.** Once the project is set up, delete everything in `README.md` and write a new one for the
    new project (what it does, how to install and run it). Keep the file: `pyproject.toml` references it
    (`readme = "README.md"`), and the build fails if it is missing.
