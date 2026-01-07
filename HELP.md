# HELP

## How to install project dependencies?

If you already have a `pyproject.toml` file, you can install the dependencies with:

```sh
uv sync
```

## How to create a new venv (virtual environment)?

Create a virtual environment with python `3.11`:

```sh
uv venv -p 3.11 
```

- adds a `.venv` folder

Initialize a uv project and a git repository:

```sh
uv init .       
```

- adds a `pyproject.toml`
- adds a `.gitignore`
- adds a `README.md`
- adds a `main.py`

Install packages

```sh
uv add pandas seaborn scikit-learn 
```

- adds the package names to the `pyproject.toml` file under the `[dependencies]` section
- downloads the packages into the `.venv` folder

## How to check if I am running in a uv environment?

```python
import sys
print(sys.executable.endswith(".venv/bin/python"))
```

## How to pull?

Situation: you `git clone <repo>` then you made changes locally. Now you want to pull the latest changes from remote without losing your local changes.

if `git pull` gives merge conflicts, do:

```sh
git stash     # save your local changes temporarily
git pull      # pull the latest changes from remote
git stash pop # re-apply your local changes
```

You might have to resolve conflicts, and commit the changes.

Or, do `git clone <repo>` elsewhere.

## AI Policy

- You may enable the in-editor **Co-pilot** for AI-assisted auto-complete. Remember however, that you are the **Pilot**. This means, you are responsible for the code.
- You may ask it to explain concepts or errors.
- You may not ask it to solve the task.
