# Reproducibility Protocol

This protocol makes the dependency-light 2D environment path easy to inspect, rerun, and
compare. It is intentionally separate from policy-training claims: training can involve
PyTorch, accelerators, and larger experiment budgets that are not required to validate the
core simulator contract.

## Scope

The public check exercises three deterministic scenario families:

| Scenario | Purpose | Default team |
| --- | --- | --- |
| `single-static` | Fixed gate traversal and single-agent observation contract | 1 |
| `multi-static` | Variable-team graph observations, slots, planning, and safety path | Caller supplied |
| `multi-dynamic` | Dynamic gate-density runtime with moving posts | 8 |

Each requested seed is reset and stepped twice with the same zero-action input. The report
compares every public rollout diagnostic from both runs. `deterministic: true` therefore
means the tested core path produced identical diagnostics on that runtime.

## Locked Core Setup

The committed `uv.lock` records the resolved core and developer dependencies. On a fresh
clone, run:

~~~powershell
uv sync --extra dev
uv run python -m aerogate info
~~~

If `uv` is unavailable, use the `pip` workflow in the README. That path observes the
version ranges in `pyproject.toml`, so it is convenient but less exact than the lockfile.

## Reference Checks

Run a small contract check first:

~~~powershell
uv run python -m aerogate smoke --scenario single-static --seed 7 --steps 8
~~~

Then create a machine-readable multi-agent report:

~~~powershell
uv run python -m aerogate reproduce --scenario multi-static --agents 4 --seeds 3 7 11 --steps 8 --output artifacts/reproducibility/multi-static.json
~~~

The JSON report contains the scenario, seeds, dimensions, reward total, episode state,
clearance, pair separation, formation error where applicable, package version, NumPy version,
and Python runtime. A diagnostic without a finite value is encoded as JSON `null`, never as a
non-standard NaN or Infinity literal. Store the report beside figures or tables when reporting
an experiment. Generated files beneath `artifacts/` are intentionally ignored.

For the dynamic task, keep the documented eight-drone reference setting:

~~~powershell
uv run python -m aerogate reproduce --scenario multi-dynamic --agents 8 --seeds 3 7 11 --steps 8
~~~

## Quality Gate

Before comparing a modified configuration or reporting a new result, run:

~~~powershell
uv run python -m pytest
uv run ruff check aerogate tests
uv run ruff format --check aerogate tests
~~~

The CI workflow runs the same test, lint, smoke, and multi-static reproduction commands. It
uploads the report as an artifact so a reviewer can inspect the exact diagnostics produced by
the hosted run.

## Interpretation Limits

This protocol validates deterministic core-environment behavior, not learned-policy quality
or physical-drone safety. GPU kernels, training schedules, Isaac Lab rendering, perception,
and real-world vehicle dynamics require their own seeds, hardware disclosure, evaluation
budget, and safety validation.
