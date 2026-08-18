# FraudGraph ML Engineering

**A method framework for graph-and-sequence fraud detection engineering across financial transaction datasets.**

> **Notice**: this is the *method-framework* repository. The core research
> implementation (graph/sequence encoders, fusion classifier, federated
> controller, dataset scheduler, and dataset feature engineering) is part of
> an in-preparation paper and is withheld until publication. See
> [NOTICE.md](NOTICE.md) and [core/](core/).

> **Publication status**: the associated paper is under review and has **not
> been published yet**. Quantitative results and the core implementation are
> therefore **not shown** in this repository; they will be released publicly
> together with the paper once it is out.

> **Open-source plan**: full open-source (complete implementation + results)
> is planned for the **end of October 2026**.

FraudGraph ML Engineering frames fraud detection as a **multi-view** problem:
the same transaction record can be read as a node in a relational graph, as
ordered behavioural sequences, and as an event stream. The framework
constructs graph, relation-sequence, and event-sequence views; fuses them into
a classifier; and orchestrates training with a federated-style controller and
a dataset scheduler — all under a strict reproducibility protocol.

The repository distributes the **method framework**: the research questions,
the protocol, the architecture, dependency-light engineering utilities, and
the public API surface of each research component. Implementations ship with
the paper.

This is a source-available engineering repository, not a redistributed dataset
or pretrained-model release. Dataset files, generated graphs, checkpoints,
TensorBoard logs, and experimental outputs remain local by design.

[View the architecture diagram source](docs/architecture.mmd)

Research framing, executable protocol mapping, evaluation rules, and
replication boundaries are documented in
[docs/research-protocol.md](docs/research-protocol.md),
[docs/experiment-catalog.md](docs/experiment-catalog.md), and
[docs/reproducibility-checklist.md](docs/reproducibility-checklist.md).
The method framework is described in [docs/methodology.md](docs/methodology.md).
Citation metadata is provided in [CITATION.cff](CITATION.cff).

## What is included

- The **method framework** — research questions, protocol, architecture, and
  engineering discipline.
- Dependency-light engineering utilities — repository paths, run artifacts,
  experiment protocol helpers, and a framework validator.
- The **withheld API surface** — `core/` declares the role and interface of
  each research component; implementations ship with the paper.
- A lightweight CI quality gate that runs on every push and pull request
  across Python 3.10 and 3.12.

The framework is described in [docs/methodology.md](docs/methodology.md) and
[docs/architecture.mmd](docs/architecture.mmd); the research protocol lives in
[docs/research-protocol.md](docs/research-protocol.md).

## Research profile

The repository is organized around three falsifiable questions:

1. Does a heterophily-aware graph encoder capture fraud signals that a sequence model misses?
2. Do relation and event sequences add stable signal beyond graph structure alone?
3. Which components remain useful when the labeled training fraction is reduced?

The answer is intentionally established by protocol, not by a single headline metric. Every mainline result should be accompanied by a seed, dataset revision, validation-selection rule, held-out test evaluation, and the relevant ablation comparison.

## Reproducibility contract

- Dataset files are acquired from their legitimate providers and are never silently downloaded by the code.
- Runtime paths resolve relative to the repository: source data under `data/`, graph caches under `data/graphs/`, and generated outputs under `artifacts/`.
- Validation selects checkpoints and thresholds; the held-out test split is evaluated only after selection.
- Seeds, protocol arguments, environment versions, and data revisions belong in the generated run summary.
- A dependency-light repository check is available before installing the full graph-learning stack:

```powershell
python scripts/validate_repository.py
```

## Repository layout

```text
src/fraud_ml_engineering/     Installable package: paths, run artifacts, protocol helpers, CLI surface
core/                         Withheld research components (API surface + role documentation)
scripts/                      Framework validation command
docs/                         Architecture, research protocol, and engineering notes
tests/                        Dependency-light structural and utility tests
```

## Environment

Use Python 3.10 through 3.12. The framework repository itself has **no ML
runtime dependency** — the included utilities, validator, and tests run on a
plain interpreter. The full graph-learning runtime is only required by the
withheld core, which ships with the paper.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

Verify the framework:

```powershell
python scripts/validate_repository.py
python -m pytest
```

## Using the framework

The framework package exposes the engineering surface: repository-local path
resolution, run-artifact creation, and experiment-protocol helpers. The CLI
documents the public command surface; invoking a withheld pipeline reports
the redaction notice.

```powershell
python -m fraud_ml_engineering --help
python -m fraud_ml_engineering --dry-run
```

The withheld core ships with the paper and plugs into this surface unchanged:
`core/` declares each component's role and interface.

## Verification

```powershell
python scripts/validate_repository.py
python -m pytest
python -m compileall -q src scripts tests
```

The test suite is deliberately dependency-light: it validates the package layout, path contract, configuration files, and absence of legacy bare internal imports. Full training requires the optional PyTorch and DGL dependencies plus source datasets.

The same checks are available as `make quality` on systems with GNU Make. GitHub Actions runs the validator, structural tests, compilation, package build, and an installed-wheel CLI smoke test on Python 3.10 and 3.12.

For contribution and review expectations, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Scope and limitations

- Dataset adapters preserve dataset-specific constraints. In particular, Ethereum Ponzi and DeFi rug-pull binary tasks require a separately sourced negative set.
- No performance figures are claimed in this repository without the associated dataset, environment, seed, and protocol artifacts.
- The code is a research engineering portfolio, not a production fraud decision service. Any operational use requires independent privacy, security, bias, calibration, monitoring, and compliance review.

## License and attribution

The repository is source-available under the terms in [LICENSE](LICENSE). The included SplitGNN integration and external data sources have separate provenance and usage considerations documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

*Bohan Yu — research project. Core implementation released with the associated paper.*
