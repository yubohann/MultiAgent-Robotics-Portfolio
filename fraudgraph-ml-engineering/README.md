# FraudGraph ML Engineering

**A reproducible engineering portfolio for graph-and-sequence fraud detection across financial transaction datasets.**

FraudGraph ML Engineering packages the complete research code behind a hybrid `SplitGNN + Transformer` fraud-detection workflow. It constructs graph, relation-sequence, and event-sequence views; trains a multimodal classifier; records experiment artifacts; and provides repeatable protocol scripts for smoke tests, ablations, tuning, and report generation.

This is a source-available engineering repository, not a redistributed dataset or pretrained-model release. Dataset files, generated graphs, checkpoints, TensorBoard logs, and experimental outputs remain local by design.

[View the architecture diagram source](docs/architecture.mmd)

Research framing, executable protocol mapping, evaluation rules, and replication boundaries are documented in [docs/research-protocol.md](docs/research-protocol.md), [docs/experiment-catalog.md](docs/experiment-catalog.md), and [docs/reproducibility-checklist.md](docs/reproducibility-checklist.md). Citation metadata is provided in [CITATION.cff](CITATION.cff).

## What is included

- A deterministic hybrid training pipeline with graph, sequence, and fusion branches.
- Dataset adapters for IEEE-CIS, Elliptic, AMLSim, credit-card fraud, DeFi, Ethereum phishing, Ethereum Ponzi, and DeFi rug-pull data.
- A vendored SplitGNN encoder integration, with attribution in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- Dataset registry, run artifacts, checkpoint compatibility, evaluation, inference, embedding analysis, and TensorBoard auditing utilities.
- Focused experiment scripts for smoke tests, mainline runs, fusion/low-label ablations, IEEE-CIS acceptance checks, tuning, and paper-package reporting.
- Pinned CPU and CUDA 12.1 environment definitions.
- A lightweight CI quality gate that runs on every push and pull request across Python 3.10 and 3.12.

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
src/fraud_ml_engineering/     Installable package and training CLI
src/.../vendor/splitgnn/      SplitGNN research encoder integration
configs/                      Dataset and experiment configurations
scripts/                      Reproducible experiment and reporting commands
data/                         Local-only datasets and generated DGL graphs
artifacts/                    Local-only checkpoints, logs, reports, and outputs
docs/                         Architecture, data contract, and engineering notes
tests/                        Fast structural and path-contract tests
```

## Environment

Use Python 3.10 through 3.12. The CPU profile is suitable for code checks and small previews. Training realistic graph workloads benefits from the CUDA 12.1 profile.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev]
python -m pip install -r requirements/requirements-cpu.txt
```

For CUDA 12.1, create the environment appropriate to the target GPU, then install the pinned profile:

```powershell
python -m pip install -r requirements/requirements-cu121.txt
python -m pip install -e .[dev] --no-deps
```

## Data setup

Place externally obtained source data beneath `data/`, following the dataset-specific layout in [docs/data-and-reproduction.md](docs/data-and-reproduction.md). Graph cache files are created under `data/graphs/`; experiment outputs go to `artifacts/`. These locations are intentionally ignored by Git.

## Run the pipeline

After preparing a dataset, use the package CLI. The exact flags available for every dataset are documented by the command itself.

```powershell
python -m fraud_ml_engineering --help

# Elliptic example
python -m fraud_ml_engineering --dataset elliptic --rounds 1 --local_epochs 1 --disable_tb

# IEEE-CIS cache construction example
python -m fraud_ml_engineering --dataset ieee --ieee_build_cache_only --ieee_data_root data/ieee_cis
```

The focused engineering workflows are available as scripts:

```powershell
python scripts/run_splitgnn_smoke_suite.py --dataset comp --device cpu
python scripts/run_hybrid_mainline_protocol.py --help
python scripts/run_hybrid_fusion_ablation.py --help
python scripts/run_hybrid_low_label_mechanism_ablation.py --help
python scripts/run_ieee_acceptance_matrix.py --help
python scripts/run_ieee_splitgnn_tuning.py --help
```

## Reproduction Workflow

Use the following sequence after placing an authorized source dataset in the documented `data/` layout:

```powershell
python scripts/validate_repository.py
python -m pytest
python scripts/run_splitgnn_smoke_suite.py --dataset comp --device cpu
```

The smoke workflow verifies the training and evaluation path on its configured lightweight dataset route. Mainline, ablation, tuning, and report commands remain in `scripts/` and use their explicit configuration files.

## Record run provenance

Create a dependency-light manifest before a long experiment to preserve the Git revision, runtime, dataset, seed, configuration reference, and exact command:

```powershell
python scripts/record_run_manifest.py --output artifacts/elliptic/manifest.json --dataset elliptic --seed 42 --config configs/experiments/onchain_main_selection.yaml -- python -m fraud_ml_engineering --dataset elliptic --rounds 20 --local_epochs 2 --disable_tb
```

The command records metadata only; it does not start training. See [docs/experiment-manifest.md](docs/experiment-manifest.md) for the schema and handling guidance.

Use [docs/comparison-report-schema.md](docs/comparison-report-schema.md) when consolidating completed results. Its report generator accepts only explicit, validation-selected records with a stated data revision and split policy; it never searches for or promotes historical best runs.

## Verification

```powershell
python scripts/validate_repository.py
python -m pytest
python -m compileall -q src scripts tests
python -m build
python -m pip install --force-reinstall --no-deps dist/*.whl
python -m fraud_ml_engineering --help
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
