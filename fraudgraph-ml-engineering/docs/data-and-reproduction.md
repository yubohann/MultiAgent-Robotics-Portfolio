# Data and Reproduction

## Data policy

This repository contains no transaction records, benchmark archives, generated graph binaries, checkpoints, or experiment results. Obtain each dataset from its legitimate source and review its usage terms before running an experiment. Store all source assets below the ignored `data/` directory.

## Expected layouts

| Dataset adapter | Default local root | Required source assets |
| --- | --- | --- |
| IEEE-CIS | `data/ieee_cis/` | IEEE-CIS transaction and identity CSV files; use `--ieee_data_root` when the files live elsewhere. |
| Elliptic | `data/elliptic/` | `elliptic_txs_classes.csv`, `elliptic_txs_features.csv`, and `elliptic_txs_edgelist.csv`. |
| AMLSim | `data/amlsim/outputs/<simulation>/` | Generated `accounts.csv`, transaction CSV, and alert/SAR files from an AMLSim simulation. |
| Credit-card fraud | `data/ccfd/` or `data/ulb/` | `creditcard.csv` containing `Time`, `Amount`, and `Class`. |
| DeFi protocol | `data/defi_protocol_ethereum/` | Source tables expected by the DeFi protocol adapter. |
| Ethereum phishing | `data/ethereum_phishing/` | Cleaned user and transaction tables accepted by the adapter. |
| Ethereum Ponzi | `data/ethereum_ponzi/` | Positive samples plus an explicit negative-address set passed with `--ethereum_ponzi_negative_users_path`. |
| DeFi rug pull | `data/defi_rug_pull/` | Curated incident data plus an explicit negative-address set passed with `--defi_rug_pull_negative_users_path`. |
| SplitGNN graph benchmarks | `data/graphs/` | DGL graph files for `amazon`, `yelp`, or `comp`. |

The code writes graph cache files to `data/graphs/` and cache shards to `data/graphs/cache/`. It writes checkpoints, TensorBoard events, summaries, audits, and reports to `artifacts/`.

## Reproduction procedure

1. Install either the CPU or CUDA 12.1 dependency profile from the root README.
2. Download an authorized data source and place it in the matching local directory, or provide its location with the dataset-specific CLI flag.
3. Start with a one-round, TensorBoard-disabled run to validate parsing, graph construction, and device placement.
4. Record the command, seed, package versions, dataset revision, and generated summary before comparing metrics.
5. Run the named experiment scripts only after the base pipeline succeeds for the same data revision.

## Reproducibility boundaries

- Experimental outputs from the previous research workspace were intentionally not published because they include large generated files and may not be distributable.
- IEEE-CIS, Elliptic, AMLSim, and on-chain source data differ in license, preprocessing, and scale. Do not compare results across them without preserving the dataset-specific protocol.
- The training code supports deterministic seeds where the underlying framework and hardware allow it, but GPU kernels and third-party libraries can still introduce nondeterminism.
