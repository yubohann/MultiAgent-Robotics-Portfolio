# Validation and Formal Admission

A captured episode becomes formal data through `rivermark_benchmark.audit.formal_dataset`, the step between raw captures and a published dataset.

## Validation

`rivermark_benchmark.audit.isaac_validate` reopens the raw artifacts and checks stage identity, collision-proxy binding, sensor synchronization, action causality, visual and LiDAR intrusion gates, contacts, route and condition realization, target-visibility evidence, and file bindings.

After validation passes, the packer creates an admission candidate.

```powershell
rivermark-isaac-pack <capture> <independent-validation.json> `
  <evaluator-manifest.json> <pack-spec.json> <candidate-output> `
  --collection-protocol .\collection-protocol.json
```

The packer recomputes the protocol identity, cell split, episode index, seed, and condition request before building a candidate.

## Candidate Contract

A source episode must contain `episode_manifest.json`, `lineage.json`, and `formal_capture_receipt.json` at its root. `lineage.json` records the ten frozen split axes. The formal receipt binds the manifest and lineage records, declares an `isaaclab` or `hardware` backend, and requires positive audits for online capture, timestamps, pose closure, action causality, sensor decode, and policy leakage.

Candidates are closed-world directories. Unbound files, symbolic links, and directories named for private truth cause rejection, and scorer-private payloads stay outside the candidate tree.

## Collection and Quarantine

```powershell
$env:PYTHONPATH = (Resolve-Path .\src)
python -m rivermark_benchmark.audit.formal_dataset collect C:\captures\episode-0001 .\rivermark `
  --trusted-receipt-identity <formal_capture_receipt_identity> `
  --supply-chain-manifest <signed-release-supply-chain.json>
```

The collector reads the source capture in place. On failure it writes a canonical reason record under `rivermark/quarantine/` with the file records and validation reasons. On success it stages a public projection and promotes it with an atomic rename.

Before staging, the collector checks the supply-chain manifest in release mode, including its SBOM and detached signature. The resulting `admission.json` records the supply-chain identity and release ID, and the dataset root commits to a single supply-chain decision.

## Split Authority

Split assignments are predeclared in the candidate manifest and stay frozen after capture. `split-plan` validates the assignments before collection, rejecting a group that spans multiple splits or reuses a trajectory lineage. After each successful collection the collector rebuilds the deterministic `manifests/split_authority.json` and `manifests/dataset_index.json`.

## Release Verification

```powershell
python -m rivermark_benchmark.audit.formal_dataset verify-dataset .\rivermark
```

Verification rechecks every payload, revalidates the manifest and admission record, rejects unbound files and symlinks, checks lineage split groups, and compares the stored index against a fresh deterministic reconstruction. A changed payload, a stale index, or an accidental private directory is a hard failure.
