# Validation and Formal Admission

A captured episode becomes formal data only through `rivermark_benchmark.formal_dataset`. This is a release-control mechanism, not a replacement for Isaac Lab or hardware collection.

## Why trust is layered

The episode manifest proves the declared ABI and file bindings. A formal capture receipt proves that a separate validation process asserted the required audits. Neither document is trustworthy on its own, because any process can write JSON. Admission therefore requires the release operator to provide an explicit allowlist of accepted `formal_capture_receipt.json` SHA-256 values. A candidate's self-reported receipt hash is never used as authorization.

## Independent validation

`rivermark_benchmark.isaac_validate` reopens the raw artifacts and checks: stage identity, collision-proxy binding, sensor synchronization, action causality, visual and LiDAR intrusion gates, contacts, route/condition realization, target-visibility evidence, provenance, and hash bindings.

Only after validation passes can the packer create an admission candidate:

```powershell
rivermark-isaac-pack <capture> <independent-validation.json> `
  <evaluator-manifest.json> <pack-spec.json> <candidate-output> `
  --collection-protocol .\collection-protocol.json
```

The packer recomputes the protocol hash, cell split, episode index, seed, and condition request before building a candidate.

## Candidate contract

A source episode must contain `episode_manifest.json`, `lineage.json`, and `formal_capture_receipt.json` at its root. `lineage.json` holds opaque SHA-256 commitments for ten frozen split axes. The formal receipt binds the manifest and lineage hashes, declares an `isaaclab` or `hardware` backend, and requires independent positive audits for online capture, timestamps, pose closure, action causality, sensor decode, and policy leakage.

Candidates are closed-world directories: unbound files, symbolic links, and directories named for private truth cause rejection. Evaluator-private payloads are never allowed beside the candidate.

## Collection and quarantine

```powershell
$env:PYTHONPATH = (Resolve-Path .\src)
python -m rivermark_benchmark.formal_dataset collect C:\captures\episode-0001 .\rivermark `
  --trusted-receipt-sha256 <formal_capture_receipt_sha256> `
  --supply-chain-manifest <signed-release-supply-chain.json>
```

The collector never moves, edits, or deletes the source capture. On failure it writes a canonical reason record under `rivermark/quarantine/` — hashes and validation reasons, not source paths or evaluator truth. On success it stages a public projection and promotes it with an atomic rename.

Before staging, the collector verifies the supply-chain manifest in release mode, including its SBOM and detached signature. The resulting `admission.json` binds the canonical supply-chain hash and release ID; a later episode cannot introduce a different decision into the same dataset root.

## Split authority

Split assignments are predeclared in the candidate manifest and cannot change after capture. `split-plan` validates the assignments before collection; a group that spans multiple splits, or a reused trajectory lineage, is rejected. After each successful collection the collector rebuilds the deterministic `manifests/split_authority.json` and `manifests/dataset_index.json`.

## Release verification

```powershell
python -m rivermark_benchmark.formal_dataset verify-dataset .\rivermark
```

Verification rehashes every payload, revalidates the manifest and admission record, rejects unbound files and symlinks, checks lineage split groups, and compares the stored index against a fresh deterministic reconstruction. A changed payload, stale index, or accidental private directory is a hard failure.
