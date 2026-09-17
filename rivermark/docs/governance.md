# Governance

This section covers asset provenance, redistribution rules, and interface stability.

## Asset provenance

Rivermark ships source, schemas, contracts, and checks only. Simulator assets are a separate runtime dependency. Users install a compatible Isaac Sim or Nucleus package under its own terms and provide local paths through an ignored asset file.

The repository keeps raw NVIDIA USD, materials, textures, the unresolved CF2X binary, composed City-Lite layers, videos, and derived payloads outside Git until a redistribution decision covers the exact artifact. Formal release validation demands a resolved license and a recorded human decision for every released asset.

`rivermark_benchmark.asset_provenance` inspects a local installation on the CPU path.

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m rivermark_benchmark.asset_provenance C:\path\to\cf2x.usd C:\path\to\rivermark.usd
```

A blocked result points to keeping the file local and leaves the legal call to a human. A passed scan still reports `license_status: unresolved` until a human records an applicable upstream license.

## Licensing

- Rivermark-authored source, schemas, and docs are Apache-2.0.
- The `isaac_drone_racer` configuration carries BSD-3-Clause for the configuration itself. Redistribution rights for the resolved binary USD and its upstream references require a separate grant.
- Isaac Sim and Isaac Lab are runtime prerequisites, obtained under their respective NVIDIA terms.
- The official Isaac Sim repository license covers its code, while its 3D models and materials carry their own terms. A public data release requires an express grant for both the upstream asset and the derived data and video scope.

## API and schema stability

The support levels are explicit.

- **Stable**. Documented commands, release-manifest schemas, and the observation ABI used by a public release. Breaking changes require a major version change and a migration note.
- **Development**. Capture internals, pilot projections, and local scorer services. These may change between minor revisions, so callers pin a commit.
- **Private**. Scorer truth, credentials, and local asset paths. These stay inside the local environment.

Compatibility in short. A patch may fix bugs while keeping the wire meaning, a minor release may add optional fields, and removing a required field or changing units, coordinate frames, action timing, or field meaning requires a new major version and a migration document. Releases supply bytes through new versions, defective shards use an recorded defect and tombstone mechanism, and a newer release carries corrected bytes.

Deprecations are announced in the changelog with a replacement, first affected version, removal version, and a migration example. Stable interfaces get at least one release with the notice before removal, and security or integrity defects may require immediate removal with the reason recorded.

## External pretraining data

External embodied data, for example Open-AoE, a smartphone-collected human manipulation dataset under Apache-2.0, may enter the research workflow as pretraining for visual or temporal representations. Its 20D MANO-hand actions are incompatible with CF2X flight actions, and the formal episode statistics stay on CF2X flight actions. A read-only audit adapter checks segment layout, calibration, array shapes, and identities, and emits a path-free provenance manifest declaring `external_pretraining_only`. A resulting checkpoint then enters the native Isaac T2 loop for scoring.
