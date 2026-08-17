# Governance

This section covers the rules that keep the benchmark honest: where assets come from, who may redistribute what, and how the interfaces stay stable.

## Asset provenance

Rivermark ships source, schemas, contracts, and checks only. Simulator assets are a separate runtime dependency: users install a compatible Isaac Sim/Nucleus package under its own terms and provide local paths through an ignored asset file.

The repository keeps raw NVIDIA USD/materials/textures, the unresolved CF2X binary, composed City-Lite layers, videos, and derived payloads out of Git until a redistribution decision covers the exact artifact. Formal release validation fails while any released asset is unresolved, internal-only, or missing a human decision record.

You can inspect a local installation without importing Isaac:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m rivermark_benchmark.asset_provenance C:\path\to\cf2x.usd C:\path\to\rivermark.usd
```

A blocked result is evidence to keep the file local, not a legal conclusion. A passed scan still reports `license_status: unresolved` until a human records an applicable upstream license.

## Licensing

- Rivermark-authored source, schemas, and docs are Apache-2.0.
- The `isaac_drone_racer` configuration is BSD-3-Clause; that configuration license does not grant redistribution rights for the resolved binary USD or its upstream references.
- Isaac Sim and Isaac Lab are runtime prerequisites, obtained under their respective NVIDIA terms.
- The official Isaac Sim repository license is not a blanket license for its 3D models and materials. A public data release requires an express grant for both the upstream asset and the derived-data/video scope.

## API and schema stability

The support levels are explicit:

- **Stable** — documented commands, release-manifest schemas, and the observation ABI used by a public release. Breaking changes require a major version change and a migration note.
- **Development** — capture internals, pilot projections, local evaluator services. These may change between minor revisions; callers should pin a commit.
- **Private** — evaluator truth, credentials, local asset paths. Never copied into a public payload or issue.

Compatibility rules in short: a patch may fix bugs without changing wire meaning; a minor release may add optional fields; removing a required field or changing units, coordinate frames, action timing, or hash meaning requires a new major version and a migration document. Releases never replace bytes in place — defective shards use a hash-bound defect/tombstone mechanism and a newer release supplies corrected bytes.

Deprecations are announced in the changelog with a replacement, first affected version, removal version, and a migration example. Stable interfaces get at least one release with the notice before removal; security or integrity defects may require immediate removal with the reason recorded.

## External pretraining data

External embodied data (for example Open-AoE, a smartphone-collected human manipulation dataset under Apache-2.0) may enter the research workflow only as **pretraining** for visual or temporal representations. Its 20D MANO-hand actions are incompatible with CF2X flight actions and may never enter the formal episode statistics. A read-only audit adapter checks segment layout, calibration, array shapes, and hashes, and emits a path-free provenance manifest declaring `external_pretraining_only`. Any resulting checkpoint must still be evaluated in the native Isaac T2 loop.
