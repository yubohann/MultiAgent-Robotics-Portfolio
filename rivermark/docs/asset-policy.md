# Asset Policy

`rivermark-benchmark` is the single source of truth for Rivermark benchmark
code, schemas, task specifications, releases, and demo manifests. Users must
vendor NVIDIA Isaac Sim and Isaac Lab under their own terms; NVIDIA Rivermark
USD and material content, CF2X USD, third-party checkpoints, private target
manifests, and raw rendered recordings stay user-installed.

Local assets are resolved in this order.

1. `RIVERMARK_ASSET_ROOT` environment variable.
2. `config/local_assets.example.json`, copied to an ignored local file.
3. an explicitly supplied command-line path.

Every resolved asset must be fingerprinted into a run receipt. Assets with an
unknown license or a missing fingerprint serve local diagnostics, while
benchmark admission and release demos require a fingerprint and a recorded
license.

The repository includes `rivermark_benchmark.ops.asset_provenance` for a bounded,
Isaac-free scan of USD metadata. It records file identities and recognizable
external references, and leaves redistribution decisions to a human. A scan of
the official Rivermark wrapper or the current CF2X binary exposes an external
Nucleus or content marker, so both remain user-installed runtime dependencies.

City-Lite authored layers may live in a separate source repository maintained by
the owner, and the NVIDIA assets they reference keep their own redistribution
terms. A public code release may reference a user-installed, version-pinned
asset root, while the resolved upstream USD, textures, and materials stay out of
Git and data shards.

The legacy MD-QD-Swarm tree stays outside the import graph. A future migration
adapter may reference a user-provided Isaac installation and scene path, while
legacy targets, scorer manifests, traces, and results remain untouched.
