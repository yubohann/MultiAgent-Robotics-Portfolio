# Asset Policy

`rivermark-benchmark` is the single source of truth for Rivermark benchmark
code, schemas, task specifications, releases, and demo manifests. It does not
vendor NVIDIA Isaac Sim, Isaac Lab, NVIDIA Rivermark USD/material content,
CF2X USD, third-party checkpoints, private target manifests, or raw rendered
recordings.

Local assets are resolved in this order:

1. `RIVERMARK_ASSET_ROOT` environment variable;
2. `config/local_assets.example.json`, copied to an ignored local file;
3. an explicitly supplied command-line path.

Every resolved asset must be fingerprinted into a run receipt. Assets with an
unknown license or a missing fingerprint may be used only for local diagnostics,
not benchmark admission or release demos.

The repository includes `rivermark_benchmark.asset_provenance` for a bounded,
Isaac-free scan of USD metadata. It records file hash and recognizable external
references; it never infers a redistribution grant. A scan of the official
Rivermark wrapper or the current CF2X binary is expected to expose an external
Nucleus/content marker, so both remain user-installed runtime dependencies.

City-Lite authored layers may be maintained by the repository owner in a
separate source repository, but they are not a grant to redistribute the
NVIDIA assets they reference. A public code release may reference a
user-installed, version-pinned asset root; it must never copy the resolved
upstream USD, textures, or materials into Git or a data shard.

The legacy MD-QD-Swarm tree is not an import dependency. A future migration
adapter may reference a user-provided Isaac installation and scene path, but it
must never read legacy targets, evaluator manifests, traces, or results.
