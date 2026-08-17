# Security And Integrity Reports

Do not disclose evaluator credentials, private target manifests, local asset
paths, or unredacted capture artifacts in a public issue. Those files are
outside the repository's public data boundary.

## Report privately

For a suspected vulnerability, credential exposure, evaluator leakage, or a
problem that could allow a result or payload to be altered, contact the
repository owner through the private GitHub security-advisory channel. Include
the affected commit or release, a minimal reproduction that contains no
secrets, and the impact. Do not test against a public evaluator or attempt to
recover hidden targets.

If private advisory access is unavailable, open a minimal issue titled
`security contact requested` without technical details and wait for a private
reply.

## Data and reproducibility issues

Use the corresponding issue template for public schema, shard, download, or
reproduction problems. Redact absolute paths, tokens, private evaluator
fields, and raw sensor payloads. A reported defect does not authorize changing
released bytes; maintainers use the hash-bound defect and tombstone mechanism
and publish corrections in a new version.

## Response boundary

The local evaluator prototype is not a public network service. It has no
public credentials, blind backend, or leaderboard. Do not infer those controls
from the presence of an in-process test service.
