# Security And Integrity Reports

Scorer credentials, private target manifests, local asset paths, and unredacted
capture artifacts belong in the private channel. Those files sit outside the
repository's public data scope.

## Report privately

For a suspected vulnerability, credential exposure, scorer leakage, or a
problem that could allow a result or payload to be altered, contact the
repository owner through the private GitHub security-advisory channel. Include
the affected commit or release, a minimal failing case free of secrets, and the
impact. Run tests on local checkouts, with hidden targets private.

When private advisory access is unavailable, open a minimal issue titled
`security contact requested` that asks for a private channel, and wait for a
private reply.

## Data and determinism issues

Use the corresponding issue template for public schema, shard, download, or
replay problems. Redact absolute paths, tokens, private scorer fields, and raw
sensor payloads. Released bytes stay fixed, and maintainers publish corrections
through the recorded defect and tombstone mechanism in a new version.

## Response scope

The local scorer prototype runs as an in-process test service on the
engineering machine. Public credentials, a blind backend, and a leaderboard
belong to a production deployment.
