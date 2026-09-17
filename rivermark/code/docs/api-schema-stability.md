# API And Schema Stability

This policy covers the public Python package, command-line tools, JSON schemas,
release manifests, and observation projections. A development fixture stays a
development fixture, and external assets keep their own redistribution terms.

## Support levels

- **Stable**. Documented commands, `rivermark/` manifest schemas, and the
  observation ABI used by a public release. Breaking changes require a major
  schema or dataset-version change and a migration note.
- **Development**. Isaac capture internals, pilot-only projections, local scorer
  services, and modules marked development-only in their manifest. These may
  change between minor revisions, so callers pin a commit.
- **Private**. Scorer truth, credentials, local asset paths, and operator
  artifacts. These stay outside the API and inside the local environment.

The owning schema or manifest sets the support level of each command and field,
and narrative documentation follows it.

## Compatibility rules

1. A patch release may fix validation, documentation, or an implementation bug
   while keeping the accepted wire meaning.
2. A minor release may add optional fields or new modalities. Readers ignore
   unknown optional fields, and writers continue to emit required fields for the
   declared schema version.
3. Removing a required field or changing units, coordinate frames, action
   timing, dtype and shape, split semantics, or hash meaning requires a new
   major schema version and an explicit migration document.
4. ABI compatibility is checked with
   `assess_observation_abi_compatibility`. A passing compatibility report
   describes ABI agreement, and formal admission and license decisions follow
   their own processes.
5. A release supplies bytes through a new version. Defective or withdrawn shards
   use the hash-bound defect and tombstone mechanism in the release manifest,
   and a newer release carries corrected bytes.

## Deprecation

Deprecations are announced in the changelog and the owning schema or document.
The notice names the replacement, first affected version, removal version or the
condition that permits removal, and a migration example. Deprecation of
development interfaces may conclude at the next minor revision. Stable
interfaces require at least one release with the deprecation notice before
removal. Security or integrity defects may require immediate fail-closed
removal, with the reason recorded in the release notes.

## Version pinning and reports

Research results bind the dataset version, source revision, ABI hash, scorer
version, configuration hash, checkpoint hash where applicable, and split
authority. Determinism reports include the exact command and environment
fingerprint. A passing CPU smoke covers the CPU path, while native Isaac and
hardware execution need their own receipts.

## Change review

Changes to schema meaning, metric definitions, split authority, privacy scope,
or release gates require focused tests and a changelog or backlog entry.
Implementation changes keep the existing owner and avoid parallel contracts.
