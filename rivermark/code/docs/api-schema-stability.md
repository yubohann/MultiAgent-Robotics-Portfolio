# API And Schema Stability

This policy applies to the public Python package, command-line tools, JSON
schemas, release manifests, and observation projections. It does not turn a
development fixture into a released dataset and does not grant permission to
redistribute external assets.

## Support levels

- **Stable:** documented commands, `rivermark/` manifest schemas, and the
  observation ABI used by a public release. Breaking changes require a major
  schema or dataset-version change and a migration note.
- **Development:** Isaac capture internals, pilot-only projections, local
  evaluator services, and modules marked development-only in their manifest.
  These may change between minor revisions; callers should pin a commit.
- **Private:** evaluator truth, credentials, local asset paths, and operator
  artifacts. They are not an API and must never be copied into a public
  payload, report, or issue.

The support level of a command or field is defined by its owning schema or
manifest. Narrative documentation cannot promote it to Stable.

## Compatibility rules

1. A patch release may fix validation, documentation, or an implementation
   bug without changing the accepted wire meaning.
2. A minor release may add optional fields or new modalities. Readers must
   ignore unknown optional fields and writers must continue to emit required
   fields for the declared schema version.
3. Removing a required field, changing units, coordinate frames, action timing,
   dtype/shape, split semantics, or hash meaning requires a new major schema
   version and an explicit migration document.
4. ABI compatibility is checked with
   `assess_observation_abi_compatibility`; a passing compatibility report is
   not a formal-admission or license decision.
5. A release never replaces bytes in place. Defective or withdrawn shards use
   the hash-bound defect/tombstone mechanism in the release manifest and a
   newer release supplies corrected bytes.

## Deprecation

Deprecations are announced in the changelog and the owning schema/document.
The notice names the replacement, first affected version, removal version (or
the condition that permits removal), and a migration example. Deprecated
development interfaces may be removed at the next minor revision; Stable
interfaces require at least one release with the deprecation notice before
removal. Security or integrity defects may require immediate fail-closed
removal, with the reason recorded in the release notes.

## Version pinning and reports

Research results must bind the dataset version, source revision, ABI hash,
evaluator version, configuration hash, checkpoint hash (when applicable), and
split authority. Reproducibility reports should include the exact command and
environment fingerprint. A passing CPU smoke is evidence for the CPU path
only; it is not evidence of native Isaac or hardware execution.

## Change review

Changes to schema meaning, metric definitions, split authority, privacy
boundaries, or release gates require focused tests and an entry in the
changelog/backlog. Changes to implementation details should preserve the
existing owner boundary instead of adding a parallel contract.
