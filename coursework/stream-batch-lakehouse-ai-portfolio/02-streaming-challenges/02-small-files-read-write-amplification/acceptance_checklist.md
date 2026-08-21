# Acceptance Checklist: Small Files And Read-Write Amplification

Student ID: `demo000000`

## 2.1 Environment

- [ ] MinIO bucket `paimon-data-demo000000` exists.
- [ ] Paimon Catalog is configured correctly.
- [ ] Explain the definition and causes of the small-file problem.
- [ ] Explain how small files harm query performance.
- [ ] Screenshot `2-1`: MinIO bucket list.

## 2.2 Reproduce Small Files

- [ ] High-frequency writes are sent to a Paimon table.
- [ ] Many small files are visible in MinIO.
- [ ] Query performance becomes slower.
- [ ] Describe the observed small-file growth trend.
- [ ] Screenshot `2-2`: small files in the Paimon table directory.

## 2.3 Compaction Optimization

- [ ] Paimon automatic compaction strategy is configured.
- [ ] Manual compaction is triggered.
- [ ] The compaction process is observed.
- [ ] Explain Paimon compaction and its key parameters.

## 2.4 Verify Optimization

- [ ] File count is significantly reduced.
- [ ] File size is closer to the recommended block size.
- [ ] Query performance improves.
- [ ] Screenshot `2-3`: file list after compaction.

## Performance Table

| Metric | Before | After | Improvement |
|---|---|---|---|
| Total files | | | |
| Average file size | | | |
| Query response time | | | |
