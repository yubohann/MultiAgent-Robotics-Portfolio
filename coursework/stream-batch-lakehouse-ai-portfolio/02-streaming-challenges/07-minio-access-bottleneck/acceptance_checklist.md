# Acceptance Checklist: MinIO Access Bottleneck

Student ID: `demo000000`

## 7.1 Environment

- [ ] MinIO is running normally.
- [ ] A large amount of data has been written to MinIO.
- [ ] Explain why object-storage access bottlenecks happen.
- [ ] Explain the performance cost of ListObjects.

## 7.2 Reproduce Problem

- [ ] Many ListObjects operations are executed.
- [ ] Query performance degradation is observed.
- [ ] Response time is recorded.
- [ ] Explain metadata-operation bottlenecks in object storage.
- [ ] Screenshot `7-1`: slow query response time.

## 7.3 Implement Optimization

- [ ] Partitioning and bucketing are used.
- [ ] Paimon data layout is optimized.
- [ ] Statistics and predicate pushdown are used.
- [ ] Explain how partitioning, bucketing, and predicate pushdown reduce scanning.

## 7.4 Verify Optimization

- [ ] Query response time is significantly reduced.
- [ ] Scanned data volume is reduced.
- [ ] Overall performance improves.
- [ ] Screenshot `7-2`: fast query response time after optimization.

## Performance Table

| Metric | Before | After | Improvement |
|---|---|---|---|
| Query response time | | | |
| Scanned data volume | | | |
| Throughput | | | |
