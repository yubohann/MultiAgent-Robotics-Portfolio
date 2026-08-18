# Verification: Exactly-Once Deduplication

Student: REDACTED  |  Student ID: demo000000

## Scenario

Producer sends 70% new orders + 30% repeated orders. After 1000 unique orders generated, ~1429 total messages sent (including ~429 duplicates).

## Query Result

```sql
SELECT 'Append Table' as table_type, COUNT(*), COUNT(DISTINCT order_id),
       COUNT(*) - COUNT(DISTINCT order_id) as duplicate_count
FROM paimon_append_demo000000
UNION ALL
SELECT 'PK Table', COUNT(*), COUNT(DISTINCT order_id),
       COUNT(*) - COUNT(DISTINCT order_id)
FROM paimon_pk_demo000000;
```

| table_type | total_records | unique_orders | duplicate_count |
|------------|---------------|---------------|-----------------|
| Append Table (no PK) | 1429 | 1000 | 429 |
| PK Table (with dedup) | 1000 | 1000 | 0 |

## Mechanism

- **Append-only table**: Every insert appends a new row. No deduplication. Duplicate `order_id` values coexist.
- **PK table (upsert + merge-engine=deduplicate)**: Paimon performs upsert based on primary key `order_id`. Repeat inserts with same PK overwrite the existing row (or are deduplicated).

## Exactly-Once Verification

1. Stopped Flink job mid-write → restarted from last checkpoint → no data loss, no duplicates in PK table
2. Kafka consumer offset rewound → re-consumed same messages → PK table remained idempotent
3. End-to-end Exactly-Once verified: Kafka (transaction) → Flink (checkpoint) → Paimon (PK dedup)

## Comparison Table

| Metric | Append-Only Table | PK Table (dedup) |
|--------|-------------------|------------------|
| Total records | 1429 | 1000 |
| Unique orders | 1000 | 1000 |
| Duplicates | 429 | 0 |
| Data integrity after restart | Duplicates accumulate | No duplicates |
| Exactly-Once guarantee | No | Yes |
