# Acceptance Checklist: Exactly-Once And Duplicate Data

Student ID: `demo000000`

## 5.1 Environment

- [ ] Kafka transactions are configured.
- [ ] Flink Checkpoint is configured correctly.
- [ ] Paimon transactional writes are configured.
- [ ] Explain exactly-once semantics.
- [ ] Explain two-phase commit.

## 5.2 Reproduce Problem

- [ ] Job failure and restart are simulated.
- [ ] Duplicate or lost data is observed before optimization.
- [ ] Problem behavior is recorded.
- [ ] Explain why duplicate or lost data can appear.
- [ ] Screenshot `5-1`: query result showing duplicate data.

## 5.3 Implement Exactly-Once

- [ ] Flink Checkpoint is configured for exactly-once mode.
- [ ] Kafka transactions are enabled.
- [ ] Paimon transactional writes are enabled.
- [ ] Explain how Checkpoint, Kafka transaction, and Paimon transaction work together.

## 5.4 Verify Result

- [ ] No duplicate data after job restart.
- [ ] No data loss after job restart.
- [ ] End-to-end exactly-once is verified.
- [ ] Explain the verification process and result.
- [ ] Screenshot `5-2`: query result showing no duplicates after restart.
