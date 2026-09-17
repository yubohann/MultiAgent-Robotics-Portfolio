# Acceptance Checklist: Flink State Bloat

Student ID: `demo000000`

## 3.1 Environment

- [ ] Flink state backend is configured correctly.
- [ ] Checkpoint directory is configured.
- [ ] Explain the definition and causes of state bloat.
- [ ] Explain how state size affects job stability.

## 3.2 Reproduce State Bloat

- [ ] A large number of keys are used for aggregation.
- [ ] State size grows continuously.
- [ ] Checkpoint duration keeps increasing.
- [ ] Describe observed state growth.
- [ ] Explain the impact on memory and checkpointing.
- [ ] Screenshot `3-1`: Flink Web UI state size before optimization.

## 3.3 Optimize State

- [ ] State TTL is configured.
- [ ] State backend optimization is used, such as RocksDB.
- [ ] State cleanup strategy is implemented.
- [ ] Explain State TTL and backend tradeoffs.

## 3.4 Verify Optimization

- [ ] State size is controlled.
- [ ] Checkpoint duration becomes stable.
- [ ] Job stability improves.
- [ ] Compare state size and checkpoint duration before and after optimization.
- [ ] Screenshot `3-2`: state monitoring after optimization.
