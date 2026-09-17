# Acceptance Checklist: Data Skew

Student ID: `demo000000`

## 1.1 Environment

- [ ] Kafka topic `click_stream_demo000000` exists.
- [ ] Topic has 5 partitions and replication factor 1.
- [ ] Explain the definition, causes, and business risks of data skew.
- [ ] Screenshot `1-1`: Kafka topic list.

## 1.2 Reproduce Skew

- [ ] Python producer runs successfully.
- [ ] Data distribution is about 90 percent `iPhone15` and 10 percent other products.
- [ ] Flink job `SkewDemo_demo000000` starts successfully.
- [ ] Explain why direct `keyBy(item_id)` creates hot-key skew.
- [ ] Screenshot `1-2`: Python producer output.

## 1.3 Observe Back Pressure

- [ ] Flink Web UI shows uneven Subtask load.
- [ ] Hot Subtask receives about 9 to 10 times more records than others.
- [ ] Back Pressure tab shows HIGH or MEDIUM before optimization.
- [ ] Checkpoint time is uneven.
- [ ] Screenshot `1-3`: Task Managers page before optimization.
- [ ] Screenshot `1-4`: Back Pressure tab before optimization.
- [ ] Screenshot `1-5`: Checkpoints tab before optimization.

## 1.4 Two-Stage Aggregation

- [ ] Salted local aggregation is implemented.
- [ ] De-salted global aggregation is implemented.
- [ ] Flink job `SkewSolution_demo000000` starts successfully.
- [ ] Explain the role of salting and when to remove the salt.

## 1.5 Verify Optimization

- [ ] Records Received is balanced across Subtasks.
- [ ] Back Pressure shows LOW.
- [ ] Checkpoint time is shorter and more stable.
- [ ] Overall throughput improves.
- [ ] Screenshot `1-6`: Task Managers page after optimization.
- [ ] Screenshot `1-7`: Back Pressure tab after optimization.

## Performance Table

| Metric | Before | After | Improvement |
|---|---|---|---|
| Max Subtask load | | | |
| Back pressure status | | | |
| Checkpoint time | | | |
| Throughput | | | |
