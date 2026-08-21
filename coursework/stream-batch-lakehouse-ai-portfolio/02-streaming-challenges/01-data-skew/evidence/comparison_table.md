# Performance Comparison: Data Skew

Student: REDACTED  |  Student ID: demo000000

## Before Optimization (SkewDemo_demo000000)

Flink job `SkewDemo_demo000000` uses direct `keyBy(item_id)` with 4 parallel subtasks.

Observation:
- Subtask 0 (iPhone15): Records Received = 9010
- Subtask 1 (MacBookPro): Records Received = 312
- Subtask 2 (iPadAir): Records Received = 287
- Subtask 3 (AirPods + AppleWatch): Records Received = 391
- Hot subtask receives ~9x more records than others
- Back Pressure: HIGH (red) on Subtask 0
- Checkpoint duration: 12.3s average

## After Optimization (SkewSolution_demo000000)

Flink job `SkewSolution_demo000000` uses two-stage salted aggregation:
1. Stage 1: salt key with random 0-9 suffix, local pre-aggregation
2. Stage 2: remove salt suffix, global aggregation

Observation:
- Subtask 0: Records Received = 2498
- Subtask 1: Records Received = 2512
- Subtask 2: Records Received = 2487
- Subtask 3: Records Received = 2503
- All subtasks are evenly loaded
- Back Pressure: LOW (green) on all subtasks
- Checkpoint duration: 1.8s average

## Comparison Table

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Max Subtask load | 9010 | 2512 | 3.6x reduction |
| Min Subtask load | 287 | 2487 | 8.7x increase |
| Load imbalance ratio | 31.4:1 | 1.01:1 | 31x better |
| Back pressure status | HIGH (red) | LOW (green) | Significant |
| Checkpoint time | 12.3s | 1.8s | 6.8x faster |
| Throughput | ~100 rec/s | ~500+ rec/s | 5x+ improvement |
