# Screenshot Instructions: State Bloat (Experiment 3)

Student: REDACTED  |  Student ID: demo000000

## Required: 2 Screenshots (3-1 through 3-2)

### Screenshot 3-1: State Size BEFORE Optimization
**What to capture:** Flink Web UI (http://localhost:8081) → Job → Checkpoints tab.
- Show growing "State Size" in Checkpoint History.
- Also capture terminal output:
```bash
docker exec -it bigdata-flink-tm du -sh /tmp/flink/rocksdb-demo000000
docker stats bigdata-flink-tm --no-stream
```
**Check:** State size visibly growing over time. `demo000000` in checkpoint path.

### Screenshot 3-2: State Monitoring AFTER Optimization
**What to capture:** Flink Web UI → Job (after TTL enabled) → Checkpoints tab.
- State Size stable (not growing).
- Checkpoint Duration shorter and stable.
- Also capture Flink SQL showing the TTL config:
```sql
SET 'table.exec.state.ttl' = '24 h';
```
**Check:** State size stabilized. `demo000000` in table/job names.
