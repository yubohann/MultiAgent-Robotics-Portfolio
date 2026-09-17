# Screenshot Instructions: Late & Out-of-Order Data (Experiment 4)

Student: REDACTED  |  Student ID: demo000000

## Required: 3 Screenshots (4-1 through 4-3)

### Screenshot 4-1: Late Data Dropped BEFORE Optimization
**What to capture:** Flink TM logs or console output showing late data being discarded.
```bash
docker logs bigdata-flink-tm | grep -E "LATE|late|dropped" | tail -20
```
Or Flink Web UI job output showing records dropped due to late arrival.
**Check:** Must show records with old timestamps being dropped. `demo000000` in topic/job name.

### Screenshot 4-2: Side Output Captured Late Data
**What to capture:** Flink console/log output showing side output stream processing late data.
```bash
docker logs bigdata-flink-tm | grep "迟到数据告警" | tail -20
```
Should show entries like:
`迟到数据告警: > (SENSOR_002,REC_00001234,26.78,1697594400000)`
**Check:** Late data successfully captured to side output. Job name `LateDataDemo_demo000000`.

### Screenshot 4-3: Final Correct Result
**What to capture:** Flink Web UI or console showing:
- Normal data window output (sensor, window, avg_temp)
- Count showing ALL records included (normal + late = total)
- Verify: before optimization had 287 records (12 dropped), after has 299 (all included).
**Check:** Complete windowed results, no data loss. `demo000000` visible.
