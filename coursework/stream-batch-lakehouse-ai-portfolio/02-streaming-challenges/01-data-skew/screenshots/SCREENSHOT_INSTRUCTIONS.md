# Screenshot Instructions: Data Skew (Experiment 1)

Student: REDACTED  |  Student ID: demo000000

## Required: 7 Screenshots (1-1 through 1-7)

### Screenshot 1-1: Kafka Topic List
**What to capture:** Terminal showing `click_stream_demo000000` in the topic list with 5 partitions.
```bash
docker exec -it bigdata-kafka /opt/kafka/bin/kafka-topics.sh --list --bootstrap-server localhost:9092
docker exec -it bigdata-kafka /opt/kafka/bin/kafka-topics.sh --describe --topic click_stream_demo000000 --bootstrap-server localhost:9092
```
**Check:** Topic name contains `demo000000`, PartitionCount: 5.

### Screenshot 1-2: Python Producer Output
**What to capture:** Terminal running `python skew_data_producer.py` showing:
- Header: `Name: REDACTED  Student ID: demo000000  Class: REDACTED`
- Distribution line: `Data distribution: iPhone15 (90%), others (10%)`
- At least 100 records sent.
```bash
cd 01-data-skew/evidence
python skew_data_producer.py
```
**Check:** `demo000000` and `REDACTED` visible in output.

### Screenshot 1-3: Task Managers Page BEFORE Optimization
**What to capture:** Flink Web UI (http://localhost:8081) → Running Job "SkewDemo_demo000000" → Task Managers tab.
- Must show uneven Subtask load: one Subtask with ~9000 records, others with ~300.
- Job name `SkewDemo_demo000000` visible.

### Screenshot 1-4: Back Pressure Tab BEFORE Optimization
**What to capture:** Flink Web UI → Job → Back Pressure tab.
- Hot Subtask showing HIGH (red).
- Job name `SkewDemo_demo000000` visible.

### Screenshot 1-5: Checkpoints Tab BEFORE Optimization
**What to capture:** Flink Web UI → Job → Checkpoints tab.
- Uneven checkpoint times visible (hot subtask ~12s, others ~1-2s).
- Job name `SkewDemo_demo000000` visible.

### Screenshot 1-6: Task Managers Page AFTER Optimization
**What to capture:** Flink Web UI → Running Job "SkewSolution_demo000000" → Task Managers tab.
- All Subtasks with balanced Records Received (~2500 each).
- Job name `SkewSolution_demo000000` visible.

### Screenshot 1-7: Back Pressure Tab AFTER Optimization
**What to capture:** Flink Web UI → Job "SkewSolution_demo000000" → Back Pressure tab.
- All Subtasks showing LOW (green).
- Job name `SkewSolution_demo000000` visible.
