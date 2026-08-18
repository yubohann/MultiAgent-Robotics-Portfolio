# Stream-Batch Business Challenges

Author: REDACTED  
Student ID: REDACTED  
Class No.: REDACTED

This lab is split into seven standalone challenge folders based on `acceptance_requirements.md`. The original full manual remains as `instruction.md`, and the overall report remains as `report.md`.

## Layout

```text
02-streaming-challenges/
├── 01-data-skew/
├── 02-small-files-read-write-amplification/
├── 03-state-bloat/
├── 04-late-out-of-order-data/
├── 05-exactly-once-duplicates/
├── 06-schema-evolution/
└── 07-minio-access-bottleneck/
```

Each challenge folder contains:

- `README.md`: independent challenge overview.
- `acceptance_checklist.md`: checklist extracted from the acceptance requirements.
- `screenshots/`: screenshot output directory.
- `evidence/`: logs, query results, command output, or other proof files.

## Student Naming Rule

All visible runtime names must include `demo000000`, including Kafka topics, Flink jobs, MinIO buckets, Paimon tables, SQL snippets, screenshots, and command output.
