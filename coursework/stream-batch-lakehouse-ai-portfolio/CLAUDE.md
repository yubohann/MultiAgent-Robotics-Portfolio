# CLAUDE.md

This repository is a redacted three-lab stream-batch big data portfolio.

## Working rules

- Treat `instruction.md` as the original assignment and `report.md` as the report draft. Do not rename or rewrite them unless the user explicitly asks.
- Preserve the redacted demo identifiers consistently across files: `demo000000` and `REDACTED`.
- Keep edits scoped. Prefer small, local changes over broad refactors.
- Use `apply_patch` for file edits.
- Do not revert unrelated user changes.
- Keep code comments short and only where they help real comprehension.
- When a task touches screenshots, startup flow, or container commands, check `01-modern-lakehouse/SCREENSHOT_GUIDE.md` and `01-modern-lakehouse/START_COMMANDS.md` first.

## Repo layout

- `01-modern-lakehouse`: Docker Compose + Kafka + MinIO + Flink + Dinky + Spark lakehouse lab.
- `02-streaming-challenges`: seven streaming/batch challenge labs and acceptance materials.
- `03-ai-recommender`: real-time recommendation system lab.

## Lab 01 conventions

- Run the stack in WSL2 with Docker Desktop.
- Use `localhost:9092` from the host for Kafka producers.
- Use `kafka:29092` and `minio:9000` from inside containers.
- Flink job submission happens on `bigdata-flink-jm`.
- Spark verification runs on `bigdata-spark-master`.
- Keep all lab-01 runtime identifiers embedded with the demo suffix.

## Good Defaults

- Prefer ASCII for new code and command snippets unless a file already uses Chinese prose.
- Keep documentation direct and operational.
- If a dependency is missing, fix the environment first and then document the exact command in the relevant guide.
