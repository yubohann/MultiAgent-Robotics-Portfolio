# Lab 04 Submission Notes

This folder contains the final submission package for Experiment 4: real-time short-video stream review and publishing.

## Included Artifacts

- Final Word report: `短视频审核实验报告_[REDACTED]_Bohan Yu.docx`
- Source manual and lab README: `instruction.md`, `README.md`
- FastAPI/Ollama demo code: `code/short-video-stream-lab`
- Screenshot evidence: `screenshots_[REDACTED]`
- Runtime and Kafka evidence logs: `logs_[REDACTED]`
- Four small real-video samples used by the runnable demo:
  - `upload_[REDACTED]_yubohan_drone_review.mp4`
  - `real_campus_sports_[REDACTED].mp4`
  - `real_low_light_review_[REDACTED].mp4`
  - `real_flash_risk_review_[REDACTED].mp4`

## Deliberately Excluded

- Local virtual environment `.venv`
- FFmpeg binary tool bundle under `tools/ffmpeg`
- Runtime SQLite state, generated media cache, thumbnails, and duplicated upload outputs
- The original 128 MB drone source file, replaced by the compressed review sample above

## Evidence Scope

The FastAPI workflow and model comparison logs cover all three real demo categories. The Kafka bonus path records one drone-video ingest event and one processed result message, which is why the Kafka result log contains a single message while the API/baseline logs cover the other videos.
