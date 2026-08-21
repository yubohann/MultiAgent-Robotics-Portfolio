# Replay Metadata Notes

This source package does not include MP4 replay files. It keeps only the CSV/JSON metadata and validation summaries needed to identify the evaluated replay episodes.

## Retained Files

| File | Purpose |
|---|---|
| `results/csv_json/single_dynamic_gate42_video_manifest.csv` | Single-agent dynamic gate42 replay metadata in CSV form. |
| `results/csv_json/single_dynamic_gate42_video_manifest.json` | Single-agent dynamic gate42 replay metadata in JSON form. |
| `results/csv_json/single_dynamic_gate42_video_validation_report.csv` | Validation summary for the selected replay episode. |

The large MP4 outputs should be handled as external artifacts when needed.
