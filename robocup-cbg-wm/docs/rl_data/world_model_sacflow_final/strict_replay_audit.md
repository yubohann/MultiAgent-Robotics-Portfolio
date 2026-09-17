# Strict SAC Flow Replay Audit

Verdict **PASS**.

This report replays the trained object-centric SAC Flow tactical actor and audits each step against strict rule and physics invariants.

## Replay Setup

- checkpoint `~\RoboCupVisionRL_AutoCommit\sim\output\rl\world_model_sacflow_seed260707_rerun\policy.pt`
- deterministic `False`
- device `cuda`
- episodes `8`
- max step translation `0.12 m`
- max step yaw delta `0.27 rad`
- static blocker tolerance `0.012 m`

## Strict Checks

- action shape is exactly 6D and bounded to [-1, 1]
- robot pose is finite, inside the arena extent, and outside static blockers
- per-step translation and yaw changes stay within differential-drive limits
- selected targets and fired targets must belong to the opponent
- own-base hit or collision is an immediate hard violation
- scores and armor only change in rule-compatible directions
- target contact grades as a warning, and a contact-induced knockdown grades as a hard violation

## Summary

| Metric | Value |
| --- | ---: |
| `episodes` | 8 |
| `yellow_win_rate` | 0.375 |
| `blue_win_rate` | 0.625 |
| `draw_or_timeout_rate` | 0.0 |
| `hard_violations` | 0 |
| `warnings` | 0 |
| `normal_hits_per_episode` | 3.75 |
| `base_wins_per_episode` | 1.0 |
| `own_target_penalties_per_episode` | 0.0 |
| `blocked_steps_per_episode` | 0.0 |
| `target_contact_events_per_episode` | 0.0 |
| `robot_contacts_per_episode` | 0.0 |
| `recovery_events_per_episode` | 0.0 |
| `block_steps_per_episode` | 0.0 |
| `base_rush_steps_per_episode` | 99.375 |
| `wall_time_s` | 87.603 |

## Output Files

- JSON summary `sim/output/replay/world_model_sacflow_strict_replay_abs/strict_replay_summary.json`
- CSV trace `sim/output/replay/world_model_sacflow_strict_replay_abs/strict_replay_trace.csv`
- JSONL event log `sim/output/replay/world_model_sacflow_strict_replay_abs/strict_replay_events.jsonl`

## Notes

Blocked steps count as prevented penetration events, because the costmap and barrier logic holds the robot outside obstacles. Actual penetration after integration is a hard violation.

A pose that touches the inflated costmap edge within the static-blocker tolerance is recorded as a warning. Physical wall penetration is a hard violation.

Pushable-box contact is allowed within the tolerance, and robot-box penetration is a hard violation.

Target contact is allowed as a non-scoring brush or contact event, and a contact-induced knockdown is a hard violation.

Robot-robot contact is allowed as a tactical event and is counted, so future training can penalize wasteful contact.
