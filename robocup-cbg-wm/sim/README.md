# IsaacLab Simulation

This folder contains the IsaacLab and Isaac Sim scene for the two-robot `RoboCup VisionRL` visual challenge simulation. It is the visual playback layer for audited RL traces and the place where arena geometry, sensors, target placement, armor blockers and video capture are assembled.

Run from the local IsaacLab checkout on Windows.

```powershell
cd <repo-root>\sim
<isaaclab-root>\isaaclab.bat -p -m arena_sim
```

Headless smoke test.

```powershell
<isaaclab-root>\isaaclab.bat -p -m arena_sim --headless --duration 5
```

Record the latest audited three-view replay.

```powershell
<isaaclab-root>\isaaclab.bat -p -m arena_sim `
  --headless --duration 40 `
  --replay_trace <repo-root>\sim\output\replay\world_model_sacflow_strict_replay_abs\strict_replay_trace.csv `
  --replay_events <repo-root>\sim\output\replay\world_model_sacflow_strict_replay_abs\strict_replay_events.jsonl `
  --replay_episode 0 `
  --record_video <repo-root>\docs\assets\最终回放_顶视角.mp4 `
  --record_view top --record_fps 30 --record_width 1600 --record_height 900
```

For the two robot first-person videos, repeat with `--record_view yellow_pov` and `--record_video <repo-root>\docs\assets\最终回放_黄车第一视角.mp4`, then with `--record_view blue_pov` and `--record_video <repo-root>\docs\assets\最终回放_蓝车第一视角.mp4`.

Live IsaacLab camera and lidar streams are opt-in, because this PC's Isaac Sim 5.1 build keeps Replicator alive during headless shutdown.

```powershell
<isaaclab-root>\isaaclab.bat -p -m arena_sim --enable_sensor_streams --enable_cameras
```

The scene uses metric dimensions from the competition material.

- 3m x 3m arena
- 0.5m wall height
- 0.5m x 0.5m start and base zones
- two 0.3m x 0.3m x 0.3m obstacles
- rule-page-aligned blue and yellow start zones and bases
- mid-field, base and start-zone fence segments matching the competition diagram
- eight normal targets placed about 45 degrees to the corner and wall geometry
- smaller yellow and blue base targets recessed inside the base, blocked until armor removal
- Tag36h11 visual target mockups with 5cm tag size and 7cm bottom height
- ground-touching blue armor blockers registered as both navigation and laser blockers
- two real PhysX dynamic pushable boxes with explicit collision, mass, gravity and high-friction physics material, and their pushed poses can also be synchronized from strict replay traces
- two robot envelopes aligned to the portfolio robot, 0.34m length, 0.24m width, 0.245m height

Each robot model includes an RGB camera, depth camera, 2D lidar ray-caster, IMU, ToF modules, bumper contacts, wheel encoders, differential-drive wheel layout, and a fixed low-power laser and shooter preview. Scripted previews and trained replay traces are checked against inflated wall, armor, target and obstacle blockers, so robots render outside static blockers and pushable boxes. Robot footprints are also resolved against each other, so yellow and blue contact registers as a collision.

Competition rule logic is active in the GUI scene.

- yellow robot enters the blue side and attacks only blue targets
- blue robot enters the yellow side and attacks only yellow targets
- the strategy layer rejects own normal target hit attempts, and own base target hits count as a hard replay violation
- a normal target hit drops the target and removes one opponent base armor plate
- an opponent base target hit drops the base target and wins the match for the firing team
- target contact registers as a brush or relocalization event
- laser hits require legal opponent target ownership, clear line of sight, distance-dependent accuracy and at least 0.80 s dwell. Normal targets use a 5-50 cm shooter-outlet range, and recessed base targets use 20-80 cm
- base armor plates are active navigation and laser blockers until removed in rule order
- pushable boxes are real dynamic rigid bodies in IsaacLab with `rigidBodyEnabled=true`, `kinematicEnabled=false`, 1.8 kg and a high-friction material, and strict replay traces provide the persistent pushed state used for audited video replay

The reinforcement-learning bridge lives in `rl/`. The selected training path is object-centric world-model SAC Flow self-play, with a flow-policy actor, centralized twin-Q critic, auxiliary object dynamics model, team-specific expert priors, Sim2Real domain randomization, contact-safe robot separation, geometry-aware action shielding and strict replay auditing before rendering in IsaacLab. The selected strict replay passes 8 audited episodes with 0 hard violations, 0 warnings, 0 own-target penalties and 1.0 base wins per episode. The USD export is written to `output/arena.usd` whenever the script starts.
