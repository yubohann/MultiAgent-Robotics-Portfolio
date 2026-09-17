# five_in_drone Asset Notes and Official Safety Distance Basis

Generated on 2026-05-04

This document describes the model position, geometry, mass and inertia, actuators and simulation configuration of the drone under `assets/five_in_drone`. It also records the official regulatory basis for the safe distance between two drones. The official safety distance stands apart from the project training rewards, the avoidance shield and the formation spacing parameters.

## 1. Model Position

`five_in_drone` is a custom 5-inch class quadrotor racing drone simulation asset. A full commercial hardware BOM for a DJI or PX4 model sits outside its scope. The `five_in_drone` name marks the 5-inch propeller class FPV racing quad. The `.dae` meshes in the current checkout are Git LFS pointers, so the 5-inch propeller diameter serves as a naming label rather than a value measured from mesh vertices.

The asset breaks into these files.

| File | Purpose |
| --- | --- |
| `five_in_drone.usd` | Top-level USD that references and composes the sensor and physics layers |
| `configuration/five_in_drone_base.usd` | Geometry and appearance layer, about 102.6 MB |
| `configuration/five_in_drone_physics.usd` | Physics layer with rigid body, joint, mass and collision approximation data |
| `configuration/five_in_drone_sensor.usd` | Sensor layer, readable tokens show one `a_five_in_drone` Xform |
| `urdf/five_in_drone.urdf` | URDF source model with the clearest record of mass, inertia and the four prop and motor positions |
| `meshes/base_link.dae` | Body mesh, a Git LFS pointer declaring 155,702,931 bytes |
| `meshes/prop.dae` | Propeller mesh, a Git LFS pointer declaring 3,141,267 bytes |

## 2. Structure and Geometry

The frame is an X configuration with one `body` link and four `prop` links on revolute joints about the Z axis.

URDF positions of the four prop and motor centers relative to the body origin.

| Joint | Child link | `xyz` in m | Axis |
| --- | --- | --- | --- |
| `m1_joint` | `prop1` | `(0.0883, 0.0883, 0.015)` | `(0, 0, 1)` |
| `m2_joint` | `prop2` | `(0.0883, -0.0883, 0.015)` | `(0, 0, 1)` |
| `m3_joint` | `prop3` | `(-0.0883, 0.0883, 0.015)` | `(0, 0, 1)` |
| `m4_joint` | `prop4` | `(-0.0883, -0.0883, 0.015)` | `(0, 0, 1)` |

Derived values.

| Parameter | Value |
| --- | --- |
| Single-axis projected arm | `0.0883 m` |
| Geometric radius from the body origin to a motor center | `0.124875 m` |
| Center distance between adjacent motors | `0.1766 m` |
| Diagonal motor center distance, wheelbase | `0.24975 m`, about `250 mm` |
| Propeller plane height | `z = 0.015 m` |

The control allocation code uses `arm_length = 0.035 m` as a control parameter, while the URDF geometry places the motor radius at `0.124875 m`. Strict physical identification and sim-to-real work start from a single reconciled value for the torque arm and the geometric arm.

## 3. Mass, Inertia and Collision

Inertia parameters stated by the URDF.

| Item | Value |
| --- | --- |
| Main body mass | `0.5 kg` |
| Center of mass origin | `(0, 0, 0)` |
| `Ixx` | `0.003 kg*m^2` |
| `Iyy` | `0.003 kg*m^2` |
| `Izz` | `0.006 kg*m^2` |

The URDF source places mass and rotational inertia in `body`, and the `prop1` to `prop4` links carry geometry and joints. Visuals and collision share one mesh set, `base_link.dae` for the body and `prop.dae` for the four props. The readable USD physics tokens include `PhysicsRigidBodyAPI`, `MassAPI`, `ArticulationRoot`, `RevoluteJoint` and `convexHull`, so the USD export uses rigid bodies, joints and convex hull collision approximations.

## 4. Joints and Rotor Directions

All four rotor joints are `revolute` about Z. The URDF sets `dynamics damping="0.0" friction="0.0"`, and the commented joint limits leave the rotation range open.

Default joint initial velocities in `assets/five_in_drone.py`.

| Joint | Initial angular velocity |
| --- | --- |
| `m1_joint` | `+200 rad/s` |
| `m2_joint` | `-200 rad/s` |
| `m3_joint` | `+200 rad/s` |
| `m4_joint` | `-200 rad/s` |

Props 1 and 3 spin in one direction and props 2 and 4 spin in the other, so the alternating directions balance yaw reaction torque.

## 5. Simulation Spawn Configuration

`FIVE_IN_DRONE` wraps an IsaacLab `ArticulationCfg` in `assets/five_in_drone.py`.

| Setting | Value or behavior |
| --- | --- |
| prim path | `{ENV_REGEX_NS}/Robot` |
| USD path | `assets/five_in_drone/five_in_drone.usd` |
| Contact sensors | `activate_contact_sensors=True` in the canonical config |
| Gravity | `disable_gravity=False`, gravity enabled |
| Gyroscopic forces | `enable_gyroscopic_forces=True` |
| Maximum depenetration velocity | `max_depenetration_velocity=10.0` |
| Self collision | `enabled_self_collisions=False` |
| Solver position iterations | `4` |
| Solver velocity iterations | `1` |
| Sleep threshold | `0.005` |
| Stabilization threshold | `0.001` |
| Actuator | Dummy implicit actuator with `stiffness=0.0` and `damping=0.0` |

`assets/five_in_drone_graph_masac_training_backup.py` is the training backup config. It loads `configuration/five_in_drone_physics.usd` directly and disables contact sensors to isolate PhysX contact reporting cost in multi-agent Graph-MASAC training.

## 6. Motor, Thrust and Torque Model

Flight power comes from an action term that computes total thrust and body-frame torque from motor angular velocity and applies them to `body`.

Single-agent control defaults come from `tasks/drone_racer/mdp/actions.py`.

| Parameter | Value | Meaning |
| --- | --- | --- |
| `thrust_coef` | `2.25e-7` | Thrust coefficient |
| `drag_coef` | `1.5e-9` | Yaw reaction torque coefficient |
| `omega_max` | `5145 rad/s` | Maximum motor angular velocity, about `49,140 RPM` |
| `init` | `(2572.5, 2572.5, 2572.5, 2572.5)` | Initial and hover reference angular velocity of the control model |
| `taus` | `0.0001 s` per motor | First-order motor response time constant |
| `max_rate` | `+50000 rad/s^2` | Maximum motor angular velocity rise rate |
| `min_rate` | `-50000 rad/s^2` | Maximum motor angular velocity fall rate |
| `use_motor_model` | Default `False` | The default path uses the direct model, and parts of the Graph-MASAC mainline force it on |

Derived values from the coefficients above.

| Operating point | Single prop thrust | Total thrust | Equivalent hover and lift mass |
| --- | --- | --- | --- |
| `omega = 2572.5 rad/s` | `1.489 N` | `5.956 N` | `0.607 kg` |
| `omega = 5145 rad/s` | `5.956 N` | `23.824 N` | `2.429 kg` |

With the reference mass `0.6076 kg` from the control model notes, the maximum total thrust to weight ratio is about 4 to 1. With the URDF mass `0.5 kg`, the ratio is about 4.86 to 1, and the reference thrust at `2572.5 rad/s` reaches about `1.21 g`. This difference between the two mass references is a model reconciliation point.

The force and torque allocation matrix reads,

```text
T  = f1 + f2 + f3 + f4
Mx =  arm_length/sqrt(2) * f1 - arm_length/sqrt(2) * f2 - arm_length/sqrt(2) * f3 + arm_length/sqrt(2) * f4
My = -arm_length/sqrt(2) * f1 - arm_length/sqrt(2) * f2 + arm_length/sqrt(2) * f3 + arm_length/sqrt(2) * f4
Mz =  drag_coef/thrust_coef * f1 - drag_coef/thrust_coef * f2 + drag_coef/thrust_coef * f3 - drag_coef/thrust_coef * f4
```

Here `fi = thrust_coef * omega_i^2`. In single-agent direct motor mode the action input maps from `[-1, 1]` to `[0, omega_max]` and then through the matrix above into body-frame total thrust and three-axis torque.

## 7. Sensors and Recorded Scope

The readable asset information shows one Xform in `five_in_drone_sensor.usd`. Camera, IMU, GPS, barometer, ESC, battery, flight controller, video link and receiver hardware parameters sit outside the recorded tokens.

The `.dae` files are Git LFS pointers, and real mesh data loads through a Git LFS pull. These items come from the mesh files themselves.

- Frame material, thickness and outer bounding box
- Propeller diameter, pitch and blade profile
- Exact geometry of the body, propeller and guard meshes
- Hardware BOM and electrical parameters

Exact outer dimensions start with a Git LFS pull and a bounding box pass through a mesh tool or the USD stage.

## 8. Official Safety Distance Basis for Two Drones

Conclusion, mainstream official regulations define the separation between two small drones as a risk standard rather than a single fixed metre figure. The official basis reads as a duty to keep clear of collision risk and to hold the separation required by air traffic control or the approved operating conditions. The project training parameters `safety_shield_trigger_dist`, `hard_dist` and formation slot spacing therefore stay separate from the official basis.

### Mainland China Basis

The Interim Regulations on the Flight Management of Unmanned Aircraft, State Council Order No. 761, effective 2024-01-01, Article 32 requires an operator to hold the separation set by the national air traffic management authority and to track other aircraft in the airspace and take collision avoidance measures during beyond visual line of sight flight. Article 33 sets the avoidance order, giving way to manned aircraft, unpowered aircraft and ground or water vehicles, single flights giving way to formation flights, and micro drones giving way to other drones.

The current public regulations in mainland China define a required separation plus right of way rules. Concrete separation values come from air traffic control approval, the rules for permitted and controlled airspace, mission and operating approvals, and later provisions from the national air traffic management authority.

Official sources,

- CAAC Interim Regulations on the Flight Management of Unmanned Aircraft, https://www.caac.gov.cn/XXGK/XXGK/FLFG/202401/t20240115_222642.html
- CAAC Civil Unmanned Aircraft Operation Safety Management Rules, CCAR-92, Ministry of Transport Order No. 1 of 2024, https://app.caac.gov.cn/XXGK/XXGK/MHGZ/202401/t20240103_222566.html

### United States FAA Part 107 Basis

FAA Part 107 likewise leaves the separation between two small drones as a risk standard. 14 CFR §107.37 gives small drones right of way duty toward all aircraft and air vehicles, with passing above, below or ahead available under a well clear condition, and it keeps every operation at a distance clear of collision risk. 14 CFR §107.35 limits one person to a single role at a time across operator, RPIC and visual observer duties for more than one drone.

Under Part 107 the official focus for two drones in the air at once is per-drone responsibility and observation coverage plus collision risk separation for any pair of drones. Fixed figures such as 2 m, 5 m or 10 m remain operational choices, and the Part 107 text states a risk standard.

Official sources,

- 14 CFR §107.37, https://www.ecfr.gov/current/title-14/chapter-I/subchapter-F/part-107/subpart-B/section-107.37
- 14 CFR §107.35, https://www.ecfr.gov/current/title-14/chapter-I/subchapter-F/part-107/subpart-B/section-107.35
- 14 CFR §107.31, https://www.ecfr.gov/current/title-14/chapter-I/subchapter-F/part-107/subpart-B/section-107.31

### EASA European Union Basis

The EASA open and specific category rules centre on VLOS, collision avoidance, risk scoring and operational authorisation. The EASA Easy Access Rules ask the remote pilot to keep a visual scan and avoid collisions, and on sighting a low-altitude aircraft that may interact with the drone, to descend below 10 m above ground and keep at least 500 m from the other aircraft, or land immediately when those conditions are out of reach.

The `500 m` figure guides collision avoidance when another low-altitude aircraft or airspace user comes into view, and formation spacing for two cooperating drones stays a separate design choice.

Official sources,

- EASA Easy Access Rules for Unmanned Aircraft Systems, Revision from July 2024, online publication, https://www.easa.europa.eu/en/document-library/easy-access-rules/online-publications/easy-access-rules-unmanned-aircraft-systems?page=5
- EASA FAQ, remote pilot responsibilities in open category, https://www.easa.europa.eu/en/faq/116468

## 9. Engineering Guidance

For simulation papers and training notes about `five_in_drone`, keep the official safety distance and the project safety distance separate in the writeup.

- Official basis, a risk standard that holds the separation required by air traffic control, regulations and operating approval.
- Project basis, training values for `hard_dist`, `trigger_dist`, slot spacing and collision radius, labeled as project engineering assumptions.
- Hardware-level mapping to a real 5-inch racing quad starts from one reconciled set across the URDF mass `0.5 kg`, the control reference mass `0.6076 kg`, the geometric arm `0.124875 m` and the control `arm_length=0.035 m`.
