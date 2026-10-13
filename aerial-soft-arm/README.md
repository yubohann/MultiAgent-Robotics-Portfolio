# AerialSoftArm

[English](README.md) | [简体中文](README.zh-CN.md)

**A rigid-soft aerial manipulator that grasps by bending — the base freezes, the soft arm takes over.**

![Isaac Sim 5.1](https://img.shields.io/badge/Isaac_Sim-5.1-0C83C9?style=flat-square&logo=nvidia)
![Isaac Lab 2.3](https://img.shields.io/badge/Isaac_Lab-2.3-76B900?style=flat-square&logo=nvidia)
![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)

<p align="center">
  <img src="media/demo_task_grasp.gif" width="58%" alt="Autonomous aerial grasp: vision lock, base coarse pose, arm extension, bend alignment, capture, lift">
  &nbsp;
  <img src="media/demo_lattice_arm.gif" width="38%" alt="Truss soft arm articulating bend and grip on hardware">
</p>
<p align="center"><em>Left: full autonomous grasp in Isaac Lab. Right: the truss soft arm on hardware.</em></p>

---

## 💡 Key insight

Aerial grasping couples three hard problems at once: the base must hold position while the arm moves, the gripper must reach millimetre alignment from a wrist camera, and contact must not destabilise the vehicle. Fighting all three with one floating base is a losing game — arm reaction torques feed through the vehicle into the vision loop, and the servo ends up chasing its own disturbance.

**We split the problem in two.** A rigid truss backbone carries load and holds pose; soft bending segments execute alignment and compliant capture. The vehicle handles coarse positioning **and then stops moving** — every remaining degree of freedom belongs to the soft arm. Capture is honest: the part is held by two fingers sustaining contact, not by a simulation attachment.

---

## ⚡ Highlights

- **Rigid-soft dual-structure manipulator** — 8-DOF truss soft arm (bend bx/by + telescoping spine + three-tendon coupling); rigid sections carry load, soft segments align and capture
- **Bend-dominant alignment with a frozen base** — eliminates the base-arm oscillation coupling that plagues free-floating manipulation; the bend-Jacobian visual servo owns the final centimetres
- **Online composite-inertia adaptation** — arm extension inflates body inertia ~2.5×; the composite inertia is recomputed every 0.5 s so the attitude loop stays converged through the whole reach
- **Contact-honest capture protocol** — attach requires both fingers to sustain contact for 0.3 s; no single-touch or gravity-assisted shortcuts
- **Millimetre-scale vision servo** — wrist-camera localisation at a ~2 mm noise floor drives final alignment below 2 cm
- **State estimation that holds up** — 15-state error-state EKF with per-sensor NIS gating cuts velocity RMS by 73% over the complementary-filter baseline
- **End-to-end autonomy** — vision lock → coarse pose → extend → align → capture → lift, plus dual-drone handover under relative-pose control

---

## 📋 Overview

![System overview](media/fig_system.jpg)
*Figure 1: The platform — quadrotor base, truss soft arm, wrist camera, two-finger gripper, and part stand.*

The system targets part retrieval in environments where a landing base is unavailable: the drone arrives, stabilises above the target, and performs the entire grasp from hover. The wrist camera locks the part, the base takes a coarse pose and holds, and the soft arm extends, bends to alignment, and closes. A second drone can receive the part mid-air on a tray.

```mermaid
flowchart LR
    accTitle: Autonomous grasp pipeline
    accDescr: Vision locks the part, the base takes a coarse pose and freezes, the arm extends, bend-Jacobian servo aligns the gripper, two fingers sustain contact, and the part is lifted.

    vision["👁 Vision lock"] --> base["🛰 Coarse pose<br/>then frozen"]
    base --> extend["🦾 Arm extension"]
    extend --> align["🎯 Bend-Jacobian<br/>visual servo"]
    align --> grip["✊ Two-finger capture<br/>0.3 s sustained contact"]
    grip --> lift["📦 Lift & hold"]

    classDef step fill:#dbeafe,stroke:#2563eb,color:#1e3a5f
    class vision,base,extend,align,grip,lift step
```

---

## 🎬 Gallery

<table>
  <tr>
    <td align="center"><img src="media/phase_hover.jpg" width="100%"><br><sub>Hover over stand</sub></td>
    <td align="center"><img src="media/phase_extend.jpg" width="100%"><br><sub>Extend soft arm</sub></td>
    <td align="center"><img src="media/phase_align.jpg" width="100%"><br><sub>Bend-align to part</sub></td>
    <td align="center"><img src="media/phase_grip.jpg" width="100%"><br><sub>Capture & lift</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="media/gripper_closeup.jpg" width="100%"><br><sub>Gripper close-up</sub></td>
    <td align="center"><img src="media/handover_hold.jpg" width="100%"><br><sub>A holds for transfer</sub></td>
    <td align="center" colspan="2"><img src="media/handover_transfer.jpg" width="100%"><br><sub>Handover to drone B</sub></td>
  </tr>
</table>

---

## 📊 Results

![Alignment error](media/fig_align_error.jpg)
*Figure 2: Horizontal alignment error over the servo window — the bend servo drives the residual to the vision noise regime before capture.*

![Per-trial outcome](media/fig_seeds.jpg)
*Figure 3: Per-trial outcome against the four success criteria.*

Across evaluation runs the pipeline converges to **sub-2 cm final alignment error** (vision noise floor ~2 mm) and lifts the 37 g, Ø28×60 mm part **0.4+ metres** after capture. The frozen-base protocol holds the hover position through arm extension, bend reaction torques of 0.5–1.0 N·m, and the capture transient — the failure mode of every free-floating baseline we tried during development. The dual-drone handover completes with the receiving drone holding a fixed tray pose while the donor releases.

---

## 📦 Repository layout

| Path | Contents |
|---|---|
| `isaac/` | Isaac Lab experiments — `fc_align.py` grasp pipeline, estimators, flight control, calibration, URDF/USD assets |
| `isaac/embodied/` | Pure-numpy flight controller, Mahony / EKF / fused estimators, vision links |
| `arm_real/` | Hardware-truss-arm MuJoCo model pipeline (meshes → XML) + textures |
| `panel/` | Keyboard operation panel with MuJoCo view and sim2real calibration export |
| `grasp_pipelines/` | Soft-Spiral grasp scripts (manual / scripted / vision) on MuJoCo |
| `arm_demo/` | MuJoCo soft-arm and aerial-grasp demos |
| `docs/` | Paper (`paper_full_en.docx`) and technical report |
| `env/` | Locked `requirements.txt` and Python version |
| `media/` | README demos and figures |

---

## 🚀 Quick start

**Isaac side** — Python 3.11, Isaac Sim 5.1 + Isaac Lab 2.3, RTX GPU (full lockfile in `env/requirements.txt`):

```powershell
$env:OMNI_KIT_ACCEPT_EULA = "YES"
$env:PYTHONIOENCODING = "utf-8"
$py = "<python from env_isaaclab>"

# Main experiment: single-drone align-grasp (~25 min headless, PASS judge line at the end)
& $py isaac\fc_align.py --headless --enable_cameras --out runs/fc_align_v7 `
  --eye 0.38,-0.45,0.18 --seed 5 --kp-vel 2.5 --kp-pos 0.4 --ki-att 0.10 --est-g 0.35

# Four-gripper grasp suite, Jacobian calibration, experiment matrix
& $py isaac\run_grasp.py --gripper diamond --headless --out runs/grasp_diamond
& $py isaac\calib_jac.py --headless
& $py isaac\run_matrix.py
```

**MuJoCo side** — `pip install mujoco pillow numpy pyyaml textual`:

```powershell
python arm_demo\aerial_grasp.py
python grasp_pipelines\spiral_grasp_scripted.py
python -m panel
```

---

## 🧩 Components

| Component | What it does |
|---|---|
| `isaac/fc_align.py` | Align-grasp state machine: vision localize → bend servo → two-finger capture → lift |
| `isaac/embodied/estimator_ekf.py` | 15-state error-state EKF (p, v, q, b_a, b_g) with GNSS / baro / gravity updates |
| `isaac/tendon_model.py` | Three-tendon ↔ joint-space coupling: tension, preload, Coulomb friction, hysteresis |
| `isaac/calib_jac.py` | Zero-gravity step calibration of the bend and CoM Jacobians |
| `arm_real/arm_lattice.py` | Truss-arm MuJoCo model with tendon routing — same kinematics as the Isaac asset |
| `panel/` | Real-time keyboard teleop, tuning, logging, and sim2real calibration export |

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@software{AerialSoftArm2026,
  author = {Yu, Bohan},
  title  = {AerialSoftArm: Rigid-Soft Aerial Manipulation in Isaac Lab},
  year   = {2026},
  url    = {https://github.com/yubohann/AerialSoftArm}
}
```

---

## License

MIT — see [LICENSE](LICENSE). The vendored Soft-Spiral subtree in [`grasp_pipelines/spiral_repo/`](grasp_pipelines/spiral_repo/) keeps its own MIT license; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
