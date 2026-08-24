# AeroGate Graph

<p align="center">
  <img src="assets/demos/formation-transition.gif" alt="多无人机在闸门场中的编队变换" width="49%" />
  <img src="assets/demos/multi-uav-dynamic-obstacle-avoidance.gif" alt="多无人机动态避障" width="49%" />
</p>

<p align="center"><em>编队变换与多机动态避障演示。</em></p>

AeroGate Graph 是一个模块化二维无人机竞速研究仿真器，覆盖图路线规划、编队控制、动态闸门导航、安全屏蔽和可复现评测。

英文展示入口：[README.md](README.md)。公共 API 和 CLI 在 `aerogate/`，共享几何与动力学在 `shared/`，单机/多机环境在 `single_gate/` 与 `multi_gate/`，测试在 `tests/`。

```powershell
uv sync --extra dev
uv run python -m aerogate info
uv run python -m aerogate smoke --scenario multi-static --agents 4 --steps 8
uv run pytest
```

项目是固定高度研究抽象；通过测试不代表真实飞行安全或硬件性能。
