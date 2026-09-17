# Rivermark Benchmark

<p align="center">
  <img src="assets/demos/rivermark-search.gif" alt="Rivermark 多智能体三维搜索" width="78%" />
</p>

<p align="center"><em>Rivermark 多智能体三维搜索演示。</em></p>

Rivermark 是面向多智能体 3D Search3D 研究的数据采集，审计和计分基础设施。它围绕确定性，数据完整性和跨范式计分组织，支持经典规划，RL 和 MARL，QD 以及 VLA 方法接入。

英文展示入口为 [README.md](README.md)。核心代码在 `code/src/`，JSON Schema 在 `code/schemas/`，CPU 测试在 `code/tests/`，采集和计分文档在 `docs/`。

CPU researcher smoke 命令。

```powershell
cd code
python -m rivermark_benchmark.researcher_entry $env:TEMP\rivermark-researcher-smoke
python -m unittest discover -s tests -v
```

Isaac Sim 和 City-Lite 资产以及计分器私有数据由运营方在本地环境维护。
