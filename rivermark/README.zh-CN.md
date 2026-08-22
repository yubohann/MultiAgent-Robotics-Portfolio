# Rivermark Benchmark

Rivermark 是面向多智能体 3D Search3D 研究的数据采集、审计和评测基础设施。它围绕可复现性、数据完整性和跨范式评测组织，支持经典规划、RL/MARL、QD 和 VLA 接口。

英文展示入口：[README.md](README.md)。核心代码在 `code/src/`，JSON Schema 在 `code/schemas/`，CPU 测试在 `code/tests/`，采集和评测文档在 `docs/`。

CPU researcher smoke：

```powershell
cd code
python -m rivermark_benchmark.researcher_entry $env:TEMP\rivermark-researcher-smoke
python -m unittest discover -s tests -v
```

Isaac Sim、City-Lite 资产和 evaluator-private 数据不随公开仓库发布。
