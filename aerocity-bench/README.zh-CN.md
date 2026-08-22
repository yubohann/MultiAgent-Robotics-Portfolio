# AeroCityBench

AeroCityBench 是一个面向 3D 多无人机搜索的开放基准，研究城市拓扑变化、目标过程变化和机群韧性变化下的受物理约束搜索能力。

英文展示入口：[README.md](README.md)。

## 主要组成

- 程序化三维城市和版本化发布配置；
- 公开任务投影与评测器私有真值边界；
- G1-U/G2-I 任务合同、检查图谱和泄漏审计；
- L0 基线、外部方法适配器与 Isaac/CF2X 预检；
- JSON Schema、运行证据、资产许可和可复现性工具。

Python 包在 `src/aerocity_bench/`，配置在 `configs/`，验证工具在 `tools/`，测试在 `tests/`。真实 Isaac 运行需要通过 `AEROCITY_ISAACLAB_ROOT` 或历史嵌套目录发现 IsaacLab；本地资产和私有评测数据不随项目发布。
