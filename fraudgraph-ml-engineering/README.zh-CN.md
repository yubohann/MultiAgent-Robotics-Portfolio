# FraudGraph ML Engineering

FraudGraph ML Engineering 是一个图与序列联合欺诈检测研究工程，组合 `SplitGNN + Transformer`，提供数据适配、训练、消融、评估、运行清单和可复现检查。

英文展示入口：[README.md](README.md)。源码在 `src/fraud_ml_engineering/`，配置在 `configs/`，脚本在 `scripts/`，测试在 `tests/`。

```powershell
python scripts/validate_repository.py
python -m pytest
python scripts/run_splitgnn_smoke_suite.py --dataset comp --device cpu
```

数据集、图缓存、checkpoint、TensorBoard 日志和实验输出保持本地，不随公开源码发布。
