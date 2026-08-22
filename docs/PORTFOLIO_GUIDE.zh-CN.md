# 作品集指南

英文版 [PORTFOLIO_GUIDE.md](PORTFOLIO_GUIDE.md) 是默认技术导航；本文件提供中文说明。

根 README 是项目索引，不代表所有子项目具有相同的证据等级。请先阅读对应项目 README，再判断其代码、结果、数据或硬件范围。

## 证据标签

| 标签 | 含义 |
|---|---|
| Framework | 公开架构、接口、文档和工具，不代表某个隐藏方法已经完成评测。 |
| Simulation or replay | 来自仿真或回放环境的证据，不等于真实硬件证据。 |
| Hardware | 只覆盖项目文档明确说明的硬件设置、协议和范围。 |
| Coursework | 教学实现或课程成果，不表述为生产软件或研究基准。 |

## 本地检查

```bash
python tools/verify_portfolio.py
python tools/run_portfolio_checks.py
```

项目注册表位于 [tools/portfolio_registry.json](../tools/portfolio_registry.json)，维护项目目录、入口文档和可运行检查的映射。
