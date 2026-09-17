# 作品集指南

英文版 [PORTFOLIO_GUIDE.md](PORTFOLIO_GUIDE.md) 是默认技术导航，本文件提供中文说明。

根 README 是项目索引，各子项目的证据等级以对应项目 README 为准。请先阅读对应项目 README，再核对代码，结果，数据与硬件范围。

## 证据标签

| 标签 | 含义 |
|---|---|
| Framework | 公开架构，接口，文档和工具。 |
| Simulation or replay | 来自仿真或回放环境的证据。 |
| Hardware | 覆盖项目文档明确说明的硬件设置，协议和范围。 |
| Coursework | 教学实现或课程成果。 |

## 本地检查

```bash
python tools/verify_portfolio.py
python tools/run_portfolio_checks.py
```

项目注册表位于 [tools/portfolio_registry.json](../tools/portfolio_registry.json)，维护项目目录，入口文档和可运行检查的映射。
