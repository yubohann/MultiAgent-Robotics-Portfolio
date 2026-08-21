# 答辩流程与验收对照表

本表依据 `~\Downloads\软件开发与管理课程设计 推荐答辩流程.xlsx` 整理，用于答辩前快速核对“PPT、报告、源码、测试、截图、Gitee 仓库”是否一致。

## 答辩环节对照

| 环节 | 时间 | 需要讲清楚的内容 | 仓库证据 | 状态 |
| --- | --- | --- | --- | --- |
| 1. 项目概况与分工 | 1 分钟 | 项目名称、负责模块、团队成员分工、Gitee 仓库与提交情况 | `README.md`、`docs/course-deliverables/stage1-startup-management.md`、Gitee 提交记录 | 已归档 |
| 2. 需求建模 | 2 分钟 | 核心业务流程、泳道图、用例图覆盖范围，重点讲 2-3 个代表流程 | `docs/course-deliverables/stage3-requirements-modeling-proof.md`、`reports/system-analysis-design/images/`、`supermarket-management-diagrams/01-环境与用例/模块级用例图/` | 已归档 |
| 3. 数据建模 | 2 分钟 | ER/ORM 实体、核心关系、关键字段，说明如何支撑业务流程 | `app/models/`、报告第 4 章、`supermarket-management-diagrams/` 中 ER/ORM 图 | 已归档 |
| 4. 界面设计 | 2 分钟 | 主要页面设计，说明页面如何承接业务流程和用户操作 | `docs/course-deliverables/stage5-ui-design-docs.md`、`app/templates/`、`app/static/js/`、报告第 5 章 | 已归档 |
| 5. 系统运行演示 | 4 分钟 | 演示供应商、会员、收银销售、库存扣减、财务流水、统计查询等完整闭环 | `run.py`、`app/routes/`、`app/services/`、`docs/course-deliverables/course-design-coverage.md` | 已归档 |
| 6. 测试与质量管理 | 2 分钟 | 测试用例、单元测试、自动化功能测试截图、代码审查记录 | `tests/`、`docs/course-deliverables/test-case-design.md`、`docs/course-deliverables/code-review-record.md`、`reports/system-analysis-design/screenshots/` | 已归档 |
| 7. 总结与问答 | 2 分钟 | 完成度、未完成问题、改进计划，以及老师追问时的证据定位 | `reports/system-analysis-design/超市管理系统_答辩PPT_20260602.pptx`、最终实验报告、本文档 | 已归档 |

## 答辩材料清单

| 材料 | 路径 | 说明 |
| --- | --- | --- |
| 课程设计报告 | `reports/system-analysis-design/超市管理系统_系统分析与设计实验报告_20260515.docx` | 已补充答辩流程、四个一致性和演示验收说明 |
| 答辩 PPT | `reports/system-analysis-design/超市管理系统_答辩PPT_20260602.pptx` | 8-10 页要求内，按 Excel 推荐答辩流程组织 |
| PPT 源与质检记录 | `reports/system-analysis-design/ppt-defense/` | 包含 `outline.json`、构建脚本和 QA 结果 |
| 可运行系统 | `run.py`、`app/`、`pyproject.toml` | `uv sync` 后执行 `uv run python run.py` |
| 自动化测试源码 | `tests/` | 覆盖登录、商品、库存、收银、财务、公告、二期模块和异常路径 |
| 测试证据 | `reports/system-analysis-design/screenshots/pytest-result.txt`、`coverage-report.txt` | 命令行结果已保留；页面截图按 `manual-screenshots.md` 人工留存 |
| 代码审查记录 | `docs/course-deliverables/code-review-record.md` | 记录问题、影响、处理结果和复查状态 |

## 四个一致性自查

| 验收重点 | 自查结论 |
| --- | --- |
| 需求与设计一致 | 12 个业务流程、12 张模块级 UML 用例图和报告第 3 章能解释系统要做什么。 |
| 设计与实现一致 | ER/ORM 图、页面设计文档、模型、路由、服务和模板均能按模块互相对应。 |
| 实现与测试一致 | `tests/` 覆盖已实现核心功能，coverage 统计结果为 100%，测试用例文档能映射到自动化代码。 |
| 个人贡献与仓库记录一致 | 分工、审查记录、PPT 发言顺序和 Gitee 提交记录可相互印证。 |

## 演示建议顺序

1. 使用 `admin / admin123` 登录后台，展示首页概览和导航结构。
2. 展示供应商维护，说明采购来源和结算周期字段。
3. 展示商品、库存和低库存预警，说明库存流水如何保留业务痕迹。
4. 展示会员维护和积分字段，说明会员参与销售闭环。
5. 在收银台完成一笔销售，随后查看销售订单、库存变化和财务流水。
6. 打开经营分析页面，说明统计图表来自已发生的业务数据。
7. 最后展示测试结果、coverage 结果和代码审查记录。
