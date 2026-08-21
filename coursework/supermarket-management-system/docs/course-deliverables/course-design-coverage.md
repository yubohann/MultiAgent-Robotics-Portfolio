# 《软件开发与管理》课程设计覆盖说明

课程设计要求二期工程包含会员管理、员工管理、销售管理、供应商管理、系统管理等模块。当前本地版本覆盖情况如下。

| 模块 | 状态 | 代码证据 | 页面 |
| --- | --- | --- | --- |
| 会员管理 | 已补齐 | `models/member.py`、`services/second_phase.py`、`routes/second_phase.py` | `/members` |
| 员工管理 | 已补齐 | `models/employee.py`、`services/second_phase.py`、`routes/second_phase.py` | `/employees` |
| 销售管理 | 已实现 | `models/sale.py`、`models/sale_item.py`、`services/sales.py`、`routes/sales.py` | `/sales` |
| 供应商管理 | 已补齐 | `models/supplier.py`、`services/second_phase.py`、`routes/second_phase.py` | `/suppliers` |
| 系统管理 | 已补齐 | `models/system_setting.py`、`services/second_phase.py`、`routes/second_phase.py` | `/system-settings` |
| 商品管理 | 已实现 | `models/product.py`、`services/products.py`、`routes/product.py` | `/product` |
| 仓库管理 | 已实现 | `models/inventory.py`、`models/inventory_log.py`、`services/inventory.py` | `/inventory` |
| 财务管理 | 已实现 | `models/finance_transaction.py`、`models/cash_reconciliation.py`、`services/finance.py` | `/finance` |

## 课程报告章节映射

| 章节 | 本地材料 |
| --- | --- |
| 需求建模 | `stage3-requirements-modeling-proof.md`、`supermarket-management-diagrams/` |
| 数据建模 | `app/models/`、`data/SQL/schema.sql`、ER/ORM 图 |
| 界面设计 | `stage5-ui-design-docs.md`、`app/templates/`、`app/static/js/` |
| 系统实施一 | `test-case-design.md`、`tests/` |
| 系统实施二 | 源码、测试、代码审查、报告归档 |

