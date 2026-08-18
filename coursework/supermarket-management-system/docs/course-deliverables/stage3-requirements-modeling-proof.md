# 阶段三：需求建模数量证明

## 泳道图清单

任务书要求至少 10 组业务流程泳道图。本仓库已归档 12 组标准泳道图，覆盖一期和二期关键场景。图中按参与者或系统组件划分泳道，横向表示业务时间顺序，跨泳道箭头表示责任交接；可编辑源文件同步保存在 `supermarket-management-diagrams-drawio-editable/02-业务流程/`。

| 序号 | 流程 | 文件 |
| --- | --- | --- |
| 1 | 销售收银 | `supermarket-management-diagrams/02-业务流程/07-销售收银流程.png` |
| 2 | 采购入库 | `supermarket-management-diagrams/02-业务流程/08-采购入库流程.png` |
| 3 | 库存盘点 | `supermarket-management-diagrams/02-业务流程/09-库存盘点流程.png` |
| 4 | 退货退款 | `supermarket-management-diagrams/02-业务流程/10-退货退款流程.png` |
| 5 | 库存预警与补货 | `supermarket-management-diagrams/02-业务流程/11-库存预警补货流程.png` |
| 6 | 会员积分 | `supermarket-management-diagrams/02-业务流程/12-会员积分流程.png` |
| 7 | 班次交接与日结 | `supermarket-management-diagrams/02-业务流程/13-班次交接日结流程.png` |
| 8 | 商品调价 | `supermarket-management-diagrams/02-业务流程/14-商品调价流程.png` |
| 9 | 管理员注册审核 | `supermarket-management-diagrams/02-业务流程/54-管理员注册审核流程.png` |
| 10 | 商品批量导入 | `supermarket-management-diagrams/02-业务流程/55-商品批量导入流程.png` |
| 11 | 财务日结对账 | `supermarket-management-diagrams/02-业务流程/56-财务日结对账流程.png` |
| 12 | 公告发布与阅读 | `supermarket-management-diagrams/02-业务流程/57-公告发布阅读流程.png` |

## UML 用例图清单

任务书要求至少 10 组 UML 用例图。现有总用例图为 `supermarket-management-diagrams/01-?????/03-??????.png`；另已补齐 12 张模块级 UML 用例图，归档于 `supermarket-management-diagrams/01-?????/??????/`、`supermarket-management-diagrams-drawio-editable/01-?????/??????/` 和 `reports/system-analysis-design/images/??????/`。下列 12 组用例均有独立图表支撑。

| 序号 | 用例组 | 参与者 | 主要用例 |
| --- | --- | --- | --- |
| 1 | 账号认证 | 管理员、收银员 | 登录、退出、注册、管理员申请审核 |
| 2 | 商品管理 | 管理员 | 新增商品、编辑商品、上下架、删除商品、批量导入 |
| 3 | 库存管理 | 管理员、库管员 | 查看库存、调整库存、查看流水、处理预警 |
| 4 | 收银结算 | 收银员 | 检索商品、加入购物车、选择支付、生成销售单 |
| 5 | 销售管理 | 管理员、店长 | 查询销售单、查看明细、筛选支付方式和状态 |
| 6 | 财务管理 | 管理员、财务人员 | 创建收支流水、日结对账、应付登记、月结 |
| 7 | 经营分析 | 管理员、店长 | 查看概览、趋势、热销商品、分类占比 |
| 8 | 公告管理 | 管理员、收银员 | 发布公告、上下线、查看公告、标记已读 |
| 9 | 智能助手 | 管理员、收银员 | 库存问答、销售问答、商品问答、帮助问答 |
| 10 | 会员管理 | 管理员、收银员 | 创建会员、维护等级、调整积分、停用会员 |
| 11 | 员工管理 | 管理员、店长 | 创建员工档案、维护岗位、维护排班、停用员工 |
| 12 | 供应商与系统管理 | 管理员、采购员 | 维护供应商、设置结算周期、维护系统参数 |

## 与代码的对应关系

| 用例组 | 页面 | 后端 |
| --- | --- | --- |
| 账号认证 | `login.html`、`register.html`、`admin_register_requests.html` | `routes/auth.py`、`services/auth.py` |
| 商品管理 | `product.html` | `routes/product.py`、`services/products.py` |
| 库存管理 | `inventory.html` | `routes/inventory.py`、`services/inventory.py` |
| 收银结算 | `cashier.html` | `routes/cashier.py`、`services/cashier.py` |
| 销售管理 | `sales.html` | `routes/sales.py`、`services/sales.py` |
| 财务管理 | `finance.html` | `routes/finance.py`、`services/finance.py` |
| 经营分析 | `analytics.html` | `routes/analytics.py`、`services/analytics.py` |
| 公告管理 | `announcements.html` | `routes/announcements.py`、`services/announcements.py` |
| 智能助手 | `assistant.html` | `routes/assistant.py`、`services/assistant.py` |
| 会员/员工/供应商/系统 | `master_data.html` | `routes/second_phase.py`、`services/second_phase.py` |
