# Supermarket Management System：项目技术说明

## 300 字项目概括

Supermarket Management System 是一个以 Flask、Flask-SQLAlchemy、SQLite、Jinja2 和原生 JavaScript 实现的课程级门店管理系统。应用从 `run.py` 创建 Flask app，`app/__init__.py` 初始化数据库、旋转日志和默认数据，再通过 `app/routes/` 注册认证、商品、库存、收银、销售、财务、公告、经营分析、智能助手以及会员、员工、供应商和系统参数等模块。数据库模型覆盖用户、分类、商品、库存流水、销售单、财务交易、应付账款、会员积分、员工排班、供应商和系统设置；页面通过模板渲染，JavaScript 负责表单、筛选、结算和交互。初始化过程会创建默认用户、分类、第二阶段演示数据，并在商品表为空时执行 `data/SQL/seed_data.sql`。项目还保留系统分析与设计报告、PNG 和 draw.io 可编辑图、答辩 PPT、测试用例和覆盖率材料。测试覆盖认证、路由访问、商品、库存、结算、财务、公告、第二阶段模块和错误路径，pyproject 的 coverage gate 目标为 100%。它适合展示数据库建模、Web 分层、业务流程和测试意识；由于属于课程设计，默认账号、演示数据和 SQLite 单机部署不能直接等同于生产零售系统。

## 项目定位

- **类型**：软件开发与管理课程设计。
- **后端**：Flask + Flask-SQLAlchemy；数据库为 SQLite。
- **前端**：Jinja2 模板、原生 JavaScript、Bootstrap 风格页面。
- **业务域**：商品、库存、结算、销售、财务、公告、分析、助手、会员、员工、供应商、系统参数。

## 技术架构

```text
browser / Jinja templates / vanilla JS
            -> Flask route blueprints
            -> services + SQLAlchemy models
            -> SQLite tables / inventory logs / finance ledgers
            -> audit logs, reports and tests
```

`create_app()` 负责 app factory、配置覆盖、数据库绑定、日志 hook、建表和默认数据；请求开始/结束会记录耗时、状态码和 IP，未处理异常进入 rotating log。路由按业务蓝图拆分，模型以关系和流水表保留库存、销售和财务的可追溯性。

## 核心代码结构

| 路径 | 作用 |
| --- | --- |
| `run.py` | 创建 app 并启动开发服务器 |
| `app/__init__.py` | app factory、SQLAlchemy、日志、默认数据和 seed 导入 |
| `app/models/` | 用户、商品、库存、销售、财务、会员、员工、供应商等模型 |
| `app/routes/` | `auth`、`product`、`inventory`、`cashier`、`sales`、`finance`、`announcements`、`analytics`、`assistant`、`second_phase` |
| `app/templates/`、`app/static/` | 页面模板、CSS、JavaScript 和 logo |
| `tests/` | API/路由、核心服务、错误路径和第二阶段测试 |
| `data/`、`scripts/` | SQLite、SQL seed 和辅助命令 |
| `reports/`、`supermarket-management-diagrams*` | 课程报告、答辩材料、PNG 与 draw.io 源文件 |

## 业务流程

1. **认证**：用户登录/注册，按 admin/cashier 等角色控制页面权限。
2. **商品**：分类、上下架、CSV/Excel 导入和基础库存初始化。
3. **库存**：汇总、流水、盘点、低库存预警和库存调整。
4. **结算**：商品搜索、购物车、库存校验、销售单和销售明细生成。
5. **财务**：收支流水、日结对账、应付账款、付款和月度快照。
6. **经营分析**：销售趋势、热销商品、分类占比和总览指标。
7. **第二阶段**：会员积分/等级、员工排班、供应商结算周期、系统参数和公告阅读状态。

## 开发过程

1. 先建立 Flask app factory、配置和最小用户/商品模型。
2. 按业务域扩展库存、销售和财务关系，使用流水和订单明细保持可追踪。
3. 增加模板、JavaScript 和 CSV/Excel 导入，形成可操作页面。
4. 加入第二阶段主数据、公告、智能助手和系统设置，补充权限与错误路径。
5. 同步维护 ER/DFD/用例/时序/状态图、答辩 PPT 和 pytest/coverage 材料。

## 涉及知识点

- Flask app factory、蓝图、请求钩子和日志轮转
- SQLAlchemy ORM、关系建模、事务和 SQLite seed
- 库存流水、订单/明细、财务台账和日结对账
- RBAC、表单校验、错误处理和页面访问控制
- Jinja2、原生 JavaScript、CSV/Excel（openpyxl）导入
- pytest、coverage、功能测试、UML/DFD/ER 建模
- 课程交付物、可编辑图源和本地开发部署

## 运行入口

```powershell
uv sync
uv run python run.py
uv run pytest
uv run coverage run -m pytest
uv run coverage report
```

服务默认监听 `http://127.0.0.1:5000`。首次启动会建表、创建演示用户/分类/第二阶段数据，并在商品为空时导入 `data/SQL/seed_data.sql`。默认账号只用于本地课程演示，不能用于公开部署。

## 测试与验证

测试文件按认证与路由、核心服务、API 功能、第二阶段和错误路径拆分。覆盖率 gate 是项目配置中的工程目标，重新运行时应同时保存 Python、依赖、数据库状态和命令；报告中的历史 coverage 数字不能替代当前 checkout 的复测。

## 学术申请中的项目价值

它能展示从需求分析到可运行业务系统的完整能力：将业务流程映射成关系模型、路由、页面、测试和设计文档，并通过库存/财务流水处理一致性问题。与机器人/ML 项目并列时，它补充了 Web 后端、数据库和软件工程基础。

## 证据边界

这是课程级单机 Flask/SQLite 系统，不包含生产级并发、灾备、支付、真实门店数据或安全审计。演示账号、数据库和 seed 数据只用于本地；覆盖率和答辩材料应以当前环境重新运行的结果为准。

## 源码审计记录

| 层次 | 代表文件/目录 | 复核内容 |
| --- | --- | --- |
| 启动 | `run.py`、`app/__init__.py` | app factory、配置覆盖、数据库绑定、日志 hook、建表和默认数据。 |
| 配置 | `config.py`、`pyproject.toml` | BASE_DIR、SQLite URI、日志轮转、依赖、pytest/coverage gate。 |
| 模型 | `app/models/` | 用户/角色、商品/分类、库存与流水、销售/明细、财务、会员、员工、供应商、系统设置。 |
| 路由 | `app/routes/auth.py` | 登录、注册、密码、session 和权限。 |
| 路由 | `product.py`、`inventory.py` | 商品 CRUD、导入、上/下架、库存汇总、流水和预警。 |
| 路由 | `cashier.py`、`sales.py` | 搜索、购物车、库存校验、订单、明细和查询。 |
| 路由 | `finance.py`、`analytics.py` | 收支、日结、应付、趋势、热销和分类分析。 |
| 路由 | `announcements.py`、`assistant.py`、`second_phase.py` | 公告、问答、会员、员工、供应商和系统参数。 |
| 前端 | `app/templates/`、`app/static/js/`、`css/` | 页面、表单、筛选、异步交互、错误反馈和访问入口。 |
| 数据 | `data/SQL/seed_data.sql`、`scripts/` | 演示数据、导入、初始化和本地维护。 |
| 验证 | `tests/`、`reports/` | API、路由、核心服务、错误路径、覆盖率、报告和图源。 |

## 一次结算调用链

1. cashier 页面通过商品搜索/库存接口加载可售商品，并把选择加入浏览器购物车。
2. 提交结算时，服务端重新查询商品和库存，不能信任客户端价格或数量。
3. 事务内校验库存、创建 Sale/SaleItem、扣减 Inventory 并写 InventoryLog；失败时回滚。
4. 销售、财务和经营分析读取订单/流水，不直接修改历史销售记录。
5. 页面返回订单号和小票文本；日志记录请求、状态和耗时，错误路径进入测试和 rotating log。

## 数据模型阅读重点

商品主数据与分类、库存当前量与库存流水、销售订单与明细、财务交易与日结应保持可解释关系；会员积分、员工状态、供应商结算周期和系统 setting 属于第二阶段扩展。阅读模型时要关注外键、唯一字段、状态值、金额精度、时间字段和删除/停用策略，而不是只看页面字段。

## 权限和错误路径

登录成功只建立身份，具体路由还需检查角色和资源状态；未登录、无权限、商品不存在、库存不足、重复会员/供应商、无效导入和财务关闭等错误都应返回稳定页面/API 错误。测试文件 `test_auth_and_routes.py`、`test_core_services.py`、`test_second_phase_errors.py` 是理解这些边界的首要入口。

## 设计交付物如何辅助代码

`supermarket-management-diagrams-drawio-editable/` 提供环境、用例、DFD、ER、架构、时序、状态和部署图的可编辑源；PNG 是展示导出；`reports/system-analysis-design/` 和答辩 PPT 解释需求到实现的映射。阅读代码时应能从图追到 route/model/template/test，若图与代码不一致，应以当前源码和测试为准并记录差异。

## 交付检查表

- [ ] app factory 可在测试配置下创建临时数据库。
- [ ] 库存、销售和财务事务有正向/异常/回滚测试。
- [ ] 路由权限、错误页面和第二阶段状态有覆盖。
- [ ] seed、默认账号和本地数据库不会被误当作生产数据。
- [ ] 报告、PNG、drawio 源和代码路径保持可追溯。

## 原理与理论基础

### 1. 业务系统分层

门店系统可以按表示层、路由/应用层、领域服务和持久化层分开。浏览器提交的是不可信输入，route 负责认证、校验和编排，service 负责业务规则，SQLAlchemy model 负责关系和事务，template/JavaScript 负责展示。分层的意义是让商品、库存和财务规则不依赖某个页面；测试可以用 test client 调用路由或服务，而不必启动真实浏览器。

### 2. 库存与结算的一致性

库存是状态，库存流水是事件。结算应在同一事务中验证可售数量、创建 Sale/SaleItem、扣减当前库存并写 InventoryLog；若任一步失败，事务回滚保持不变量 stock>=0。订单明细保存成交价格，不能在商品调价后重算历史订单。低库存预警是读取规则，不应改变已结算事实。

### 3. 财务与业务可追溯性

销售、收款、费用、应付和日结形成财务事件链。日结可以看作给定营业日的聚合和封存，封存后修改应留下调整记录而非覆盖原值。金额、时间、状态和关联订单必须有明确精度和生命周期，否则经营分析会与结算页面产生不可解释差异。

### 4. 权限与失败安全

RBAC 不是只在前端隐藏按钮；服务器必须在每个敏感 route 检查身份、角色、资源状态和输入。密码应使用哈希，CSV/Excel 导入需限制字段和异常，默认账号只用于本地。失败请求应返回可理解错误并记录 request id/耗时，不能吞掉异常或把数据库错误显示给用户。

### 5. 理论假设和失效模式

课程系统假设单进程、低并发、SQLite 可用、演示数据可信。并发结算、进程崩溃、数据库损坏、支付回调、跨门店库存和敏感数据泄漏超出当前合同。测试和 coverage 只能证明已覆盖的路径，不代表所有生产故障；因此文档必须把演示能力与上线能力分开。
