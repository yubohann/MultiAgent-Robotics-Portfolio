# 阶段六：测试用例设计与自动化映射

任务书要求测试用例文档不少于 10 份。本文件列出 17 个功能测试用例，并给出自动化测试映射。

| 编号 | 测试点 | 前置条件 | 操作 | 预期结果 | 自动化映射 |
| --- | --- | --- | --- | --- | --- |
| TC01 | 管理员登录成功 | 已有管理员账号 | 输入 `admin/admin123` 登录 | 进入首页，Session 写入管理员身份 | `tests/test_auth_and_routes.py` |
| TC02 | 登录失败 | 输入错误密码 | 提交登录 | 页面提示失败，不写入用户会话 | `tests/test_auth_and_routes.py` |
| TC03 | 管理员注册审核 | 有待审核申请 | 管理员通过申请 | 创建管理员账号，申请状态为 approved | `tests/test_auth_and_routes.py` |
| TC04 | 商品新增 | 管理员登录 | 提交合法商品 | 商品和初始库存创建成功 | `tests/test_core_services.py` |
| TC05 | 商品编码重复 | 已存在商品编码 | 再次创建同编码商品 | 返回失败提示 | `tests/test_core_services.py` |
| TC06 | 库存调整 | 已存在商品 | 修改库存数量 | 库存数量更新并写入流水 | `tests/test_core_services.py` |
| TC07 | 收银库存不足 | 商品库存不足 | 提交超过库存的购物车 | 阻止结算，返回库存不足 | `tests/test_core_services.py` |
| TC08 | 收银结算成功 | 商品库存充足 | 提交购物车和支付方式 | 生成销售单、销售明细并扣减库存 | `tests/test_core_services.py` |
| TC09 | 销售订单明细 | 已有销售单 | 查询订单详情 | 返回订单头、明细、金额和支付方式 | `tests/test_core_services.py` |
| TC10 | 财务日结对账 | 已有销售收入 | 保存某日实收金额 | 返回系统金额、实收金额和差异 | `tests/test_core_services.py` |
| TC11 | 公告发布与已读 | 管理员发布公告 | 收银员查看并标记已读 | 未读数减少，公告记录保留 | `tests/test_core_services.py` |
| TC12 | 角色权限控制 | 未登录或收银员登录 | 访问管理页面 | 未登录跳转登录，收银员被拒绝 | `tests/test_auth_and_routes.py` |
| TC13 | 会员管理 | 管理员登录 | 新增会员、调整积分、停用 | 会员状态和积分正确变化 | `tests/test_second_phase.py` |
| TC14 | 员工/供应商/系统管理 | 管理员登录 | 新增、编辑、停用或保存参数 | 主数据变更正确落库 | `tests/test_second_phase.py` |
| TC15 | Excel 商品导入 | 管理员准备合法 Excel | 调用导入服务 | 商品、库存和导入流水同步创建 | `tests/test_core_services.py` |
| TC16 | 二期模块异常路径 | 输入缺失、重复、非法状态等数据 | 调用会员、员工、供应商、系统参数服务 | 返回明确失败信息，数据库回滚 | `tests/test_second_phase_errors.py` |
| TC17 | 二期 API 功能流 | 管理员登录 | 调用会员、员工、供应商、系统参数 API | API 返回成功且列表可查询 | `tests/test_api_functional.py` |

## 自动化命令

```powershell
uv run pytest
uv run coverage run -m pytest
uv run coverage report
```

当前覆盖率统计范围为 `app.models` 与 `app.services.second_phase`，对应课程二期核心服务和后端数据模型；`coverage report` 已设置 `fail_under = 100`，未达到 100% 会直接失败。当前命令行记录见 `reports/system-analysis-design/screenshots/coverage-report.txt`，结果为 `TOTAL 530 0 100%`。

## 截图留存

自动化测试运行截图建议放入：

- `reports/system-analysis-design/screenshots/pytest-result.png`
- `reports/system-analysis-design/screenshots/coverage-report.png`
- `reports/system-analysis-design/screenshots/manual-pages/`

具体截图清单见 `manual-screenshots.md`。
