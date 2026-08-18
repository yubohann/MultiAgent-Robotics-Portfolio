# 手动截图清单

以下截图建议在本地启动系统后人工保存。部分命令行截图我可以代跑并生成文本记录，但页面状态截图最好你亲自确认画面。

## 需要手动截图

| 编号 | 截图内容 | 建议路径 |
| --- | --- | --- |
| S01 | 登录页 | `reports/system-analysis-design/screenshots/manual-pages/01-login.png` |
| S02 | 首页概览 | `reports/system-analysis-design/screenshots/manual-pages/02-index.png` |
| S03 | 商品管理页，含新增/导入按钮 | `reports/system-analysis-design/screenshots/manual-pages/03-product.png` |
| S04 | 库存管理页，含库存摘要和预警 | `reports/system-analysis-design/screenshots/manual-pages/04-inventory.png` |
| S05 | 收银台，购物车中有商品 | `reports/system-analysis-design/screenshots/manual-pages/05-cashier.png` |
| S06 | 销售管理页，打开订单明细弹窗 | `reports/system-analysis-design/screenshots/manual-pages/06-sales-detail.png` |
| S07 | 财务管理页，含日结对账区域 | `reports/system-analysis-design/screenshots/manual-pages/07-finance.png` |
| S08 | 数据分析页，图表加载完成 | `reports/system-analysis-design/screenshots/manual-pages/08-analytics.png` |
| S09 | 公告管理页，含公告发布表单 | `reports/system-analysis-design/screenshots/manual-pages/09-announcements.png` |
| S10 | 智能助手页，发送一条库存问题后有回复 | `reports/system-analysis-design/screenshots/manual-pages/10-assistant.png` |
| S11 | 会员管理页，含积分调整入口 | `reports/system-analysis-design/screenshots/manual-pages/11-members.png` |
| S12 | 员工管理页 | `reports/system-analysis-design/screenshots/manual-pages/12-employees.png` |
| S13 | 供应商管理页 | `reports/system-analysis-design/screenshots/manual-pages/13-suppliers.png` |
| S14 | 系统参数页 | `reports/system-analysis-design/screenshots/manual-pages/14-system-settings.png` |
| S15 | pytest 测试通过结果 | `reports/system-analysis-design/screenshots/pytest-result.png` |
| S16 | coverage 覆盖率报告 | `reports/system-analysis-design/screenshots/coverage-report.png` |

## 截图前启动命令

```powershell
cd ~\supermarket-management-system\supermarket-management-system
uv run python run.py
```

管理员登录：`admin / admin123`。

