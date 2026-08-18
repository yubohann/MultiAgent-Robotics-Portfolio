from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
DRAWIO_ROOT = ROOT / "supermarket-management-diagrams-drawio-editable"
REPORT_IMAGE_ROOT = ROOT / "reports" / "system-analysis-design" / "images"
PNG_MIRROR_ROOT = ROOT / "supermarket-management-diagrams"
REPORT_ROOT = ROOT / "reports" / "system-analysis-design"
DRAWIO_EXE = Path(r"C:\Program Files\draw.io\draw.io.exe")

PAGE_W = 2600
PAGE_H = 1700
FONT = "Microsoft YaHei"

INK = "#172033"
MUTED = "#586174"
BORDER = "#9aa6b2"
GRID = "#d8dde5"
BLUE = "#1565c0"
BLUE_FILL = "#e3f2fd"
GREEN = "#2e7d32"
GREEN_FILL = "#e8f5e9"
AMBER = "#f57c00"
AMBER_FILL = "#fff9c4"
ORANGE = "#e65100"
ORANGE_FILL = "#fff3e0"
RED = "#c62828"
RED_FILL = "#ffebee"
PURPLE = "#6a1b9a"
PURPLE_FILL = "#f3e5f5"
GRAY = "#455a64"
GRAY_FILL = "#eceff1"
WHITE = "#ffffff"


def x(value: str) -> str:
    return escape(str(value), {'"': "&quot;"}).replace("\n", "&#xa;")


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def style(parts: list[str] | tuple[str, ...]) -> str:
    base = [
        "whiteSpace=wrap",
        "html=1",
        f"fontFamily={FONT}",
        "fontColor=" + INK,
        "align=center",
        "verticalAlign=middle",
        "spacing=8",
        "spacingTop=4",
        "spacingBottom=4",
    ]
    return ";".join([*base, *parts]) + ";"


def box_style(fill: str = WHITE, stroke: str = BORDER, font_size: int = 24, bold: bool = False, extra: str = "") -> str:
    parts = [
        "rounded=0",
        "fillColor=" + fill,
        "strokeColor=" + stroke,
        "strokeWidth=2",
        "fontSize=" + str(font_size),
    ]
    if bold:
        parts.append("fontStyle=1")
    if extra:
        parts.extend([p for p in extra.split(";") if p])
    return style(parts)


def swim_style(fill: str = WHITE, stroke: str = BORDER, font_size: int = 24) -> str:
    return style([
        "swimlane",
        "startSize=48",
        "fillColor=" + fill,
        "strokeColor=" + stroke,
        "strokeWidth=2",
        "fontSize=" + str(font_size),
        "fontStyle=1",
        "container=1",
        "collapsible=0",
        "recursiveResize=0",
    ])


def edge_style(font_size: int = 19, dashed: bool = False, color: str = "#5b6472") -> str:
    parts = [
        "edgeStyle=orthogonalEdgeStyle",
        "rounded=0",
        "orthogonalLoop=1",
        "jettySize=auto",
        "html=1",
        f"fontFamily={FONT}",
        "fontSize=" + str(font_size),
        "fontColor=" + MUTED,
        "strokeColor=" + color,
        "strokeWidth=2",
        "endArrow=classic",
        "endFill=1",
    ]
    if dashed:
        parts.append("dashed=1")
    return ";".join(parts) + ";"


@dataclass(frozen=True)
class Diagram:
    no: int
    folder: str
    cn_name: str
    title: str
    kind: str
    key: str

    @property
    def base_name(self) -> str:
        return f"{self.no:02d}-{self.cn_name}"


class Graph:
    def __init__(self, title: str, width: int = PAGE_W, height: int = PAGE_H):
        self.title = title
        self.width = width
        self.height = height
        self.cells: list[str] = [
            '<mxCell id="0" />',
            '<mxCell id="1" parent="0" />',
        ]
        self.next_id = 2

    def _id(self) -> str:
        value = str(self.next_id)
        self.next_id += 1
        return value

    def vertex(
        self,
        value: str,
        px: int,
        py: int,
        w: int,
        h: int,
        st: str,
        parent: str = "1",
    ) -> str:
        cell_id = self._id()
        self.cells.append(
            f'<mxCell id="{cell_id}" value="{x(value)}" style="{st}" vertex="1" parent="{parent}">'
            f'<mxGeometry x="{px}" y="{py}" width="{w}" height="{h}" as="geometry" />'
            f'</mxCell>'
        )
        return cell_id

    def edge(
        self,
        source: str,
        target: str,
        label: str = "",
        points: list[tuple[int, int]] | None = None,
        dashed: bool = False,
        exit_pos: tuple[float, float] | None = None,
        entry_pos: tuple[float, float] | None = None,
        color: str = "#5b6472",
    ) -> str:
        cell_id = self._id()
        st = edge_style(dashed=dashed, color=color)
        if exit_pos:
            st += f"exitX={exit_pos[0]};exitY={exit_pos[1]};exitDx=0;exitDy=0;"
        if entry_pos:
            st += f"entryX={entry_pos[0]};entryY={entry_pos[1]};entryDx=0;entryDy=0;"
        if points:
            pts = "".join(f'<mxPoint x="{px}" y="{py}" />' for px, py in points)
            geometry = f'<mxGeometry relative="1" as="geometry"><Array as="points">{pts}</Array></mxGeometry>'
        else:
            geometry = '<mxGeometry relative="1" as="geometry" />'
        self.cells.append(
            f'<mxCell id="{cell_id}" value="{x(label)}" style="{st}" edge="1" parent="1" source="{source}" target="{target}">'
            f'{geometry}</mxCell>'
        )
        return cell_id

    def xml(self) -> str:
        body = "\n".join(self.cells)
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<mxfile host="drawio" modified="2026-06-02T00:00:00.000Z" agent="Codex drawio-skill" version="28.0.6" type="device">\n'
            f'  <diagram id="page-1" name="{x(self.title)}">\n'
            f'    <mxGraphModel dx="1600" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{self.width}" pageHeight="{self.height}" math="0" shadow="0">\n'
            "      <root>\n"
            f"{body}\n"
            "      </root>\n"
            "    </mxGraphModel>\n"
            "  </diagram>\n"
            "</mxfile>\n"
        )


DIAGRAMS: list[Diagram] = [
    Diagram(1, "01-环境与用例", "系统环境与边界", "系统环境与边界", "context", "system"),
    Diagram(2, "01-环境与用例", "角色职责矩阵", "角色职责矩阵", "matrix", "roles"),
    Diagram(3, "01-环境与用例", "业务用例总图", "业务用例总图", "usecase", "system"),
    Diagram(4, "01-环境与用例", "功能分解结构", "功能分解结构", "tree", "system"),
    Diagram(5, "01-环境与用例", "前端信息架构", "前端信息架构", "architecture", "frontend"),
    Diagram(6, "01-环境与用例", "业务能力地图", "业务能力地图", "matrix", "system"),
    Diagram(7, "02-业务流程", "销售收银流程", "销售收银流程", "process", "cashier"),
    Diagram(8, "02-业务流程", "采购入库流程", "采购入库流程", "process", "purchase"),
    Diagram(9, "02-业务流程", "库存盘点流程", "库存盘点流程", "process", "inventory_count"),
    Diagram(10, "02-业务流程", "退货退款流程", "退货退款流程", "process", "return_refund"),
    Diagram(11, "02-业务流程", "库存预警补货流程", "库存预警补货流程", "process", "replenishment"),
    Diagram(12, "02-业务流程", "会员积分流程", "会员积分流程", "process", "member"),
    Diagram(13, "02-业务流程", "班次交接日结流程", "班次交接日结流程", "process", "finance_close"),
    Diagram(14, "02-业务流程", "商品调价流程", "商品调价流程", "process", "product_price"),
    Diagram(15, "03-数据流", "DFD零层图", "DFD 零层图", "dfd", "system"),
    Diagram(16, "03-数据流", "销售一层DFD", "销售一层 DFD", "dfd", "sales"),
    Diagram(17, "03-数据流", "库存一层DFD", "库存一层 DFD", "dfd", "inventory"),
    Diagram(18, "03-数据流", "采购一层DFD", "采购一层 DFD", "dfd", "purchase"),
    Diagram(19, "03-数据流", "会员一层DFD", "会员一层 DFD", "dfd", "member"),
    Diagram(20, "03-数据流", "报表分析一层DFD", "报表分析一层 DFD", "dfd", "analytics"),
    Diagram(21, "04-数据模型", "领域概念模型", "领域概念模型", "erd", "domain"),
    Diagram(22, "04-数据模型", "核心ER模型", "核心 ER 模型", "erd", "core"),
    Diagram(23, "04-数据模型", "商品主数据模型", "商品主数据模型", "erd", "product"),
    Diagram(24, "04-数据模型", "库存流水模型", "库存流水模型", "erd", "inventory"),
    Diagram(25, "04-数据模型", "销售订单模型", "销售订单模型", "erd", "sales"),
    Diagram(26, "04-数据模型", "采购供应商模型", "采购供应商模型", "erd", "purchase"),
    Diagram(27, "04-数据模型", "会员积分模型", "会员积分模型", "erd", "member"),
    Diagram(28, "04-数据模型", "用户权限模型", "用户权限模型", "erd", "auth"),
    Diagram(29, "05-技术架构", "总体技术架构", "总体技术架构", "architecture", "overall"),
    Diagram(30, "05-技术架构", "前端应用架构", "前端应用架构", "architecture", "frontend"),
    Diagram(31, "05-技术架构", "后端应用架构", "后端应用架构", "architecture", "backend"),
    Diagram(32, "05-技术架构", "API模块地图", "API 模块地图", "api", "api"),
    Diagram(33, "05-技术架构", "认证权限架构", "认证权限架构", "architecture", "auth"),
    Diagram(34, "05-技术架构", "设备外部集成", "设备外部集成", "architecture", "integration"),
    Diagram(35, "05-技术架构", "智能助手集成", "智能助手集成", "architecture", "assistant"),
    Diagram(36, "05-技术架构", "报表分析架构", "报表分析架构", "architecture", "analytics"),
    Diagram(37, "06-时序图", "登录认证时序", "登录认证时序", "sequence", "auth"),
    Diagram(38, "06-时序图", "销售结算时序", "销售结算时序", "sequence", "cashier"),
    Diagram(39, "06-时序图", "采购入库时序", "采购入库时序", "sequence", "purchase"),
    Diagram(40, "06-时序图", "库存预警时序", "库存预警时序", "sequence", "inventory"),
    Diagram(41, "06-时序图", "会员积分时序", "会员积分时序", "sequence", "member"),
    Diagram(42, "06-时序图", "退货退款时序", "退货退款时序", "sequence", "return_refund"),
    Diagram(43, "07-状态图", "商品生命周期状态", "商品生命周期状态", "state", "product"),
    Diagram(44, "07-状态图", "采购订单状态", "采购订单状态", "state", "purchase"),
    Diagram(45, "07-状态图", "盘点任务状态", "盘点任务状态", "state", "inventory_count"),
    Diagram(46, "07-状态图", "销售订单状态", "销售订单状态", "state", "sales"),
    Diagram(47, "07-状态图", "会员账户状态", "会员账户状态", "state", "member"),
    Diagram(48, "08-部署安全质量", "本地开发部署", "本地开发部署", "deployment", "local"),
    Diagram(49, "08-部署安全质量", "生产部署视图", "生产部署视图", "deployment", "production"),
    Diagram(50, "08-部署安全质量", "安全信任边界", "安全信任边界", "architecture", "security"),
    Diagram(51, "08-部署安全质量", "可观测审计", "可观测审计", "architecture", "audit"),
    Diagram(52, "08-部署安全质量", "测试覆盖矩阵", "测试覆盖矩阵", "matrix", "testing"),
    Diagram(53, "08-部署安全质量", "发布交付物清单", "发布交付物清单", "matrix", "release"),
    Diagram(54, "02-业务流程", "管理员注册审核流程", "管理员注册审核流程", "process", "admin_signup"),
    Diagram(55, "02-业务流程", "商品批量导入流程", "商品批量导入流程", "process", "product_import"),
    Diagram(56, "02-业务流程", "财务日结对账流程", "财务日结对账流程", "process", "finance_reconcile"),
    Diagram(57, "02-业务流程", "公告发布阅读流程", "公告发布阅读流程", "process", "announcement"),
]


MODULE_USE_CASES = [
    ("01-账号认证用例图", "账号认证用例图", "auth"),
    ("02-商品管理用例图", "商品管理用例图", "product"),
    ("03-库存管理用例图", "库存管理用例图", "inventory"),
    ("04-收银结算用例图", "收银结算用例图", "cashier"),
    ("05-销售管理用例图", "销售管理用例图", "sales"),
    ("06-财务管理用例图", "财务管理用例图", "finance"),
    ("07-经营分析用例图", "经营分析用例图", "analytics"),
    ("08-公告管理用例图", "公告管理用例图", "announcement"),
    ("09-智能助手用例图", "智能助手用例图", "assistant"),
    ("10-会员管理用例图", "会员管理用例图", "member"),
    ("11-员工管理用例图", "员工管理用例图", "employee"),
    ("12-供应商系统管理用例图", "供应商系统管理用例图", "supplier_system"),
]


def add_title(g: Graph, title: str, subtitle: str = "接口｜服务｜数据表｜校验/事务｜验收证据") -> None:
    g.vertex(title, 60, 35, g.width - 120, 70, box_style("#f8fafc", "#334155", 38, True, "align=left;spacingLeft=22;"))
    g.vertex(subtitle, 60, 105, g.width - 120, 40, box_style("#ffffff", "#ffffff", 20, False, "align=left;fontColor=#586174;spacingLeft=22;"))


def add_section(g: Graph, title: str, x0: int, y0: int, w: int, h: int, fill: str = "#ffffff") -> str:
    return g.vertex(title, x0, y0, w, h, swim_style(fill, "#9aa6b2", 24))


def _detail_row_height(value: str) -> int:
    visual_lines = value.count("\n") + 1
    visual_lines += max(0, len(value) // 80)
    return max(82, min(250, 34 * visual_lines + 28))


def add_detail_table(g: Graph, rows: list[tuple[str, str]], x0: int, y0: int, w: int, title: str = "实现明细") -> int:
    label_w = 170
    title_h = 60
    heights = [_detail_row_height(value) for _, value in rows]
    total_h = title_h + sum(heights)
    g.vertex(title, x0, y0, w, title_h, box_style("#172033", "#172033", 24, True, "fontColor=#ffffff;"))
    y = y0 + title_h
    for idx, (label, value) in enumerate(rows):
        row_h = heights[idx]
        g.vertex(label, x0, y, label_w, row_h, box_style("#f1f5f9", "#cbd5e1", 20, True, "align=left;spacingLeft=14;"))
        g.vertex(value, x0 + label_w, y, w - label_w, row_h, box_style("#ffffff", "#cbd5e1", 19, False, "align=left;spacingLeft=16;verticalAlign=middle;"))
        y += row_h
    g.vertex("", x0, y0, w, total_h, box_style("none", "#334155", 18, False, "fillColor=none;strokeWidth=2;pointerEvents=0;"))
    return total_h


def add_legend(g: Graph, items: list[tuple[str, str, str]], x0: int, y0: int) -> None:
    for idx, (name, fill, stroke) in enumerate(items):
        x0_item = x0 + idx * 250
        g.vertex("", x0_item, y0, 36, 22, box_style(fill, stroke, 16))
        g.vertex(name, x0_item + 44, y0 - 8, 195, 38, box_style("#ffffff", "#ffffff", 18, False, "align=left;fontColor=#586174;"))


DETAILS = {
    "system": {
        "page": "base.html / index.html / 业务页面",
        "routes": "auth、product、inventory、cashier、sales、finance、analytics、announcements、assistant、second_phase",
        "service": "routes 调度 service；service 负责校验、事务、DTO；models 映射 18 张表",
        "tables": "users、products、inventory、sales、finance、members、employees、suppliers、system_settings 等",
        "checks": "登录态、角色权限、唯一约束、库存非负、金额精度、失败 rollback、coverage 100%",
    },
    "auth": {
        "page": "login.html / register.html / admin_register_requests.html",
        "routes": "/、/login、/register、/admin/register-requests、approve、reject、/logout",
        "service": "register_user、login_user、get_admin_signup_requests、review_admin_signup_request",
        "tables": "users、admin_signup_requests",
        "checks": "password_hash 校验；管理员申请 pending/approved/rejected；session 保存 user_id/role；admin_required 拦截",
    },
    "product": {
        "page": "product.html / product.js",
        "routes": "/product、/api/products GET/POST、/api/products/<id> PUT/DELETE、offline、online、import、/api/categories",
        "service": "get_products、create_product、update_product、offline_product、online_product、delete_product、import_products_from_csv/excel",
        "tables": "products、categories、inventory、inventory_logs",
        "checks": "product_code 唯一；barcode 唯一；售价/进价合法；导入初始化库存；异常 rollback",
    },
    "inventory": {
        "page": "inventory.html / inventory.js",
        "routes": "/inventory、/api/inventory/summary、list、logs、alerts",
        "service": "set_inventory_quantity、get_inventory_summary、get_inventory_list、get_inventory_logs、get_inventory_alerts",
        "tables": "inventory、inventory_logs、products、users",
        "checks": "quantity >= 0；记录 before/after；change_type 枚举；低库存 quantity <= min_stock；operator_id 留痕",
    },
    "cashier": {
        "page": "cashier.html / cashier.js",
        "routes": "/cashier、/api/cashier/products、/api/cashier/checkout",
        "service": "search_cashier_products、checkout_cashier_order、_normalize_cart_items、_generate_order_no、set_inventory_quantity",
        "tables": "products、inventory、sales、sale_items、inventory_logs、users",
        "checks": "只查上架商品；购物车数量>0；库存充足；折扣金额合法；同一事务写销售单、明细、扣库存、流水",
    },
    "sales": {
        "page": "sales.html / sales.js",
        "routes": "/sales、/api/sales/orders、/api/sales/orders/<id>",
        "service": "get_sales_orders、get_sales_order_detail",
        "tables": "sales、sale_items、products、users",
        "checks": "按日期/支付方式/状态筛选；订单明细合计；不存在返回空；status completed/refunded/cancelled",
    },
    "finance": {
        "page": "finance.html / finance.js",
        "routes": "/finance、overview、transactions、reconciliation、payables、payment、closings、close-month",
        "service": "get_finance_overview、create_finance_transaction、save_reconciliation、create_payable、record_payable_payment、close_finance_period",
        "tables": "finance_transactions、cash_reconciliations、supplier_payables、payable_payments、finance_period_closings、sales",
        "checks": "transaction_no 唯一；reconcile_date+payment_method 唯一；difference 自动计算；应付状态 unpaid/partial/paid/overdue",
    },
    "analytics": {
        "page": "analytics.html / analytics.js / index.html",
        "routes": "/analytics、/api/analytics/overview、trend、top-products、category-distribution",
        "service": "get_dashboard_overview、get_sales_overview、get_sales_trend、get_top_products、get_category_distribution",
        "tables": "sales、sale_items、products、inventory、categories",
        "checks": "按日期聚合；热销商品 limit；分类占比；空库自动 seed；统计 JSON 可直接驱动图表",
    },
    "announcement": {
        "page": "announcements.html / base.js",
        "routes": "/announcements、create、toggle、/api/announcements、unread-count、read、read-all",
        "service": "get_announcements_for_user、get_unread_announcement_count、mark_announcement_read、create_announcement、set_announcement_publish_status",
        "tables": "announcements、announcement_reads、users",
        "checks": "target_role 控制可见；is_published 控制上下线；announcement_id+user_id 唯一；已读状态可追溯",
    },
    "assistant": {
        "page": "assistant.html / assistant.js / assistant.css",
        "routes": "/assistant、/api/assistant/chat",
        "service": "generate_assistant_reply、_get_faq_preset_reply、_build_inventory_reply、_build_sales_reply、_try_generate_by_llm",
        "tables": "products、inventory、sales、sale_items",
        "checks": "空消息拒绝；FAQ/库存/销售/商品意图识别；OPENAI_API_KEY 可选；LLM 失败降级本地回复",
    },
    "member": {
        "page": "master_data.html / master-data.js / /members",
        "routes": "/members、/api/members GET/POST/PUT、status、points",
        "service": "get_members、create_member、update_member、adjust_member_points、set_member_status",
        "tables": "members、sales",
        "checks": "member_no 唯一；phone 唯一；points >= 0；level normal/silver/gold/vip；status active/inactive",
    },
    "employee": {
        "page": "master_data.html / master-data.js / /employees",
        "routes": "/employees、/api/employees GET/POST/PUT、status",
        "service": "get_employees、create_employee、update_employee、set_employee_status",
        "tables": "employees",
        "checks": "employee_no 唯一；phone 唯一；position 必填；排班可维护；status active/inactive",
    },
    "supplier": {
        "page": "master_data.html / master-data.js / /suppliers",
        "routes": "/suppliers、/api/suppliers GET/POST/PUT、status",
        "service": "get_suppliers、create_supplier、update_supplier、set_supplier_status",
        "tables": "suppliers、supplier_payables、payable_payments",
        "checks": "supplier_code 唯一；settlement_cycle weekly/monthly/quarterly；应付付款后重算状态",
    },
    "supplier_system": {
        "page": "master_data.html / /suppliers / /system-settings",
        "routes": "/api/suppliers、/api/system-settings GET/POST/PUT",
        "service": "create_supplier、set_supplier_status、get_system_settings、upsert_system_setting",
        "tables": "suppliers、supplier_payables、system_settings",
        "checks": "supplier_code 唯一；setting_key 唯一；配置写入 updated_at；统一 JSON 返回",
    },
    "system_setting": {
        "page": "master_data.html / /system-settings",
        "routes": "/system-settings、/api/system-settings、/api/system-settings/<key>",
        "service": "get_system_settings、upsert_system_setting",
        "tables": "system_settings",
        "checks": "setting_key 唯一；setting_value 必填；description 可维护；updated_at 自动更新",
    },
    "frontend": {
        "page": "base.html、product、inventory、cashier、sales、finance、analytics、master_data",
        "routes": "页面路由 render_template；API 路由返回 JSON；JS fetch 分层调用",
        "service": "表格、筛选、弹窗、分页、消息提示、主题初始化、公告角标",
        "tables": "前端不直连数据库；全部通过 Flask route/service 访问",
        "checks": "登录导航；表单必填；失败 message；统一布局；二期主数据共用模板",
    },
    "backend": {
        "page": "routes/*.py / services/*.py / models/*.py",
        "routes": "register_routes(app) 注册蓝图；login_required/admin_required/cashier_required",
        "service": "参数清洗、业务校验、事务提交/回滚、DTO 格式化、错误 success=false",
        "tables": "SQLAlchemy Model + schema.sql + seed_data.sql",
        "checks": "唯一约束；库存/积分非负；金额 Decimal；异常路径测试覆盖",
    },
    "api": {
        "page": "前端 fetch + Flask JSON contract",
        "routes": "GET 列表/详情；POST 新增/动作；PUT 更新；DELETE 删除/下架",
        "service": "分页、筛选、排序、状态变更、DTO 格式化",
        "tables": "products、sales、inventory、finance、members、employees、suppliers、system_settings",
        "checks": "参数类型转换；权限装饰器；空结果返回空列表；失败返回 message",
    },
    "security": {
        "page": "login/register/admin pages + protected business pages",
        "routes": "login_required、admin_required、cashier_required",
        "service": "Werkzeug password hash；session user_id/role；管理员注册审核",
        "tables": "users、admin_signup_requests、announcement_reads",
        "checks": "密码不落明文；未登录 redirect login；管理员申请先入队再审核；角色最小权限",
    },
    "audit": {
        "page": "库存流水、公告已读、财务日结、应付付款",
        "routes": "/api/inventory/logs、/api/finance/closings、/api/announcements/read-all、/api/finance/payables/<id>/payment",
        "service": "set_inventory_quantity、close_finance_period、mark_all_announcements_read、record_payable_payment",
        "tables": "inventory_logs、finance_period_closings、announcement_reads、payable_payments",
        "checks": "operator_id 留痕；before/after 快照；read_at/paid_at/closed_at 时间戳；可按页面追溯",
    },
    "testing": {
        "page": "tests/ + coverage config",
        "routes": "pytest、coverage run -m pytest、coverage report",
        "service": "test_core_services、test_second_phase、test_second_phase_errors、test_auth_and_routes、test_api_functional",
        "tables": "临时 SQLite；fixture 初始化；模型隔离",
        "checks": "TOTAL 530/0/100%；正常路径+异常路径；fail_under=100；截图仅需人工留存",
    },
    "release": {
        "page": "README.md、docs/、reports/、diagrams/",
        "routes": "本地运行说明、截图清单、课程阶段材料、图表归档",
        "service": "uv run pytest；uv run coverage report；draw.io 导出 png；docx 嵌图同步",
        "tables": "schema.sql、seed_data.sql、报告图片、可编辑 drawio",
        "checks": "不提交 .venv/db/log；本轮只本地改；提交 Gitee 前复核 git status 和截图",
    },
}


PROCESS_STEPS = {
    "cashier": [
        ("收银员", "打开收银台\n输入条码/名称", "/cashier\n/api/cashier/products"),
        ("前端", "加入购物车\n输入数量/折扣", "cashier.js\ncart validation"),
        ("服务", "规范化购物车\n生成订单号", "_normalize_cart_items\n_generate_order_no"),
        ("服务", "校验商品状态\n校验库存充足", "products.status=1\ninventory.quantity"),
        ("数据库", "写销售主表\n写销售明细", "sales\nsale_items"),
        ("数据库", "扣减库存\n写库存流水", "inventory\ninventory_logs"),
        ("结果", "返回订单号\n刷新销售/库存", "order_no\nsuccess=true"),
    ],
    "purchase": [
        ("管理员", "选择供应商\n登记到货明细", "供应商档案\n商品编码"),
        ("服务", "校验商品存在\n校验供应商状态", "products\nsuppliers"),
        ("服务", "调用库存调整\nchange_type=in", "set_inventory_quantity"),
        ("数据库", "增加库存\n写入入库流水", "inventory\ninventory_logs"),
        ("数据库", "登记应付账款\n生成 bill_no", "supplier_payables"),
        ("服务", "事务提交\n失败整体回滚", "db.session.commit\nrollback"),
        ("结果", "入库完成\n预警状态刷新", "库存列表\n低库存提醒"),
    ],
    "inventory_count": [
        ("管理员", "创建盘点批次\n录入实盘数量", "inventory page"),
        ("服务", "读取账面库存\n计算差异数量", "get_inventory_list"),
        ("服务", "复核差异\n确认调整原因", "reason=count"),
        ("服务", "执行库存调整\nchange_type=adjust", "set_inventory_quantity"),
        ("数据库", "更新库存\n记录 before/after", "inventory\ninventory_logs"),
        ("结果", "输出差异清单\n关闭盘点任务", "count result"),
    ],
    "return_refund": [
        ("收银员", "选择销售单\n读取明细", "/api/sales/orders/<id>"),
        ("服务", "校验可退数量\n校验订单状态", "sale_items\nsales.status"),
        ("服务", "登记退款金额\n更新订单状态", "refunded/cancelled"),
        ("数据库", "库存回补\nchange_type=return", "inventory\ninventory_logs"),
        ("数据库", "财务记录退款\n关联 order_no", "finance_transactions"),
        ("结果", "返回退款结果\n刷新销售单", "success=true"),
    ],
    "replenishment": [
        ("系统", "扫描库存预警\nquantity <= min_stock", "/api/inventory/alerts"),
        ("管理员", "确认补货建议\n选择供应商", "suppliers active"),
        ("采购", "登记采购到货\n录入数量", "purchase receiving"),
        ("服务", "校验商品/供应商\n计算应付金额", "products\nsuppliers"),
        ("数据库", "入库流水\n应付账款", "inventory_logs\nsupplier_payables"),
        ("结果", "解除低库存\n刷新预警列表", "alerts updated"),
    ],
    "member": [
        ("收银员", "识别会员\n读取会员档案", "/api/members"),
        ("服务", "结算后计算积分\n按实际金额换算", "checkout result"),
        ("服务", "调整积分余额\n校验非负", "adjust_member_points"),
        ("数据库", "写 members.points\n更新时间", "members"),
        ("结果", "返回积分余额\n显示会员等级", "points\nlevel"),
    ],
    "finance_close": [
        ("财务", "选择日结日期\n选择支付方式", "/api/finance/reconciliation"),
        ("服务", "汇总当日销售\nexpected_amount", "_sales_amount_between"),
        ("财务", "录入实收金额\n填写备注", "actual_amount\nnote"),
        ("服务", "计算差异金额\n保存对账", "save_reconciliation"),
        ("数据库", "写现金对账表\n唯一日期+方式", "cash_reconciliations"),
        ("结果", "输出日结差异\n进入月结统计", "difference_amount"),
    ],
    "product_price": [
        ("管理员", "编辑售价/进价\n提交商品更新", "PUT /api/products/<id>"),
        ("服务", "校验商品存在\n校验价格字段", "update_product"),
        ("数据库", "更新 products\n保留库存不变", "products.updated_at"),
        ("结果", "刷新商品列表\n收银台使用新价", "/api/cashier/products"),
    ],
    "admin_signup": [
        ("申请人", "提交注册申请\n填写账号密码", "/register"),
        ("服务", "密码哈希\n写待审记录", "register_user"),
        ("管理员", "查看待审列表\n批准/驳回", "/admin/register-requests"),
        ("服务", "审核通过创建用户\n驳回写原因", "review_admin_signup_request"),
        ("数据库", "更新申请状态\n写 users", "admin_signup_requests\nusers"),
        ("结果", "返回审核结果\n申请人可登录", "approved/rejected"),
    ],
    "product_import": [
        ("管理员", "上传 CSV/Excel\n选择导入文件", "/api/products/import"),
        ("服务", "解析表头行\n逐行读取字段", "import_products_from_csv/excel"),
        ("服务", "校验编码/条码唯一\n校验价格库存", "product_code\nbarcode"),
        ("数据库", "批量写 products\n初始化 inventory", "products\ninventory"),
        ("数据库", "写导入流水\nchange_type=import", "inventory_logs"),
        ("结果", "返回成功/错误数\n异常回滚", "success_count\nerrors"),
    ],
    "finance_reconcile": [
        ("财务", "选择日期/支付方式\n查询应收", "GET reconciliation"),
        ("服务", "按销售单汇总\n生成 expected", "sales actual_amount"),
        ("财务", "录入实收金额\n提交对账", "POST reconciliation"),
        ("服务", "计算 difference\n保存或更新记录", "save_reconciliation"),
        ("数据库", "唯一键 upsert\n保留创建人", "cash_reconciliations"),
        ("结果", "展示差异\n进入审计留存", "difference_amount"),
    ],
    "announcement": [
        ("管理员", "填写标题内容\n选择目标角色", "/announcements/create"),
        ("服务", "校验必填字段\n创建公告", "create_announcement"),
        ("数据库", "写 announcements\nis_published=1", "announcements"),
        ("用户", "读取公告列表\n查看未读数量", "/api/announcements\nunread-count"),
        ("服务", "标记已读/全部已读\n避免重复记录", "mark_announcement_read\nmark_all_announcement_read"),
        ("数据库", "写已读表\n唯一公告+用户", "announcement_reads"),
        ("结果", "更新角标\n保留阅读时间", "read_at"),
    ],
}


ER_TABLES = {
    "users": ["PK user_id", "username UNIQUE", "password_hash", "real_name", "role admin/cashier", "is_active", "created_at"],
    "admin_signup_requests": ["PK request_id", "username UNIQUE", "password_hash", "status pending/approved/rejected", "reviewed_by FK users", "reviewed_at", "reject_reason"],
    "announcements": ["PK announcement_id", "title", "content", "level normal/important", "target_role all/admin/cashier", "is_published", "created_by FK users"],
    "announcement_reads": ["PK read_id", "announcement_id FK", "user_id FK", "read_at", "UNIQUE announcement_id+user_id"],
    "categories": ["PK category_id", "category_name", "parent_id", "sort_order", "created_at"],
    "products": ["PK product_id", "barcode UNIQUE", "product_code UNIQUE", "product_name", "category_id FK", "unit", "purchase_price", "selling_price", "min_stock", "status 0/1"],
    "inventory": ["PK/FK product_id", "quantity CHECK >=0", "last_check_time", "updated_at"],
    "inventory_logs": ["PK log_id", "product_id FK", "change_type", "quantity_change", "quantity_before", "quantity_after", "reason", "operator_id FK", "created_at"],
    "sales": ["PK sale_id", "order_no UNIQUE", "cashier_id FK", "total_amount", "discount_amount", "actual_amount", "payment_method", "status", "created_at"],
    "sale_items": ["PK item_id", "sale_id FK", "product_id FK", "quantity CHECK >0", "unit_price", "purchase_price", "subtotal"],
    "finance_transactions": ["PK transaction_id", "transaction_no UNIQUE", "transaction_type", "category", "amount", "payment_method", "related_order_no", "operator_id FK"],
    "cash_reconciliations": ["PK reconciliation_id", "reconcile_date", "payment_method", "expected_amount", "actual_amount", "difference_amount", "UNIQUE date+method"],
    "supplier_payables": ["PK payable_id", "supplier_name", "bill_no UNIQUE", "total_amount", "paid_amount", "due_date", "status", "created_by FK"],
    "payable_payments": ["PK payment_id", "payable_id FK", "amount", "payment_method", "paid_at", "operator_id FK"],
    "finance_period_closings": ["PK close_id", "period_month UNIQUE", "total_sales", "other_income", "expense_amount", "gross_profit", "net_profit", "closed_by FK"],
    "members": ["PK member_id", "member_no UNIQUE", "member_name", "phone UNIQUE", "level", "points CHECK >=0", "status", "registered_at"],
    "employees": ["PK employee_id", "employee_no UNIQUE", "employee_name", "position", "phone UNIQUE", "work_schedule", "status"],
    "suppliers": ["PK supplier_id", "supplier_code UNIQUE", "supplier_name", "contact_person", "phone", "settlement_cycle", "status"],
    "system_settings": ["PK setting_id", "setting_key UNIQUE", "setting_value", "description", "updated_at"],
}


ER_SETS = {
    "domain": ["users", "products", "inventory", "sales", "finance_transactions", "members", "employees", "suppliers", "system_settings"],
    "core": list(ER_TABLES.keys()),
    "product": ["categories", "products", "inventory", "inventory_logs", "users"],
    "inventory": ["products", "inventory", "inventory_logs", "users", "categories"],
    "sales": ["users", "sales", "sale_items", "products", "inventory", "inventory_logs", "members"],
    "purchase": ["suppliers", "supplier_payables", "payable_payments", "products", "inventory", "inventory_logs", "users"],
    "member": ["members", "sales", "sale_items", "products", "users"],
    "auth": ["users", "admin_signup_requests", "announcements", "announcement_reads"],
}


ER_RELATIONS = {
    "core": [
        "categories 1-N products",
        "products 1-1 inventory",
        "products 1-N inventory_logs",
        "users 1-N inventory_logs",
        "users 1-N sales",
        "sales 1-N sale_items",
        "products 1-N sale_items",
        "users 1-N finance_transactions",
        "users 1-N cash_reconciliations",
        "supplier_payables 1-N payable_payments",
        "users 1-N finance_period_closings",
        "announcements 1-N announcement_reads",
        "users 1-N announcement_reads",
    ],
    "product": ["categories 1-N products", "products 1-1 inventory", "products 1-N inventory_logs", "users 1-N inventory_logs"],
    "inventory": ["products 1-1 inventory", "products 1-N inventory_logs", "users 1-N inventory_logs"],
    "sales": ["users 1-N sales", "sales 1-N sale_items", "products 1-N sale_items", "products 1-1 inventory"],
    "purchase": ["suppliers 1-N supplier_payables", "supplier_payables 1-N payable_payments", "products 1-N inventory_logs"],
    "member": ["members 通过收银流程关联 sales", "sales 1-N sale_items", "products 1-N sale_items"],
    "auth": ["users 1-N admin_signup_requests reviewed_by", "users 1-N announcements created_by", "announcements 1-N announcement_reads", "users 1-N announcement_reads"],
    "domain": ["商品-库存-销售形成经营闭环", "用户贯穿操作审计", "财务承接销售与采购应付", "二期主数据支撑会员、员工、供应商、系统参数"],
}


def detail_rows(key: str) -> list[tuple[str, str]]:
    detail = DETAILS.get(key, DETAILS["system"])
    return [
        ("页面/脚本", detail["page"]),
        ("路由/API", detail["routes"]),
        ("服务函数", detail["service"]),
        ("数据表", detail["tables"]),
        ("校验/事务", detail["checks"]),
    ]


def build_process(d: Diagram) -> Graph:
    g = Graph(d.title, 3200, 2100)
    add_title(g, d.title, "流程步骤｜责任方｜实现入口｜落表与事务")
    steps = PROCESS_STEPS[d.key]
    role_palette = {
        "收银员": (BLUE_FILL, BLUE),
        "管理员": (BLUE_FILL, BLUE),
        "申请人": (BLUE_FILL, BLUE),
        "财务": (BLUE_FILL, BLUE),
        "采购": (BLUE_FILL, BLUE),
        "用户": (BLUE_FILL, BLUE),
        "系统": (PURPLE_FILL, PURPLE),
        "前端": (ORANGE_FILL, ORANGE),
        "服务": (PURPLE_FILL, PURPLE),
        "数据库": (GREEN_FILL, GREEN),
        "结果": (GRAY_FILL, GRAY),
    }
    nodes: list[str] = []
    flow_x, flow_y = 70, 190
    step_w, step_h, step_gap = 300, 210, 20
    for idx, (role, title, tech) in enumerate(steps):
        fill, stroke = role_palette.get(role, (GRAY_FILL, GRAY))
        sx = flow_x + idx * (step_w + step_gap)
        g.vertex(f"{idx + 1}", sx, flow_y, step_w, 48, box_style("#172033", "#172033", 23, True, "fontColor=#ffffff;"))
        label = f"{role}\n{title}\n{tech}"
        nodes.append(g.vertex(label, sx, flow_y + 48, step_w, step_h - 48, box_style(fill, stroke, 21, True, "align=left;spacingLeft=14;verticalAlign=middle;")))
    for a, b in zip(nodes, nodes[1:]):
        g.edge(a, b, "", exit_pos=(1, 0.5), entry_pos=(0, 0.5))

    matrix_x, matrix_y = 70, 500
    label_w = 170
    cell_w = 300
    row_h = 118
    g.vertex("步骤实现矩阵", matrix_x, matrix_y, label_w + len(steps) * cell_w, 58, box_style("#172033", "#172033", 24, True, "fontColor=#ffffff;"))
    for idx in range(len(steps)):
        sx = matrix_x + label_w + idx * cell_w
        g.vertex(f"步骤 {idx + 1}", sx, matrix_y + 58, cell_w, 58, box_style("#f1f5f9", "#cbd5e1", 20, True))
    rows = [
        ("责任方", [role for role, _, _ in steps]),
        ("业务动作", [title for _, title, _ in steps]),
        ("接口/函数", [tech for _, _, tech in steps]),
        ("验收关注", [
            "输入合法" if idx < 2 else
            "服务校验" if role == "服务" else
            "写库留痕" if role == "数据库" else
            "返回结果"
            for idx, (role, _, _) in enumerate(steps)
        ]),
    ]
    for row_idx, (label, values) in enumerate(rows):
        y = matrix_y + 116 + row_idx * row_h
        g.vertex(label, matrix_x, y, label_w, row_h, box_style("#f1f5f9", "#cbd5e1", 20, True, "align=left;spacingLeft=14;"))
        for idx, value in enumerate(values):
            sx = matrix_x + label_w + idx * cell_w
            g.vertex(value, sx, y, cell_w, row_h, box_style("#ffffff", "#cbd5e1", 18, False, "align=left;spacingLeft=12;verticalAlign=middle;"))

    add_detail_table(g, detail_rows(d.key), 2340, 190, 830, "实现明细")
    exception_rows = [
        ("异常处理", "参数缺失、唯一冲突、库存不足、金额不合法时返回 success=false/message"),
        ("事务边界", "涉及销售、库存、流水、财务的写操作位于同一 service 事务内，失败 rollback"),
        ("可追溯点", "operator_id、created_at、before/after、order_no、bill_no、read_at 等字段留痕"),
    ]
    add_detail_table(g, exception_rows, 70, 1080, 2230, "验收追问点")
    code_rows = [
        ("路由文件", "app/routes/*.py 中对应蓝图接口"),
        ("服务文件", "app/services/*.py 中对应业务函数"),
        ("数据模型", "app/models/*.py 与 data/SQL/schema.sql"),
        ("测试证据", "tests/ 覆盖正常路径、异常路径、权限路径"),
    ]
    add_detail_table(g, code_rows, 2340, 980, 830, "代码定位")
    add_legend(g, [("页面/人员", BLUE_FILL, BLUE), ("前端交互", ORANGE_FILL, ORANGE), ("后端服务", PURPLE_FILL, PURPLE), ("数据库", GREEN_FILL, GREEN), ("结果审计", GRAY_FILL, GRAY)], 2340, 1780)
    return g


def build_erd(d: Diagram) -> Graph:
    if d.key == "core":
        return build_core_erd(d)
    else:
        g = Graph(d.title, 2700, 1800)
        add_title(g, d.title, "ER 模型｜表字段、主外键、唯一约束、关系清单")
        tables = ER_SETS.get(d.key, ER_SETS["domain"])
        cols = 3
        x0, y0 = 90, 190
        card_w, card_h = 650, 340
        gap_x, gap_y = 70, 70
    for idx, table in enumerate(tables):
        cx = x0 + (idx % cols) * (card_w + gap_x)
        cy = y0 + (idx // cols) * (card_h + gap_y)
        header = g.vertex(table, cx, cy, card_w, 56, box_style(BLUE_FILL, BLUE, 24, True, "align=left;spacingLeft=16;"))
        fields = ER_TABLES[table]
        body = "\n".join(fields[:10])
        g.vertex(body, cx, cy + 56, card_w, card_h - 56, box_style("#ffffff", BLUE, 20, False, "align=left;verticalAlign=top;spacingLeft=18;spacingTop=12;"))
        _ = header
    relation_key = d.key if d.key in ER_RELATIONS else "domain"
    relations = ER_RELATIONS.get(relation_key, ER_RELATIONS["domain"])
    rel_text = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(relations))
    y_rel = 1940 if d.key == "core" else 1340
    add_detail_table(g, [("关系", rel_text), ("约束", "唯一：username/product_code/barcode/order_no/transaction_no/bill_no/period_month/member_no/employee_no/supplier_code/setting_key\n检查：库存>=0、积分>=0、销售明细数量>0、角色/状态/支付方式枚举")], 90, y_rel, g.width - 180, "关系与约束清单")
    return g


def build_core_erd(d: Diagram) -> Graph:
    g = Graph(d.title, 3300, 2300)
    add_title(g, d.title, "核心 ER 模型｜业务域分区｜关键字段｜关系与约束")
    domains = [
        ("账号与公告域", 80, 190, 760, 670, ["users", "admin_signup_requests", "announcements", "announcement_reads"], BLUE_FILL, BLUE),
        ("商品库存销售域", 900, 190, 980, 670, ["categories", "products", "inventory", "inventory_logs", "sales", "sale_items"], GREEN_FILL, GREEN),
        ("财务采购域", 80, 920, 1080, 570, ["finance_transactions", "cash_reconciliations", "supplier_payables", "payable_payments", "finance_period_closings"], ORANGE_FILL, ORANGE),
        ("二期主数据域", 1220, 920, 860, 570, ["members", "employees", "suppliers", "system_settings"], PURPLE_FILL, PURPLE),
    ]
    table_positions: dict[str, tuple[int, int, int, int]] = {}
    for title, x0, y0, w, h, tables, fill, stroke in domains:
        add_section(g, title, x0, y0, w, h, "#ffffff")
        cols = 2 if len(tables) <= 4 else 3
        card_w = int((w - 70 - (cols - 1) * 25) / cols)
        card_h = 185
        for idx, table in enumerate(tables):
            cx = x0 + 35 + (idx % cols) * (card_w + 25)
            cy = y0 + 75 + (idx // cols) * (card_h + 30)
            table_positions[table] = (cx, cy, card_w, card_h)
            g.vertex(table, cx, cy, card_w, 48, box_style(fill, stroke, 23, True, "align=left;spacingLeft=12;"))
            fields = "\n".join(ER_TABLES[table][:6])
            g.vertex(fields, cx, cy + 48, card_w, card_h - 48, box_style("#ffffff", stroke, 19, False, "align=left;verticalAlign=top;spacingLeft=14;spacingTop=10;"))

    relation_text = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(ER_RELATIONS["core"]))
    constraints = (
        "唯一：username、product_code、barcode、order_no、transaction_no、bill_no、period_month、member_no、employee_no、supplier_code、setting_key\n"
        "检查：inventory.quantity>=0；members.points>=0；sale_items.quantity>0；role/status/payment_method 使用枚举\n"
        "外键：销售、库存流水、财务、公告已读均保留操作者或业务主表引用，支持追溯"
    )
    add_detail_table(g, [("关系", relation_text), ("约束", constraints)], 2160, 170, 1120, "关系与约束清单")
    all_tables = "、".join(ER_TABLES.keys())
    add_detail_table(g, [("核心表", all_tables), ("说明", "主图展示关键字段；完整字段已在 schema.sql、models/*.py 与各主题 ER 图中展开。")], 80, 1580, 3200, "字段归档")
    return g


def build_architecture(d: Diagram) -> Graph:
    g = Graph(d.title, 2800, 1800)
    add_title(g, d.title, "架构图｜分层、模块、接口、数据表、权限边界")
    if d.key == "api":
        return build_api_map(d)
    layers = [
        ("用户与终端", 90, 190, 2450, 210, BLUE_FILL, BLUE, [
            ("管理员", "商品/库存/财务/公告/二期主数据"),
            ("收银员", "收银/销售/公告/助手"),
            ("店长/财务", "分析/对账/验收"),
            ("浏览器", "Jinja2 页面 + JS fetch"),
        ]),
        ("Flask 路由层", 90, 470, 2450, 255, ORANGE_FILL, ORANGE, [
            ("auth", "/login /register /logout"),
            ("product", "/api/products /import /categories"),
            ("inventory", "/summary /list /logs /alerts"),
            ("cashier/sales", "/checkout /orders /detail"),
            ("finance", "/overview /reconciliation /payables"),
            ("second_phase", "/members /employees /suppliers /settings"),
        ]),
        ("服务与业务规则", 90, 795, 2450, 300, PURPLE_FILL, PURPLE, [
            ("权限", "login_required/admin_required"),
            ("商品库存", "create/update/import + set_inventory_quantity"),
            ("收银事务", "sales + sale_items + inventory_logs"),
            ("财务事务", "transactions + reconciliation + payables"),
            ("智能助手", "本地规则 + 可选 LLM 降级"),
            ("测试", "pytest + coverage 100%"),
        ]),
        ("数据与交付", 90, 1165, 2450, 260, GREEN_FILL, GREEN, [
            ("SQLite", "schema.sql + seed_data.sql"),
            ("18 张表", "业务表、审计表、二期主数据"),
            ("报告", "docx + 57 主图 + 12 用例图"),
            ("可编辑图", "中文 .drawio + PNG"),
        ]),
    ]
    previous_nodes: list[str] = []
    for title, lx, ly, lw, lh, fill, stroke, items in layers:
        add_section(g, title, lx, ly, lw, lh, "#ffffff")
        nodes = []
        cols = len(items)
        card_w = int((lw - 80 - (cols - 1) * 28) / cols)
        for idx, (name, info) in enumerate(items):
            nx = lx + 40 + idx * (card_w + 28)
            nodes.append(g.vertex(f"{name}\n{info}", nx, ly + 72, card_w, lh - 95, box_style(fill, stroke, 21, True)))
        if previous_nodes:
            for p, n in zip(previous_nodes[: min(len(previous_nodes), len(nodes))], nodes[: min(len(previous_nodes), len(nodes))]):
                g.edge(p, n, "", exit_pos=(0.5, 1), entry_pos=(0.5, 0))
        previous_nodes = nodes
    add_detail_table(g, detail_rows(d.key if d.key in DETAILS else "system"), 90, 1490, 2450, "模块实现明细")
    return g


def build_api_map(d: Diagram) -> Graph:
    g = Graph(d.title, 3300, 2100)
    add_title(g, d.title, "API 模块地图｜页面入口、HTTP 方法、服务函数、落表")
    groups = [
        ("认证", "auth.py", ["/", "/login GET/POST", "/register GET/POST", "/admin/register-requests", "approve/reject", "/logout"], "login_user / register_user / review_admin_signup_request", "users / admin_signup_requests"),
        ("商品", "product.py", ["/product", "/api/products GET/POST", "/api/products/<id> PUT/DELETE", "offline/online", "/api/products/import", "/api/categories"], "get/create/update/offline/online/delete/import", "products / categories / inventory"),
        ("库存", "inventory.py", ["/inventory", "/api/inventory/summary", "/api/inventory/list", "/api/inventory/logs", "/api/inventory/alerts"], "get_inventory_summary/list/logs/alerts", "inventory / inventory_logs"),
        ("收银销售", "cashier.py + sales.py", ["/cashier", "/api/cashier/products", "/api/cashier/checkout", "/sales", "/api/sales/orders", "/api/sales/orders/<id>"], "search_cashier_products / checkout / get_sales_orders", "sales / sale_items / inventory_logs"),
        ("财务", "finance.py", ["/finance", "overview", "transactions GET/POST", "reconciliation GET/POST", "payables GET/POST", "payment / closings / close-month"], "overview / transaction / reconciliation / payable / closing", "finance_transactions / cash_reconciliations / payables"),
        ("二期主数据", "second_phase.py", ["/members /employees /suppliers /system-settings", "CRUD APIs", "status APIs", "points API", "settings GET/POST/PUT"], "members/employees/suppliers/settings services", "members / employees / suppliers / system_settings"),
        ("公告", "announcements.py", ["/announcements", "create/toggle", "/api/announcements", "unread-count", "read/read-all"], "create / publish / read / unread_count", "announcements / announcement_reads"),
        ("分析助手", "analytics.py + assistant.py", ["/analytics", "overview/trend/top-products/category", "/assistant", "/api/assistant/chat"], "analytics aggregation / assistant intent replies", "sales / sale_items / products / inventory"),
    ]
    card_w, card_h = 760, 390
    for idx, (name, route_file, apis, service, tables) in enumerate(groups):
        cx = 90 + (idx % 4) * 800
        cy = 190 + (idx // 4) * 460
        g.vertex(f"{name}\n{route_file}", cx, cy, card_w, 72, box_style(BLUE_FILL, BLUE, 25, True, "align=left;spacingLeft=16;"))
        body = f"API\n" + "\n".join(f"• {api}" for api in apis) + f"\n\n服务\n{service}\n\n落表\n{tables}"
        g.vertex(body, cx, cy + 72, card_w, card_h - 72, box_style("#ffffff", BLUE, 19, False, "align=left;verticalAlign=top;spacingLeft=18;spacingTop=12;"))
    add_detail_table(g, detail_rows("api"), 90, 1160, 3120, "统一 API 合同")
    return g


def build_dfd(d: Diagram) -> Graph:
    g = Graph(d.title, 2800, 1800)
    add_title(g, d.title, "数据流图｜输入、处理、数据存储、输出、审计")
    detail = DETAILS.get(d.key, DETAILS.get("analytics"))
    actor = g.vertex("外部参与者\n管理员 / 收银员 / 店长 / 财务", 110, 420, 300, 150, box_style("#ffffff", GRAY, 22, True))
    route = g.vertex(f"输入/API\n{detail['routes']}", 540, 360, 450, 250, box_style(ORANGE_FILL, ORANGE, 21, True, "align=left;spacingLeft=16;"))
    validate = g.vertex(f"权限与参数校验\n{detail['checks']}", 1130, 220, 520, 260, box_style(AMBER_FILL, AMBER, 20, True, "align=left;spacingLeft=16;"))
    service = g.vertex(f"业务处理\n{detail['service']}", 1130, 620, 520, 260, box_style(PURPLE_FILL, PURPLE, 20, True, "align=left;spacingLeft=16;"))
    store = g.vertex(f"数据存储\n{detail['tables']}", 1830, 360, 560, 300, box_style(GREEN_FILL, GREEN, 21, True, "align=left;spacingLeft=16;"))
    output = g.vertex("输出\n分页列表 / 明细 DTO / 统计 JSON / message", 1130, 1010, 520, 170, box_style(BLUE_FILL, BLUE, 21, True))
    audit = g.vertex("审计与状态\n库存流水 / 公告已读 / 财务对账 / 操作时间戳", 1830, 850, 560, 180, box_style(GRAY_FILL, GRAY, 21, True))
    for src, dst, label in [
        (actor, route, "请求"),
        (route, validate, "session/role/参数"),
        (validate, service, "通过"),
        (service, store, "读写"),
        (store, service, "结果集"),
        (service, output, "DTO"),
        (service, audit, "留痕"),
        (output, actor, "响应"),
    ]:
        g.edge(src, dst, label)
    add_detail_table(g, detail_rows(d.key), 90, 1340, 2380, "实现明细")
    return g


SEQUENCE_STEPS = {
    "auth": [
        ("用户", "浏览器", "提交 username/password"),
        ("浏览器", "auth.route", "POST /login"),
        ("auth.route", "auth.service", "login_user"),
        ("auth.service", "users", "查询用户+password_hash"),
        ("auth.service", "auth.route", "返回 user 或错误"),
        ("auth.route", "浏览器", "写 session 后跳转首页"),
    ],
    "cashier": [
        ("收银员", "cashier.js", "提交购物车"),
        ("cashier.js", "cashier.route", "POST /api/cashier/checkout"),
        ("cashier.route", "cashier.service", "checkout_cashier_order"),
        ("cashier.service", "products/inventory", "校验上架与库存"),
        ("cashier.service", "sales/sale_items", "写销售单和明细"),
        ("cashier.service", "inventory_logs", "扣库存并写流水"),
        ("cashier.route", "cashier.js", "返回 order_no"),
    ],
    "purchase": [
        ("管理员", "采购页面", "录入到货"),
        ("采购页面", "库存服务", "调用 set_inventory_quantity"),
        ("库存服务", "products", "校验商品"),
        ("库存服务", "inventory", "增加库存"),
        ("库存服务", "inventory_logs", "记录入库流水"),
        ("库存服务", "supplier_payables", "登记应付"),
        ("采购页面", "管理员", "显示入库完成"),
    ],
    "inventory": [
        ("管理员", "inventory.js", "请求低库存"),
        ("inventory.js", "inventory.route", "GET /api/inventory/alerts"),
        ("inventory.route", "inventory.service", "get_inventory_alerts"),
        ("inventory.service", "products/inventory", "quantity <= min_stock"),
        ("inventory.service", "inventory.route", "返回预警列表"),
        ("inventory.route", "inventory.js", "渲染预警卡片"),
    ],
    "member": [
        ("收银员", "master_data/cashier", "选择会员"),
        ("页面", "second_phase.route", "POST /api/members/<id>/points"),
        ("route", "second_phase.service", "adjust_member_points"),
        ("service", "members", "校验 points>=0 并更新"),
        ("service", "route", "返回积分余额"),
        ("route", "页面", "刷新会员等级/积分"),
    ],
    "return_refund": [
        ("收银员", "sales.js", "选择订单"),
        ("sales.js", "sales.route", "GET /api/sales/orders/<id>"),
        ("sales.route", "sales.service", "get_sales_order_detail"),
        ("sales.service", "sales/sale_items", "读取订单和明细"),
        ("退货服务", "inventory", "回补库存"),
        ("退货服务", "finance_transactions", "登记退款"),
        ("页面", "收银员", "展示退款结果"),
    ],
}


def build_sequence(d: Diagram) -> Graph:
    g = Graph(d.title, 2900, 2200)
    add_title(g, d.title, "时序图｜参与者、调用顺序、落表与返回值")
    steps = SEQUENCE_STEPS[d.key]
    participants = []
    for s in steps:
        for item in (s[0], s[1]):
            if item not in participants:
                participants.append(item)
    px = {}
    base_x = 100
    gap = int((2500) / max(1, len(participants) - 1)) if len(participants) > 1 else 300
    for idx, part in enumerate(participants):
        x_part = base_x + idx * gap
        px[part] = x_part
        g.vertex(part, x_part, 190, 230, 70, box_style(BLUE_FILL, BLUE, 22, True))
        g.vertex("", x_part + 112, 280, 6, 760, box_style("#ffffff", "#9aa6b2", 12, False, "dashed=1;fillColor=#ffffff;strokeWidth=2;"))
    y0 = 320
    for idx, (src, dst, msg) in enumerate(steps):
        y = y0 + idx * 105
        src_x = px[src] + 115
        dst_x = px[dst] + 115
        if src_x < dst_x:
            sx, tx = src_x, dst_x
        else:
            sx, tx = src_x, dst_x
        label = f"{idx + 1}. {msg}"
        msg_box = g.vertex(label, min(sx, tx) + 18, y - 28, abs(tx - sx) - 36 if abs(tx - sx) > 240 else 230, 50, box_style("#ffffff", "#ffffff", 18, False, "fontColor=#586174;"))
        _ = msg_box
        # Use small invisible endpoints for straight, readable arrows.
        a = g.vertex("", sx - 1, y, 2, 2, box_style("none", "none", 1, False, "fillColor=none;strokeColor=none;"))
        b = g.vertex("", tx - 1, y, 2, 2, box_style("none", "none", 1, False, "fillColor=none;strokeColor=none;"))
        g.edge(a, b, "", color="#344054")
    matrix_rows = []
    for idx, (src, dst, msg) in enumerate(steps):
        matrix_rows.append((f"{idx + 1}", f"{src} -> {dst}：{msg}"))
    add_detail_table(g, matrix_rows, 120, 900, 2540, "调用清单")
    add_detail_table(g, detail_rows(d.key), 120, 1560, 2540, "实现明细")
    return g


STATE_SETS = {
    "product": [
        ("草稿/新建", "create_product\n初始化 inventory"),
        ("上架", "status=1\n可被收银检索"),
        ("调价/维护", "update_product\n更新价格字段"),
        ("下架", "offline_product\nstatus=0"),
        ("删除", "delete_product\n无销售引用时删除"),
    ],
    "purchase": [
        ("待采购", "低库存预警生成建议"),
        ("已下单", "确认供应商与数量"),
        ("部分到货", "分批入库并记流水"),
        ("已入库", "库存增加"),
        ("已结算", "应付账款 paid"),
    ],
    "inventory_count": [
        ("待盘点", "创建任务"),
        ("盘点中", "录入实盘"),
        ("待复核", "计算差异"),
        ("已调整", "set_inventory_quantity"),
        ("已关闭", "输出差异清单"),
    ],
    "sales": [
        ("购物车", "前端暂存商品"),
        ("待结算", "校验数量/折扣"),
        ("已完成", "status=completed"),
        ("已退款", "status=refunded"),
        ("已取消", "status=cancelled"),
    ],
    "member": [
        ("新建", "create_member"),
        ("启用", "status=active"),
        ("积分变动", "adjust_member_points"),
        ("升级/降级", "level 字段维护"),
        ("停用", "status=inactive"),
    ],
}


def build_state(d: Diagram) -> Graph:
    g = Graph(d.title, 2700, 1800)
    add_title(g, d.title, "状态图｜状态、触发动作、持久化字段")
    states = STATE_SETS[d.key]
    nodes = []
    start_x = 150
    for idx, (name, detail) in enumerate(states):
        x0 = start_x + idx * 480
        nodes.append(g.vertex(f"{name}\n{detail}", x0, 360, 360, 150, box_style(BLUE_FILL if idx == 0 else GREEN_FILL if idx == len(states) - 1 else "#ffffff", BLUE if idx == 0 else GREEN if idx == len(states) - 1 else GRAY, 23, True)))
    for a, b in zip(nodes, nodes[1:]):
        g.edge(a, b, "触发动作", exit_pos=(1, 0.5), entry_pos=(0, 0.5))
    rules = [
        ("状态字段", DETAILS.get(d.key, DETAILS["product"])["tables"]),
        ("状态约束", DETAILS.get(d.key, DETAILS["product"])["checks"]),
        ("异常路径", "不满足约束时不改变状态，service 返回 success=false/message，事务 rollback"),
    ]
    add_detail_table(g, rules, 150, 760, 2220, "状态落库与约束")
    return g


def build_matrix(d: Diagram) -> Graph:
    g = Graph(d.title, 3000, 1900)
    add_title(g, d.title, "矩阵图｜职责、实现入口、验收证据")
    if d.key == "roles":
        headers = ["职责域", "管理员", "店长/财务", "收银员", "实现入口"]
        rows = [
            ["账号与权限", "审核/维护", "查看", "登录/退出", "auth.py + users"],
            ["商品库存", "增删改/导入", "盘点/预警", "查询", "product.py + inventory.py"],
            ["收银销售", "查看/配置", "销售分析", "结算/退款", "cashier.py + sales.py"],
            ["财务对账", "维护", "日结/应付/月结", "无", "finance.py"],
            ["公告助手", "发布/管理", "查看", "阅读/问答", "announcements.py + assistant.py"],
            ["二期主数据", "会员/员工/供应商/参数", "查看", "会员识别", "second_phase.py"],
        ]
    elif d.key == "testing":
        headers = ["模块", "模型/服务", "覆盖场景", "测试文件", "结果"]
        rows = [
            ["商品/库存", "products.py / inventory.py", "新增、导入、上下架、库存流水", "test_core_services.py", "100%"],
            ["收银销售", "cashier.py / sales.py", "结账、扣库存、订单明细", "test_core_services.py", "100%"],
            ["财务", "finance.py", "交易、对账、应付、付款、月结", "test_core_services.py", "100%"],
            ["会员", "second_phase.py", "CRUD、积分、状态", "test_second_phase.py", "100%"],
            ["员工/供应商", "second_phase.py", "唯一校验、状态切换", "test_second_phase.py", "100%"],
            ["异常路径", "models + services", "重复键、负积分、空输入、权限访问", "test_*_errors/api/auth.py", "100%"],
        ]
    elif d.key == "release":
        headers = ["交付项", "本地路径", "格式", "状态", "验收点"]
        rows = [
            ["最终报告", "reports/system-analysis-design", ".docx", "已同步图表", "57 张主图"],
            ["可编辑图", "supermarket-management-diagrams-drawio-editable", ".drawio", "中文命名", "可在 draw.io 打开"],
            ["报告图片", "reports/system-analysis-design/images", ".png", "中文命名", "57 主图 + 12 用例"],
            ["自动化测试", "tests/", ".py", "已完成", "pytest 通过"],
            ["覆盖率记录", "screenshots/*.txt", ".txt/.png", "txt 已生成", "coverage 100%"],
            ["课程文档", "docs/course-deliverables", ".md", "已补齐", "阶段一至七材料"],
        ]
    else:
        headers = ["能力域", "页面/脚本", "接口", "数据表", "二期覆盖"]
        rows = [
            ["商品主数据", "product.html/js", "/api/products", "products/categories", "已完成"],
            ["库存运营", "inventory.html/js", "/api/inventory/*", "inventory/logs", "已完成"],
            ["销售收银", "cashier/sales", "/api/cashier /api/sales", "sales/items", "已完成"],
            ["财务经营", "finance/analytics", "/api/finance /api/analytics", "finance/cash/payables", "已完成"],
            ["公告助手", "announcements/assistant", "/api/announcements /chat", "announcements/reads", "已完成"],
            ["二期主数据", "master_data.html/js", "/api/members/employees/suppliers/settings", "members/employees/suppliers/settings", "已完成"],
        ]
    col_w = [420, 560, 560, 560, 520]
    x0, y0 = 110, 210
    for col, head in enumerate(headers):
        g.vertex(head, x0 + sum(col_w[:col]), y0, col_w[col], 70, box_style("#172033", "#172033", 23, True, "fontColor=#ffffff;"))
    for row_idx, row in enumerate(rows):
        for col, value in enumerate(row):
            fill = "#f8fafc" if row_idx % 2 == 0 else "#ffffff"
            g.vertex(value, x0 + sum(col_w[:col]), y0 + 70 + row_idx * 112, col_w[col], 112, box_style(fill, "#cbd5e1", 21, col == 0, "align=left;spacingLeft=16;"))
    if d.key == "testing":
        g.vertex("coverage report：TOTAL 530 / 0 / 100%，fail_under = 100", 110, 980, 1400, 80, box_style(GREEN_FILL, GREEN, 24, True, "align=left;spacingLeft=18;"))
    add_detail_table(g, detail_rows(d.key), 110, 1160, 2620, "实现明细")
    return g


def build_context(d: Diagram) -> Graph:
    g = Graph(d.title, 2800, 1800)
    add_title(g, d.title, "上下文图｜参与者、系统边界、外部交付物")
    admin = g.vertex("管理员\n商品/库存/财务/公告/二期主数据", 130, 310, 330, 130, box_style("#ffffff", GRAY, 23, True))
    cashier = g.vertex("收银员\n收银/销售/公告/助手", 130, 560, 330, 130, box_style("#ffffff", GRAY, 23, True))
    manager = g.vertex("店长/财务\n分析/对账/验收", 130, 810, 330, 130, box_style("#ffffff", GRAY, 23, True))
    browser = g.vertex("浏览器\nJinja2 页面 + JavaScript fetch", 650, 520, 420, 180, box_style(BLUE_FILL, BLUE, 24, True))
    app = g.vertex("超市管理系统\nFlask routes + services + models\n登录、商品、库存、收银、销售、财务、公告、分析、助手、二期主数据", 1230, 380, 600, 420, box_style("#ffffff", INK, 25, True))
    db = g.vertex("SQLite 数据库\nschema.sql + seed_data.sql\n18 张核心业务表", 2060, 300, 470, 180, box_style(GREEN_FILL, GREEN, 24, True))
    llm = g.vertex("可选 LLM\nOPENAI_API_KEY\n失败自动降级本地规则", 2060, 610, 470, 170, box_style(GRAY_FILL, GRAY, 23, True))
    deliver = g.vertex("课程交付物\nREADME / docs / reports\n中文 drawio + PNG", 2060, 910, 470, 190, box_style(ORANGE_FILL, ORANGE, 23, True))
    for actor in [admin, cashier, manager]:
        g.edge(actor, browser, "操作页面", exit_pos=(1, 0.5), entry_pos=(0, 0.5))
    g.edge(browser, app, "HTTP/JSON")
    g.edge(app, db, "SQLAlchemy 读写")
    g.edge(app, llm, "助手可选调用", dashed=True)
    g.edge(app, deliver, "验收材料", dashed=True)
    add_detail_table(g, detail_rows("system"), 130, 1220, 2400, "系统边界明细")
    return g


def build_usecase(d: Diagram) -> Graph:
    g = Graph(d.title, 3000, 1900)
    add_title(g, d.title, "UML 用例总图｜角色、用例、实现入口")
    boundary = add_section(g, "系统边界：超市管理系统", 460, 210, 1900, 930, "#ffffff")
    actors = [
        ("管理员", 120, 310),
        ("收银员", 120, 790),
        ("店长/财务", 2470, 550),
    ]
    actor_ids = {name: g.vertex(name, x0, y0, 190, 78, box_style("#ffffff", GRAY, 23, True)) for name, x0, y0 in actors}
    cases = [
        ("登录/注册审核", "管理员", "auth.py"),
        ("商品新增/导入/调价", "管理员", "product.py"),
        ("库存查询/盘点/预警", "管理员", "inventory.py"),
        ("收银结算/退款", "收银员", "cashier.py"),
        ("销售订单查询", "收银员", "sales.py"),
        ("财务交易/日结/月结", "店长/财务", "finance.py"),
        ("公告发布/已读", "管理员", "announcements.py"),
        ("经营分析报表", "店长/财务", "analytics.py"),
        ("智能助手问答", "收银员", "assistant.py"),
        ("会员积分管理", "管理员", "second_phase.py"),
        ("员工档案管理", "管理员", "second_phase.py"),
        ("供应商与系统参数", "管理员", "second_phase.py"),
    ]
    for idx, (case, actor, impl) in enumerate(cases):
        cx = 560 + (idx % 3) * 600
        cy = 310 + (idx // 3) * 190
        case_id = g.vertex(f"{case}\n{impl}", cx, cy, 430, 105, box_style(BLUE_FILL if actor == "管理员" else ORANGE_FILL if actor == "收银员" else GREEN_FILL, BLUE if actor == "管理员" else ORANGE if actor == "收银员" else GREEN, 21, True, "ellipse;"))
        g.edge(actor_ids[actor], case_id, "", dashed=True)
    _ = boundary
    add_detail_table(g, detail_rows("system"), 120, 1260, 2640, "用例实现明细")
    return g


def build_tree(d: Diagram) -> Graph:
    g = Graph(d.title, 2900, 1800)
    add_title(g, d.title, "功能分解结构｜模块、页面、API、落表")
    root = g.vertex("超市管理系统", 1250, 190, 420, 90, box_style(BLUE_FILL, BLUE, 28, True))
    modules = [
        ("商品管理", "product.html/js\n/api/products/import\nproducts/categories"),
        ("库存管理", "inventory.html/js\n/api/inventory/logs\ninventory_logs"),
        ("收银销售", "cashier/sales 页面\ncheckout/order APIs\nsales/sale_items"),
        ("财务分析", "finance/analytics 页面\nreconciliation/payables\nfinance_transactions"),
        ("公告助手", "announcements/assistant\nread/chat APIs\nannouncements/reads"),
        ("二期主数据", "master_data.html/js\nmembers/employees/suppliers\nsystem_settings"),
    ]
    for idx, (name, info) in enumerate(modules):
        x0 = 130 + idx * 460
        node = g.vertex(name, x0, 430, 350, 80, box_style("#ffffff", INK, 24, True))
        g.edge(root, node, "", exit_pos=(0.5, 1), entry_pos=(0.5, 0))
        g.vertex(info, x0, 560, 350, 220, box_style(GRAY_FILL, GRAY, 21, False, "align=left;verticalAlign=top;spacingLeft=16;spacingTop=14;"))
    add_detail_table(g, detail_rows("system"), 130, 1020, 2580, "功能实现明细")
    return g


def build_deployment(d: Diagram) -> Graph:
    g = Graph(d.title, 2800, 1800)
    add_title(g, d.title, "部署图｜运行环境、初始化、验证、交付")
    if d.key == "production":
        nodes = [
            ("Gitee 仓库\nmain 分支\nREADME + docs + reports", BLUE_FILL, BLUE),
            ("部署主机\nWindows/Linux\nuv + Python 3.13", ORANGE_FILL, ORANGE),
            ("应用进程\nuv run python run.py\ncreate_app()", PURPLE_FILL, PURPLE),
            ("SQLite 数据\nschema.sql\nseed_data.sql", GREEN_FILL, GREEN),
            ("浏览器访问\n/login\n业务页面", BLUE_FILL, BLUE),
            ("验收验证\npytest/coverage\n截图/报告/图表", GRAY_FILL, GRAY),
        ]
    else:
        nodes = [
            ("本地工作区\nsupermarket-management-system", BLUE_FILL, BLUE),
            ("依赖环境\nuv sync\nPython 3.13", ORANGE_FILL, ORANGE),
            ("启动应用\nuv run python run.py\nFlask Debug", PURPLE_FILL, PURPLE),
            ("数据库初始化\ndata/SQL/schema.sql\nseed_data.sql", GREEN_FILL, GREEN),
            ("自动化验证\nuv run pytest\ncoverage report", GRAY_FILL, GRAY),
            ("交付物目录\nreports/images\ndrawio-editable/docs", BLUE_FILL, BLUE),
            ("人工截图\nmanual-pages/*.png\npytest/coverage.png", ORANGE_FILL, ORANGE),
        ]
    ids = []
    for idx, (label, fill, stroke) in enumerate(nodes):
        x0 = 120 + (idx % 4) * 650
        y0 = 280 + (idx // 4) * 360
        ids.append(g.vertex(label, x0, y0, 480, 170, box_style(fill, stroke, 23, True)))
    for a, b in zip(ids, ids[1:]):
        g.edge(a, b, "", exit_pos=(1, 0.5), entry_pos=(0, 0.5))
    add_detail_table(g, detail_rows("release" if d.key == "local" else "system"), 120, 1120, 2440, "部署验收明细")
    return g


def build_module_usecase(title: str, key: str) -> Graph:
    g = Graph(title, 2800, 1800)
    add_title(g, title, "模块级用例矩阵｜角色｜用例｜实现入口｜校验点")
    detail = DETAILS.get(key, DETAILS["product"])
    routes = detail["routes"].split("、")
    services = detail["service"].split("、")
    tables = detail["tables"].split("、")
    checks = detail["checks"].split("；")
    cases = [
        ("列表/查询", routes[0] if routes else detail["routes"]),
        ("新增/更新", routes[1] if len(routes) > 1 else detail["routes"]),
        ("状态动作", services[0] if services else detail["service"]),
        ("业务处理", services[1] if len(services) > 1 else detail["service"]),
        ("数据落表", " / ".join(tables[:2])),
        ("校验异常", checks[0] if checks else detail["checks"]),
    ]
    g.vertex("角色", 110, 210, 260, 70, box_style("#172033", "#172033", 24, True, "fontColor=#ffffff;"))
    g.vertex("管理员\n配置、维护、审核", 110, 300, 260, 150, box_style(BLUE_FILL, BLUE, 22, True))
    g.vertex("业务人员\n查询、办理、确认", 110, 490, 260, 150, box_style(ORANGE_FILL, ORANGE, 22, True))
    g.vertex("系统边界：超市管理系统", 430, 210, 1450, 70, box_style("#172033", "#172033", 24, True, "fontColor=#ffffff;"))
    for idx, (name, info) in enumerate(cases):
        cx = 470 + (idx % 3) * 470
        cy = 320 + (idx // 3) * 220
        fill, stroke = (BLUE_FILL, BLUE) if idx in {0, 1, 2, 5} else (ORANGE_FILL, ORANGE)
        g.vertex(f"{name}\n{info}", cx, cy, 410, 145, box_style(fill, stroke, 22, True, "align=left;spacingLeft=16;"))
    dependency_rows = [
        ("include", "列表/查询 -> 业务处理；新增/更新 -> 状态动作；业务处理 -> 数据落表；校验异常贯穿所有用例"),
        ("角色权限", "管理员侧负责维护和审核；业务人员侧负责查询、办理、确认，具体权限由 login_required/admin_required 控制"),
        ("验收证据", "页面模板、JS fetch、routes、services、models、tests 均可定位到对应实现"),
    ]
    add_detail_table(g, dependency_rows, 110, 790, 1770, "用例依赖")
    add_detail_table(g, detail_rows(key), 1950, 210, 700, "实现明细")
    return g


def build_diagram(d: Diagram) -> Graph:
    if d.kind == "process":
        return build_process(d)
    if d.kind == "erd":
        return build_erd(d)
    if d.kind in {"architecture", "api"}:
        return build_architecture(d)
    if d.kind == "sequence":
        return build_sequence(d)
    if d.kind == "state":
        return build_state(d)
    if d.kind == "dfd":
        return build_dfd(d)
    if d.kind == "matrix":
        return build_matrix(d)
    if d.kind == "context":
        return build_context(d)
    if d.kind == "usecase":
        return build_usecase(d)
    if d.kind == "tree":
        return build_tree(d)
    if d.kind == "deployment":
        return build_deployment(d)
    return build_architecture(d)


def ensure_clean_outputs() -> None:
    targets = [
        DRAWIO_ROOT,
        PNG_MIRROR_ROOT,
        REPORT_IMAGE_ROOT,
    ]
    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
        for pattern in ("*.drawio", "*.png", "*.svg"):
            for file in target.rglob(pattern):
                file.unlink()
        for readme in target.rglob("README.md"):
            readme.unlink()
        # Remove empty old English folders from the previous generator.
        for child in sorted(target.rglob("*"), reverse=True):
            if child.is_dir():
                try:
                    child.rmdir()
                except OSError:
                    pass


def write_drawio(path: Path, graph: Graph) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(graph.xml(), encoding="utf-8")


def export_png(drawio_path: Path, png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    if png_path.exists():
        png_path.unlink()
    command = [
        str(DRAWIO_EXE),
        "--export",
        "--format",
        "png",
        "--border",
        "24",
        "--scale",
        "2",
        "--output",
        str(png_path),
        str(drawio_path),
    ]
    subprocess.run(command, check=True, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(50):
        if png_path.exists() and png_path.stat().st_size > 0:
            return
        time.sleep(0.2)
    raise RuntimeError(f"PNG export failed: {png_path}")


def validate_drawio_files() -> None:
    import xml.etree.ElementTree as ET

    for file in DRAWIO_ROOT.rglob("*.drawio"):
        ET.parse(file)
        text = file.read_text(encoding="utf-8")
        if 'edge="1"' in text and '<mxGeometry relative="1" as="geometry"' not in text:
            raise RuntimeError(f"edge geometry missing in {file}")


def write_readme_for_folder(folder: Path, files: list[tuple[str, str]], label: str) -> None:
    lines = [
        f"# {label}",
        "",
        "图表使用 drawio-skill 的企业风格规则生成，文件名为中文，便于报告引用和人工验收。",
        "",
        "## 文件清单",
        "",
    ]
    for filename, title in files:
        lines.append(f"- `{filename}`：{title}")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readmes() -> None:
    grouped: dict[str, list[Diagram]] = {}
    for d in DIAGRAMS:
        grouped.setdefault(d.folder, []).append(d)
    for folder, diagrams in grouped.items():
        drawio_files = [(f"{d.base_name}.drawio", d.title) for d in sorted(diagrams, key=lambda item: item.no)]
        png_files = [(f"{d.base_name}.png", d.title) for d in sorted(diagrams, key=lambda item: item.no)]
        write_readme_for_folder(DRAWIO_ROOT / folder, drawio_files, "draw.io 可编辑图表")
        write_readme_for_folder(PNG_MIRROR_ROOT / folder, png_files, "PNG 图表")
    module_drawio = [(f"{name}.drawio", title) for name, title, _ in MODULE_USE_CASES]
    module_png = [(f"{name}.png", title) for name, title, _ in MODULE_USE_CASES]
    write_readme_for_folder(DRAWIO_ROOT / "01-环境与用例" / "模块级用例图", module_drawio, "模块级 UML 用例图 draw.io")
    write_readme_for_folder(PNG_MIRROR_ROOT / "01-环境与用例" / "模块级用例图", module_png, "模块级 UML 用例图 PNG")


def main_pngs() -> list[Path]:
    return [REPORT_IMAGE_ROOT / f"{d.base_name}.png" for d in sorted(DIAGRAMS, key=lambda item: item.no)]


def sync_docx_images() -> None:
    docx_files = [p for p in REPORT_ROOT.glob("*.docx") if not p.name.endswith(".bak.docx")]
    if not docx_files:
        return
    docx_path = docx_files[0]
    pngs = main_pngs()
    if len([p for p in pngs if p.exists()]) != len(DIAGRAMS):
        raise RuntimeError("main PNG count is incomplete; skip docx sync")
    backup = docx_path.with_suffix(".docx.bak-before-chinese-drawio")
    if not backup.exists():
        shutil.copy2(docx_path, backup)

    with zipfile.ZipFile(docx_path, "r") as zin:
        names = zin.namelist()
        doc = zin.read("word/document.xml").decode("utf-8")
        rels = zin.read("word/_rels/document.xml.rels").decode("utf-8")
        rids = re.findall(r'<[^>]*blip\b[^>]*(?:\w+:)?embed="([^"]+)"', doc)
        relmap = {m.group(1): m.group(2) for m in re.finditer(r'<Relationship[^>]*Id="([^"]+)"[^>]*Target="([^"]+)"', rels)}
        image_targets = [relmap[rid] for rid in rids if rid in relmap and relmap[rid].startswith("media/")]
        # First embedded picture is usually the cover/logo. Replace the following 57 report diagrams.
        if len(image_targets) < len(DIAGRAMS) + 1:
            raise RuntimeError(f"docx has only {len(image_targets)} embedded images; expected at least {len(DIAGRAMS)+1}")
        replace_targets = image_targets[1 : 1 + len(DIAGRAMS)]

        fd, tmp_name = tempfile.mkstemp(suffix=".docx", dir=str(REPORT_ROOT))
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zout:
                for name in names:
                    data = zin.read(name)
                    target_short = name.replace("word/", "", 1) if name.startswith("word/media/") else None
                    if target_short in replace_targets:
                        index = replace_targets.index(target_short)
                        data = pngs[index].read_bytes()
                    zout.writestr(name, data)
            with zipfile.ZipFile(tmp_path, "r") as test_zip:
                bad = test_zip.testzip()
                if bad:
                    raise RuntimeError(f"docx zip validation failed at {bad}")
            shutil.move(str(tmp_path), str(docx_path))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


def generate(export: bool = True, sync_report: bool = True) -> None:
    ensure_clean_outputs()
    for d in sorted(DIAGRAMS, key=lambda item: item.no):
        graph = build_diagram(d)
        drawio_path = DRAWIO_ROOT / d.folder / f"{d.base_name}.drawio"
        report_png = REPORT_IMAGE_ROOT / f"{d.base_name}.png"
        mirror_png = PNG_MIRROR_ROOT / d.folder / f"{d.base_name}.png"
        write_drawio(drawio_path, graph)
        if export:
            export_png(drawio_path, report_png)
            mirror_png.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(report_png, mirror_png)

    module_drawio_dir = DRAWIO_ROOT / "01-环境与用例" / "模块级用例图"
    module_report_dir = REPORT_IMAGE_ROOT / "模块级用例图"
    module_mirror_dir = PNG_MIRROR_ROOT / "01-环境与用例" / "模块级用例图"
    for name, title, key in MODULE_USE_CASES:
        graph = build_module_usecase(title, key)
        drawio_path = module_drawio_dir / f"{name}.drawio"
        report_png = module_report_dir / f"{name}.png"
        mirror_png = module_mirror_dir / f"{name}.png"
        write_drawio(drawio_path, graph)
        if export:
            export_png(drawio_path, report_png)
            mirror_png.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(report_png, mirror_png)

    write_readmes()
    validate_drawio_files()
    if export and sync_report:
        sync_docx_images()


def cli() -> None:
    parser = argparse.ArgumentParser(description="生成中文 draw.io 企业模板图表并导出 PNG")
    parser.add_argument("--no-export", action="store_true", help="只生成 .drawio，不导出 PNG")
    parser.add_argument("--no-docx-sync", action="store_true", help="不替换报告 docx 中的图片")
    args = parser.parse_args()
    if not DRAWIO_EXE.exists() and not args.no_export:
        raise SystemExit(f"draw.io not found: {DRAWIO_EXE}")
    generate(export=not args.no_export, sync_report=not args.no_docx_sync)
    print(f"generated {len(DIAGRAMS)} main diagrams and {len(MODULE_USE_CASES)} module use-case diagrams")


if __name__ == "__main__":
    cli()
