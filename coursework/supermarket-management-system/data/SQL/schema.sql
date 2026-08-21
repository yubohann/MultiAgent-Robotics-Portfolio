-- ============================================
-- 超市管理系统数据库 - SQLite
-- 版本: 1.0
-- 设计: 支持商品/库存/销售/分析
-- ============================================

-- 1. 用户表（管理员 + 收银员）
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    real_name VARCHAR(50),
    role VARCHAR(20) DEFAULT 'cashier' CHECK(role IN ('admin', 'cashier')),
    is_active INTEGER DEFAULT 1 CHECK(is_active IN (0, 1)),
    created_at DATETIME DEFAULT (datetime('now', 'localtime'))
);

-- 1.1 管理员注册申请表（待审核）
CREATE TABLE IF NOT EXISTS admin_signup_requests (
    request_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'approved', 'rejected')),
    reviewed_by INTEGER,
    reviewed_at DATETIME,
    reject_reason VARCHAR(200),
    created_at DATETIME DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (reviewed_by) REFERENCES users(user_id) ON DELETE SET NULL
);

-- 1.2 公告表
CREATE TABLE IF NOT EXISTS announcements (
    announcement_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title VARCHAR(120) NOT NULL,
    content VARCHAR(1000) NOT NULL,
    level VARCHAR(20) NOT NULL DEFAULT 'normal' CHECK(level IN ('normal', 'important')),
    target_role VARCHAR(20) NOT NULL DEFAULT 'all' CHECK(target_role IN ('all', 'admin', 'cashier')),
    is_published INTEGER NOT NULL DEFAULT 1 CHECK(is_published IN (0, 1)),
    created_by INTEGER,
    created_at DATETIME DEFAULT (datetime('now', 'localtime')),
    updated_at DATETIME,
    FOREIGN KEY (created_by) REFERENCES users(user_id) ON DELETE SET NULL
);

-- 1.3 公告已读记录表
CREATE TABLE IF NOT EXISTS announcement_reads (
    read_id INTEGER PRIMARY KEY AUTOINCREMENT,
    announcement_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    read_at DATETIME DEFAULT (datetime('now', 'localtime')),
    UNIQUE (announcement_id, user_id),
    FOREIGN KEY (announcement_id) REFERENCES announcements(announcement_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- 2. 商品分类表（支持层级）
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name VARCHAR(50) NOT NULL,
    parent_id INTEGER DEFAULT 0,
    sort_order INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT (datetime('now', 'localtime'))
);

-- 3. 商品表
CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    barcode VARCHAR(50) UNIQUE,
    product_code VARCHAR(50) UNIQUE NOT NULL,
    product_name VARCHAR(100) NOT NULL,
    category_id INTEGER,
    unit VARCHAR(20) DEFAULT '件',
    purchase_price DECIMAL(10,2) DEFAULT 0.00,
    selling_price DECIMAL(10,2) NOT NULL,
    min_stock INTEGER DEFAULT 10,
    status INTEGER DEFAULT 1 CHECK(status IN (0, 1)),
    created_at DATETIME DEFAULT (datetime('now', 'localtime')),
    updated_at DATETIME,
    FOREIGN KEY (category_id) REFERENCES categories(category_id) ON DELETE SET NULL
);

-- 4. 库存表（一对一，简化设计）
CREATE TABLE IF NOT EXISTS inventory (
    product_id INTEGER PRIMARY KEY,
    quantity INTEGER NOT NULL DEFAULT 0 CHECK(quantity >= 0),
    last_check_time DATETIME,
    updated_at DATETIME,
    FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE
);

-- 5. 库存变动日志表（核心追踪表）
CREATE TABLE IF NOT EXISTS inventory_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    change_type VARCHAR(20) NOT NULL CHECK(change_type IN ('in', 'out', 'adjust', 'sale', 'return', 'create', 'import')),
    quantity_change INTEGER NOT NULL,
    quantity_before INTEGER NOT NULL,
    quantity_after INTEGER NOT NULL,
    reason VARCHAR(200),
    operator_id INTEGER,
    created_at DATETIME DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE,
    FOREIGN KEY (operator_id) REFERENCES users(user_id) ON DELETE SET NULL
);

-- 6. 销售订单表
CREATE TABLE IF NOT EXISTS sales (
    sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_no VARCHAR(50) UNIQUE NOT NULL,
    cashier_id INTEGER NOT NULL,
    total_amount DECIMAL(10,2) NOT NULL,
    discount_amount DECIMAL(10,2) DEFAULT 0.00,
    actual_amount DECIMAL(10,2) NOT NULL,
    payment_method VARCHAR(20) CHECK(payment_method IN ('cash', 'wechat', 'alipay', 'card')),
    status VARCHAR(20) DEFAULT 'completed' CHECK(status IN ('completed', 'refunded', 'cancelled')),
    created_at DATETIME DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (cashier_id) REFERENCES users(user_id) ON DELETE RESTRICT
);

-- 7. 销售明细表（价格快照）
CREATE TABLE IF NOT EXISTS sale_items (
    item_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sale_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL CHECK(quantity > 0),
    unit_price DECIMAL(10,2) NOT NULL,
    purchase_price DECIMAL(10,2) NOT NULL,
    subtotal DECIMAL(10,2) NOT NULL,
    FOREIGN KEY (sale_id) REFERENCES sales(sale_id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE RESTRICT
);

-- 8. 财务流水表（收入/支出）
CREATE TABLE IF NOT EXISTS finance_transactions (
    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_no VARCHAR(50) UNIQUE NOT NULL,
    transaction_type VARCHAR(20) NOT NULL CHECK(transaction_type IN ('income', 'expense')),
    category VARCHAR(50) NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    payment_method VARCHAR(20) CHECK(payment_method IN ('cash', 'wechat', 'alipay', 'card', 'bank_transfer')),
    related_order_no VARCHAR(50),
    description VARCHAR(300),
    occurred_at DATETIME DEFAULT (datetime('now', 'localtime')),
    operator_id INTEGER,
    created_at DATETIME DEFAULT (datetime('now', 'localtime')),
    updated_at DATETIME,
    FOREIGN KEY (operator_id) REFERENCES users(user_id) ON DELETE SET NULL
);

-- 9. 日结对账表
CREATE TABLE IF NOT EXISTS cash_reconciliations (
    reconciliation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    reconcile_date DATE NOT NULL,
    payment_method VARCHAR(20) NOT NULL CHECK(payment_method IN ('cash', 'wechat', 'alipay', 'card')),
    expected_amount DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    actual_amount DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    difference_amount DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    note VARCHAR(300),
    created_by INTEGER,
    created_at DATETIME DEFAULT (datetime('now', 'localtime')),
    updated_at DATETIME,
    UNIQUE (reconcile_date, payment_method),
    FOREIGN KEY (created_by) REFERENCES users(user_id) ON DELETE SET NULL
);

-- 10. 供应商应付款表
CREATE TABLE IF NOT EXISTS supplier_payables (
    payable_id INTEGER PRIMARY KEY AUTOINCREMENT,
    supplier_name VARCHAR(100) NOT NULL,
    bill_no VARCHAR(60) UNIQUE,
    total_amount DECIMAL(10,2) NOT NULL,
    paid_amount DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    due_date DATE NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'unpaid' CHECK(status IN ('unpaid', 'partial', 'paid', 'overdue')),
    note VARCHAR(300),
    created_by INTEGER,
    created_at DATETIME DEFAULT (datetime('now', 'localtime')),
    updated_at DATETIME,
    FOREIGN KEY (created_by) REFERENCES users(user_id) ON DELETE SET NULL
);

-- 11. 应付款支付记录表
CREATE TABLE IF NOT EXISTS payable_payments (
    payment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    payable_id INTEGER NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    payment_method VARCHAR(20) CHECK(payment_method IN ('cash', 'wechat', 'alipay', 'card', 'bank_transfer')),
    paid_at DATETIME DEFAULT (datetime('now', 'localtime')),
    remark VARCHAR(200),
    operator_id INTEGER,
    created_at DATETIME DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (payable_id) REFERENCES supplier_payables(payable_id) ON DELETE CASCADE,
    FOREIGN KEY (operator_id) REFERENCES users(user_id) ON DELETE SET NULL
);

-- 12. 财务关账快照表
CREATE TABLE IF NOT EXISTS finance_period_closings (
    close_id INTEGER PRIMARY KEY AUTOINCREMENT,
    period_month VARCHAR(7) UNIQUE NOT NULL,
    total_sales DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    other_income DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    expense_amount DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    gross_profit DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    net_profit DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    cash_inflow DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    cash_outflow DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    note VARCHAR(300),
    closed_by INTEGER,
    closed_at DATETIME DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (closed_by) REFERENCES users(user_id) ON DELETE SET NULL
);

-- 13. 会员账户表
CREATE TABLE IF NOT EXISTS members (
    member_id INTEGER PRIMARY KEY AUTOINCREMENT,
    member_no VARCHAR(50) UNIQUE NOT NULL,
    member_name VARCHAR(80) NOT NULL,
    phone VARCHAR(30) UNIQUE,
    level VARCHAR(20) DEFAULT 'normal' CHECK(level IN ('normal', 'silver', 'gold', 'vip')),
    points INTEGER NOT NULL DEFAULT 0 CHECK(points >= 0),
    status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'inactive')),
    registered_at DATETIME DEFAULT (datetime('now', 'localtime')),
    updated_at DATETIME
);

-- 14. 员工档案表
CREATE TABLE IF NOT EXISTS employees (
    employee_id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_no VARCHAR(50) UNIQUE NOT NULL,
    employee_name VARCHAR(80) NOT NULL,
    position VARCHAR(50) NOT NULL,
    phone VARCHAR(30) UNIQUE,
    work_schedule VARCHAR(120),
    status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'inactive')),
    hired_at DATETIME DEFAULT (datetime('now', 'localtime')),
    updated_at DATETIME
);

-- 15. 供应商档案表
CREATE TABLE IF NOT EXISTS suppliers (
    supplier_id INTEGER PRIMARY KEY AUTOINCREMENT,
    supplier_code VARCHAR(50) UNIQUE NOT NULL,
    supplier_name VARCHAR(120) NOT NULL,
    contact_person VARCHAR(80),
    phone VARCHAR(30),
    settlement_cycle VARCHAR(30) NOT NULL DEFAULT 'monthly' CHECK(settlement_cycle IN ('weekly', 'monthly', 'quarterly')),
    status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'inactive')),
    created_at DATETIME DEFAULT (datetime('now', 'localtime')),
    updated_at DATETIME
);

-- 16. 系统参数表
CREATE TABLE IF NOT EXISTS system_settings (
    setting_id INTEGER PRIMARY KEY AUTOINCREMENT,
    setting_key VARCHAR(80) UNIQUE NOT NULL,
    setting_value VARCHAR(300) NOT NULL,
    description VARCHAR(300),
    updated_at DATETIME DEFAULT (datetime('now', 'localtime'))
);

-- ============================================
-- 索引优化
-- ============================================
CREATE INDEX IF NOT EXISTS idx_products_barcode ON products(barcode);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category_id);
CREATE INDEX IF NOT EXISTS idx_products_status ON products(status);
CREATE INDEX IF NOT EXISTS idx_admin_signup_requests_status ON admin_signup_requests(status);
CREATE INDEX IF NOT EXISTS idx_admin_signup_requests_created ON admin_signup_requests(created_at);
CREATE INDEX IF NOT EXISTS idx_announcements_target_role ON announcements(target_role);
CREATE INDEX IF NOT EXISTS idx_announcements_published_created ON announcements(is_published, created_at);
CREATE INDEX IF NOT EXISTS idx_announcement_reads_user ON announcement_reads(user_id);
CREATE INDEX IF NOT EXISTS idx_announcement_reads_announcement ON announcement_reads(announcement_id);
CREATE INDEX IF NOT EXISTS idx_inventory_logs_product ON inventory_logs(product_id);
CREATE INDEX IF NOT EXISTS idx_inventory_logs_type ON inventory_logs(change_type);
CREATE INDEX IF NOT EXISTS idx_inventory_logs_created ON inventory_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_sales_cashier ON sales(cashier_id);
CREATE INDEX IF NOT EXISTS idx_sales_created ON sales(created_at);
CREATE INDEX IF NOT EXISTS idx_sales_status ON sales(status);
CREATE INDEX IF NOT EXISTS idx_sale_items_sale ON sale_items(sale_id);
CREATE INDEX IF NOT EXISTS idx_sale_items_product ON sale_items(product_id);
CREATE INDEX IF NOT EXISTS idx_finance_transactions_type ON finance_transactions(transaction_type);
CREATE INDEX IF NOT EXISTS idx_finance_transactions_occurred ON finance_transactions(occurred_at);
CREATE INDEX IF NOT EXISTS idx_finance_transactions_operator ON finance_transactions(operator_id);
CREATE INDEX IF NOT EXISTS idx_cash_reconciliations_date ON cash_reconciliations(reconcile_date);
CREATE INDEX IF NOT EXISTS idx_cash_reconciliations_method ON cash_reconciliations(payment_method);
CREATE INDEX IF NOT EXISTS idx_supplier_payables_due_date ON supplier_payables(due_date);
CREATE INDEX IF NOT EXISTS idx_supplier_payables_status ON supplier_payables(status);
CREATE INDEX IF NOT EXISTS idx_payable_payments_payable ON payable_payments(payable_id);
CREATE INDEX IF NOT EXISTS idx_finance_period_closings_month ON finance_period_closings(period_month);
CREATE INDEX IF NOT EXISTS idx_members_no ON members(member_no);
CREATE INDEX IF NOT EXISTS idx_members_phone ON members(phone);
CREATE INDEX IF NOT EXISTS idx_members_status ON members(status);
CREATE INDEX IF NOT EXISTS idx_employees_no ON employees(employee_no);
CREATE INDEX IF NOT EXISTS idx_employees_status ON employees(status);
CREATE INDEX IF NOT EXISTS idx_suppliers_code ON suppliers(supplier_code);
CREATE INDEX IF NOT EXISTS idx_suppliers_status ON suppliers(status);
CREATE INDEX IF NOT EXISTS idx_system_settings_key ON system_settings(setting_key);
