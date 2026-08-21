const statusLabels = {
    completed: '完成',
    refunded: '已退款',
    cancelled: '已取消'
};

const paymentLabels = {
    cash: '现金',
    wechat: '微信',
    alipay: '支付宝',
    card: '银行卡'
};

const statusClasses = {
    completed: 'bg-green-50 text-green-700',
    refunded: 'bg-amber-50 text-amber-700',
    cancelled: 'bg-red-50 text-red-700'
};

const paymentClasses = {
    cash: 'bg-slate-100 text-slate-700',
    wechat: 'bg-emerald-50 text-emerald-700',
    alipay: 'bg-blue-50 text-blue-700',
    card: 'bg-violet-50 text-violet-700'
};

const modal = document.getElementById('orderModal');
const orderTableBody = document.getElementById('orderTableBody');
const paginationControls = document.getElementById('paginationControls');
const paginationInfo = document.getElementById('paginationInfo');
const resultInfo = document.getElementById('resultInfo');
const searchInput = document.getElementById('searchInput');
const cashierFilter = document.getElementById('cashierFilter');
const statusFilter = document.getElementById('statusFilter');
const paymentFilter = document.getElementById('paymentFilter');
const startDateFilter = document.getElementById('startDateFilter');
const endDateFilter = document.getElementById('endDateFilter');
const searchBtn = document.getElementById('searchBtn');
const resetBtn = document.getElementById('resetBtn');
const closeModalBtn = document.getElementById('closeModalBtn');
const orderMeta = document.getElementById('orderMeta');
const orderItemsBody = document.getElementById('orderItemsBody');

let currentPage = 1;
const perPage = 10;

function formatMoney(value) {
    return `¥${Number(value || 0).toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function formatDateTime(value) {
    return value || '-';
}

function getFilters() {
    return {
        search: searchInput.value.trim(),
        cashier_id: cashierFilter.value,
        status: statusFilter.value,
        payment_method: paymentFilter.value,
        start_date: startDateFilter.value,
        end_date: endDateFilter.value
    };
}

function setDefaultDateRange() {
    const today = new Date();
    const start = new Date(today.getFullYear(), today.getMonth(), 1);
    startDateFilter.value = start.toISOString().slice(0, 10);
    endDateFilter.value = today.toISOString().slice(0, 10);
}

function renderStatusBadge(status) {
    const label = statusLabels[status] || status || '-';
    const className = statusClasses[status] || 'bg-slate-100 text-slate-700';
    return `<span class="inline-flex items-center rounded-full px-3 py-1 text-xs font-medium ${className}">${label}</span>`;
}

function renderPaymentBadge(method) {
    const label = paymentLabels[method] || method || '-';
    const className = paymentClasses[method] || 'bg-slate-100 text-slate-700';
    return `<span class="inline-flex items-center rounded-full px-3 py-1 text-xs font-medium ${className}">${label}</span>`;
}

function renderTable(items) {
    if (!items.length) {
        orderTableBody.innerHTML = '<tr><td colspan="10" class="px-6 py-12 text-center text-slate-400">暂无订单数据</td></tr>';
        return;
    }

    orderTableBody.innerHTML = items.map((item) => `
        <tr class="hover:bg-slate-50 transition-colors">
            <td class="px-6 py-4 text-sm font-medium text-slate-800">${item.order_no}</td>
            <td class="px-6 py-4 text-sm text-slate-600">${item.cashier_name}</td>
            <td class="px-6 py-4 text-sm text-slate-600">${item.item_count}</td>
            <td class="px-6 py-4 text-sm text-slate-600">${formatMoney(item.total_amount)}</td>
            <td class="px-6 py-4 text-sm text-slate-600">${formatMoney(item.discount_amount)}</td>
            <td class="px-6 py-4 text-sm text-slate-600">${formatMoney(item.actual_amount)}</td>
            <td class="px-6 py-4 text-sm">${renderPaymentBadge(item.payment_method)}</td>
            <td class="px-6 py-4 text-sm">${renderStatusBadge(item.status)}</td>
            <td class="px-6 py-4 text-sm text-slate-600 whitespace-nowrap">${formatDateTime(item.created_at)}</td>
            <td class="px-6 py-4 text-sm">
                <button class="view-detail-btn px-3 py-1.5 rounded-lg bg-blue-50 text-primary hover:bg-blue-100 transition-colors" data-id="${item.sale_id}">查看</button>
            </td>
        </tr>
    `).join('');

    document.querySelectorAll('.view-detail-btn').forEach((button) => {
        button.addEventListener('click', () => openOrderDetail(button.dataset.id));
    });
}

function renderPagination(current, totalPages) {
    if (totalPages <= 1) {
        paginationControls.innerHTML = '';
        return;
    }

    const buttons = [];
    buttons.push(`<button class="px-3 py-1.5 rounded-lg border border-slate-200 text-sm ${current === 1 ? 'text-slate-300 cursor-not-allowed' : 'text-slate-700 hover:bg-slate-50'}" ${current === 1 ? 'disabled' : ''} data-page="${current - 1}">上一页</button>`);

    const startPage = Math.max(1, current - 2);
    const endPage = Math.min(totalPages, current + 2);
    for (let page = startPage; page <= endPage; page += 1) {
        buttons.push(`<button class="px-3 py-1.5 rounded-lg border text-sm ${page === current ? 'bg-primary text-white border-primary' : 'border-slate-200 text-slate-700 hover:bg-slate-50'}" data-page="${page}">${page}</button>`);
    }

    buttons.push(`<button class="px-3 py-1.5 rounded-lg border border-slate-200 text-sm ${current === totalPages ? 'text-slate-300 cursor-not-allowed' : 'text-slate-700 hover:bg-slate-50'}" ${current === totalPages ? 'disabled' : ''} data-page="${current + 1}">下一页</button>`);
    paginationControls.innerHTML = buttons.join('');

    paginationControls.querySelectorAll('button[data-page]').forEach((button) => {
        button.addEventListener('click', () => {
            const page = Number(button.dataset.page);
            if (page >= 1 && page <= totalPages) {
                loadOrders(page);
            }
        });
    });
}

async function loadOrders(page = 1) {
    currentPage = page;
    const filters = getFilters();
    const params = new URLSearchParams({ page, per_page: perPage });

    Object.entries(filters).forEach(([key, value]) => {
        if (value) {
            params.set(key, value);
        }
    });

    orderTableBody.innerHTML = '<tr><td colspan="10" class="px-6 py-12 text-center text-slate-400">加载中...</td></tr>';

    try {
        const response = await fetch(`/api/sales/orders?${params.toString()}`);
        const data = await response.json();

        renderTable(data.items || []);
        renderPagination(data.current_page || 1, data.pages || 0);
        paginationInfo.textContent = `共 ${data.total || 0} 条记录`;
        resultInfo.textContent = `当前显示第 ${data.current_page || 1} 页，共 ${data.pages || 0} 页`;
    } catch (error) {
        console.error('加载订单失败:', error);
        orderTableBody.innerHTML = '<tr><td colspan="10" class="px-6 py-12 text-center text-red-500">加载失败，请稍后重试</td></tr>';
        paginationControls.innerHTML = '';
        paginationInfo.textContent = '共 0 条记录';
        resultInfo.textContent = '加载失败';
    }
}

function openModal() {
    modal.classList.remove('hidden');
    modal.classList.add('flex');
}

function closeModal() {
    modal.classList.add('hidden');
    modal.classList.remove('flex');
}

async function openOrderDetail(saleId) {
    try {
        const response = await fetch(`/api/sales/orders/${saleId}`);
        const data = await response.json();

        if (!data.success) {
            alert(data.message || '订单不存在');
            return;
        }

        const order = data.order;
        orderMeta.innerHTML = `
            <div class="rounded-2xl bg-slate-50 p-4">
                <p class="text-xs text-slate-500 mb-1">订单号</p>
                <p class="text-sm font-semibold text-slate-800">${order.order_no}</p>
            </div>
            <div class="rounded-2xl bg-slate-50 p-4">
                <p class="text-xs text-slate-500 mb-1">收银员</p>
                <p class="text-sm font-semibold text-slate-800">${order.cashier_name}</p>
            </div>
            <div class="rounded-2xl bg-slate-50 p-4">
                <p class="text-xs text-slate-500 mb-1">支付方式</p>
                <p class="text-sm font-semibold text-slate-800">${order.payment_method_label}</p>
            </div>
            <div class="rounded-2xl bg-slate-50 p-4">
                <p class="text-xs text-slate-500 mb-1">状态</p>
                <p class="text-sm font-semibold text-slate-800">${order.status_label}</p>
            </div>
            <div class="rounded-2xl bg-slate-50 p-4">
                <p class="text-xs text-slate-500 mb-1">应收金额</p>
                <p class="text-sm font-semibold text-slate-800">${formatMoney(order.total_amount)}</p>
            </div>
            <div class="rounded-2xl bg-slate-50 p-4">
                <p class="text-xs text-slate-500 mb-1">优惠金额</p>
                <p class="text-sm font-semibold text-slate-800">${formatMoney(order.discount_amount)}</p>
            </div>
            <div class="rounded-2xl bg-slate-50 p-4">
                <p class="text-xs text-slate-500 mb-1">实收金额</p>
                <p class="text-sm font-semibold text-slate-800">${formatMoney(order.actual_amount)}</p>
            </div>
            <div class="rounded-2xl bg-slate-50 p-4">
                <p class="text-xs text-slate-500 mb-1">下单时间</p>
                <p class="text-sm font-semibold text-slate-800">${order.created_at}</p>
            </div>
        `;

        if (!order.items.length) {
            orderItemsBody.innerHTML = '<tr><td colspan="6" class="px-4 py-8 text-center text-slate-400">暂无明细</td></tr>';
        } else {
            orderItemsBody.innerHTML = order.items.map((item) => `
                <tr>
                    <td class="px-4 py-3 text-sm text-slate-800">${item.product_name}</td>
                    <td class="px-4 py-3 text-sm text-slate-600">${item.product_code}</td>
                    <td class="px-4 py-3 text-sm text-slate-600">${item.quantity}</td>
                    <td class="px-4 py-3 text-sm text-slate-600">${formatMoney(item.unit_price)}</td>
                    <td class="px-4 py-3 text-sm text-slate-600">${formatMoney(item.purchase_price)}</td>
                    <td class="px-4 py-3 text-sm text-slate-600">${formatMoney(item.subtotal)}</td>
                </tr>
            `).join('');
        }

        document.getElementById('modalSubtitle').textContent = `${order.item_count} 个商品项目`;
        openModal();
    } catch (error) {
        console.error('加载订单详情失败:', error);
        alert('加载订单详情失败');
    }
}

function resetFilters() {
    searchInput.value = '';
    cashierFilter.value = '';
    statusFilter.value = '';
    paymentFilter.value = '';
    setDefaultDateRange();
    loadOrders(1);
}

searchBtn.addEventListener('click', () => loadOrders(1));
resetBtn.addEventListener('click', resetFilters);
closeModalBtn.addEventListener('click', closeModal);
modal.addEventListener('click', (event) => {
    if (event.target === modal) {
        closeModal();
    }
});

[searchInput, cashierFilter, statusFilter, paymentFilter, startDateFilter, endDateFilter].forEach((element) => {
    element.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            loadOrders(1);
        }
    });
});

setDefaultDateRange();
loadOrders(1);
