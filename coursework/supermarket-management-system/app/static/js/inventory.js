let inventoryPage = 1;
let logPage = 1;

document.addEventListener('DOMContentLoaded', function () {
    loadCategories();
    loadInventorySummary();
    loadInventoryList();
    loadInventoryLogs();
    loadInventoryAlerts();

    document.getElementById('inventorySearch').addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            inventoryPage = 1;
            loadInventoryList();
        }
    });

    document.getElementById('inventoryCategoryFilter').addEventListener('change', function () {
        inventoryPage = 1;
        loadInventoryList();
    });

    document.getElementById('stockStateFilter').addEventListener('change', function () {
        inventoryPage = 1;
        loadInventoryList();
    });

    document.getElementById('logSearch').addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            logPage = 1;
            loadInventoryLogs();
        }
    });

    document.getElementById('changeTypeFilter').addEventListener('change', function () {
        logPage = 1;
        loadInventoryLogs();
    });

    document.getElementById('startDateFilter').addEventListener('change', function () {
        logPage = 1;
        loadInventoryLogs();
    });

    document.getElementById('endDateFilter').addEventListener('change', function () {
        logPage = 1;
        loadInventoryLogs();
    });
});

function reloadAll() {
    inventoryPage = 1;
    logPage = 1;
    loadInventorySummary();
    loadInventoryList();
    loadInventoryLogs();
    loadInventoryAlerts();
}

function loadCategories() {
    fetch('/api/categories')
        .then((res) => res.json())
        .then((data) => {
            const select = document.getElementById('inventoryCategoryFilter');
            data.categories.forEach((cat) => {
                select.innerHTML += `<option value="${cat.id}">${cat.name}</option>`;
            });
        });
}

function loadInventorySummary() {
    fetch('/api/inventory/summary')
        .then((res) => res.json())
        .then((data) => {
            const summary = data.summary || {};
            document.getElementById('summaryTotalProducts').textContent = summary.total_products ?? 0;
            document.getElementById('summaryTotalQuantity').textContent = summary.total_quantity ?? 0;
            document.getElementById('summaryLowStockCount').textContent = summary.low_stock_count ?? 0;
            document.getElementById('summaryOutStockCount').textContent = summary.out_stock_count ?? 0;
        });
}

function loadInventoryList() {
    const params = new URLSearchParams({
        page: inventoryPage,
        per_page: 10,
        search: document.getElementById('inventorySearch').value.trim(),
        category_id: document.getElementById('inventoryCategoryFilter').value,
        stock_state: document.getElementById('stockStateFilter').value
    });

    fetch(`/api/inventory/list?${params.toString()}`)
        .then((res) => res.json())
        .then((data) => {
            renderInventoryTable(data.items || []);
            renderInventoryPagination(data);
        });
}

function loadInventoryLogs() {
    const params = new URLSearchParams({
        page: logPage,
        per_page: 10,
        search: document.getElementById('logSearch').value.trim(),
        change_type: document.getElementById('changeTypeFilter').value,
        start_date: document.getElementById('startDateFilter').value,
        end_date: document.getElementById('endDateFilter').value
    });

    fetch(`/api/inventory/logs?${params.toString()}`)
        .then((res) => res.json())
        .then((data) => {
            renderLogTable(data.items || []);
            renderLogPagination(data);
        });
}

function loadInventoryAlerts() {
    fetch('/api/inventory/alerts?limit=12')
        .then((res) => res.json())
        .then((data) => {
            const alerts = data.alerts || [];
            document.getElementById('alertLowCount').textContent = alerts.filter((item) => item.alert_level === 'warning').length;
            document.getElementById('alertOutCount').textContent = alerts.filter((item) => item.alert_level === 'danger').length;

            const list = document.getElementById('alertList');
            if (!alerts.length) {
                list.innerHTML = `
                    <div class="col-span-full text-center text-slate-400 py-12">
                        <div class="text-lg font-medium text-slate-500">当前没有库存预警</div>
                        <div class="text-sm text-slate-400 mt-1">所有商品库存都高于预警线。</div>
                    </div>
                `;
                return;
            }

            list.innerHTML = alerts.map((item) => {
                const badgeClass = item.alert_level === 'danger' ? 'bg-red-50 text-red-700 border-red-200' : 'bg-amber-50 text-amber-700 border-amber-200';
                const titleClass = item.alert_level === 'danger' ? 'text-red-700' : 'text-amber-700';
                return `
                    <div class="rounded-2xl border ${badgeClass} bg-white p-5 shadow-sm">
                        <div class="flex items-start justify-between gap-4">
                            <div>
                                <div class="inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium ${badgeClass}">${item.alert_level === 'danger' ? '缺货预警' : '低库存预警'}</div>
                                <h3 class="mt-3 text-lg font-semibold text-slate-800">${item.product_name}</h3>
                                <p class="text-sm text-slate-500 mt-1">${item.product_code} · ${item.category_name}</p>
                            </div>
                            <div class="text-right">
                                <div class="text-3xl font-bold ${titleClass}">${item.quantity}</div>
                                <div class="text-xs text-slate-500">预警线 ${item.min_stock}</div>
                            </div>
                        </div>
                        <div class="mt-4 grid grid-cols-2 gap-3 text-sm text-slate-600">
                            <div class="rounded-xl bg-slate-50 px-3 py-2">缺口：<span class="font-semibold text-slate-800">${item.stock_gap}</span></div>
                            <div class="rounded-xl bg-slate-50 px-3 py-2">更新时间：<span class="font-semibold text-slate-800">${item.updated_at}</span></div>
                        </div>
                    </div>
                `;
            }).join('');
        });
}

function renderInventoryTable(items) {
    const tbody = document.getElementById('inventoryTableBody');
    if (!items.length) {
        tbody.innerHTML = '<tr><td colspan="7" class="px-6 py-12 text-center text-slate-400">暂无数据</td></tr>';
        return;
    }

    tbody.innerHTML = items.map((item) => {
        const stateText = item.stock_status === 'out' ? '缺货' : item.stock_status === 'low' ? '低库存' : '正常';
        const stateClass = item.stock_status === 'out' ? 'bg-red-50 text-red-700' : item.stock_status === 'low' ? 'bg-amber-50 text-amber-700' : 'bg-green-50 text-green-700';
        const quantityClass = item.stock_status === 'out' ? 'text-red-600' : item.stock_status === 'low' ? 'text-amber-600' : 'text-green-600';

        return `
            <tr class="hover:bg-slate-50 transition-colors">
                <td class="px-6 py-4 text-sm text-slate-700">${item.product_code}</td>
                <td class="px-6 py-4 text-sm font-medium text-slate-800">${item.product_name}</td>
                <td class="px-6 py-4 text-sm text-slate-600">${item.category_name}</td>
                <td class="px-6 py-4 text-sm font-semibold ${quantityClass}">${item.quantity}</td>
                <td class="px-6 py-4 text-sm text-slate-700">${item.min_stock}</td>
                <td class="px-6 py-4"><span class="inline-flex px-3 py-1 rounded-full text-xs font-medium ${stateClass}">${stateText}</span></td>
                <td class="px-6 py-4 text-sm text-slate-500">${item.updated_at}</td>
            </tr>
        `;
    }).join('');
}

function renderLogTable(items) {
    const tbody = document.getElementById('logTableBody');
    if (!items.length) {
        tbody.innerHTML = '<tr><td colspan="8" class="px-6 py-12 text-center text-slate-400">暂无数据</td></tr>';
        return;
    }

    tbody.innerHTML = items.map((item) => {
        const changeClass = item.quantity_change > 0 ? 'text-green-600' : item.quantity_change < 0 ? 'text-red-600' : 'text-slate-600';
        return `
            <tr class="hover:bg-slate-50 transition-colors">
                <td class="px-6 py-4 text-sm text-slate-500 whitespace-nowrap">${item.created_at}</td>
                <td class="px-6 py-4 text-sm text-slate-800">
                    <div class="font-medium">${item.product_name}</div>
                    <div class="text-xs text-slate-500">${item.product_code}</div>
                </td>
                <td class="px-6 py-4 text-sm text-slate-600">${item.change_type_label}</td>
                <td class="px-6 py-4 text-sm font-semibold ${changeClass}">${item.quantity_change > 0 ? '+' : ''}${item.quantity_change}</td>
                <td class="px-6 py-4 text-sm text-slate-600">${item.quantity_before}</td>
                <td class="px-6 py-4 text-sm text-slate-600">${item.quantity_after}</td>
                <td class="px-6 py-4 text-sm text-slate-600">${item.reason}</td>
                <td class="px-6 py-4 text-sm text-slate-600">${item.operator_name}</td>
            </tr>
        `;
    }).join('');
}

function renderInventoryPagination(data) {
    document.getElementById('inventoryPaginationInfo').textContent = `共 ${data.total || 0} 条记录`;
    document.getElementById('inventoryPaginationControls').innerHTML = renderPaginationButtons(data, 'goToInventoryPage');
}

function renderLogPagination(data) {
    document.getElementById('logPaginationInfo').textContent = `共 ${data.total || 0} 条记录`;
    document.getElementById('logPaginationControls').innerHTML = renderPaginationButtons(data, 'goToLogPage');
}

function renderPaginationButtons(data, callbackName) {
    const currentPage = data.current_page || 1;
    const totalPages = data.pages || 0;
    if (!totalPages) {
        return '';
    }

    let html = '';
    if (currentPage > 1) {
        html += `<button onclick="${callbackName}(${currentPage - 1})" class="px-3 py-1.5 border border-slate-200 rounded-lg hover:bg-slate-50 text-sm">上一页</button>`;
    }

    const start = Math.max(1, currentPage - 2);
    const end = Math.min(totalPages, currentPage + 2);

    for (let i = start; i <= end; i += 1) {
        if (i === currentPage) {
            html += `<button class="px-3 py-1.5 rounded-lg bg-primary text-white text-sm">${i}</button>`;
        } else {
            html += `<button onclick="${callbackName}(${i})" class="px-3 py-1.5 border border-slate-200 rounded-lg hover:bg-slate-50 text-sm">${i}</button>`;
        }
    }

    if (currentPage < totalPages) {
        html += `<button onclick="${callbackName}(${currentPage + 1})" class="px-3 py-1.5 border border-slate-200 rounded-lg hover:bg-slate-50 text-sm">下一页</button>`;
    }

    return html;
}

function goToInventoryPage(page) {
    inventoryPage = page;
    loadInventoryList();
}

function goToLogPage(page) {
    logPage = page;
    loadInventoryLogs();
}
