const txState = { page: 1, pages: 1 };

function formatMoney(value) {
    return `¥${Number(value || 0).toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function notify(message) {
    window.alert(message);
}

function toDateValue(dateObj) {
    const y = dateObj.getFullYear();
    const m = String(dateObj.getMonth() + 1).padStart(2, '0');
    const d = String(dateObj.getDate()).padStart(2, '0');
    return `${y}-${m}-${d}`;
}

async function loadOverview() {
    const period = document.getElementById('overviewPeriod').value;
    const response = await fetch(`/api/finance/overview?period=${period}`);
    const result = await response.json();
    if (!result.success) {
        return;
    }

    const data = result.overview;
    document.getElementById('cardCashInflow').textContent = formatMoney(data.cash_inflow);
    document.getElementById('cardCashOutflow').textContent = formatMoney(data.cash_outflow);
    document.getElementById('cardNetProfit').textContent = formatMoney(data.net_profit);
    document.getElementById('cardPendingPayables').textContent = formatMoney(data.pending_payables);
    document.getElementById('cardUnreconciled').textContent = data.unreconciled_count;
}

function getTxQuery(page = 1) {
    const params = new URLSearchParams({ page: String(page), per_page: '8' });
    const mapping = {
        search: document.getElementById('txSearch').value.trim(),
        transaction_type: document.getElementById('txFilterType').value,
        start_date: document.getElementById('txStartDate').value,
        end_date: document.getElementById('txEndDate').value
    };
    Object.entries(mapping).forEach(([key, value]) => {
        if (value) {
            params.set(key, value);
        }
    });
    return params;
}

async function loadTransactions(page = 1) {
    const tbody = document.getElementById('txTableBody');
    tbody.innerHTML = '<tr><td colspan="8" class="px-4 py-8 text-center text-slate-400">加载中...</td></tr>';

    const response = await fetch(`/api/finance/transactions?${getTxQuery(page).toString()}`);
    const result = await response.json();
    if (!result.success) {
        tbody.innerHTML = '<tr><td colspan="8" class="px-4 py-8 text-center text-red-500">加载失败</td></tr>';
        return;
    }

    txState.page = result.current_page || 1;
    txState.pages = result.pages || 1;

    if (!result.items.length) {
        tbody.innerHTML = '<tr><td colspan="8" class="px-4 py-8 text-center text-slate-400">暂无流水记录</td></tr>';
    } else {
        tbody.innerHTML = result.items.map((item) => {
            const typeClass = item.transaction_type === 'income' ? 'text-emerald-600' : 'text-rose-600';
            const symbol = item.transaction_type === 'income' ? '+' : '-';
            return `
                <tr>
                    <td class="px-4 py-3 text-sm text-slate-700">${item.transaction_no}</td>
                    <td class="px-4 py-3 text-sm ${typeClass}">${item.transaction_type_label}</td>
                    <td class="px-4 py-3 text-sm text-slate-700">${item.category}</td>
                    <td class="px-4 py-3 text-sm font-medium ${typeClass}">${symbol}${formatMoney(item.amount)}</td>
                    <td class="px-4 py-3 text-sm text-slate-600">${item.payment_method_label}</td>
                    <td class="px-4 py-3 text-sm text-slate-600">${item.operator_name}</td>
                    <td class="px-4 py-3 text-sm text-slate-600 whitespace-nowrap">${item.occurred_at}</td>
                    <td class="px-4 py-3 text-sm text-slate-600">${item.description}</td>
                </tr>
            `;
        }).join('');
    }

    document.getElementById('txPageInfo').textContent = `共 ${result.total || 0} 条，第 ${txState.page}/${txState.pages} 页`;
    document.getElementById('txPrevBtn').disabled = txState.page <= 1;
    document.getElementById('txNextBtn').disabled = txState.page >= txState.pages;
}

async function createTransaction() {
    const occurredAt = document.getElementById('txOccurredAt').value;
    const payload = {
        transaction_type: document.getElementById('txType').value,
        category: document.getElementById('txCategory').value,
        amount: document.getElementById('txAmount').value,
        payment_method: document.getElementById('txPaymentMethod').value,
        occurred_at: occurredAt ? `${occurredAt.replace('T', ' ')}:00` : '',
        description: '',
        related_order_no: ''
    };

    const response = await fetch('/api/finance/transactions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    const result = await response.json();

    if (!result.success) {
        notify(result.message || '新增失败');
        return;
    }

    document.getElementById('txCategory').value = '';
    document.getElementById('txAmount').value = '';
    notify(result.message || '新增成功');
    await loadTransactions(1);
    await loadOverview();
}

async function loadReconciliation() {
    const date = document.getElementById('reconDate').value;
    const response = await fetch(`/api/finance/reconciliation?date=${encodeURIComponent(date)}`);
    const result = await response.json();

    const summary = result.summary || {};
    document.getElementById('reconSummary').innerHTML = `
        <div class="bg-slate-50 rounded-xl p-3 border border-slate-100"><p class="text-xs text-slate-500">应收合计</p><p class="text-lg font-semibold text-slate-800">${formatMoney(summary.expected_total)}</p></div>
        <div class="bg-slate-50 rounded-xl p-3 border border-slate-100"><p class="text-xs text-slate-500">实收合计</p><p class="text-lg font-semibold text-slate-800">${formatMoney(summary.actual_total)}</p></div>
        <div class="bg-slate-50 rounded-xl p-3 border border-slate-100"><p class="text-xs text-slate-500">总差额</p><p class="text-lg font-semibold text-slate-800">${formatMoney(summary.difference_total)}</p></div>
        <div class="bg-slate-50 rounded-xl p-3 border border-slate-100"><p class="text-xs text-slate-500">异常笔数</p><p class="text-lg font-semibold text-slate-800">${summary.mismatch_count || 0}</p></div>
    `;

    const tbody = document.getElementById('reconTableBody');
    if (!result.items || !result.items.length) {
        tbody.innerHTML = '<tr><td colspan="7" class="px-4 py-8 text-center text-slate-400">当天暂无可对账数据</td></tr>';
        return;
    }

    tbody.innerHTML = result.items.map((item) => `
        <tr>
            <td class="px-4 py-3 text-sm text-slate-700">${item.payment_method_label}</td>
            <td class="px-4 py-3 text-sm text-slate-600">${item.order_count}</td>
            <td class="px-4 py-3 text-sm text-slate-700">${formatMoney(item.expected_amount)}</td>
            <td class="px-4 py-3 text-sm text-slate-700"><input type="number" min="0" step="0.01" value="${item.actual_amount}" data-recon-actual="${item.payment_method}" class="w-32 px-2 py-1.5 border border-slate-200 rounded-lg text-sm"></td>
            <td class="px-4 py-3 text-sm ${item.difference_amount === 0 ? 'text-emerald-600' : 'text-rose-600'}">${formatMoney(item.difference_amount)}</td>
            <td class="px-4 py-3 text-sm text-slate-700"><input type="text" value="${item.note || ''}" data-recon-note="${item.payment_method}" class="w-full px-2 py-1.5 border border-slate-200 rounded-lg text-sm"></td>
            <td class="px-4 py-3 text-sm"><button data-recon-save="${item.payment_method}" class="px-3 py-1.5 bg-primary text-white rounded-lg hover:bg-blue-700 transition-colors">保存</button></td>
        </tr>
    `).join('');

    document.querySelectorAll('button[data-recon-save]').forEach((btn) => {
        btn.addEventListener('click', async () => {
            await saveReconciliation(btn.getAttribute('data-recon-save'));
        });
    });
}

async function saveReconciliation(paymentMethod) {
    const date = document.getElementById('reconDate').value;
    const actualInput = document.querySelector(`input[data-recon-actual="${paymentMethod}"]`);
    const noteInput = document.querySelector(`input[data-recon-note="${paymentMethod}"]`);
    const payload = {
        date,
        payment_method: paymentMethod,
        actual_amount: actualInput ? actualInput.value : '0',
        note: noteInput ? noteInput.value : ''
    };

    const response = await fetch('/api/finance/reconciliation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    const result = await response.json();
    notify(result.message || '操作完成');

    if (result.success) {
        await loadReconciliation();
        await loadOverview();
    }
}

async function createPayable() {
    const payload = {
        supplier_name: document.getElementById('payableSupplier').value,
        bill_no: document.getElementById('payableBillNo').value,
        total_amount: document.getElementById('payableAmount').value,
        due_date: document.getElementById('payableDueDate').value,
        note: document.getElementById('payableNote').value
    };

    const response = await fetch('/api/finance/payables', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    const result = await response.json();

    notify(result.message || '操作完成');
    if (!result.success) {
        return;
    }

    document.getElementById('payableSupplier').value = '';
    document.getElementById('payableBillNo').value = '';
    document.getElementById('payableAmount').value = '';
    document.getElementById('payableNote').value = '';

    await loadPayables();
    await loadOverview();
}

async function loadPayables() {
    const params = new URLSearchParams({ page: '1', per_page: '20' });
    const search = document.getElementById('payableSearch').value.trim();
    const status = document.getElementById('payableStatus').value;
    if (search) {
        params.set('search', search);
    }
    if (status) {
        params.set('status', status);
    }

    const response = await fetch(`/api/finance/payables?${params.toString()}`);
    const result = await response.json();
    const tbody = document.getElementById('payableTableBody');

    if (!result.success || !result.items || !result.items.length) {
        tbody.innerHTML = '<tr><td colspan="8" class="px-4 py-8 text-center text-slate-400">暂无应付款记录</td></tr>';
        return;
    }

    const statusClass = {
        unpaid: 'bg-slate-100 text-slate-700',
        partial: 'bg-amber-50 text-amber-700',
        paid: 'bg-emerald-50 text-emerald-700',
        overdue: 'bg-rose-50 text-rose-700'
    };

    tbody.innerHTML = result.items.map((item) => `
        <tr>
            <td class="px-4 py-3 text-sm text-slate-700">${item.supplier_name}</td>
            <td class="px-4 py-3 text-sm text-slate-600">${item.bill_no}</td>
            <td class="px-4 py-3 text-sm text-slate-700">${formatMoney(item.total_amount)}</td>
            <td class="px-4 py-3 text-sm text-slate-700">${formatMoney(item.paid_amount)}</td>
            <td class="px-4 py-3 text-sm font-medium ${item.remaining_amount > 0 ? 'text-rose-600' : 'text-emerald-600'}">${formatMoney(item.remaining_amount)}</td>
            <td class="px-4 py-3 text-sm text-slate-600">${item.due_date}</td>
            <td class="px-4 py-3 text-sm"><span class="inline-flex px-2 py-1 rounded-full text-xs ${statusClass[item.status] || 'bg-slate-100 text-slate-700'}">${item.status_label}</span></td>
            <td class="px-4 py-3 text-sm">
                <button data-payable-pay="${item.payable_id}" data-remaining="${item.remaining_amount}" class="px-3 py-1.5 rounded-lg bg-primary text-white hover:bg-blue-700 transition-colors ${item.remaining_amount <= 0 ? 'opacity-50 pointer-events-none' : ''}">登记付款</button>
            </td>
        </tr>
    `).join('');

    document.querySelectorAll('button[data-payable-pay]').forEach((btn) => {
        btn.addEventListener('click', async () => {
            const payableId = Number(btn.getAttribute('data-payable-pay'));
            const remaining = Number(btn.getAttribute('data-remaining') || 0);
            const amount = window.prompt(`请输入付款金额（剩余 ${formatMoney(remaining)}）`);
            if (!amount) {
                return;
            }
            await payPayable(payableId, amount);
        });
    });
}

async function payPayable(payableId, amount) {
    const payload = {
        amount,
        payment_method: 'bank_transfer'
    };

    const response = await fetch(`/api/finance/payables/${payableId}/payment`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    const result = await response.json();
    notify(result.message || '操作完成');

    if (result.success) {
        await loadPayables();
        await loadOverview();
    }
}

async function loadClosings() {
    const response = await fetch('/api/finance/closings?limit=12');
    const result = await response.json();
    const tbody = document.getElementById('closingTableBody');

    if (!result.success || !result.items || !result.items.length) {
        tbody.innerHTML = '<tr><td colspan="6" class="px-4 py-8 text-center text-slate-400">暂无关账记录</td></tr>';
        return;
    }

    tbody.innerHTML = result.items.map((item) => `
        <tr>
            <td class="px-4 py-3 text-sm text-slate-700">${item.period_month}</td>
            <td class="px-4 py-3 text-sm text-slate-600">${formatMoney(item.total_sales)}</td>
            <td class="px-4 py-3 text-sm text-slate-600">${formatMoney(item.other_income)}</td>
            <td class="px-4 py-3 text-sm text-slate-600">${formatMoney(item.expense_amount)}</td>
            <td class="px-4 py-3 text-sm font-medium ${item.net_profit >= 0 ? 'text-emerald-600' : 'text-rose-600'}">${formatMoney(item.net_profit)}</td>
            <td class="px-4 py-3 text-sm text-slate-600">${item.closed_at}</td>
        </tr>
    `).join('');
}

async function closeMonth() {
    const payload = {
        period_month: document.getElementById('closeMonth').value,
        note: document.getElementById('closeNote').value
    };

    const response = await fetch('/api/finance/close-month', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    const result = await response.json();
    notify(result.message || '操作完成');

    if (result.success) {
        await loadClosings();
    }
}

function bindEvents() {
    document.getElementById('refreshOverviewBtn').addEventListener('click', loadOverview);
    document.getElementById('overviewPeriod').addEventListener('change', loadOverview);

    document.getElementById('txCreateBtn').addEventListener('click', createTransaction);
    document.getElementById('txSearch').addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            loadTransactions(1);
        }
    });
    document.getElementById('txFilterType').addEventListener('change', () => loadTransactions(1));
    document.getElementById('txStartDate').addEventListener('change', () => loadTransactions(1));
    document.getElementById('txEndDate').addEventListener('change', () => loadTransactions(1));
    document.getElementById('txPrevBtn').addEventListener('click', () => {
        if (txState.page > 1) {
            loadTransactions(txState.page - 1);
        }
    });
    document.getElementById('txNextBtn').addEventListener('click', () => {
        if (txState.page < txState.pages) {
            loadTransactions(txState.page + 1);
        }
    });

    document.getElementById('loadReconBtn').addEventListener('click', loadReconciliation);

    document.getElementById('createPayableBtn').addEventListener('click', createPayable);
    document.getElementById('payableQueryBtn').addEventListener('click', loadPayables);

    document.getElementById('closeMonthBtn').addEventListener('click', closeMonth);
}

function applyDefaultValues() {
    const now = new Date();
    document.getElementById('reconDate').value = toDateValue(now);
    document.getElementById('txOccurredAt').value = `${toDateValue(now)}T${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;
    document.getElementById('txStartDate').value = toDateValue(new Date(now.getFullYear(), now.getMonth(), 1));
    document.getElementById('txEndDate').value = toDateValue(now);
    document.getElementById('payableDueDate').value = toDateValue(new Date(now.getFullYear(), now.getMonth(), now.getDate() + 7));
    document.getElementById('closeMonth').value = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`;
}

document.addEventListener('DOMContentLoaded', async () => {
    applyDefaultValues();
    bindEvents();

    await loadOverview();
    await loadTransactions(1);
    await loadReconciliation();
    await loadPayables();
    await loadClosings();
});
