let currentPage = 1;
let currentSearch = '';
let currentStatus = '';
let currentRecords = [];

const config = window.MASTER_DATA_CONFIG;

document.addEventListener('DOMContentLoaded', () => {
    renderHead();
    renderForm();
    bindEvents();
    loadRecords();
});

function bindEvents() {
    document.getElementById('addButton').addEventListener('click', () => showModal());
    document.getElementById('closeButton').addEventListener('click', closeModal);
    document.getElementById('saveButton').addEventListener('click', saveRecord);
    document.getElementById('searchInput').addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            currentSearch = event.target.value;
            currentPage = 1;
            loadRecords();
        }
    });

    const statusFilter = document.getElementById('statusFilter');
    if (statusFilter) {
        statusFilter.addEventListener('change', (event) => {
            currentStatus = event.target.value;
            currentPage = 1;
            loadRecords();
        });
    }

    const closePointsButton = document.getElementById('closePointsButton');
    if (closePointsButton) {
        closePointsButton.addEventListener('click', closePointsModal);
        document.getElementById('savePointsButton').addEventListener('click', savePoints);
    }
}

function renderHead() {
    const head = document.getElementById('tableHead');
    const columns = config.columns.map((column) => (
        `<th class="px-6 py-4 text-left text-sm font-semibold text-slate-700">${column.label}</th>`
    ));
    columns.push('<th class="px-6 py-4 text-left text-sm font-semibold text-slate-700">操作</th>');
    head.innerHTML = columns.join('');
}

function renderForm() {
    const container = document.getElementById('formFields');
    container.innerHTML = config.fields.map((field) => {
        const required = field.required ? 'required' : '';
        const star = field.required ? '<span class="text-red-500">*</span>' : '';
        if (field.type === 'select') {
            const options = (field.options || []).map((option) => (
                `<option value="${option.value}">${option.label}</option>`
            )).join('');
            return `
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-2">${field.label} ${star}</label>
                    <select id="field_${field.name}" ${required} class="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-primary outline-none">${options}</select>
                </div>
            `;
        }
        return `
            <div>
                <label class="block text-sm font-medium text-slate-700 mb-2">${field.label} ${star}</label>
                <input id="field_${field.name}" type="${field.type || 'text'}" ${required} class="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-primary outline-none">
            </div>
        `;
    }).join('');
}

function loadRecords() {
    const params = new URLSearchParams({
        page: currentPage,
        search: currentSearch,
        status: currentStatus
    });

    fetch(`${config.api_base}?${params}`)
        .then((response) => response.json())
        .then((data) => {
            renderRows(data.items || []);
            renderPagination(data);
        });
}

function renderRows(items) {
    currentRecords = items;
    const body = document.getElementById('tableBody');
    if (!items.length) {
        body.innerHTML = `<tr><td colspan="${config.columns.length + 1}" class="px-6 py-12 text-center text-slate-500">暂无数据</td></tr>`;
        return;
    }

    body.innerHTML = items.map((item) => {
        const cells = config.columns.map((column) => {
            let value = item[column.name] ?? '';
            if (column.name === 'status') {
                const active = value === 'active';
                const klass = active ? 'bg-green-100 text-green-700' : 'bg-slate-100 text-slate-600';
                value = `<span class="px-3 py-1 rounded-full text-xs font-medium ${klass}">${active ? '启用' : '停用'}</span>`;
            }
            return `<td class="px-6 py-4 text-sm text-slate-700">${value}</td>`;
        }).join('');

        const id = item[config.id_field];
        const statusAction = config.disable_status ? '' : `
            <button onclick="toggleStatus(${id}, '${item.status === 'active' ? 'inactive' : 'active'}')" class="text-yellow-600 hover:text-yellow-700 text-sm">
                ${item.status === 'active' ? '停用' : '启用'}
            </button>
        `;
        const pointsAction = config.extra_action === 'points'
            ? `<button onclick="showPointsModal(${id})" class="text-green-600 hover:text-green-700 text-sm">积分</button>`
            : '';

        return `
            <tr class="hover:bg-slate-50 transition-colors" data-record-id="${id}">
                ${cells}
                <td class="px-6 py-4">
                    <div class="flex gap-2">
                        <button onclick="editRecord(${id})" class="text-primary hover:text-primary/80 text-sm">编辑</button>
                        ${statusAction}
                        ${pointsAction}
                    </div>
                </td>
            </tr>
        `;
    }).join('');
}

function renderPagination(data) {
    document.getElementById('totalCount').textContent = data.total || 0;
    const pagination = document.getElementById('pagination');
    let html = '';

    if (data.current_page > 1) {
        html += `<button onclick="goToPage(${data.current_page - 1})" class="px-3 py-1 border border-slate-300 rounded hover:bg-slate-50">上一页</button>`;
    }
    for (let i = 1; i <= (data.pages || 1); i += 1) {
        html += i === data.current_page
            ? `<button class="px-3 py-1 bg-primary text-white rounded">${i}</button>`
            : `<button onclick="goToPage(${i})" class="px-3 py-1 border border-slate-300 rounded hover:bg-slate-50">${i}</button>`;
    }
    if (data.current_page < data.pages) {
        html += `<button onclick="goToPage(${data.current_page + 1})" class="px-3 py-1 border border-slate-300 rounded hover:bg-slate-50">下一页</button>`;
    }
    pagination.innerHTML = html;
}

function goToPage(page) {
    currentPage = page;
    loadRecords();
}

function showModal(record = null) {
    document.getElementById('modalTitle').textContent = record ? '编辑' : '新增';
    document.getElementById('recordId').value = record ? record[config.id_field] : '';
    config.fields.forEach((field) => {
        document.getElementById(`field_${field.name}`).value = record ? (record[field.name] ?? '') : '';
    });
    document.getElementById('dataModal').classList.remove('hidden');
}

function closeModal() {
    document.getElementById('dataModal').classList.add('hidden');
}

function editRecord(id) {
    const record = currentRecords.find((item) => String(item[config.id_field]) === String(id));
    if (record) {
        showModal(record);
    }
}

function collectData() {
    const data = {};
    config.fields.forEach((field) => {
        const input = document.getElementById(`field_${field.name}`);
        data[field.name] = field.type === 'number' ? parseInt(input.value || '0', 10) : input.value;
    });
    return data;
}

function saveRecord() {
    const id = document.getElementById('recordId').value;
    const method = id ? 'PUT' : 'POST';
    let url = config.api_base;
    if (id) {
        const keyField = config.active_page === 'system_settings' ? 'setting_key' : config.id_field;
        const keyValue = keyField === 'setting_key' ? document.getElementById('field_setting_key').value : id;
        url = `${config.api_base}/${encodeURIComponent(keyValue)}`;
    }

    fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(collectData())
    })
        .then((response) => response.json())
        .then((result) => {
            alert(result.message);
            if (result.success) {
                closeModal();
                loadRecords();
            }
        });
}

function toggleStatus(id, status) {
    fetch(`${config.api_base}/${id}/status`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status })
    })
        .then((response) => response.json())
        .then((result) => {
            alert(result.message);
            loadRecords();
        });
}

function showPointsModal(id) {
    document.getElementById('pointsRecordId').value = id;
    document.getElementById('pointsDelta').value = '';
    document.getElementById('pointsReason').value = '';
    document.getElementById('pointsModal').classList.remove('hidden');
}

function closePointsModal() {
    document.getElementById('pointsModal').classList.add('hidden');
}

function savePoints() {
    const id = document.getElementById('pointsRecordId').value;
    fetch(`${config.api_base}/${id}/points`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            points_delta: parseInt(document.getElementById('pointsDelta').value || '0', 10),
            reason: document.getElementById('pointsReason').value
        })
    })
        .then((response) => response.json())
        .then((result) => {
            alert(result.message);
            if (result.success) {
                closePointsModal();
                loadRecords();
            }
        });
}
