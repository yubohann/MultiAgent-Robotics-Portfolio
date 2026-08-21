let currentPage = 1;
let currentSearch = '';
let currentCategory = '';
let currentStatus = '';

document.addEventListener('DOMContentLoaded', function () {
    loadCategories();
    loadProducts();

    document.getElementById('searchInput').addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            currentSearch = this.value;
            currentPage = 1;
            loadProducts();
        }
    });

    document.getElementById('categoryFilter').addEventListener('change', function () {
        currentCategory = this.value;
        currentPage = 1;
        loadProducts();
    });

    document.getElementById('statusFilter').addEventListener('change', function () {
        currentStatus = this.value;
        currentPage = 1;
        loadProducts();
    });
});

function loadCategories() {
    fetch('/api/categories')
        .then((res) => res.json())
        .then((data) => {
            const select = document.getElementById('categoryFilter');
            const modalSelect = document.getElementById('categoryId');

            data.categories.forEach((cat) => {
                select.innerHTML += `<option value="${cat.id}">${cat.name}</option>`;
                modalSelect.innerHTML += `<option value="${cat.id}">${cat.name}</option>`;
            });
        });
}

function loadProducts() {
    const params = new URLSearchParams({
        page: currentPage,
        search: currentSearch,
        category_id: currentCategory,
        status: currentStatus
    });

    fetch(`/api/products?${params}`)
        .then((res) => res.json())
        .then((data) => {
            renderTable(data.items);
            renderPagination(data);
        });
}

function renderTable(products) {
    const tbody = document.getElementById('productTableBody');

    if (products.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="px-6 py-12 text-center text-slate-500">暂无数据</td></tr>';
        return;
    }

    tbody.innerHTML = products.map((product) => {
        const stockClass = product.stock_status === 'out' ? 'text-red-600'
            : product.stock_status === 'low' ? 'text-yellow-600' : 'text-green-600';
        const statusText = product.status === 1 ? '在售' : '已下架';
        const statusClass = product.status === 1 ? 'bg-green-100 text-green-700' : 'bg-slate-100 text-slate-600';

        const actionButton = product.status === 1
            ? `<button onclick="offlineProduct(${product.product_id})" class="text-yellow-600 hover:text-yellow-700 text-sm">下架</button>`
            : `<button onclick="onlineProduct(${product.product_id})" class="text-green-600 hover:text-green-700 text-sm">上架</button>`;

        return `
            <tr class="hover:bg-slate-50 transition-colors">
                <td class="px-6 py-4 text-sm text-slate-700">${product.product_code}</td>
                <td class="px-6 py-4 text-sm font-medium text-slate-800">${product.product_name}</td>
                <td class="px-6 py-4 text-sm text-slate-600">${product.category_name}</td>
                <td class="px-6 py-4 text-sm text-slate-700">¥${product.selling_price.toFixed(2)}</td>
                <td class="px-6 py-4 text-sm ${stockClass} font-medium">${product.quantity}</td>
                <td class="px-6 py-4">
                    <span class="px-3 py-1 rounded-full text-xs font-medium ${statusClass}">${statusText}</span>
                </td>
                <td class="px-6 py-4">
                    <div class="flex gap-2">
                        <button onclick="editProduct(${product.product_id})" class="text-primary hover:text-primary/80 text-sm">编辑</button>
                        ${actionButton}
                        <button onclick="deleteProduct(${product.product_id})" class="text-red-600 hover:text-red-700 text-sm">删除</button>
                    </div>
                </td>
            </tr>
        `;
    }).join('');
}

function renderPagination(data) {
    document.getElementById('totalCount').textContent = data.total;

    const pagination = document.getElementById('pagination');
    let html = '';

    if (data.current_page > 1) {
        html += `<button onclick="goToPage(${data.current_page - 1})" class="px-3 py-1 border border-slate-300 rounded hover:bg-slate-50">上一页</button>`;
    }

    for (let i = 1; i <= data.pages; i += 1) {
        if (i === data.current_page) {
            html += `<button class="px-3 py-1 bg-primary text-white rounded">${i}</button>`;
        } else {
            html += `<button onclick="goToPage(${i})" class="px-3 py-1 border border-slate-300 rounded hover:bg-slate-50">${i}</button>`;
        }
    }

    if (data.current_page < data.pages) {
        html += `<button onclick="goToPage(${data.current_page + 1})" class="px-3 py-1 border border-slate-300 rounded hover:bg-slate-50">下一页</button>`;
    }

    pagination.innerHTML = html;
}

function goToPage(page) {
    currentPage = page;
    loadProducts();
}

function showAddModal() {
    document.getElementById('modalTitle').textContent = '添加商品';
    document.getElementById('productForm').reset();
    document.getElementById('productId').value = '';
    document.getElementById('productModal').classList.remove('hidden');
}

function editProduct(productId) {
    fetch('/api/products?page=1&per_page=100')
        .then((res) => res.json())
        .then((data) => {
            const product = data.items.find((p) => p.product_id === productId);
            if (!product) {
                return;
            }

            document.getElementById('modalTitle').textContent = '编辑商品';
            document.getElementById('productId').value = product.product_id;
            document.getElementById('productCode').value = product.product_code;
            document.getElementById('barcode').value = product.barcode;
            document.getElementById('productName').value = product.product_name;
            document.getElementById('categoryId').value = '';
            document.getElementById('unit').value = product.unit;
            document.getElementById('purchasePrice').value = product.purchase_price;
            document.getElementById('sellingPrice').value = product.selling_price;
            document.getElementById('quantity').value = product.quantity;
            document.getElementById('minStock').value = product.min_stock;

            document.getElementById('productModal').classList.remove('hidden');
        });
}

function saveProduct() {
    const productId = document.getElementById('productId').value;
    const data = {
        product_code: document.getElementById('productCode').value,
        barcode: document.getElementById('barcode').value,
        product_name: document.getElementById('productName').value,
        category_id: document.getElementById('categoryId').value || null,
        unit: document.getElementById('unit').value,
        purchase_price: parseFloat(document.getElementById('purchasePrice').value) || 0,
        selling_price: parseFloat(document.getElementById('sellingPrice').value),
        quantity: parseInt(document.getElementById('quantity').value, 10) || 0,
        min_stock: parseInt(document.getElementById('minStock').value, 10) || 10
    };

    const url = productId ? `/api/products/${productId}` : '/api/products';
    const method = productId ? 'PUT' : 'POST';

    fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    })
        .then((res) => res.json())
        .then((result) => {
            if (result.success) {
                alert(result.message);
                closeModal();
                loadProducts();
            } else {
                alert(result.message);
            }
        });
}

function offlineProduct(productId) {
    if (!confirm('确定要下架该商品吗？下架后可以重新上架。')) {
        return;
    }

    fetch(`/api/products/${productId}/offline`, { method: 'POST' })
        .then((res) => res.json())
        .then((result) => {
            alert(result.message);
            loadProducts();
        });
}

function onlineProduct(productId) {
    if (!confirm('确定要上架该商品吗？')) {
        return;
    }

    fetch(`/api/products/${productId}/online`, { method: 'POST' })
        .then((res) => res.json())
        .then((result) => {
            alert(result.message);
            loadProducts();
        });
}

function deleteProduct(productId) {
    if (!confirm('⚠️ 警告：此操作将永久删除该商品及其库存记录，不可恢复！确定要继续吗？')) {
        return;
    }

    fetch(`/api/products/${productId}`, { method: 'DELETE' })
        .then((res) => res.json())
        .then((result) => {
            alert(result.message);
            loadProducts();
        });
}

function closeModal() {
    document.getElementById('productModal').classList.add('hidden');
}

function showImportModal() {
    document.getElementById('importModal').classList.remove('hidden');
    document.getElementById('importFile').value = '';
    document.getElementById('fileName').classList.add('hidden');
    document.getElementById('importBtn').disabled = true;
}

function closeImportModal() {
    document.getElementById('importModal').classList.add('hidden');
}

document.getElementById('importFile').addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (file) {
        document.getElementById('fileName').textContent = `已选择: ${file.name}`;
        document.getElementById('fileName').classList.remove('hidden');
        document.getElementById('importBtn').disabled = false;
    }
});

const dropZone = document.getElementById('dropZone');

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('border-primary', 'bg-blue-50');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('border-primary', 'bg-blue-50');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('border-primary', 'bg-blue-50');

    const file = e.dataTransfer.files[0];
    if (file) {
        document.getElementById('importFile').files = e.dataTransfer.files;
        document.getElementById('fileName').textContent = `已选择: ${file.name}`;
        document.getElementById('fileName').classList.remove('hidden');
        document.getElementById('importBtn').disabled = false;
    }
});

function importProducts() {
    const fileInput = document.getElementById('importFile');
    const file = fileInput.files[0];

    if (!file) {
        alert('请选择文件');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    document.getElementById('importBtn').disabled = true;
    document.getElementById('importBtn').textContent = '导入中...';

    fetch('/api/products/import', {
        method: 'POST',
        body: formData
    })
        .then((res) => res.json())
        .then((result) => {
            document.getElementById('importBtn').disabled = false;
            document.getElementById('importBtn').textContent = '开始导入';

            if (result.success) {
                let message = result.message;
                if (result.errors && result.errors.length > 0) {
                    message += `\n\n错误详情：\n${result.errors.join('\n')}`;
                }
                alert(message);
                closeImportModal();
                loadProducts();
            } else {
                alert(result.message);
            }
        })
        .catch((err) => {
            alert(`导入失败：${err.message}`);
            document.getElementById('importBtn').disabled = false;
            document.getElementById('importBtn').textContent = '开始导入';
        });
}

function downloadTemplate() {
    const csvContent = 'product_code,barcode,product_name,category_id,unit,purchase_price,selling_price,quantity,min_stock\nP001,6901234567890,可口可乐330ml,1,瓶,2.50,3.50,100,50\nP002,,雪碧330ml,1,瓶,2.00,3.00,80,50\nP003,,矿泉水550ml,1,瓶,1.00,2.00,200,100';

    const blob = new Blob(['\ufeff' + csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = '商品导入模板.csv';
    link.click();
}
