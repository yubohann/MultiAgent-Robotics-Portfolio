const productTableBody = document.getElementById('productTableBody');
const keywordInput = document.getElementById('keywordInput');
const searchBtn = document.getElementById('searchBtn');
const cartContainer = document.getElementById('cartContainer');
const emptyCartText = document.getElementById('emptyCartText');
const totalQuantity = document.getElementById('totalQuantity');
const totalAmount = document.getElementById('totalAmount');
const actualAmount = document.getElementById('actualAmount');
const discountInput = document.getElementById('discountInput');
const paymentMethod = document.getElementById('paymentMethod');
const checkoutBtn = document.getElementById('checkoutBtn');
const printBtn = document.getElementById('printBtn');
const clearCartBtn = document.getElementById('clearCartBtn');
const messageBox = document.getElementById('messageBox');

const paymentLabels = {
    cash: '现金',
    wechat: '微信',
    alipay: '支付宝',
    card: '银行卡'
};

let cart = [];
let productIndex = {};
let isCheckingOut = false;
let lastOrder = null;

const formatMoney = (value) => `¥${Number(value || 0).toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

const setMessage = (text, type = 'info') => {
    messageBox.textContent = text || '';
    if (type === 'error') {
        messageBox.className = 'text-sm text-red-500 min-h-[20px]';
    } else if (type === 'success') {
        messageBox.className = 'text-sm text-green-600 min-h-[20px]';
    } else {
        messageBox.className = 'text-sm text-slate-500 min-h-[20px]';
    }
};

const renderProducts = (products) => {
    productIndex = {};
    products.forEach((item) => {
        productIndex[item.product_id] = item;
    });

    if (!products.length) {
        productTableBody.innerHTML = '<tr><td colspan="5" class="px-4 py-10 text-center text-slate-400">未查询到可售商品</td></tr>';
        return;
    }

    productTableBody.innerHTML = products.map((item) => `
        <tr class="hover:bg-slate-50 transition-colors">
            <td class="px-4 py-3 text-sm text-slate-800">${item.product_name}</td>
            <td class="px-4 py-3 text-sm text-slate-600">${item.product_code}${item.barcode ? `<div class="text-xs text-slate-400 mt-1">${item.barcode}</div>` : ''}</td>
            <td class="px-4 py-3 text-sm text-slate-600">${formatMoney(item.selling_price)}</td>
            <td class="px-4 py-3 text-sm text-slate-600">${item.quantity}</td>
            <td class="px-4 py-3 text-sm">
                <button class="add-cart-btn px-3 py-1.5 rounded-lg bg-primary text-white hover:bg-blue-700 transition-colors" data-product-id="${item.product_id}">加入</button>
            </td>
        </tr>
    `).join('');

    document.querySelectorAll('.add-cart-btn').forEach((btn) => {
        btn.addEventListener('click', () => {
            const productId = Number(btn.dataset.productId);
            const product = productIndex[productId];
            if (product) {
                addToCart(product);
            }
        });
    });
};

const loadProducts = async () => {
    const keyword = keywordInput.value.trim();
    productTableBody.innerHTML = '<tr><td colspan="5" class="px-4 py-10 text-center text-slate-400">查询中...</td></tr>';

    try {
        const params = new URLSearchParams({ keyword, limit: 30 });
        const response = await fetch(`/api/cashier/products?${params.toString()}`);
        const data = await response.json();
        if (!response.ok || !data.success) {
            throw new Error(data.message || '查询失败');
        }
        renderProducts(data.products || []);
    } catch (error) {
        productTableBody.innerHTML = '<tr><td colspan="5" class="px-4 py-10 text-center text-red-500">商品查询失败，请稍后重试</td></tr>';
        setMessage(error.message || '商品查询失败', 'error');
    }
};

const addToCart = (product) => {
    const existing = cart.find((item) => item.product_id === product.product_id);
    if (existing) {
        if (existing.quantity >= product.quantity) {
            setMessage(`库存不足：${product.product_name}`, 'error');
            return;
        }
        existing.quantity += 1;
    } else {
        cart.push({
            product_id: product.product_id,
            product_name: product.product_name,
            product_code: product.product_code,
            unit: product.unit,
            selling_price: Number(product.selling_price),
            stock: Number(product.quantity),
            quantity: 1
        });
    }
    setMessage(`已加入购物车：${product.product_name}`, 'success');
    renderCart();
};

const changeQuantity = (productId, delta) => {
    const item = cart.find((row) => row.product_id === productId);
    if (!item) {
        return;
    }
    const next = item.quantity + delta;
    if (next <= 0) {
        cart = cart.filter((row) => row.product_id !== productId);
        renderCart();
        return;
    }
    if (next > item.stock) {
        setMessage(`库存不足：${item.product_name}`, 'error');
        return;
    }
    item.quantity = next;
    renderCart();
};

const renderCart = () => {
    if (!cart.length) {
        cartContainer.innerHTML = '<p id="emptyCartText" class="text-sm text-slate-400 text-center py-8">购物车为空</p>';
    } else {
        cartContainer.innerHTML = cart.map((item) => `
            <div class="rounded-xl border border-slate-200 p-3">
                <div class="flex items-start justify-between gap-3">
                    <div>
                        <p class="text-sm font-medium text-slate-800">${item.product_name}</p>
                        <p class="text-xs text-slate-500 mt-1">${item.product_code} · ${formatMoney(item.selling_price)}/${item.unit}</p>
                    </div>
                    <button class="text-xs text-danger hover:text-danger/80" data-remove="${item.product_id}">移除</button>
                </div>
                <div class="mt-3 flex items-center justify-between">
                    <div class="inline-flex items-center rounded-lg border border-slate-200 overflow-hidden">
                        <button class="px-3 py-1.5 text-slate-600 hover:bg-slate-100" data-delta="-1" data-product="${item.product_id}">-</button>
                        <span class="px-3 py-1.5 text-sm text-slate-700 border-x border-slate-200">${item.quantity}</span>
                        <button class="px-3 py-1.5 text-slate-600 hover:bg-slate-100" data-delta="1" data-product="${item.product_id}">+</button>
                    </div>
                    <p class="text-sm font-semibold text-slate-800">${formatMoney(item.quantity * item.selling_price)}</p>
                </div>
            </div>
        `).join('');
    }

    cartContainer.querySelectorAll('button[data-delta]').forEach((button) => {
        button.addEventListener('click', () => {
            const productId = Number(button.dataset.product);
            const delta = Number(button.dataset.delta);
            changeQuantity(productId, delta);
        });
    });

    cartContainer.querySelectorAll('button[data-remove]').forEach((button) => {
        button.addEventListener('click', () => {
            const productId = Number(button.dataset.remove);
            cart = cart.filter((item) => item.product_id !== productId);
            renderCart();
        });
    });

    updateSummary();
};

const updateSummary = () => {
    const qty = cart.reduce((sum, item) => sum + item.quantity, 0);
    const total = cart.reduce((sum, item) => sum + item.quantity * item.selling_price, 0);
    const discount = Math.max(0, Number(discountInput.value || 0));
    const actual = Math.max(0, total - discount);

    totalQuantity.textContent = String(qty);
    totalAmount.textContent = formatMoney(total);
    actualAmount.textContent = formatMoney(actual);

    checkoutBtn.disabled = !cart.length || isCheckingOut;
};

const doCheckout = async () => {
    if (!cart.length || isCheckingOut) {
        return;
    }

    const discount = Math.max(0, Number(discountInput.value || 0));
    isCheckingOut = true;
    checkoutBtn.disabled = true;
    setMessage('正在结算，请稍候...');

    try {
        const payload = {
            payment_method: paymentMethod.value,
            discount_amount: discount,
            items: cart.map((item) => ({ product_id: item.product_id, quantity: item.quantity }))
        };
        const response = await fetch('/api/cashier/checkout', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.message || '结算失败');
        }

        lastOrder = data.order;
        cart = [];
        discountInput.value = '0';
        renderCart();
        printBtn.disabled = false;
        setMessage(`结算成功，订单号：${data.order.order_no}`, 'success');
        await loadProducts();
    } catch (error) {
        setMessage(error.message || '结算失败，请稍后重试', 'error');
    } finally {
        isCheckingOut = false;
        updateSummary();
    }
};

const printReceipt = () => {
    if (!lastOrder) {
        setMessage('暂无可打印小票，请先完成结算', 'error');
        return;
    }

    const lines = lastOrder.items.map((item) => `
        <tr>
            <td>${item.product_name}</td>
            <td style="text-align:center;">${item.quantity}</td>
            <td style="text-align:right;">${Number(item.unit_price).toFixed(2)}</td>
            <td style="text-align:right;">${Number(item.subtotal).toFixed(2)}</td>
        </tr>
    `).join('');

    const receiptHtml = `
        <html>
        <head>
            <meta charset="utf-8" />
            <title>小票-${lastOrder.order_no}</title>
            <style>
                body { font-family: 'Microsoft YaHei', sans-serif; padding: 10px; }
                .ticket { width: 280px; margin: 0 auto; font-size: 12px; color: #111; }
                .center { text-align: center; }
                .line { border-top: 1px dashed #555; margin: 8px 0; }
                table { width: 100%; border-collapse: collapse; }
                th, td { padding: 2px 0; font-size: 12px; }
                .right { text-align: right; }
            </style>
        </head>
        <body>
            <div class="ticket">
                <h3 class="center" style="margin: 0 0 6px;">超市管理系统</h3>
                <p class="center" style="margin: 0;">销售小票</p>
                <div class="line"></div>
                <p style="margin: 0 0 4px;">订单号：${lastOrder.order_no}</p>
                <p style="margin: 0 0 4px;">时间：${lastOrder.created_at}</p>
                <p style="margin: 0 0 4px;">支付方式：${paymentLabels[lastOrder.payment_method] || lastOrder.payment_method}</p>
                <div class="line"></div>
                <table>
                    <thead>
                        <tr>
                            <th style="text-align:left;">商品</th>
                            <th style="text-align:center;">数量</th>
                            <th style="text-align:right;">单价</th>
                            <th style="text-align:right;">小计</th>
                        </tr>
                    </thead>
                    <tbody>${lines}</tbody>
                </table>
                <div class="line"></div>
                <p class="right" style="margin: 0 0 4px;">应收：${Number(lastOrder.total_amount).toFixed(2)}</p>
                <p class="right" style="margin: 0 0 4px;">优惠：${Number(lastOrder.discount_amount).toFixed(2)}</p>
                <p class="right" style="margin: 0; font-weight: bold;">实收：${Number(lastOrder.actual_amount).toFixed(2)}</p>
                <div class="line"></div>
                <p class="center" style="margin: 8px 0 0;">欢迎下次光临</p>
            </div>
        </body>
        </html>
    `;

    const printWindow = window.open('', '_blank');
    if (!printWindow) {
        setMessage('打印窗口被浏览器拦截，请允许弹窗后重试', 'error');
        return;
    }
    printWindow.document.open();
    printWindow.document.write(receiptHtml);
    printWindow.document.close();
    printWindow.focus();
    printWindow.print();
};

searchBtn.addEventListener('click', loadProducts);
keywordInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
        loadProducts();
    }
});

discountInput.addEventListener('input', updateSummary);
checkoutBtn.addEventListener('click', doCheckout);
printBtn.addEventListener('click', printReceipt);
clearCartBtn.addEventListener('click', () => {
    cart = [];
    renderCart();
    setMessage('已清空购物车');
});

renderCart();
loadProducts();
