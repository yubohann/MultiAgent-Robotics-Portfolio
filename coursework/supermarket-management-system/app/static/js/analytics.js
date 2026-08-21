let salesTrendChart = null;
let categoryPieChart = null;
let currentPeriod = 'month';
let isAscending = false;

document.addEventListener('DOMContentLoaded', function () {
    if (typeof Chart === 'undefined') {
        console.error('Chart.js 未加载');
        return;
    }

    initCharts();

    const periodFilter = document.getElementById('periodFilter');
    if (periodFilter) {
        periodFilter.addEventListener('change', function (e) {
            currentPeriod = e.target.value;
            loadData();
        });
    }

    const sortBySelect = document.getElementById('sortBySelect');
    if (sortBySelect) {
        sortBySelect.addEventListener('change', function () {
            loadTopProducts();
        });
    }

    const toggleBtn = document.getElementById('toggleSortBtn');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', function () {
            isAscending = !isAscending;
            this.textContent = isAscending ? '查看热销' : '查看滞销';
            document.getElementById('productRankTitle').textContent = isAscending ? '滞销商品 TOP 10' : '热销商品 TOP 10';
            loadTopProducts();
        });
    } else {
        console.error('未找到toggleSortBtn按钮元素');
    }
});

function initCharts() {
    loadData();
}

function loadData() {
    loadSalesOverview();
    loadSalesTrend();
    loadTopProducts();
    loadCategoryDistribution();
}

async function loadSalesOverview() {
    try {
        const response = await fetch(`/api/analytics/overview?period=${currentPeriod}`);
        const data = await response.json();

        document.getElementById('totalSales').textContent = `¥${data.total_sales.toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
        document.getElementById('orderCount').textContent = data.order_count;
        document.getElementById('avgOrderValue').textContent = `¥${data.avg_order_value.toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
        document.getElementById('grossProfit').textContent = `¥${data.gross_profit.toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
        document.getElementById('profitRate').textContent = `利润率 ${data.profit_rate}%`;

        const growthEl = document.getElementById('salesGrowth');
        if (data.growth_rate > 0) {
            growthEl.className = 'text-xs text-green-600 mt-1';
            growthEl.textContent = `↑ ${data.growth_rate}% 较上期`;
        } else if (data.growth_rate < 0) {
            growthEl.className = 'text-xs text-red-600 mt-1';
            growthEl.textContent = `↓ ${Math.abs(data.growth_rate)}% 较上期`;
        } else {
            growthEl.className = 'text-xs text-slate-500 mt-1';
            growthEl.textContent = '持平 较上期';
        }
    } catch (error) {
        console.error('加载销售概览失败:', error);
    }
}

async function loadSalesTrend() {
    try {
        const response = await fetch('/api/analytics/trend?days=30');
        const result = await response.json();
        const data = result.trend;

        const ctx = document.getElementById('salesTrendChart').getContext('2d');

        if (salesTrendChart) {
            salesTrendChart.destroy();
        }

        salesTrendChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.map((item) => item.date.substring(5)),
                datasets: [{
                    label: '销售额',
                    data: data.map((item) => item.sales),
                    borderColor: '#3B82F6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true,
                    pointRadius: 3,
                    pointHoverRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                return `销售额: ¥${context.parsed.y.toLocaleString('zh-CN', { minimumFractionDigits: 2 })}`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: function (value) {
                                return `¥${value}`;
                            }
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error('加载销售趋势失败:', error);
    }
}

async function loadTopProducts() {
    try {
        const sortBy = document.getElementById('sortBySelect').value;
        const sortParam = isAscending ? `${sortBy}_asc` : sortBy;

        const response = await fetch(`/api/analytics/top-products?limit=10&period=${currentPeriod}&sort_by=${sortParam}`);
        const result = await response.json();
        const products = result.products;

        const tbody = document.getElementById('productRankTable');

        if (products.length === 0) {
            tbody.innerHTML = '<tr class="border-b border-slate-100"><td colspan="5" class="py-8 text-center text-slate-400">暂无数据</td></tr>';
            return;
        }

        tbody.innerHTML = products.map((product, index) => {
            const rankIcon = index === 0 ? '🥇' : index === 1 ? '🥈' : index === 2 ? '🥉' : product.rank;
            return `
                <tr class="border-b border-slate-100 hover:bg-slate-50">
                    <td class="py-3 px-4 text-sm">${rankIcon}</td>
                    <td class="py-3 px-4 text-sm font-medium text-slate-800">${product.product_name}</td>
                    <td class="py-3 px-4 text-sm text-slate-600">${product.quantity_sold}</td>
                    <td class="py-3 px-4 text-sm text-slate-600">¥${product.total_sales.toLocaleString('zh-CN', { minimumFractionDigits: 2 })}</td>
                    <td class="py-3 px-4 text-sm text-slate-600">${product.percentage}</td>
                </tr>
            `;
        }).join('');
    } catch (error) {
        console.error('加载商品排行失败:', error);
    }
}

async function loadCategoryDistribution() {
    try {
        const response = await fetch(`/api/analytics/category-distribution?period=${currentPeriod}`);
        const result = await response.json();
        const categories = result.categories;

        const ctx = document.getElementById('categoryPieChart').getContext('2d');

        if (categoryPieChart) {
            categoryPieChart.destroy();
        }

        const colors = [
            '#3B82F6', '#10B981', '#F59E0B', '#EF4444',
            '#8B5CF6', '#EC4899', '#06B6D4', '#84CC16'
        ];

        categoryPieChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: categories.map((c) => c.category_name),
                datasets: [{
                    data: categories.map((c) => c.percentage),
                    backgroundColor: colors.slice(0, categories.length),
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: {
                            padding: 15,
                            usePointStyle: true,
                            pointStyle: 'circle'
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                const label = context.label || '';
                                const value = context.parsed || 0;
                                return `${label}: ${value}%`;
                            }
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error('加载分类占比失败:', error);
    }
}
