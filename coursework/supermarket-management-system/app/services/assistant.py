import re
import json
from urllib import error, request

from flask import current_app
from sqlalchemy import func

from app import db
from app.models import Category, Inventory, Product
from app.services.analytics import get_sales_overview, get_top_products
from app.services.inventory import get_inventory_alerts, get_inventory_summary


_FAQ_PRESET_ANSWERS = {
    '如何添加新商品': {
        'answer': '\n'.join([
            '添加新商品可以按下面步骤操作：',
            '1. 进入“商品管理”页面。',
            '2. 点击“新增商品”。',
            '3. 填写商品编码、名称、售价、分类、最小库存等信息。',
            '4. 如需初始化库存，填写初始数量后保存。',
            '5. 保存成功后可在商品列表搜索验证。',
        ]),
        'suggestions': ['如何批量导入商品', '怎样更新库存数量', '设置库存预警值'],
    },
    '怎样更新库存数量': {
        'answer': '\n'.join([
            '更新库存数量的建议流程：',
            '1. 进入“库存管理”页面。',
            '2. 找到目标商品后执行库存调整。',
            '3. 输入调整后的库存数量并填写调整原因。',
            '4. 保存后系统会自动记录库存流水，便于追踪。',
        ]),
        'suggestions': ['查询低库存商品', '设置库存预警值', '查看商品库存状态'],
    },
    '如何导出销售报表': {
        'answer': '\n'.join([
            '当前版本暂未提供页面一键导出按钮。',
            '你可以先在“销售管理”页按时间、收银员、支付方式筛选数据。',
            '如需导出文件，建议对接接口 `/api/sales/orders` 拉取数据后生成 Excel/CSV。',
            '如果你愿意，我可以下一步直接帮你补一个“导出销售报表”按钮。',
        ]),
        'suggestions': ['查看本周销售趋势', '今天销售怎么样', '查询热销商品'],
    },
    '设置库存预警值': {
        'answer': '\n'.join([
            '库存预警值使用商品的“最小库存（min_stock）”字段控制：',
            '1. 进入“商品管理”页面。',
            '2. 编辑对应商品。',
            '3. 将“最小库存”设置为目标阈值并保存。',
            '4. 当库存小于等于该值时，会在库存预警中显示。',
        ]),
        'suggestions': ['查询低库存商品', '怎样更新库存数量', '查看商品库存状态'],
    },
}


def generate_assistant_reply(message):
    """根据用户问题生成助手回复。"""
    text = (message or '').strip()
    if not text:
        return {
            'success': False,
            'answer': '请输入问题后再发送。',
            'suggestions': ['查询低库存商品', '查看今日销售概览', '如何添加新商品'],
        }

    if len(text) > 500:
        return {
            'success': False,
            'answer': '问题长度不能超过 500 个字符。',
            'suggestions': ['精简后重试'],
        }

    faq_reply = _get_faq_preset_reply(text)
    if faq_reply:
        return faq_reply

    llm_answer = _try_generate_by_llm(text)
    if llm_answer:
        return {
            'success': True,
            'answer': llm_answer,
            'suggestions': _build_suggestions(text),
        }

    if _is_inventory_query(text):
        return {
            'success': True,
            'answer': _build_inventory_reply(text),
            'suggestions': ['查询低库存商品', '查看商品库存状态', '如何更新库存数量'],
        }

    if _is_sales_query(text):
        return {
            'success': True,
            'answer': _build_sales_reply(text),
            'suggestions': ['查看本周销售趋势', '查询热销商品', '导出销售报表'],
        }

    if _is_product_query(text):
        return {
            'success': True,
            'answer': _build_product_reply(text),
            'suggestions': ['如何添加新商品', '如何导入商品', '设置库存预警值'],
        }

    if _is_help_query(text):
        return {
            'success': True,
            'answer': _build_help_reply(),
            'suggestions': ['如何添加新商品', '怎样更新库存数量', '如何导出销售报表'],
        }

    return {
        'success': True,
        'answer': _build_default_reply(),
        'suggestions': ['查询库存预警', '查看今日销售', '如何添加新商品'],
    }


def _is_inventory_query(text):
    return any(word in text for word in ('库存', '补货', '缺货', '预警'))


def _is_sales_query(text):
    return any(word in text for word in ('销售', '营收', '订单', '利润', '报表', '趋势'))


def _is_product_query(text):
    return any(word in text for word in ('商品', '上架', '下架', '分类', '条码', '编码', '导入'))


def _is_help_query(text):
    return any(word in text for word in ('帮助', '怎么', '如何', '不会', '指南', '说明'))


def _normalize_question(text):
    normalized = re.sub(r'[\s\?？!！。,.，:：;；"\'“”‘’]+', '', text or '')
    return normalized.strip()


def _get_faq_preset_reply(text):
    normalized_text = _normalize_question(text)
    for question, payload in _FAQ_PRESET_ANSWERS.items():
        if normalized_text == _normalize_question(question):
            return {
                'success': True,
                'answer': payload['answer'],
                'suggestions': payload['suggestions'],
            }
    return None


def _build_suggestions(text):
    if _is_inventory_query(text):
        return ['查询低库存商品', '查看商品库存状态', '如何更新库存数量']
    if _is_sales_query(text):
        return ['查看本周销售趋势', '查询热销商品', '导出销售报表']
    if _is_product_query(text):
        return ['如何添加新商品', '如何导入商品', '设置库存预警值']
    if _is_help_query(text):
        return ['如何添加新商品', '怎样更新库存数量', '如何导出销售报表']
    return ['查询库存预警', '查看今日销售', '如何添加新商品']


def _try_generate_by_llm(text):
    provider = (current_app.config.get('LLM_PROVIDER') or '').strip().lower()
    api_key = (current_app.config.get('LLM_API_KEY') or '').strip()
    if not api_key:
        return None

    base_url = (current_app.config.get('LLM_BASE_URL') or '').strip().rstrip('/')
    chat_path = (current_app.config.get('LLM_CHAT_PATH') or '/chat/completions').strip()
    default_model = 'deepseek-ai/DeepSeek-V3' if provider == 'siliconflow' else 'gpt-4o-mini'
    model = (current_app.config.get('LLM_MODEL') or default_model).strip()
    timeout = int(current_app.config.get('LLM_TIMEOUT') or 30)
    site_url = (current_app.config.get('LLM_SITE_URL') or '').strip()
    site_name = (current_app.config.get('LLM_SITE_NAME') or '').strip()

    if not base_url:
        return None

    endpoint = base_url + chat_path if chat_path.startswith('/') else f'{base_url}/{chat_path}'
    payload = {
        'model': model,
        'messages': [
            {
                'role': 'system',
                'content': (
                    '你是超市管理系统的智能助手。'
                    '请用中文简洁回答，优先提供可执行建议，避免编造系统不存在的功能。'
                ),
            },
            {'role': 'user', 'content': text},
        ],
        'temperature': 0.3,
    }

    body = json.dumps(payload).encode('utf-8')
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    # 硅基流动兼容 OpenAI 接口，附带来源信息便于平台识别调用方。
    if provider == 'siliconflow':
        if site_url:
            headers['Referer'] = site_url
        if site_name:
            headers['X-Title'] = site_name

    req = request.Request(
        endpoint,
        data=body,
        method='POST',
        headers=headers,
    )

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode('utf-8')
            data = json.loads(raw)
            return _extract_llm_content(data)
    except error.URLError:
        return None
    except ValueError:
        return None
    except Exception:
        return None


def _extract_llm_content(data):
    choices = data.get('choices') if isinstance(data, dict) else None
    if not choices:
        return None

    first = choices[0] or {}
    message = first.get('message') if isinstance(first, dict) else None
    if not isinstance(message, dict):
        return None

    content = message.get('content')
    if isinstance(content, str):
        content = content.strip()
        return content if content else None

    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get('text'), str):
                chunks.append(item['text'])
        merged = ''.join(chunks).strip()
        return merged if merged else None

    return None


def _build_inventory_reply(text):
    summary = get_inventory_summary()
    lines = [
        '已为您整理当前库存概览：',
        f"- 在库商品数：{summary['total_products']}",
        f"- 总库存数量：{summary['total_quantity']}",
        f"- 低库存商品：{summary['low_stock_count']}",
        f"- 缺货商品：{summary['out_stock_count']}",
    ]

    matched_products = _search_product_stock(text)
    if matched_products:
        lines.append('')
        lines.append('匹配到以下商品库存：')
        for item in matched_products:
            lines.append(
                f"- {item['product_name']}（编码 {item['product_code']}）：当前 {item['quantity']}，预警值 {item['min_stock']}"
            )
    else:
        alerts = get_inventory_alerts(limit=5)
        if alerts:
            lines.append('')
            lines.append('建议优先补货（最多 5 条）：')
            for item in alerts:
                lines.append(
                    f"- {item['product_name']}：当前 {item['quantity']}，建议补 {item['stock_gap']}"
                )

    return '\n'.join(lines)


def _search_product_stock(text, limit=5):
    candidates = _extract_keywords(text)
    if not candidates:
        return []

    query = db.session.query(
        Product.product_name,
        Product.product_code,
        Product.min_stock,
        func.coalesce(Inventory.quantity, 0).label('quantity'),
    ).outerjoin(
        Inventory, Product.product_id == Inventory.product_id
    )

    filters = []
    for keyword in candidates:
        like_key = f'%{keyword}%'
        filters.append(Product.product_name.like(like_key))
        filters.append(Product.product_code.like(like_key))

    rows = query.filter(db.or_(*filters)).order_by(Product.product_id.desc()).limit(limit).all()

    items = []
    for row in rows:
        items.append({
            'product_name': row.product_name,
            'product_code': row.product_code,
            'quantity': int(row.quantity or 0),
            'min_stock': int(row.min_stock or 0),
        })

    return items


def _extract_keywords(text):
    words = re.findall(r'[A-Za-z0-9\u4e00-\u9fff]{2,}', text)
    stop_words = {
        '库存', '查询', '查看', '商品', '现在', '目前', '多少', '一下', '帮我', '请问', '预警',
        '销售', '报表', '统计', '趋势', '分析', '如何', '怎么', '导出',
    }
    keywords = []
    for word in words:
        if word in stop_words:
            continue
        if word not in keywords:
            keywords.append(word)
    return keywords[:4]


def _build_sales_reply(text):
    period = 'month'
    period_label = '本月'
    if '今天' in text or '今日' in text:
        period = 'today'
        period_label = '今日'
    elif '周' in text:
        period = 'week'
        period_label = '近 7 天'

    overview = get_sales_overview(period)
    top_products = get_top_products(limit=3, period=period, sort_by='sales')

    lines = [
        f'以下是{period_label}销售概览：',
        f"- 销售额：¥{overview['total_sales']}",
        f"- 订单数：{overview['order_count']}",
        f"- 客单价：¥{overview['avg_order_value']}",
        f"- 毛利润：¥{overview['gross_profit']}（利润率 {overview['profit_rate']}%）",
    ]

    if top_products:
        lines.append('')
        lines.append('销售额 Top 3 商品：')
        for item in top_products:
            lines.append(f"- {item['product_name']}：¥{item['total_sales']}（售出 {item['quantity_sold']}）")

    return '\n'.join(lines)


def _build_product_reply(text):
    total_products = Product.query.count()
    category_count = Category.query.count()

    lines = [
        '商品管理可以这样使用：',
        '- 添加商品：进入“商品管理”，点击“新增商品”，填写编码、名称、售价等信息。',
        '- 批量导入：支持 CSV / Excel，字段需包含 product_code、product_name、selling_price。',
        '- 上下架：在商品列表中可快速下架/上架，避免误售。',
        f'- 当前系统内共有 {total_products} 个商品，{category_count} 个分类。',
    ]

    if '热销' in text or '推荐' in text:
        hot_items = get_top_products(limit=5, period='week', sort_by='quantity')
        if hot_items:
            lines.append('')
            lines.append('近 7 天热销推荐：')
            for item in hot_items:
                lines.append(f"- {item['product_name']}：售出 {item['quantity_sold']}，销售额 ¥{item['total_sales']}")

    return '\n'.join(lines)


def _build_help_reply():
    return '\n'.join([
        '我可以协助以下常见场景：',
        '- 库存：查询低库存、缺货、指定商品库存。',
        '- 销售：查看今日/本周/本月销售与热销商品。',
        '- 商品：新增、导入、上下架、分类管理说明。',
        '',
        '示例提问：',
        '- 今天销售怎么样？',
        '- 哪些商品需要补货？',
        '- 如何批量导入商品？',
    ])


def _build_default_reply():
    return '\n'.join([
        '我已收到您的问题。当前版本支持库存、销售、商品管理三类智能问答。',
        '您可以换个问法试试，例如：',
        '- 查询低库存商品',
        '- 查看今日销售概览',
        '- 如何添加新商品',
    ])
