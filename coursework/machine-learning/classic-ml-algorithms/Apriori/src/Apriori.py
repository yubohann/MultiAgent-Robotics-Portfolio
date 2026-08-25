# 作者：23计算1Bohan Yu 

import csv
import itertools
import os
import argparse

def load_groceries_data(file_path):
    # 读取购物篮数据，每行一个事务，商品用逗号分隔
    transactions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        baskets = {}
        for row in reader:
            member = row['Member_number']
            item = row['itemDescription']
            if member not in baskets:
                baskets[member] = set()
            baskets[member].add(item)
        transactions = list(baskets.values())
    return transactions

def get_frequent_itemsets(transactions, min_support):
    item_counts = {}
    num_transactions = len(transactions)
    for transaction in transactions:
        for item in transaction:
            itemset = frozenset([item])
            item_counts[itemset] = item_counts.get(itemset, 0) + 1
    frequent_itemsets = {item for item, count in item_counts.items() if count / num_transactions >= min_support}
    result = [frequent_itemsets]
    k = 2
    while True:
        prev_frequent = result[-1]
        candidates = set()
        prev_list = list(prev_frequent)
        for i in range(len(prev_list)):
            for j in range(i+1, len(prev_list)):
                union = prev_list[i] | prev_list[j]
                if len(union) == k:
                    candidates.add(union)
        candidate_counts = {itemset: 0 for itemset in candidates}
        for transaction in transactions:
            for candidate in candidates:
                if candidate.issubset(transaction):
                    candidate_counts[candidate] += 1
        frequent_k = {itemset for itemset, count in candidate_counts.items() if count / num_transactions >= min_support}
        if not frequent_k:
            break
        result.append(frequent_k)
        k += 1
    all_frequent = set()
    for level in result:
        all_frequent |= level
    return all_frequent

def generate_rules(frequent_itemsets, transactions, min_confidence):
    rules = []
    num_transactions = len(transactions)
    itemset_support = {}
    for itemset in frequent_itemsets:
        count = sum(1 for transaction in transactions if itemset.issubset(transaction))
        itemset_support[itemset] = count / num_transactions
    for itemset in frequent_itemsets:
        if len(itemset) < 2:
            continue
        for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                if not consequent:
                    continue
                confidence = itemset_support[itemset] / itemset_support.get(antecedent, 1e-9)
                if confidence >= min_confidence:
                    rules.append((set(antecedent), set(consequent), confidence))
    return rules

if __name__ == '__main__':
    # (3) 数据集说明
    # 来源：https://www.kaggle.com/heeraldedhia/groceries-dataset
    # 记录数：约9835条，每条为一个购物篮
    # 属性数：商品种类约169种
    # 数据内容：每行表示一笔交易，内容为购买的商品列表

    parser = argparse.ArgumentParser(description='Apriori 算法实现（纯 Python，无第三方库）')
    parser.add_argument('-f', '--file', help='Groceries 数据集 CSV 文件路径', default=None)
    parser.add_argument('-s', '--min_support', help='最小支持度（小数）', type=float, default=0.01)
    parser.add_argument('-c', '--min_confidence', help='最小置信度（小数）', type=float, default=0.3)
    args = parser.parse_args()

    # 尝试使用用户提供路径，或使用相对于当前脚本的默认数据路径
    if args.file:
        file_path = args.file
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 默认位于 repo 的 ../data/archive/Groceries_dataset.csv
        file_path = os.path.normpath(os.path.join(script_dir, '..', 'data', 'archive', 'Groceries_dataset.csv'))

    if not os.path.exists(file_path):
        print(f"找不到数据文件: {file_path}")
        print("请将 `Groceries_dataset.csv` 放到 `Apriori/data/archive/`，或使用 -f 指定文件路径。")
        raise SystemExit(1)

    transactions = load_groceries_data(file_path)
    print(f'总记录数: {len(transactions)}')
    unique_items = set(itertools.chain.from_iterable(transactions))
    print(f'属性数: {len(unique_items)}')
    print('数据内容示例:', next(iter(transactions)))

    # 挖掘频繁项集
    min_support = args.min_support
    frequent_itemsets = get_frequent_itemsets(transactions, min_support)
    print('频繁项集数量:', len(frequent_itemsets))
    for itemset in list(frequent_itemsets)[:10]:  # 只显示前10个
        print(set(itemset))

    # 生成关联规则
    min_confidence = args.min_confidence
    rules = generate_rules(frequent_itemsets, transactions, min_confidence)
    print('关联规则数量:', len(rules))
    for antecedent, consequent, confidence in rules[:10]:  # 只显示前10条
        print(f'{antecedent} => {consequent} (置信度: {confidence:.2f})')