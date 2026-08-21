# 作者：23计算1Bohan Yu

import os
import argparse
from collections import defaultdict, deque

class PageRank:
    
    def __init__(self, damping_factor=0.85, max_iterations=100, tolerance=1e-6):
        """
        Args:
            damping_factor: 阻尼因子（通常 0.85），表示用户跟随链接的概率
            max_iterations: 最大迭代次数
            tolerance: 收敛容差（PageRank 值变化）
        """
        self.damping_factor = damping_factor
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # 图结构
        self.graph = defaultdict(list)          # 出边（邻接表）
        self.reverse_graph = defaultdict(list)  # 入边（反向图）
        self.nodes = set()                      # 所有节点
        self.out_degree = defaultdict(int)      # 出度
        
        # PageRank 值
        self.pagerank = {}
        self.pagerank_history = []

    def add_edge(self, from_node, to_node):
        """添加一条有向边（from_node -> to_node）
        
        Args:
            from_node: 源节点
            to_node: 目标节点
        """
        if from_node != to_node:  # 忽略自环
            self.graph[from_node].append(to_node)
            self.reverse_graph[to_node].append(from_node)
            
            self.nodes.add(from_node)
            self.nodes.add(to_node)
            
            self.out_degree[from_node] += 1

    def load_graph_from_file(self, file_path):
        """从文件加载图结构
        
        文件格式：
        - 以 # 开头的行是注释
        - 每行包含 "from_node to_node"
        
        Args:
            file_path: 输入文件路径
        """
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # 跳过空行和注释
                    if not line or line.startswith('#'):
                        continue
                    
                    try:
                        parts = line.split()
                        if len(parts) >= 2:
                            from_node = int(parts[0])
                            to_node = int(parts[1])
                            self.add_edge(from_node, to_node)
                    except (ValueError, IndexError):
                        continue
        
        except FileNotFoundError:
            print(f"错误：文件 {file_path} 未找到")
            return False
        except Exception as e:
            print(f"错误：读取文件失败 - {e}")
            return False
        
        return True

    def _initialize_pagerank(self):
        """初始化 PageRank 值
        
        所有网页的初始 PageRank 值相等 = 1 / N
        """
        n = len(self.nodes)
        if n == 0:
            return
        
        initial_pr = 1.0 / n
        self.pagerank = {node: initial_pr for node in self.nodes}

    def _compute_pagerank_iteration(self):
        """计算一次 PageRank 迭代
        
        PageRank(p) = (1-d) / N + d * Σ(PageRank(q) / out_degree(q))
        
        其中：
        - d: 阻尼因子
        - N: 网页总数
        - q: 指向 p 的网页
        - out_degree(q): q 的出度
        """
        new_pagerank = {}
        n = len(self.nodes)
        base_value = (1 - self.damping_factor) / n
        
        for node in self.nodes:
            # 计算来自指向该节点的所有网页的 PageRank 贡献
            rank_sum = 0.0
            
            for in_neighbor in self.reverse_graph[node]:
                out_degree = self.out_degree[in_neighbor]
                if out_degree > 0:
                    rank_sum += self.pagerank[in_neighbor] / out_degree
            
            # 更新 PageRank
            new_pagerank[node] = base_value + self.damping_factor * rank_sum
        
        return new_pagerank

    def compute(self):
        """计算 PageRank
        
        使用迭代方法直到收敛
        """
        self._initialize_pagerank()
        self.pagerank_history = []
        
        for iteration in range(self.max_iterations):
            # 进行一次迭代
            new_pagerank = self._compute_pagerank_iteration()
            
            # 计算 PageRank 值的最大变化
            max_change = max(abs(new_pagerank[node] - self.pagerank[node]) 
                           for node in self.nodes)
            
            self.pagerank = new_pagerank
            self.pagerank_history.append(max_change)
            
            # 检查收敛
            if max_change < self.tolerance:
                print(f"PageRank 在第 {iteration + 1} 次迭代后收敛")
                break
        else:
            print(f"PageRank 未在 {self.max_iterations} 次迭代内收敛")

    def get_top_k_nodes(self, k=10):
        """获取 PageRank 值最高的 K 个节点
        
        Args:
            k: 返回节点数量
            
        Returns:
            [(node, pagerank), ...] 列表，按 PageRank 值降序排列
        """
        sorted_nodes = sorted(self.pagerank.items(), 
                             key=lambda x: x[1], reverse=True)
        return sorted_nodes[:k]

    def get_pagerank(self, node):
        """获取某个节点的 PageRank 值
        
        Args:
            node: 节点标识
            
        Returns:
            PageRank 值，若节点不存在返回 None
        """
        return self.pagerank.get(node, None)

    def get_statistics(self):
        """获取图的统计信息
        
        Returns:
            包含节点数、边数等信息的字典
        """
        n_nodes = len(self.nodes)
        n_edges = sum(len(neighbors) for neighbors in self.graph.values())
        
        # 计算平均出度
        avg_out_degree = n_edges / n_nodes if n_nodes > 0 else 0
        
        # 计算有向无环性等信息
        in_degrees = [len(neighbors) for neighbors in self.reverse_graph.values()]
        out_degrees = [self.out_degree[node] for node in self.nodes]
        
        avg_in_degree = sum(in_degrees) / n_nodes if n_nodes > 0 else 0
        
        return {
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'avg_in_degree': avg_in_degree,
            'avg_out_degree': avg_out_degree,
            'max_in_degree': max(in_degrees) if in_degrees else 0,
            'max_out_degree': max(out_degrees) if out_degrees else 0,
            'dangling_nodes': sum(1 for node in self.nodes if self.out_degree[node] == 0),
            'iterations': len(self.pagerank_history)
        }

    def export_pagerank(self, file_path):
        """导出 PageRank 结果到文件
        
        Args:
            file_path: 输出文件路径
        """
        try:
            with open(file_path, 'w') as f:
                f.write("NodeID\tPageRank\n")
                for node in sorted(self.pagerank.keys()):
                    f.write(f"{node}\t{self.pagerank[node]:.10f}\n")
            print(f"已导出 PageRank 结果到 {file_path}")
        except Exception as e:
            print(f"错误：导出失败 - {e}")

    def print_summary(self):
        """打印 PageRank 计算总结"""
        stats = self.get_statistics()
        
        print("\n=== 图统计 ===")
        print(f"节点数: {stats['n_nodes']}")
        print(f"边数: {stats['n_edges']}")
        print(f"平均入度: {stats['avg_in_degree']:.4f}")
        print(f"平均出度: {stats['avg_out_degree']:.4f}")
        print(f"最大入度: {stats['max_in_degree']}")
        print(f"最大出度: {stats['max_out_degree']}")
        print(f"悬挂节点数: {stats['dangling_nodes']}")
        print(f"收敛迭代次数: {stats['iterations']}")
        
        print("\n=== 参数 ===")
        print(f"阻尼因子: {self.damping_factor}")
        print(f"收敛容差: {self.tolerance}")
        
        print("\n=== PageRank 统计 ===")
        pr_values = list(self.pagerank.values())
        print(f"最小 PageRank: {min(pr_values):.10f}")
        print(f"最大 PageRank: {max(pr_values):.10f}")
        print(f"平均 PageRank: {sum(pr_values) / len(pr_values):.10f}")
        
        print("\n=== Top 10 节点 ===")
        for i, (node, pr) in enumerate(self.get_top_k_nodes(10), 1):
            print(f"{i:2d}. 节点 {node:8d}: PageRank = {pr:.10f}")


def load_web_graph(file_path, max_nodes=None):
    """加载网络图数据
    
    Args:
        file_path: 数据文件路径
        max_nodes: 最大加载节点数（用于测试）
        
    Returns:
        PageRank 对象
    """
    pr = PageRank()
    
    try:
        edge_count = 0
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # 跳过注释和空行
                if not line or line.startswith('#'):
                    continue
                
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        from_node = int(parts[0])
                        to_node = int(parts[1])
                        
                        pr.add_edge(from_node, to_node)
                        edge_count += 1
                        
                        # 限制加载的边数（用于大数据集）
                        if max_nodes is not None and edge_count % 100000 == 0:
                            print(f"已加载 {edge_count} 条边...")
                            if edge_count >= max_nodes:
                                print(f"达到最大边数限制 ({max_nodes})")
                                break
                
                except (ValueError, IndexError):
                    continue
        
        print(f"共加载 {edge_count} 条边")
    
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到")
        return None
    except Exception as e:
        print(f"错误：读取文件失败 - {e}")
        return None
    
    return pr if pr.nodes else None


def create_sample_graph():
    """创建示例图（用于测试）
    
    简单的网页图示例：
    1 -> 2, 3
    2 -> 3
    3 -> 1
    4 -> 3
    5 -> 4, 6
    6 -> 5
    """
    pr = PageRank()
    
    edges = [
        (1, 2), (1, 3),
        (2, 3),
        (3, 1),
        (4, 3),
        (5, 4), (5, 6),
        (6, 5)
    ]
    
    for from_node, to_node in edges:
        pr.add_edge(from_node, to_node)
    
    return pr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PageRank 算法')
    parser.add_argument('-f', '--file', type=str, default=None,
                        help='网络图数据文件路径')
    parser.add_argument('-d', '--damping', type=float, default=0.85,
                        help='阻尼因子（0-1）')
    parser.add_argument('-i', '--iterations', type=int, default=100,
                        help='最大迭代次数')
    parser.add_argument('-t', '--tolerance', type=float, default=1e-6,
                        help='收敛容差')
    parser.add_argument('-k', '--top_k', type=int, default=10,
                        help='显示 Top K 节点')
    parser.add_argument('--max_edges', type=int, default=None,
                        help='最多加载的边数（用于大数据集测试）')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='输出 PageRank 结果到文件')
    parser.add_argument('--sample', action='store_true',
                        help='使用示例图而不加载文件')
    
    args = parser.parse_args()
    
    # 加载图
    if args.sample:
        print("使用示例图...")
        pr = create_sample_graph()
    elif args.file:
        print(f"加载网络图数据 ({args.file})...")
        pr = load_web_graph(args.file, max_nodes=args.max_edges)
    else:
        # 默认路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_file = os.path.join(script_dir, '..', 'data', 'web-Google.txt')
        
        if os.path.exists(default_file):
            print(f"加载网络图数据 ({default_file})...")
            pr = load_web_graph(default_file, max_nodes=args.max_edges)
        else:
            print(f"默认数据文件未找到: {default_file}")
            print("使用示例图进行演示...")
            pr = create_sample_graph()
    
    if pr is None:
        print("无法加载图数据，程序退出")
        exit(1)
    
    # 计算 PageRank
    print(f"\n计算 PageRank (d={args.damping}, max_iter={args.iterations})...")
    pr.compute()
    
    # 显示结果
    pr.print_summary()
    
    # 导出结果
    if args.output:
        pr.export_pagerank(args.output)
