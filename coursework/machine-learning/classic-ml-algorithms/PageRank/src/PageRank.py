# Author: Bohan Yu

import argparseimport osfrom collections import defaultdictclass PageRank:
    
    def __init__(self, damping_factor=0.85, max_iterations=100, tolerance=1e-6):
        """Initialize the damping factor, iteration limit and tolerance."""
        self.damping_factor = damping_factor
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        

        self.graph = defaultdict(list)
        self.reverse_graph = defaultdict(list)
        self.nodes = set()
        self.out_degree = defaultdict(int)
        

        self.pagerank = {}
        self.pagerank_history = []

    def add_edge(self, from_node, to_node):
        """Add a directed edge from one node to another."""
        if from_node != to_node:
            self.graph[from_node].append(to_node)
            self.reverse_graph[to_node].append(from_node)
            
            self.nodes.add(from_node)
            self.nodes.add(to_node)
            
            self.out_degree[from_node] += 1

    def load_graph_from_file(self, file_path):
        """Load a directed graph from a text file."""
        try:
            with open(file_path) as f:
                for _line_num, line in enumerate(f, 1):
                    line = line.strip()
                    

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
        """Initialize every node to an equal PageRank value."""
        n = len(self.nodes)
        if n == 0:
            return
        
        initial_pr = 1.0 / n
        self.pagerank = {node: initial_pr for node in self.nodes}

    def _compute_pagerank_iteration(self):
        """Run one PageRank iteration."""
        new_pagerank = {}
        n = len(self.nodes)
        base_value = (1 - self.damping_factor) / n
        
        for node in self.nodes:

            rank_sum = 0.0
            
            for in_neighbor in self.reverse_graph[node]:
                out_degree = self.out_degree[in_neighbor]
                if out_degree > 0:
                    rank_sum += self.pagerank[in_neighbor] / out_degree
            

            new_pagerank[node] = base_value + self.damping_factor * rank_sum
        
        return new_pagerank

    def compute(self):
        """Compute PageRank by iteration until convergence."""
        self._initialize_pagerank()
        self.pagerank_history = []
        
        for iteration in range(self.max_iterations):

            new_pagerank = self._compute_pagerank_iteration()
            

            max_change = max(abs(new_pagerank[node] - self.pagerank[node]) 
                           for node in self.nodes)
            
            self.pagerank = new_pagerank
            self.pagerank_history.append(max_change)
            

            if max_change < self.tolerance:
                print(f"PageRank 在第 {iteration + 1} 次迭代后收敛")
                break
        else:
            print(f"PageRank 未在 {self.max_iterations} 次迭代内收敛")

    def get_top_k_nodes(self, k=10):
        """Return the k nodes with the highest PageRank."""
        sorted_nodes = sorted(self.pagerank.items(), 
                             key=lambda x: x[1], reverse=True)
        return sorted_nodes[:k]

    def get_pagerank(self, node):
        """Return the PageRank value of one node."""
        return self.pagerank.get(node, None)

    def get_statistics(self):
        """Return graph statistics such as node and edge counts."""
        n_nodes = len(self.nodes)
        n_edges = sum(len(neighbors) for neighbors in self.graph.values())
        

        avg_out_degree = n_edges / n_nodes if n_nodes > 0 else 0
        

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
        """Write PageRank results to a file."""
        try:
            with open(file_path, 'w') as f:
                f.write("NodeID\tPageRank\n")
                for node in sorted(self.pagerank.keys()):
                    f.write(f"{node}\t{self.pagerank[node]:.10f}\n")
            print(f"已导出 PageRank 结果到 {file_path}")
        except Exception as e:
            print(f"错误：导出失败 - {e}")

    def print_summary(self):
        """Print a PageRank summary."""
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
    """Load a web graph from a dataset file."""
    pr = PageRank()
    
    try:
        edge_count = 0
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                

                if not line or line.startswith('#'):
                    continue
                
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        from_node = int(parts[0])
                        to_node = int(parts[1])
                        
                        pr.add_edge(from_node, to_node)
                        edge_count += 1
                        
                        # Cap loaded edges for large graphs.
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
    """Create the sample web graph used for the demo."""
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
    

    if args.sample:
        print("使用示例图...")
        pr = create_sample_graph()
    elif args.file:
        print(f"加载网络图数据 ({args.file})...")
        pr = load_web_graph(args.file, max_nodes=args.max_edges)
    else:

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
    

    print(f"\n计算 PageRank (d={args.damping}, max_iter={args.iterations})...")
    pr.compute()
    

    pr.print_summary()
    

    if args.output:
        pr.export_pagerank(args.output)
