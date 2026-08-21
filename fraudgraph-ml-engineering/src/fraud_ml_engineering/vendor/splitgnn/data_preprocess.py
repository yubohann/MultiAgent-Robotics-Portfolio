"""SplitGNN 数据预处理脚本。

总说明：
1. 将原始 `.mat` 数据转换为 DGL 异构图。
2. 划分 train / valid / test 掩码。
3. 为 `homo` 边生成监督标签和训练掩码。

其中 `comp` 数据集已经自带 DGL 图，因此这里只做存在性确认。
"""

import argparse
import os
import sys
import copy
import dgl
import torch
import numpy as np
import scipy.io as scio
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')


def first_existing_path(*candidates):
    # 在多个可能路径中找到第一个实际存在的文件。
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f'Cannot find any expected dataset file: {candidates}')


def generate_edges_labels(edges, labels, train_idx):
    # 同标签边记为 +1，异标签边记为 -1，同时标记哪些边属于训练集。
    row, col = edges
    edge_labels = []
    edge_train_mask = []
    train_idx = set(train_idx)
    for i, j in zip(row, col):
        i = i.item()
        j = j.item()
        if labels[i] == labels[j]:
            edge_labels.append(1)
        else:
            edge_labels.append(-1)
        if i in train_idx and j in train_idx:
            edge_train_mask.append(1)
        else:
            edge_train_mask.append(0)
    edge_labels = torch.Tensor(edge_labels).long()
    edge_train_mask = torch.Tensor(edge_train_mask).bool()
    return edge_labels, edge_train_mask


if __name__ == '__main__':
    dataset_path = DATA_DIR + os.sep
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='amazon')
    args = parser.parse_args()
    print('**********************************')
    print(f'Generate {args.dataset}')
    print('**********************************')
    if args.dataset == 'yelp':
        # 生成 YelpChi 的 DGL 图。
        if os.path.exists(dataset_path+'yelp.dgl'):
            print('Dataset yelp has been created')
            sys.exit()
        print('Convert to DGL Graph.')
        yelp_path = first_existing_path(
            os.path.join(DATA_DIR, 'YelpChi.mat'),
            os.path.join(DATA_DIR, 'YelpChi', 'YelpChi.mat')
        )
        yelp = scio.loadmat(yelp_path)
        feats = yelp['features'].todense()
        features = torch.from_numpy(feats)
        lbs = yelp['label'][0]
        labels = torch.from_numpy(lbs)
        homo = yelp['homo']
        homo = homo+homo.transpose()
        homo = homo.tocoo()
        rur = yelp['net_rur']
        rur = rur+rur.transpose()
        rur = rur.tocoo()
        rtr = yelp['net_rtr']
        rtr = rtr+rtr.transpose()
        rtr = rtr.tocoo()
        rsr = yelp['net_rsr']
        rsr = rsr+rsr.transpose()
        rsr = rsr.tocoo()
        
        yelp_graph_structure = {
            ('r','homo','r'):(torch.tensor(homo.row), torch.tensor(homo.col)),
            ('r','u','r'):(torch.tensor(rur.row), torch.tensor(rur.col)),
            ('r','t','r'):(torch.tensor(rtr.row), torch.tensor(rtr.col)),
            ('r','s','r'):(torch.tensor(rsr.row), torch.tensor(rsr.col))
        }
        yelp_graph = dgl.heterograph(yelp_graph_structure)
        yelp_graph.nodes['r'].data['feature'] = features
        yelp_graph.nodes['r'].data['label'] = labels
        print('Generate dataset partition.')
        train_ratio = 0.4
        test_ratio = 0.67
        index = list(range(len(lbs)))
        dataset_l = len(lbs)
        train_idx, rest_idx, train_lbs, rest_lbs = train_test_split(index, lbs, stratify=lbs, train_size=train_ratio, random_state=2, shuffle=True)
        valid_idx, test_idx, _,_ = train_test_split(rest_idx, rest_lbs, stratify=rest_lbs, test_size=test_ratio, random_state=2, shuffle=True)
        train_mask = torch.zeros(dataset_l, dtype=torch.bool)
        train_mask[np.array(train_idx)] = True
        valid_mask = torch.zeros(dataset_l, dtype=torch.bool)
        valid_mask[np.array(valid_idx)] = True
        test_mask = torch.zeros(dataset_l, dtype=torch.bool)
        test_mask[np.array(test_idx)] = True
        
        yelp_graph.nodes['r'].data['train_mask'] = train_mask
        yelp_graph.nodes['r'].data['valid_mask'] = valid_mask
        yelp_graph.nodes['r'].data['test_mask'] = test_mask
        
        print('Generate edge labels.')
        homo_edges = yelp_graph.edges(etype='homo')
        homo_labels, homo_train_mask = generate_edges_labels(homo_edges, lbs, train_idx)
        yelp_graph.edges['homo'].data['label'] = homo_labels
        yelp_graph.edges['homo'].data['train_mask'] = homo_train_mask
        
        dgl.save_graphs(dataset_path+'yelp.dgl', yelp_graph)
        print(f'yelp dataset\'s num nodes:{yelp_graph.num_nodes("r")}, \
            rur edges:{yelp_graph.num_edges("u")}, \
            rtr edges:{yelp_graph.num_edges("t")}, \
            rsr edges:{yelp_graph.num_edges("s")}')
        print(f'Edge train num:{homo_train_mask.sum().item()}, pos num:{(homo_labels[homo_train_mask]==1).sum().item()}')
        
    elif args.dataset == 'amazon':
        # 生成 Amazon 的 DGL 图。
        if os.path.exists(dataset_path+'amazon.dgl'):
            print('dataset amazon has been created')
            sys.exit()
        print('Convert to DGL Graph.')
        amazon_path = first_existing_path(
            os.path.join(DATA_DIR, 'Amazon.mat'),
            os.path.join(DATA_DIR, 'Amazon', 'Amazon.mat')
        )
        amazon = scio.loadmat(amazon_path)
        feats = amazon['features'].todense()
        features = torch.from_numpy(feats).float()
        lbs = amazon['label'][0]
        labels = torch.from_numpy(lbs).long()
        homo = amazon['homo']
        homo = homo+homo.transpose()
        homo = homo.tocoo()
        upu = amazon['net_upu']
        upu = upu+upu.transpose()
        upu = upu.tocoo()
        usu = amazon['net_usu']
        usu = usu+usu.transpose()
        usu = usu.tocoo()
        uvu = amazon['net_uvu']
        uvu = uvu+uvu.transpose()
        uvu = uvu.tocoo()
        
        amazon_graph_structure = {
            ('r','homo','r'):(torch.tensor(homo.row), torch.tensor(homo.col)),
            ('r','p','r'):(torch.tensor(upu.row), torch.tensor(upu.col)),
            ('r','s','r'):(torch.tensor(usu.row), torch.tensor(usu.col)),
            ('r','v','r'):(torch.tensor(uvu.row), torch.tensor(uvu.col))
        }
        amazon_graph = dgl.heterograph(amazon_graph_structure)
        amazon_graph.nodes['r'].data['feature'] = features
        amazon_graph.nodes['r'].data['label'] = labels
        print('Generate dataset partition.')
        train_ratio = 0.4
        test_ratio = 0.67
        index = list(range(3305, len(labels)))
        dataset_l = len(lbs)
        train_idx, rest_idx, train_lbs, rest_lbs = train_test_split(index, lbs[3305:], stratify=lbs[3305:], train_size=train_ratio, random_state=2, shuffle=True)
        valid_idx, test_idx, _,_ = train_test_split(rest_idx, rest_lbs, stratify=rest_lbs, test_size=test_ratio, random_state=2, shuffle=True)
        train_mask = torch.zeros(dataset_l, dtype=torch.bool)
        train_mask[np.array(train_idx)] = True
        valid_mask = torch.zeros(dataset_l, dtype=torch.bool)
        valid_mask[np.array(valid_idx)] = True
        test_mask = torch.zeros(dataset_l, dtype=torch.bool)
        test_mask[np.array(test_idx)] = True
        
        amazon_graph.nodes['r'].data['train_mask'] = train_mask
        amazon_graph.nodes['r'].data['valid_mask'] = valid_mask
        amazon_graph.nodes['r'].data['test_mask'] = test_mask
        
        print('Generate edge labels.')
        homo_edges = amazon_graph.edges(etype='homo')
        homo_labels, homo_train_mask = generate_edges_labels(homo_edges, lbs, train_idx)
        amazon_graph.edges['homo'].data['label'] = homo_labels
        amazon_graph.edges['homo'].data['train_mask'] = homo_train_mask
        
        dgl.save_graphs(dataset_path+'amazon.dgl', amazon_graph)
        print(f'amazon dataset\'s num nodes:{amazon_graph.num_nodes("r")}, \
            upu edges:{amazon_graph.num_edges("p")}, \
            usu edges:{amazon_graph.num_edges("s")}, \
            uvu edges:{amazon_graph.num_edges("v")}')
        print(f'Edge train num:{homo_train_mask.sum().item()}, pos num:{(homo_labels[homo_train_mask]==1).sum().item()}')

    elif args.dataset == 'comp':
        # `comp` 原本就是 DGL 图，因此不需要再做转换。
        comp_path = first_existing_path(
            os.path.join(DATA_DIR, 'comp.dgl'),
            os.path.join(DATA_DIR, 'FDCompCN', 'comp.dgl')
        )
        print(f'FDCompCN graph already exists: {comp_path}')
        print('No preprocessing is required for dataset "comp".')
    else:
        raise ValueError(f'Unsupported dataset: {args.dataset}')

    print('***************endl****************')
