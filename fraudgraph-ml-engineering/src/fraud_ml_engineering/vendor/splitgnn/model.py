"""Core SplitGNN components used by the hybrid fraud pipeline."""

import copy

import dgl
import dgl.function as fn
import numpy as np
import scipy
import sympy
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .utils import hinge_loss
except ImportError:  # pragma: no cover - fallback for local imports
    from utils import hinge_loss


def calculate_theta2(d):
    """Generate polynomial coefficients for the convolution filters."""
    thetas = []
    x = sympy.symbols("x")
    for i in range(d + 1):
        f = sympy.poly((x / 2) ** i * (1 - x / 2) ** (d - i) / (scipy.special.beta(i + 1, d + 1 - i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for j in range(d + 1):
            inv_coeff.append(float(coeff[d - j]))
        thetas.append(inv_coeff)
    return thetas


def sample_balanced_index(pos_index, neg_index):
    """Draw a balanced subset so one side does not dominate the edge loss."""
    if len(pos_index) == 0 or len(neg_index) == 0:
        return None
    sample_size = min(len(pos_index), len(neg_index))
    pos_sample = np.random.choice(pos_index, size=sample_size, replace=False)
    neg_sample = np.random.choice(neg_index, size=sample_size, replace=False)
    index = np.concatenate([pos_sample, neg_sample])
    index.sort()
    return index


class PolyConv(nn.Module):
    """Polynomial convolution block used by SplitGNN."""

    def __init__(
        self,
        in_feats,
        out_feats,
        relation_aware,
        thetas,
        K,
        activation=F.leaky_relu,
        lin=False,
        bias=True,
    ):
        super().__init__()
        self._theta = thetas
        self._k = len(self._theta[0])
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.relation_aware = relation_aware
        self.K = K
        self.linear = nn.Linear(in_feats * len(thetas), out_feats, bias)
        self.linear1 = nn.Linear(in_feats * len(thetas), out_feats, bias)
        self.transh = nn.Linear(in_feats, out_feats, bias)
        self.lin = lin
        self.time_decay = nn.Parameter(torch.tensor(0.75))

    def forward(self, graph, feat):
        def unnLaplacian(input_feat, D_invsqrt, graph, flag):
            graph.ndata["h"] = input_feat * D_invsqrt
            if flag == 0:
                graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            elif flag == 1:
                graph.update_all(self.message_positive, fn.sum("p", "h"))
            elif flag == 2:
                graph.update_all(self.message_negative, fn.sum("n", "h"))
            return input_feat - graph.ndata.pop("h") * D_invsqrt

        with graph.local_scope():
            graph.ndata["feat"] = feat
            graph.apply_edges(self.sign_edges)
            graph.apply_edges(self.judge_edges)

            graph.update_all(message_func=fn.copy_e("positive", "positive"), reduce_func=self.positive_reduce)
            graph.update_all(message_func=fn.copy_e("negative", "negative"), reduce_func=self.negative_reduce)

            positive_in_degrees = graph.ndata["positive_in_degree"]
            negative_in_degrees = graph.ndata["negative_in_degree"]

            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5).unsqueeze(-1).to(feat.device)
            D_invsqrt_positive = torch.pow(positive_in_degrees.float().clamp(min=1), -0.5).unsqueeze(-1).to(feat.device)
            D_invsqrt_negative = torch.pow(negative_in_degrees.float().clamp(min=1), -0.5).unsqueeze(-1).to(feat.device)

            base_feat = graph.ndata["feat"]
            hs_o = []
            hs_p = []
            hs_n = []
            transh = self.transh(feat)

            # Important: each theta branch must start from the original features.
            for theta in self._theta:
                cur_feat = base_feat
                h_o = theta[0] * cur_feat
                for k in range(1, self._k):
                    cur_feat = unnLaplacian(cur_feat, D_invsqrt, graph, 0)
                    h_o += theta[k] * cur_feat
                hs_o.append(h_o)

            for theta in self._theta[0 : self.K + 1]:
                cur_feat = base_feat
                h_p = theta[0] * cur_feat
                for k in range(1, self._k):
                    cur_feat = unnLaplacian(cur_feat, D_invsqrt_positive, graph, 1)
                    h_p += theta[k] * cur_feat
                hs_p.append(h_p)

            for theta in self._theta[self.K + 1 :]:
                cur_feat = base_feat
                h_n = theta[0] * cur_feat
                for k in range(1, self._k):
                    cur_feat = unnLaplacian(cur_feat, D_invsqrt_negative, graph, 2)
                    h_n += theta[k] * cur_feat
                hs_n.append(h_n)

            hs_o = torch.cat(hs_o, dim=1)
            if self.K != len(self._theta) - 1 and self.K != -1:
                hs_p = torch.cat(hs_p, dim=1)
                hs_n = torch.cat(hs_n, dim=1)
                hs_pn = torch.cat([hs_p, hs_n], dim=1)
            elif self.K == -1:
                hs_pn = torch.cat(hs_n, dim=1)
            else:
                hs_pn = torch.cat(hs_p, dim=1)

        if self.lin:
            hs_o = self.linear(hs_o)
            hs_o = self.activation(hs_o)
            hs_pn = self.linear1(hs_pn)
            hs_pn = self.activation(hs_pn)

        return hs_o, hs_pn, transh

    def sign_edges(self, edges):
        src = edges.src["feat"]
        dst = edges.dst["feat"]
        score = self.relation_aware(src, dst)
        return {"sign": torch.sign(score)}

    def judge_edges(self, edges):
        return {"positive": (edges.data["sign"] >= 0).float(), "negative": (edges.data["sign"] < 0).float()}

    def positive_reduce(self, nodes):
        return {"positive_in_degree": nodes.mailbox["positive"].sum(1)}

    def negative_reduce(self, nodes):
        return {"negative_in_degree": nodes.mailbox["negative"].sum(1)}

    def message_positive(self, edges):
        mask = (edges.data["sign"] >= 0).float().view(-1, 1)
        if "log_delta_t" in edges.data:
            log_delta_t = edges.data["log_delta_t"].float().view(-1, 1)
            time_weight = torch.exp(-F.softplus(self.time_decay) * log_delta_t)
        elif "delta_t" in edges.data:
            delta_t = edges.data["delta_t"].float().view(-1, 1)
            time_weight = 1.0 / (1.0 + delta_t)
        else:
            time_weight = torch.ones_like(mask)
        masked_src_feats = edges.src["h"] * mask * time_weight
        return {"p": masked_src_feats}

    def message_negative(self, edges):
        mask = (edges.data["sign"] < 0).float().view(-1, 1)
        if "log_delta_t" in edges.data:
            log_delta_t = edges.data["log_delta_t"].float().view(-1, 1)
            time_weight = torch.exp(-F.softplus(self.time_decay) * log_delta_t)
        elif "delta_t" in edges.data:
            delta_t = edges.data["delta_t"].float().view(-1, 1)
            time_weight = 1.0 / (1.0 + delta_t)
        else:
            time_weight = torch.ones_like(mask)
        masked_src_feats = edges.src["h"] * mask * time_weight
        return {"n": masked_src_feats}


class RelationAware(nn.Module):
    """Edge scoring block used to separate positive and negative relations."""

    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.d_liner = nn.Linear(input_dim, output_dim)
        self.f_liner = nn.Linear(3 * output_dim, 1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, dst):
        src = self.d_liner(src)
        dst = self.d_liner(dst)
        diff = src - dst
        e_feats = torch.cat([src, dst, diff], dim=1)
        e_feats = self.dropout(e_feats)
        score = self.f_liner(e_feats).squeeze(-1)
        score = self.tanh(score)
        return score


class MultiRelationSplitGNNLayer(nn.Module):
    """Build relation-specific branches and fuse them into one SplitGNN layer."""

    def __init__(self, input_dim, output_dim, dataset, dropout, thetas, K, if_sum=False):
        super().__init__()
        self.relation = copy.deepcopy(dataset.etypes)
        self.relation.remove("homo")
        self.n_relation = len(self.relation)
        self.liner = nn.Linear(self.n_relation * output_dim * 3, output_dim)
        self.linear = nn.Linear(input_dim, output_dim)
        self.relation_aware = RelationAware(input_dim, output_dim, dropout)
        self.minelayers = nn.ModuleDict()
        self.dropout = nn.Dropout(dropout)
        for e in self.relation:
            self.minelayers[e] = PolyConv(input_dim, output_dim, self.relation_aware, thetas, K, lin=True)

    def forward(self, g, h):
        # Run each relation branch independently, then concatenate the results.
        hs_o = []
        hs_lh = []
        hs_trans = []
        for e in self.relation:
            h_o, h_lh, h_trans = self.minelayers[e](g.edge_type_subgraph([e]), h)
            hs_o.append(h_o)
            hs_lh.append(h_lh)
            hs_trans.append(h_trans)
        h = torch.cat([torch.cat(hs_o, dim=1), torch.cat(hs_lh, dim=1), torch.cat(hs_trans, dim=1)], dim=1)
        h = self.dropout(h)
        h = self.liner(h)
        return h

    def loss(self, g, h):
        with g.local_scope():
            g.ndata["feat"] = h
            agg_h = self.forward(g, h)

            # ``homo`` edges carry supervision for the edge discrepancy loss.
            g.apply_edges(self.score_edges, etype="homo")
            edges_score = g.edges["homo"].data["score"]
            edge_train_mask = g.edges["homo"].data["train_mask"].bool()
            edge_diff_loss = torch.tensor(0.0, device=agg_h.device)
            if edge_train_mask.any():
                edge_train_label = g.edges["homo"].data["label"][edge_train_mask]
                edge_train_pos = edge_train_label == 1
                edge_train_neg = edge_train_label == -1
                edge_train_pos_index = edge_train_pos.nonzero().flatten().detach().cpu().numpy()
                edge_train_neg_index = edge_train_neg.nonzero().flatten().detach().cpu().numpy()
                index = sample_balanced_index(edge_train_pos_index, edge_train_neg_index)
                if index is not None:
                    edge_train_score = edges_score[edge_train_mask]
                    edge_diff_loss = hinge_loss(edge_train_label[index], edge_train_score[index])

            return agg_h, edge_diff_loss

    def score_edges(self, edges):
        src = edges.src["feat"]
        dst = edges.dst["feat"]
        score = self.relation_aware(src, dst)
        return {"score": score}


class SplitGNN(nn.Module):
    """Full SplitGNN model: encoder plus linear classifier."""

    def __init__(self, args, g):
        super().__init__()
        self.encoder = SplitGNNEncoder(args, g)
        self.n_class = args.n_class
        self.classifier = nn.Linear(self.encoder.output_dim, self.n_class)

    def forward(self, g):
        return self.classifier(self.encoder(g))

    def node_embeddings(self, g):
        return self.encoder(g)

    def loss(self, g):
        # Node classification loss and edge supervision loss are trained together.
        h, edge_loss = self.encoder.forward_with_edge_loss(g)
        logits = self.classifier(h)
        train_mask = g.ndata["train_mask"].bool()
        train_logits = logits[train_mask]
        train_label = g.ndata["label"][train_mask]
        train_pos = (train_label == 1).nonzero().flatten().detach().cpu().numpy()
        train_neg = (train_label == 0).nonzero().flatten().detach().cpu().numpy()
        index = sample_balanced_index(train_pos, train_neg)
        if index is None:
            model_loss = F.cross_entropy(train_logits, train_label)
        else:
            model_loss = F.cross_entropy(train_logits[index], train_label[index])
        return model_loss + self.encoder.gamma * edge_loss


class SplitGNNEncoder(nn.Module):
    """Reusable pure graph encoder for the hybrid model."""

    def __init__(self, args, g):
        super().__init__()
        self.node_type = g.ntypes[0]
        self.input_dim = g.nodes[self.node_type].data["feature"].shape[1]
        self.intra_dim = args.intra_dim
        self.gamma = args.gamma
        self.C = args.C
        self.K = args.K
        self.thetas = calculate_theta2(d=self.C)
        self.mine_layer = MultiRelationSplitGNNLayer(self.intra_dim, self.intra_dim, g, args.dropout, self.thetas, self.K)
        self.linear = nn.Linear(self.input_dim, self.intra_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.LeakyReLU()
        self.output_dim = self.intra_dim

    def forward(self, g):
        # Project node features first, then apply the SplitGNN block.
        feats = g.ndata["feature"].float()
        h = self.linear(feats)
        h = self.mine_layer(g, h)
        h = self.relu(h)
        h = self.dropout(h)
        return h

    def forward_with_edge_loss(self, g):
        # Same as ``forward`` but also returns the auxiliary edge loss.
        feats = g.ndata["feature"].float()
        h = self.linear(feats)
        h, edge_loss = self.mine_layer.loss(g, h)
        h = self.relu(h)
        h = self.dropout(h)
        return h, edge_loss
